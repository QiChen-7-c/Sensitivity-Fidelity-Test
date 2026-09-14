import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

# from libs.tools import get_pos_lst


class _RB2DDataView:
    """Lightweight proxy for accessing full trajectories on demand."""

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def shape(self):
        return self._dataset._data_shape

    def __getitem__(self, idx):
        return self._dataset.get_full_trajectory(int(idx))


class _NpySource:
    def __init__(self, file_path):
        self.file_path = file_path
        self._data = np.load(file_path, mmap_mode="r")
        if self._data.ndim != 5:
            raise ValueError(
                f"Expected 5D data in {file_path}, got shape {self._data.shape}."
            )

    @property
    def shape(self):
        return self._data.shape

    def load_window(self, local_f_id, t_slice, x_slice, y_slice):
        return np.asarray(
            self._data[local_f_id, t_slice, x_slice, y_slice, :],
            dtype=np.float32,
        )

    def load_full(self, local_f_id, time_slice):
        return np.array(self._data[local_f_id, time_slice, :, :, :], copy=True)

    def load_stats_chunk(self, t_slice):
        return np.asarray(self._data[:, t_slice, :, :, :], dtype=np.float64)


class _NpzSource:
    def __init__(self, file_path):
        self.file_path = file_path
        archive = np.load(file_path)
        try:
            self._data = np.stack(
                [archive["p"], archive["b"], archive["u"], archive["v"]],
                axis=-1,
            )
        except KeyError as exc:
            raise KeyError(
                f"Legacy npz file {file_path} must contain p/b/u/v arrays."
            ) from exc
        finally:
            archive.close()

        if self._data.ndim != 5:
            raise ValueError(
                f"Expected stacked npz data to be 5D in {file_path}, got shape {self._data.shape}."
            )

    @property
    def shape(self):
        return self._data.shape

    def load_window(self, local_f_id, t_slice, x_slice, y_slice):
        return np.asarray(
            self._data[local_f_id, t_slice, x_slice, y_slice, :],
            dtype=np.float32,
        )

    def load_full(self, local_f_id, time_slice):
        return np.array(self._data[local_f_id, time_slice, :, :, :], copy=True)

    def load_stats_chunk(self, t_slice):
        return np.asarray(self._data[:, t_slice, :, :, :], dtype=np.float64)

class RB2D_Dataset(Dataset):
    """ Rayleigh-Bernard 2D Dataset with Different Initial Conditions.
    Loads two clips clip_t1 and clip_t2 a simulation.
    clip_t1 is from time t1 and clip_t2 from time t2, where t1 < t2.
    The two clips are spatially aligned.
    """
    def __init__(self, data_folder, data_filenames, nx, ny, nt_in, nt_out, stride_t, normalize_channels,
        subsample_rate=1, BC=True):
        """
        Args:
          data_folder: string, path to the data folder
          data_filenames: list of strings, a list of .npz data filenames. The files must have the same data shape.
          nx: int, number of 'pixels' in x dimension.
          nz: int, number of 'pixels' in z dimension.
          nt_in: int, number of frames in time of input.
          nt_out: int, number of frames in time of output.
          stride_t: int, number of frames between the first frame of each clip. stride_t=nt means that the clips are non-overlapping and next to each other.
          normalize_channels: bool, whether to normalize the range of each channel to [0, 1].
          subsample_rate: int, subsample rate. 1 means use all frames. 2 means take every other frame
          BC: bool, whether to apply different Boundary Conditions
        """
        self.data_folder = data_folder
        self.data_filenames = data_filenames
        self.nx = nx
        self.ny = ny
        self.nt_in = nt_in
        self.nt_out = nt_out
        self.stride_t = stride_t
        self.normalize_channels = normalize_channels
        self.subsample_rate = subsample_rate
        self.BC = BC  # whether to apply BC

        self._arrays, raw_shape = RB2D_Dataset._open_data_files(data_folder, data_filenames)
        nf_data, nt_data, nx_data, ny_data, nc_data = raw_shape

        # load coordinates (before any temporal slicing)
        self.x_pos, self.y_pos, self.t_coord = self._load_coordinates(
            data_folder=data_folder,
            nx=nx_data,
            ny=ny_data,
            nt=nt_data,
        )

        # keep only frames with time >= 10 (with tolerance for float precision)
        min_time = 10.0
        time_tol = 1e-4
        t_np = self.t_coord.cpu().numpy()
        time_mask = t_np >= (min_time - time_tol)
        if not np.any(time_mask):
            raise ValueError("No frames satisfy time >= 12.")

        self._time_start = int(np.flatnonzero(time_mask)[0])
        self._time_slice = slice(self._time_start, None, self.subsample_rate)
        self.t_coord = self.t_coord[self._time_slice]
        nt_data = int(self.t_coord.shape[0])

        # assert nx, nz, nt, and stride_t are viable
        if (nx > nx_data) or (ny > ny_data) or (nt_in + nt_out + stride_t > nt_data):
            raise ValueError('Resolution in each spatial temporal dimension x ({}), z({}), t_in + t_out + stride_t ({} + {} +{})'
                             'must not exceed dataset limits x ({}) z ({}) t ({})'.format(
                                 nx, ny, nt_in, nt_out, stride_t, nx_data, ny_data, nt_data))

        self.nf_start_range = np.arange(nf_data)
        self.nx_start_range = np.arange(0, nx_data-nx+1)
        self.ny_start_range = np.arange(0, ny_data-ny+1)
        self.nt_start_range = np.arange(0, nt_data-max(nt_in, nt_out)-stride_t+1) # t start from 100 to get stable simulation data
        self.rand_grid = np.stack(np.meshgrid(self.nf_start_range,
                                              self.nt_start_range,
                                              self.ny_start_range,
                                              self.nx_start_range, indexing='ij'), axis=-1)
        self.rand_start_id = self.rand_grid.reshape([-1, 4])

        self._data_shape = (nf_data, nt_data, nx_data, ny_data, nc_data)
        if self.normalize_channels:
            self._mean, self._std = self._load_or_compute_stats()
        else:
            self._mean = np.zeros((nc_data,), dtype=np.float32)
            self._std = np.ones((nc_data,), dtype=np.float32)

        # # positional embedding. (may need to be updated for time dimension)
        # size_lst = [(nx, ny)]
        # length = [3.0, 1.0]  # Rayleigh-Bénard length-to-height ratio
        # self.pos_lst = get_pos_lst(size_lst, length)[0]
        # length = [3.0, 1.0]  # Rayleigh-Bénard length-to-height
        # self.x_pos = torch.from_numpy(np.linspace(0, length[0], nx, endpoint=False))
        # self.y_pos = torch.from_numpy(np.linspace(0, length[1], ny))
        # self.t_coord = torch.from_numpy(np.linspace(0, 60, 600, endpoint=False)[::self.subsample_rate])  # assuming total 600 frames in 60 time units

        self.data = _RB2DDataView(self)
        print(f'data.shape [f, t, x, y, c] = {torch.Size(self._data_shape)}')
        print(f'len(dataset): {len(self)}')

    def __len__(self):
        return self.rand_start_id.shape[0]

    def __getitem__(self, idx):
        """Loads the pair of clips, clip_t1 and clip_t2, corresponding to idx.
        clip_t1 is from time t1 and clip_t2 from time t2, where t1 < t2.
        The two clips are spatially aligned.

        Args:
          idx: int, index of the crop to return. must be smaller than len(self).

        Returns:
          clip_t1: array of shape [nt_in, nz, nx, channel].
          clip_t2: array of shape [nt_out, nz, nx, channel], where 4 are the phys channels pbuw.
          grid_point_with_coord: array of shape [nt, nz, nx, 3], where 3 are the the t-z-x coordinates normalized between [0,1].
        """
        f_id, t_id, y_id, x_id = self.rand_start_id[idx]
        file_idx, local_f_id = self._resolve_file_index(int(f_id))
        raw_t_start = self._time_start + int(t_id) * self.subsample_rate
        x_slice = slice(int(x_id), int(x_id) + self.nx)
        y_slice = slice(int(y_id), int(y_id) + self.ny)

        clip_t1 = self._load_window(
            file_idx=file_idx,
            local_f_id=local_f_id,
            t_start=raw_t_start,
            length=self.nt_in,
            x_slice=x_slice,
            y_slice=y_slice,
        )
        clip_t2 = self._load_window(
            file_idx=file_idx,
            local_f_id=local_f_id,
            t_start=raw_t_start + self.stride_t * self.subsample_rate,
            length=self.nt_out,
            x_slice=x_slice,
            y_slice=y_slice,
        )
        t_coord = self.t_coord[t_id:t_id+self.stride_t+self.nt_out].to(dtype=torch.float32)
        x_coord = self.x_pos[x_id:x_id+self.nx].to(dtype=torch.float32)
        y_coord = self.y_pos[y_id:y_id+self.ny].to(dtype=torch.float32)

        if self.normalize_channels:
            clip_t1 = self.normalize_grid(clip_t1)
            clip_t2 = self.normalize_grid(clip_t2)
        if self.BC == False:
            return clip_t1, clip_t2, t_coord, x_coord, y_coord  
        else:
            top_bc = clip_t1[:, :, 0, 0]        # bc of temperature
            bottom_bc = clip_t1[:, :, -1, 0]
            return clip_t1, clip_t2, t_coord, x_coord, y_coord, top_bc, bottom_bc


    @staticmethod
    def _open_data_files(data_folder, data_filenames):
        arrays = []
        reference_shape = None
        total_samples = 0

        for data_filename in data_filenames:
            file_path = os.path.join(data_folder, data_filename)
            suffix = Path(file_path).suffix.lower()
            if suffix == ".npy":
                data = _NpySource(file_path)
            elif suffix == ".npz":
                data = _NpzSource(file_path)
            else:
                raise ValueError(
                    f"Unsupported data file format '{suffix}' for {file_path}. "
                    "Expected .npy or legacy .npz."
                )

            if reference_shape is None:
                _, nt, nx, ny, nc = data.shape
                reference_shape = (nt, nx, ny, nc)
            else:
                if data.shape[1:] != reference_shape:
                    raise ValueError(
                        "All data files must have the same [t, x, y, c] shape. "
                        f"Expected {reference_shape}, got {data.shape[1:]} for {file_path}."
                    )

            arrays.append(data)
            total_samples += int(data.shape[0])

        if reference_shape is None:
            raise ValueError("No data files were provided.")

        nt, nx, ny, nc = reference_shape
        return arrays, (total_samples, nt, nx, ny, nc)
    

    @staticmethod
    def _load_coordinates(data_folder, nx, ny, nt):
        """Load spatial (x/z) and temporal coordinates if available."""
        x_path = os.path.join(data_folder, "x.npy")
        z_path = os.path.join(data_folder, "z.npy")
        t_path = os.path.join(data_folder, "time.npy")
        if not all(os.path.exists(p) for p in (x_path, z_path, t_path)):
            raise FileNotFoundError(
                f"Expected coordinate files x.npy, z.npy, time.npy in {data_folder}."
            )

        x_pos_np = np.load(x_path)
        z_pos_np = np.load(z_path)
        t_coord_np = np.load(t_path)
        if x_pos_np.shape[0] != nx or z_pos_np.shape[0] != ny:
            raise ValueError(
                f"Coordinate length mismatch: x {x_pos_np.shape[0]} vs {nx}, z {z_pos_np.shape[0]} vs {ny}."
            )
        if t_coord_np.shape[0] != nt:
            raise ValueError(
                f"Time coordinate length mismatch: {t_coord_np.shape[0]} vs {nt}."
            )
        return (
            torch.from_numpy(np.asarray(x_pos_np)),
            torch.from_numpy(np.asarray(z_pos_np)),
            torch.from_numpy(np.asarray(t_coord_np)),
        )

    def _resolve_file_index(self, f_id):
        running = 0
        for file_idx, array in enumerate(self._arrays):
            next_running = running + int(array.shape[0])
            if f_id < next_running:
                return file_idx, f_id - running
            running = next_running
        raise IndexError(f"f_id {f_id} is out of range for dataset of size {self._data_shape[0]}.")

    def _load_window(self, file_idx, local_f_id, t_start, length, x_slice, y_slice):
        array = self._arrays[file_idx]
        t_stop = t_start + length * self.subsample_rate
        window = array.load_window(
            local_f_id=local_f_id,
            t_slice=slice(t_start, t_stop, self.subsample_rate),
            x_slice=x_slice,
            y_slice=y_slice,
        )
        return torch.from_numpy(window)

    def get_full_trajectory(self, f_id):
        file_idx, local_f_id = self._resolve_file_index(int(f_id))
        full = self._arrays[file_idx].load_full(local_f_id, self._time_slice)
        return torch.from_numpy(full)

    def _stats_cache_path(self):
        signature = hashlib.sha1()
        signature.update(os.path.abspath(self.data_folder).encode("utf-8"))
        for name in self.data_filenames:
            signature.update(name.encode("utf-8"))
        signature.update(str(self._time_start).encode("utf-8"))
        signature.update(str(self.subsample_rate).encode("utf-8"))
        return Path(self.data_folder) / f".rb2d_stats_{signature.hexdigest()[:16]}.npz"

    def _load_or_compute_stats(self):
        cache_path = self._stats_cache_path()
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached["mean"], cached["std"]

        nc = self._data_shape[-1]
        sum_channels = np.zeros((nc,), dtype=np.float64)
        sum_sq_channels = np.zeros((nc,), dtype=np.float64)
        sample_count = 0
        chunk_size = 16

        for array in self._arrays:
            nt_raw = int(array.shape[1])
            step = self.subsample_rate * chunk_size
            for raw_start in range(self._time_start, nt_raw, step):
                raw_stop = min(nt_raw, raw_start + step)
                chunk = array.load_stats_chunk(
                    slice(raw_start, raw_stop, self.subsample_rate)
                )
                if chunk.size == 0:
                    continue
                sum_channels += np.sum(chunk, axis=(0, 1, 2, 3))
                sum_sq_channels += np.sum(np.square(chunk), axis=(0, 1, 2, 3))
                sample_count += int(np.prod(chunk.shape[:-1]))

        if sample_count == 0:
            raise ValueError("Failed to compute normalization statistics: empty sample set.")

        mean = sum_channels / sample_count
        var = np.maximum(sum_sq_channels / sample_count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)

        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp_path, mean=mean, std=std)
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

        return mean, std


    @staticmethod
    def _normalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            mean_val = torch.as_tensor(mean, device=array.device, dtype=array.dtype)
            std_val = torch.as_tensor(std, device=array.device, dtype=array.dtype)
        else:
            mean_val = np.asarray(mean, dtype=array.dtype)
            std_val = np.asarray(std, dtype=array.dtype)
        return (array - mean_val) / std_val

    @staticmethod
    def _denormalize_array(array, mean, std):
        """denormalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            mean_val = torch.as_tensor(mean, device=array.device, dtype=array.dtype)
            std_val = torch.as_tensor(std, device=array.device, dtype=array.dtype)
        else:
            mean_val = np.asarray(mean, dtype=array.dtype)
            std_val = np.asarray(std, dtype=array.dtype)
        return array * std_val + mean_val

    @property
    def channel_mean(self):
        """channel-wise mean of dataset."""
        return self._mean

    @property
    def channel_std(self):
        """channel-wise mean of dataset."""
        return self._std

    def normalize_grid(self, grid):
        """Normalize grid.

        Args:
          grid: np array or torch tensor of shape [...,3], 3 are the num. of phys channels.
        Returns:
          channel normalized grid of same shape as input.
        """
        expand_shape = (1,) * (grid.ndim - 1) + (self.channel_mean.shape[0],)
        mean_bc = self.channel_mean.reshape(expand_shape)
        std_bc = self.channel_std.reshape(expand_shape)
        return self._normalize_array(grid, mean_bc, std_bc)

    def denormalize_grid(self, grid):
        """Denormalize grid.

        Args:
          grid: np array or torch tensor of shape [...,3], 3 are the num. of phys channels.
        Returns:
          channel denormalized grid of same shape as input.
        """
        expand_shape = (1,) * (grid.ndim - 1) + (self.channel_mean.shape[0],)
        mean_bc = self.channel_mean.reshape(expand_shape)
        std_bc = self.channel_std.reshape(expand_shape)
        return self._denormalize_array(grid, mean_bc, std_bc)
    

