def resolve_local_batch_size(global_batch_size: int, world_size: int) -> int:
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}.")
    if global_batch_size < 1:
        raise ValueError(
            f"Global batch size must be >= 1, got {global_batch_size}."
        )
    if global_batch_size < world_size:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be >= world_size "
            f"({world_size}) so each rank receives at least one sample."
        )
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by "
            f"world_size ({world_size}) for equal per-rank batches."
        )

    return global_batch_size // world_size
