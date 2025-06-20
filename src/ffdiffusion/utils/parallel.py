import jax
from typing import Tuple, Optional


def generate_sharding(num_devices: int) -> Tuple[Optional[jax.sharding.Sharding], Optional[jax.sharding.Sharding]]:
    if num_devices <= 1:
        return None, None

    mesh = jax.make_mesh((num_devices,), ("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return data_sharding, replicated_sharding
