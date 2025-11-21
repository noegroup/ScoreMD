from typing import Tuple, Callable
import jax
import jax.numpy as jnp


def create_langevin_step_function(
    force: Callable[[jnp.ndarray], jnp.ndarray], mass: jnp.ndarray, gamma: float, num_steps: int, dt: float, kbT: float
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Implementation of this function is based on the Langevin integrator in OpenMM.
    http://docs.openmm.org/8.2.0/userguide/theory/04_integrators.html#langevinintegator

    Args:
        force: A function defining the forces acting on the system. It takes the current positions and returns the forces.
        For molecular systems the positions the potential takes are in nanometers.
        mass: The mass of the system. For molecular systems this is in atomic mass units.
        gamma: The friction coefficient, for molecular systems this is in units of inverse picoseconds. (i.e., we divide by gamma)
        num_steps: The number of steps that are performed at once.
        dt: The time step, again in ps for molecular systems.
        kbT: The thermal energy, in kJ/mol for molecular systems.

    Returns:
        A function that takes the current positions and velocities and maps them to new positions and velocities.
        For molecular systems all units are in nanometers, ps and kJ/mol.

    """
    assert num_steps >= 1, f"Number of steps should be at least 1. Got {num_steps}."

    def step_single(
        x: jnp.ndarray, v: jnp.ndarray, key: jax.random.PRNGKey, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one step of forward langevin as implemented in openmm."""
        assert x.shape == v.shape, f"Position and velocity should have the same shape. Shape {x.shape} != {v.shape}."
        alpha = jnp.exp(-gamma * dt)
        f_scale = (1 - alpha) / gamma
        new_v_det = alpha * v + f_scale * force(x, **kwargs) / mass
        new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha**2) / mass) * jax.random.normal(key, x.shape)

        return x + dt * new_v, new_v

    if num_steps == 1:
        return step_single

    def step_n(x: jnp.ndarray, v: jnp.ndarray, key: jax.random.PRNGKey, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for _ in range(num_steps):
            key, step_key = jax.random.split(key)
            x, v = step_single(x, v, step_key, **kwargs)
        return x, v

    return step_n


def simulate(
    x0: jnp.ndarray,
    v0: jnp.ndarray,
    step: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    n_steps: int,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    @jax.jit
    def step_fn(carry, _):
        x, v, key = carry
        key, step_key = jax.random.split(key)
        new_x, new_v = step(x, v, step_key)
        return (new_x, new_v, key), (new_x, new_v)

    init_carry = (x0, v0, key)
    _, (trajectory, velocities) = jax.lax.scan(step_fn, init_carry, None, length=n_steps)

    return trajectory, velocities
