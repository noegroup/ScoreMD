import jax.numpy as jnp
import wandb as wandb_lib
from ffdiffusion.data.dataset import ALDPDataset, CGMinipeptideDataset, CoarseGrainingLevel
import matplotlib.pyplot as plt
from ffdiffusion.utils.evaluation import js_divergence, pairwise_distances, phi_psi_metrics
import nglview as nv
import logging
from typing import Callable, Optional
import jax
from ffdiffusion.evaluate.utils import once_on_every_and_all, empty_if_none
from ffdiffusion.rmsd import kabsch_align_many
from tqdm import tqdm
import numpy as onp
from ffdiffusion.simulation import simulate
from ffdiffusion.utils.plots import plot_fes_angles

log = logging.getLogger(__name__)


def evaluate_iid_samples(
    dataset: ALDPDataset | CGMinipeptideDataset,
    ground_truth_samples: jnp.ndarray,
    q_samples: jnp.ndarray,
    write_animation: Callable[[jnp.ndarray, str], None],
    wandb: bool,
    prefix: str,
    out_dir: str,
) -> dict:
    log.info(f"Evaluating iid samples for {prefix}")
    plt.figure(clear=True)
    dataset.plot_phi_psi(q_samples, title=f"Histogram {prefix} (iid)")
    plt.savefig(f"{out_dir}/{prefix}_iid_phi_psi_histogram.png", bbox_inches="tight")
    plt.close()

    target_phi, target_psi = dataset.get_phi_psi(ground_truth_samples)
    sampled_phi, sampled_psi = dataset.get_phi_psi(q_samples)

    plt.figure(clear=True)
    plot_fes_angles(target_phi, dataset.kbT, label="Target")
    plot_fes_angles(sampled_phi, dataset.kbT, label="Sampled", linestyle="--")
    plt.xlabel(r"$\varphi$")
    plt.legend()
    plt.savefig(f"{out_dir}/{prefix}_iid_phi_free_energy.png", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    plot_fes_angles(target_psi, dataset.kbT, label="Target")
    plot_fes_angles(sampled_psi, dataset.kbT, label="Sampled", linestyle="--")
    plt.xlabel(r"$\psi$")
    plt.legend()
    plt.savefig(f"{out_dir}/{prefix}_iid_psi_free_energy.png", bbox_inches="tight")
    plt.close()

    if wandb:
        wandb_lib.save(f"{out_dir}/{prefix}_target_phi_psi.png", base_path=out_dir)
        wandb_lib.log(
            {f"eval/{prefix}_iid_phi_psi_histogram": wandb_lib.Image(f"{out_dir}/{prefix}_iid_phi_psi_histogram.png")}
        )
        wandb_lib.log(
            {f"eval/{prefix}_iid_phi_free_energy": wandb_lib.Image(f"{out_dir}/{prefix}_iid_phi_free_energy.png")}
        )
        wandb_lib.log(
            {f"eval/{prefix}_iid_psi_free_energy": wandb_lib.Image(f"{out_dir}/{prefix}_iid_psi_free_energy.png")}
        )

    try:
        write_animation(q_samples[:64], f"{out_dir}/{prefix}_iid_samples.pdb")

        view = nv.show_file(f"{out_dir}/{prefix}_iid_samples.pdb")
        html_file_path = f"{out_dir}/{prefix}_iid_samples.html"
        nv.write_html(html_file_path, view)

        if wandb:
            wandb_lib.save(f"{out_dir}/{prefix}_iid_samples.pdb", base_path=out_dir)
            wandb_lib.save(f"{out_dir}/{prefix}_iid_samples.html", base_path=out_dir)
    except Exception as e:
        log.warning(f"Failed to write animation: {e}")

    target_phi_psi = jnp.stack([target_phi, target_psi], axis=1)
    phi_psi_samples = jnp.stack([sampled_phi, sampled_psi], axis=1)

    rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(target_phi, target_psi, sampled_phi, sampled_psi)

    return {
        f"eval/{prefix}_iid_js_divergence": js_divergence(target_phi_psi, phi_psi_samples, bins=100),
        f"eval/{prefix}_iid_rms_fe_sq_error": rms_fe_sq_error,
        f"eval/{prefix}_iid_rms_mjs_error": rms_mjs_error,
    }


def simulate_molecule(
    dataset: ALDPDataset | CGMinipeptideDataset,
    force: Callable[[jnp.ndarray], jnp.ndarray],
    initial_positions: jnp.ndarray,
    features: Optional[jnp.ndarray],
    n_samples: int,
    n_steps: int,
    langevin_dt: Optional[float],
    seed: int,
):
    key = jax.random.PRNGKey(seed)
    key, velocity_key = jax.random.split(key)

    n_points = initial_positions.shape[0]
    initial_positions = initial_positions.reshape(n_points, *dataset.sample_shape)
    initial_velocities = jnp.sqrt(dataset.kbT / dataset.mass)[None, ...] * jax.random.normal(
        velocity_key, (n_points, *dataset.sample_shape)
    )

    def adjusted_force(x, features):
        # flatten the input and convert back to (n_atoms, 3)
        return force(x.reshape(1, -1), features).reshape(*dataset.sample_shape)

    @jax.jit
    def multi_step(x, v, key):
        # step each trajectory individually
        trajectory_step = jax.vmap(dataset.create_langevin_step_function(n_steps, force=adjusted_force, dt=langevin_dt))

        # we need to take care of the keys
        keys = jax.random.split(key, n_points)
        return trajectory_step(x, v, keys, features=features)

    trajectories, velocities = simulate(initial_positions, initial_velocities, multi_step, n_samples, key)
    # trajectories now has shape (n_samples + 1, n_points, *dataset.sample_shape), we change that
    trajectories = jnp.swapaxes(trajectories, 0, 1).reshape(n_points, n_samples + 1, -1)
    velocities = jnp.swapaxes(velocities, 0, 1).reshape(n_points, n_samples + 1, -1)
    # new shape is (n_points, n_samples + 1, *dataset.sample_shape)

    # filter out nan values
    nan_entries = jnp.isnan(trajectories).any(axis=(1, 2))
    if nan_entries.sum() > 0:
        log.warning(f"Found {nan_entries.sum()} NaN entries. Filtering them out...")

    return trajectories[~nan_entries], velocities[~nan_entries]


def evaluate_langevin_samples(
    dataset: ALDPDataset | CGMinipeptideDataset,
    baseline: jnp.ndarray,
    trajectories: jnp.ndarray,
    velocities: jnp.ndarray,
    max_num_openmm_evaluations: int,
    write_animation: Callable[[jnp.ndarray, str], None],
    wandb: bool,
    prefix: str,
    out_dir: str,
) -> dict:
    # store the trajectories and velocities
    onp.save(f"{out_dir}/{prefix}_langevin_trajectories.npy", trajectories)
    onp.save(f"{out_dir}/{prefix}_langevin_velocities.npy", velocities)
    log.info(
        f"Saved langevin trajectories and velocities to {out_dir}/{prefix}_langevin_trajectories.npy and {out_dir}/{prefix}_langevin_velocities.npy"
    )

    target_phi, target_psi = dataset.get_phi_psi(baseline)

    initial_positions_phi, initial_positions_psi = dataset.get_phi_psi(trajectories[:, 0])
    plt.figure(clear=True)
    plt.title("Starting positions")
    for i, (phi, psi) in enumerate(zip(initial_positions_phi, initial_positions_psi)):
        plt.plot(phi, psi, markersize=8, marker="*", label=f"Trajectory {prefix} {i}")
    if trajectories.shape[0] < 10:
        plt.legend()
    plt.xlim(-jnp.pi, jnp.pi)
    plt.ylim(-jnp.pi, jnp.pi)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\psi$")
    plt.gca().set_box_aspect(1)
    plt.savefig(f"{out_dir}/{prefix}_langevin_start_positions.png", bbox_inches="tight")
    plt.close()

    if wandb:
        wandb_lib.log(
            {
                f"eval/{prefix}_langevin_start_positions": wandb_lib.Image(
                    f"{out_dir}/{prefix}_langevin_start_positions.png"
                )
            }
        )

    base_frame = trajectories[0, 0]

    @jax.vmap
    def align_trajectory(trajectory):
        return kabsch_align_many(trajectory, base_frame)[0]

    # align each trajectory to the base frame
    trajectories = align_trajectory(trajectories)

    def _plot_phi_psi_histogram(trajectories: jnp.ndarray, i: Optional[int]):
        # plot the phi psi histogram of all trajectories individually and together
        plt.figure(clear=True)
        current_phi, current_psi = dataset.plot_phi_psi(
            trajectories.reshape(-1, *dataset.sample_shape),
            title=f"Histogram {prefix} (simulated){empty_if_none(i, f' - Trajectory {i}')}",
            highlight=None if i is None else [0, -1],
        )
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_psi{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()

        if wandb:
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_psi_histogram{empty_if_none(i, f'_{i}')}": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_psi{empty_if_none(i, f'_{i}')}.png"
                    )
                }
            )

        # now we plot the marginals
        plt.figure(clear=True)
        plt.title(f"$\\varphi$ Histogram {prefix} (simulated) {empty_if_none(i, f' - Trajectory {i}')}")
        plot_fes_angles(target_phi, dataset.kbT, label="Target")
        plot_fes_angles(current_phi, dataset.kbT, label="Sampled", linestyle="--")
        plt.xlabel(r"$\varphi$")
        plt.legend()
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_free_energy{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()

        if wandb and i is None:  # only log full trajectory
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_free_energy": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_free_energy.png"
                    )
                }
            )

        # now we do psi
        plt.figure(clear=True)
        plt.title(f"$\\psi$ Histogram {prefix} (simulated) {empty_if_none(i, f' - Trajectory {i}')}")
        plot_fes_angles(target_psi, dataset.kbT, label="Target")
        plot_fes_angles(current_psi, dataset.kbT, label="Sampled", linestyle="--")
        plt.xlabel(r"$\psi$")
        plt.legend()
        plt.savefig(f"{out_dir}/{prefix}_langevin_psi_free_energy{empty_if_none(i, f'_{i}')}.png", bbox_inches="tight")
        plt.close()

        if wandb and i is None:  # only log full trajectory
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_psi_free_energy": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_psi_free_energy.png"
                    )
                }
            )

    once_on_every_and_all(trajectories, _plot_phi_psi_histogram, desc="Plotting phi psi histogram")

    # see how many transitions are happening
    for i, t in enumerate(trajectories):
        phi, psi = dataset.get_phi_psi(t)
        plt.figure(clear=True)
        plt.title(f"$\\varphi$ over time {prefix} {i}")
        plt.scatter(range(len(phi)), phi, label=f"Trajectory {prefix} {i}", c=psi, s=10)
        plt.xlabel("Step")
        plt.ylabel(r"$\varphi$")
        plt.ylim(-jnp.pi, jnp.pi)
        plt.savefig(f"{out_dir}/{prefix}_langevin_phi_over_t_{i}.png", bbox_inches="tight")
        plt.close()

        if wandb:
            wandb_lib.log(
                {
                    f"eval/{prefix}_langevin_phi_over_t_{i}": wandb_lib.Image(
                        f"{out_dir}/{prefix}_langevin_phi_over_t_{i}.png"
                    )
                }
            )

    target_distances = pairwise_distances(dataset.train.data)

    def _plot_bond_lengths(trajectories: jnp.ndarray, i: Optional[int]):
        distances = pairwise_distances(trajectories)

        for a0, a1 in dataset.bonds:
            a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
            # plot the histogram of the distances
            plt.figure(clear=True)
            plt.title(f"Histogram of distances between {a0_name} and {a1_name} {prefix} {i}")
            plt.hist(target_distances[:, a0, a1], bins=100, density=True, label="Target")
            plt.hist(distances[:, a0, a1], bins=100, density=True, label="Sampled", alpha=0.7)
            plt.xlim(target_distances[:, a0, a1].min(), target_distances[:, a0, a1].max())
            plt.legend()
            plt.savefig(
                f"{out_dir}/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}.png",
                bbox_inches="tight",
            )
            plt.close()

            if wandb:
                wandb_lib.log(
                    {
                        f"eval/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}": wandb_lib.Image(
                            f"{out_dir}/{prefix}_langevin_bond_lengths_{a0}_{a1}{empty_if_none(i, f'_{i}')}.png"
                        )
                    }
                )

    once_on_every_and_all(trajectories, _plot_bond_lengths, desc="Plotting bond lengths")

    if hasattr(dataset, "coarse_graining_level") and dataset.coarse_graining_level == CoarseGrainingLevel.NONE:
        ground_truth_energies = []
        for i, t in enumerate(trajectories):
            ground_truth_energies.append(
                jnp.array(
                    [
                        dataset.energy(x)
                        for x in tqdm(
                            t[:max_num_openmm_evaluations],
                            desc=f"Computing ground truth energies with openmm {i + 1}/{len(trajectories)}",
                        )
                    ]
                )
            )

        if wandb:
            for i in range(len(ground_truth_energies)):
                wandb_lib.define_metric(f"eval/{prefix}_energy_{i}", step_metric=f"eval/{prefix}_langevin_steps")

            for j in range(len(ground_truth_energies[0])):
                logging_dict = {"eval/{prefix}_langevin_steps": j}
                for i in range(len(ground_truth_energies)):
                    logging_dict |= {f"eval/{prefix}_energy_{i}": ground_truth_energies[i][j]}
                wandb_lib.log(logging_dict)

        for i, ground_truth_energy in enumerate(ground_truth_energies):
            plt.figure(clear=True)
            plt.plot(ground_truth_energy)
            plt.xlabel("Step")
            plt.ylabel("Energy")
            plt.savefig(f"{out_dir}/{prefix}_langevin_energy_{i}.png", bbox_inches="tight")
            plt.close()

    for i, t in enumerate(trajectories):
        try:
            write_animation(t[:1000], f"{out_dir}/{prefix}_langevin_samples_{i}.pdb")

            if wandb:
                wandb_lib.save(f"{out_dir}/{prefix}_langevin_samples_{i}.pdb", base_path=out_dir)
        except Exception as e:
            log.warning(f"Failed to write {i}-th animation: {e}")

    target_phi_psi = jnp.stack([*dataset.get_phi_psi(dataset.train.data)], axis=1)
    sampled_phi_psi = jnp.stack([*dataset.get_phi_psi(trajectories.reshape(-1, *dataset.sample_shape))], axis=1)

    rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(
        target_phi, target_psi, sampled_phi_psi[:, 0], sampled_phi_psi[:, 1]
    )

    return {
        f"eval/{prefix}_langevin_js_divergence": js_divergence(target_phi_psi, sampled_phi_psi, bins=100),
        f"eval/{prefix}_langevin_rms_fe_sq_error": rms_fe_sq_error,
        f"eval/{prefix}_langevin_rms_mjs_error": rms_mjs_error,
    }
