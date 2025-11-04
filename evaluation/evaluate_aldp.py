from itertools import chain
import jax
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as onp
import os
from dataclasses import dataclass
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from scoremd.utils.evaluation import pairwise_distances
from scoremd.utils.plots import plot_fes_angles
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm


@dataclass
class DataSource:
    path: str
    file_name_prefix: str
    include_in_comparisons: bool
    plot_ramachandran: bool = False
    linestyle: str = "--"

    def __post_init__(self):
        self.path = os.path.expanduser(self.path)


data_sources: dict[str, DataSource] = {
    "Diffusion": DataSource(
        path="./multirun/2025-05-01/11-15-15/0",
        file_name_prefix="baseline",
        include_in_comparisons=True,
    ),
    "Conservative Diffusion Model": DataSource(
        path="./multirun/2025-05-01/11-15-15/0",
        file_name_prefix="baseline",
        include_in_comparisons=False,
        plot_ramachandran=True,
    ),
    "Score-based Diffusion Model": DataSource(
        path="./outputs/aldp/2025-04-26/10-03-51",
        file_name_prefix="baseline_score",
        include_in_comparisons=False,
        plot_ramachandran=True,
    ),
    "Two For One": DataSource(
        path="./multirun/2025-05-01/11-14-58/0",
        file_name_prefix="two_for_one",
        include_in_comparisons=True,
    ),
    "Mixture": DataSource(
        path="./multirun/2025-05-01/11-18-52/0",
        file_name_prefix="mixture",
        include_in_comparisons=True,
    ),
    "Fokker-Planck": DataSource(
        path="./multirun/2025-06-19/09-24-45/0",
        file_name_prefix="fp",
        include_in_comparisons=True,
    ),
    "Both": DataSource(
        path="./multirun/2025-07-02/11-19-08/0",
        file_name_prefix="both",
        include_in_comparisons=True,
    ),
}

iid_path = "aldp_iid_samples.npy"
langevin_path = "aldp_langevin_trajectories.npy"
scalar_fp_path = "scalar_fp_error.npy"
out_dir = os.path.expanduser("./evaluation/aldp")
bins = 60
bw_method = 0.05

dataset = ALDPDataset(coarse_graining_level=CoarseGrainingLevel.FULL, limit_samples=50_000, validation=False)
dataset_full = ALDPDataset(coarse_graining_level=CoarseGrainingLevel.FULL, validation=False)
target_phi, target_psi = dataset.get_2d_features(dataset.train.data)

plt.style.use("tableau-colorblind10")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

auto_extension = "pdf"
plot_free_energy_bar = False


def get_method_title(method_name):
    if method_name == "iid":
        return "iid"
    elif method_name == "langevin":
        return "sim"

    raise NotImplementedError(f"Unknown method name: {method_name}")


def plot_samples(data_sources, samples, prefix, vmin, vmax):
    # plot ramachandran plots
    plt.figure(clear=True)
    dataset.plot_2d(
        dataset.train.data,
        title="Reference",
        cmap="turbo_r",
        bins=bins,
        vmin=vmin,
        vmax=vmax,
        free_energy_bar=plot_free_energy_bar,
    )

    plt.savefig(f"{out_dir}/reference_phi_psi.{auto_extension}", bbox_inches="tight")
    plt.close()

    for name, data_info in data_sources.items():
        plt.figure(clear=True)
        dataset.plot_2d(
            samples[name],
            title=f"{name} ({get_method_title(prefix)})",
            cmap="turbo_r",
            bins=bins,
            vmin=vmin,
            vmax=vmax,
            free_energy_bar=plot_free_energy_bar,
        )

        plt.savefig(f"{out_dir}/{data_info.file_name_prefix}_{prefix}_phi_psi.{auto_extension}", bbox_inches="tight")
        if data_info.plot_ramachandran:
            plt.title(f"{name} ({get_method_title(prefix)})")
            plt.savefig(f"{out_dir}/{data_info.file_name_prefix}_{prefix}_phi_psi.pdf", bbox_inches="tight")
        plt.close()

    # plot all free energies in a single plot
    offset = 0
    plt.figure(clear=True)
    plot_fes_angles(target_phi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5)
    for i, (name, data_info) in enumerate(data_sources.items()):
        if not data_info.include_in_comparisons:
            offset += 1
            continue

        sampled_phi, sampled_psi = dataset.get_2d_features(samples[name])
        plot_fes_angles(
            sampled_phi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--", color=colors[i - offset]
        )

    plt.ylim(top=35)
    plt.xlabel(r"$\varphi$")
    plt.legend()
    plt.title(r"Free energy projection of $\varphi$")
    plt.savefig(f"{out_dir}/{prefix}_phi_free_energy.pdf", bbox_inches="tight")
    plt.close()

    # we now do a slim verison of the free energy plot
    offset = 0
    phi_fes = {}
    plt.figure(clear=True, figsize=(6.4, 2.8))
    grid, phi_fes["reference"] = plot_fes_angles(
        target_phi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5
    )
    for i, (name, data_info) in enumerate(data_sources.items()):
        if not data_info.include_in_comparisons:
            offset += 1
            continue

        sampled_phi, sampled_psi = dataset.get_2d_features(samples[name])
        grid, phi_fes[name] = plot_fes_angles(
            sampled_phi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--", color=colors[i - offset]
        )

    plt.ylim(top=35)
    plt.xlabel(r"$\varphi$")
    plt.savefig(f"{out_dir}/{prefix}_phi_free_energy_slim.{auto_extension}", bbox_inches="tight")
    plt.close()

    # now we plot the free energy differences
    plt.figure(clear=True, figsize=(6.4, 2.8))
    plt.plot(grid, 0 * phi_fes["reference"], label="Reference", color="black", linewidth=5)
    offset = 0
    for i, (name, data_info) in enumerate(data_sources.items()):
        if not data_info.include_in_comparisons:
            offset += 1
            continue

        plt.plot(
            grid,
            phi_fes[name] - phi_fes["reference"],
            label=name,
            linestyle="--",
            linewidth=3,
            color=colors[i - offset],
        )

    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\Delta$ Energy / $k_BT$")
    plt.xticks(
        [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    plt.ylim(bottom=-10, top=10)
    plt.savefig(f"{out_dir}/{prefix}_phi_free_energy_diff.{auto_extension}", bbox_inches="tight")
    plt.close()

    offset = 0
    plt.figure(clear=True)
    plot_fes_angles(target_psi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5)
    for i, (name, data_info) in enumerate(data_sources.items()):
        if not data_info.include_in_comparisons:
            offset += 1
            continue

        sampled_phi, sampled_psi = dataset.get_2d_features(samples[name])
        plot_fes_angles(
            sampled_psi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--", color=colors[i - offset]
        )

    plt.xlabel(r"$\psi$")
    plt.legend()
    plt.savefig(f"{out_dir}/{prefix}_psi_free_energy.svg", bbox_inches="tight")
    plt.title(r"Free energy projection of $\psi$")
    plt.savefig(f"{out_dir}/{prefix}_psi_free_energy.pdf", bbox_inches="tight")
    plt.close()

    # look at bonds
    for bond_index, (a0, a1) in enumerate(dataset.bonds):
        a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
        ground_truth_bonds = pairwise_distances(dataset_full.train.data)[:, a0, a1] * 10  # convert to angstroms
        grid = jnp.linspace(ground_truth_bonds.min(), ground_truth_bonds.max(), 100)

        offset = 0
        plt.figure(clear=True)
        # plt.hist(ground_truth_bonds, bins=100, density=True, color=colors[0], rasterized=True)
        plt.plot(grid, gaussian_kde(ground_truth_bonds)(grid), label="Reference", linewidth=5, color="black")
        for i, (name, data_info) in enumerate(data_sources.items()):
            if not data_info.include_in_comparisons:
                offset += 1
                continue

            current_bonds = pairwise_distances(samples[name])[:, a0, a1] * 10  # convert to angstroms
            grid = jnp.linspace(
                min(current_bonds.min(), ground_truth_bonds.min()),
                max(current_bonds.max(), ground_truth_bonds.max()),
                100,
            )

            plt.plot(
                grid,
                gaussian_kde(current_bonds)(grid),
                label=name,
                linewidth=3,
                linestyle="--",
                color=colors[i - offset],
            )

        if bond_index == 0:
            plt.legend()
            plt.savefig(f"{out_dir}/{prefix}_bond_lengths_{bond_index}_slim.pdf", bbox_inches="tight")

        plt.title(f"${a0_name}-{a1_name}$ bond length")
        plt.xlabel(r"Length in $\AA$")
        plt.savefig(f"{out_dir}/{prefix}_bond_lengths_{bond_index}.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    mpl.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.labelsize": 18}
    )  # We only use this labelsize for standalone plots, not the ones I created manually

    # for the big figure
    # mpl.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 24})
    # auto_extension = "svg"
    # plot_free_energy_bar = False

    iid_samples = {}
    langevin_samples = {}
    scalar_fp_error = {}

    os.makedirs(out_dir, exist_ok=True)

    for name, data_info in tqdm(data_sources.items(), desc="Loading data"):
        data = onp.load(os.path.join(data_info.path, "out", iid_path))
        iid_samples[name] = data
        num_iid_samples = data.shape[0]
        data = onp.load(os.path.join(data_info.path, "out", langevin_path))
        data = data.reshape(-1, *dataset.sample_shape)
        # random permuation of data with jax, and limit to 1000 samples
        data = jax.random.permutation(jax.random.PRNGKey(0), data)[:num_iid_samples]
        langevin_samples[name] = data

        if os.path.exists(os.path.join(data_info.path, "out", scalar_fp_path)):
            data = onp.load(os.path.join(data_info.path, "out", scalar_fp_path))
            scalar_fp_error[name] = data

    # figure out vmin and vmax for histogram plots
    vmin, vmax = float("inf"), float("-inf")
    for d in chain(iid_samples.values(), langevin_samples.values(), [dataset.train.data]):
        phi, psi = dataset.get_2d_features(d)
        H, _, _ = jnp.histogram2d(phi, psi, bins=bins)
        # Update min/max values directly
        nonzero_min = H[H > 0].min() if jnp.any(H > 0) else vmin
        vmin = min(vmin, nonzero_min)
        vmax = max(vmax, H.max())

    plot_samples(data_sources, iid_samples, "iid", vmin, vmax)
    plot_samples(data_sources, langevin_samples, "langevin", vmin, vmax)

    # plot fp error

    plt.figure(clear=True)
    for name, error in scalar_fp_error.items():
        if not data_sources[name].include_in_comparisons:
            continue

        error = jnp.where(jnp.abs(error) < 1e-5, jnp.nan, error)
        plt.plot(
            jnp.linspace(0, 1, len(error) - 2),
            error[1:-1],
            label=name,
            linewidth=3,
            linestyle=data_sources[name].linestyle,
        )
    plt.legend()
    plt.yscale("log")
    plt.xlabel(r"Diffusion time $t$")
    plt.ylabel("Fokker-Planck Error")
    plt.xlim(0, 1)
    plt.savefig(f"{out_dir}/scalar_fp_error.pdf", bbox_inches="tight")
    plt.close()
