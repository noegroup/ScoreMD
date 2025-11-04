from itertools import chain
import jax
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as onp
import os
from dataclasses import dataclass
from scoremd.data.dataset.minipeptide import CGMinipeptideDataset
from scoremd.utils.evaluation import js_divergence, pairwise_distances, phi_psi_metrics
from scoremd.utils.file import get_persistent_storage
from scoremd.utils.plots import plot_fes_angles
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm
import json


@dataclass
class DataSource:
    path: str
    file_name_prefix: str
    supports_langevin: bool = True

    def __post_init__(self):
        self.path = os.path.expanduser(self.path)


data_sources: dict[str, DataSource] = {
    "Diffusion": DataSource(
        path="./outputs/minipeptides/2025-05-04/23-52-31",
        file_name_prefix="baseline",
    ),
    "Two For One": DataSource(
        path="./outputs/minipeptides/2025-05-05/02-12-06",
        file_name_prefix="two_for_one",
    ),
    "Mixture": DataSource(
        path="./outputs/minipeptides/2025-05-05/15-02-30",
        file_name_prefix="mixture",
    ),
    "Fokker-Planck": DataSource(
        path="./outputs/minipeptides/2025-07-05/14-14-18",
        file_name_prefix="fp",
    ),
    "Both": DataSource(
        path="./outputs/minipeptides/2025-06-19/08-52-27",
        file_name_prefix="both",
    ),
    "Transferable BG": DataSource(
        path="../minipeptides/tbg",
        file_name_prefix="tbg",
        supports_langevin=False,
    ),
}

out_dir = os.path.expanduser("./evaluation/minipeptides")
bins = 60
bw_method = 0.05
limit_samples = 49_000

peptides = ["KS", "AT", "LW", "KQ", "NY", "IM", "TD", "HT", "NF", "RL", "ET", "AC", "RV", "HP", "AP"]

plt.style.use("tableau-colorblind10")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

main_figure_peptide = "KS"

base_dir = get_persistent_storage()
dataset = CGMinipeptideDataset(
    pdb_directory=f"{base_dir}/minipeptides/pdbs",
    train_path=f"{base_dir}/minipeptides/train.npy",
    val_path=f"{base_dir}/minipeptides/val.npy",
    test_path=f"{base_dir}/minipeptides/test.npy",
    name="minipeptides",
    limit_peptides=peptides,
)


def load_numpy(path):
    loaded = onp.load(path)

    if path.endswith(".npz"):
        loaded = loaded["samples_np"]

    return loaded.reshape(-1, 30)


def plot_peptide(samples, peptide, prefix, vmin, vmax, extension="svg", title=None):
    plt.figure(clear=True)
    dataset.plot_2d(samples, title=title, cmap="turbo_r", bins=bins, vmin=vmin, vmax=vmax)
    plt.savefig(f"{out_dir}/{peptide}/{prefix}_phi_psi.{extension}", bbox_inches="tight")
    plt.close()


def plot_samples(data_sources, samples, reference, prefix, vmin, vmax):
    for peptide in peptides:
        os.makedirs(os.path.join(out_dir, peptide), exist_ok=True)
        plot_peptide(
            reference[peptide],
            peptide,
            "reference",
            vmin[peptide],
            vmax[peptide],
            extension="svg" if peptide == main_figure_peptide else "pdf",
            title=None if peptide == main_figure_peptide else "Reference",
        )
        for name, data_info in data_sources.items():
            plot_peptide(
                samples[name][peptide],
                peptide,
                f"{data_info.file_name_prefix}_{prefix}",
                vmin[peptide],
                vmax[peptide],
                extension="svg" if peptide == main_figure_peptide else "pdf",
                title=None if peptide == main_figure_peptide else name,
            )

        target_phi, target_psi = dataset.get_2d_features(reference[peptide])
        # free energy plot
        plt.figure(clear=True)
        plot_fes_angles(target_phi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5)
        for name, data_info in data_sources.items():
            sampled_phi, _ = dataset.get_2d_features(samples[name][peptide])
            plot_fes_angles(sampled_phi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--")

        if peptide == "NF" or peptide == "NY" or peptide == "AC" or peptide == "RV" or peptide == "ET":
            plt.ylim(top=40)

        plt.xlabel(r"$\varphi$")
        plt.title(r"Free energy projection of $\varphi$")
        plt.savefig(f"{out_dir}/{peptide}/{prefix}_phi_free_energy.pdf", bbox_inches="tight")
        plt.close()

        if peptide == main_figure_peptide:
            # we do a special free energy plot, that is a bit smaller
            phi_fes = {}
            plt.figure(clear=True, figsize=(6.4, 2.8))
            grid, phi_fes["reference"] = plot_fes_angles(
                target_phi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5
            )
            for name, data_info in data_sources.items():
                sampled_phi, _ = dataset.get_2d_features(samples[name][peptide])
                grid, phi_fes[name] = plot_fes_angles(
                    sampled_phi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--", linewidth=3
                )

            plt.ylim(top=35)
            plt.xlabel(r"$\varphi$")
            plt.savefig(f"{out_dir}/{peptide}/{prefix}_phi_free_energy_slim.svg", bbox_inches="tight")
            plt.close()

            # now we plot the free energy differences
            # model - reference, where reference should be plotted the same way, but with black as we had before
            plt.figure(clear=True, figsize=(6.4, 2.8))
            plt.plot(grid, 0 * phi_fes["reference"], label="Reference", color="black", linewidth=5)
            for name, data_info in data_sources.items():
                plt.plot(grid, phi_fes[name] - phi_fes["reference"], label=name, linestyle="--", linewidth=3)

            plt.xlabel(r"$\varphi$")
            plt.ylabel(r"$\Delta$ Energy / $k_BT$")
            plt.xticks(
                [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
                [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
            )

            plt.ylim(bottom=-10, top=10)
            plt.xlim(-jnp.pi, jnp.pi)
            plt.savefig(f"{out_dir}/{peptide}/{prefix}_phi_free_energy_diff.svg", bbox_inches="tight")
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=18)
            plt.savefig(f"{out_dir}/{peptide}/{prefix}_phi_free_energy_diff_legend.svg", bbox_inches="tight")
            plt.close()

        plt.figure(clear=True)
        plot_fes_angles(target_psi, dataset.kbT, bw_method=bw_method, label="Reference", color="black", linewidth=5)
        for name, data_info in data_sources.items():
            _, sampled_psi = dataset.get_2d_features(samples[name][peptide])
            plot_fes_angles(sampled_psi, dataset.kbT, bw_method=bw_method, label=name, linestyle="--")
        plt.xlabel(r"$\psi$")

        # save for the main figure
        if peptide == main_figure_peptide:
            legend = plt.legend()
            plt.savefig(f"{out_dir}/{peptide}/{prefix}_psi_free_energy.svg", bbox_inches="tight")
            legend.remove()

        plt.title(r"Free energy projection of $\psi$")
        plt.savefig(f"{out_dir}/{peptide}/{prefix}_psi_free_energy.pdf", bbox_inches="tight")
        plt.close()

        # plot CA-CA distance
        ca_distance = pairwise_distances(reference[peptide])[:, 1, 6] * 10  # convert to angstroms
        plt.figure(clear=True)
        # plt.hist(ca_distance, bins=150, density=True, color=colors[0], rasterized=True)
        xlim = (3.4, 4.4)
        grid = jnp.linspace(*xlim, 150)
        plt.plot(grid, gaussian_kde(ca_distance)(grid), label="Reference", linewidth=5, color="black")

        for i, (name, data_info) in enumerate(data_sources.items()):
            sampled_ca_distance = pairwise_distances(samples[name][peptide])[:, 1, 6] * 10  # convert to angstroms

            # filter out values outside the xlim
            if ((sampled_ca_distance >= xlim[0]) & (sampled_ca_distance <= xlim[1])).sum() > 0:
                sampled_ca_distance = sampled_ca_distance[
                    (sampled_ca_distance >= xlim[0]) & (sampled_ca_distance <= xlim[1])
                ]

            plt.plot(
                grid, gaussian_kde(sampled_ca_distance)(grid), label=name, linewidth=3, color=colors[i], linestyle="--"
            )

        plt.legend()
        plt.xlim(*xlim)
        plt.title(r"$C_\alpha-C_\alpha$ distance")
        plt.xlabel(r"Distance in $\AA$")
        plt.savefig(f"{out_dir}/{peptide}/{prefix}_ca_distance.pdf", bbox_inches="tight")
        plt.close()


def compute_metrics(samples, reference):
    metrics = {}
    for peptide in peptides:
        target_phi, target_psi = dataset.get_2d_features(reference[peptide])
        phi_model, psi_model = dataset.get_2d_features(samples[peptide])
        rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(target_phi, target_psi, phi_model, psi_model)
        js_div = js_divergence(
            jnp.concatenate([target_phi[None], target_psi[None]]).T,
            jnp.concatenate([phi_model[None], psi_model[None]]).T,
            bins=64,
        )
        metrics[peptide] = {"fe_sq": rms_fe_sq_error, "mjs": rms_mjs_error, "js": js_div}

    return metrics


def print_metrics(metrics):
    model_names = list(metrics.keys())
    aggregated_metrics = {name: {"fe_sq": [], "mjs": [], "js": []} for name in model_names}

    mean_metrics = {}
    std_metrics = {}
    # Collect metrics across all peptides
    for name in model_names:
        for peptide in peptides:
            aggregated_metrics[name]["fe_sq"].append(metrics[name][peptide]["fe_sq"])
            aggregated_metrics[name]["mjs"].append(metrics[name][peptide]["mjs"])
            aggregated_metrics[name]["js"].append(metrics[name][peptide]["js"])

    # Print aggregated metrics
    for name in model_names:
        fe_sq_values = jnp.array(aggregated_metrics[name]["fe_sq"])
        mjs_values = jnp.array(aggregated_metrics[name]["mjs"])
        js_values = jnp.array(aggregated_metrics[name]["js"])

        print(f"{name}:")
        print(f"  FE_SQ: {jnp.mean(fe_sq_values):.3f} ± {jnp.std(fe_sq_values):.3f}")
        print(f"  MJS: {jnp.mean(mjs_values):.4f} ± {jnp.std(mjs_values):.4f}")
        print(f"  JS: {jnp.mean(js_values):.4f} ± {jnp.std(js_values):.4f}")

        mean_metrics[name] = {
            "fe_sq": jnp.mean(fe_sq_values),
            "mjs": jnp.mean(mjs_values),
            "js": jnp.mean(js_values),
        }
        std_metrics[name] = {
            "fe_sq": jnp.std(fe_sq_values),
            "mjs": jnp.std(mjs_values),
            "js": jnp.std(js_values),
        }

    return mean_metrics, std_metrics


if __name__ == "__main__":
    mpl.rcParams.update({"font.size": 14, "axes.titlesize": 22, "axes.labelsize": 18})
    iid_samples = {}
    langevin_samples = {}
    scalar_fp_error = {}
    scalar_ac_fp_error = {}

    os.makedirs(out_dir, exist_ok=True)

    # create reference data from test set
    reference = {}
    current_idx = 0
    raw_test = dataset.test
    for peptide, peptide_length in zip(raw_test.peptides, raw_test.peptide_lengths):
        next_idx = current_idx + peptide_length
        reference[peptide] = raw_test.data[current_idx:next_idx]
        current_idx = next_idx

    for name, data_info in tqdm(data_sources.items(), desc="Loading data"):
        iid_samples[name] = {}
        langevin_samples[name] = {}
        for peptide in peptides:
            if os.path.isdir(os.path.join(data_info.path, "out", "test")):
                output_name = "test"
            elif os.path.isdir(os.path.join(data_info.path, "out", "validation")):
                output_name = "validation"
            else:
                raise ValueError(f"No test or validation directory found for {name} in {data_info.path}")
            data = load_numpy(os.path.join(data_info.path, "out", output_name, f"{peptide}_iid_samples.npy"))
            iid_samples[name][peptide] = jax.random.permutation(jax.random.PRNGKey(0), data)[:limit_samples]

            # load langevin samples
            if data_info.supports_langevin:
                data = load_numpy(
                    os.path.join(data_info.path, "out", output_name, f"{peptide}_langevin_trajectories.npy")
                )
                langevin_samples[name][peptide] = jax.random.permutation(jax.random.PRNGKey(0), data)[:limit_samples]

        # load scalar fp error
        if os.path.exists(os.path.join(data_info.path, "out", "scalar_fp_error.npy")):
            scalar_fp_error[name] = onp.load(os.path.join(data_info.path, "out", "scalar_fp_error.npy"))

    print("Finished loading data")

    # plot ramachandran plots
    # figure out vmin and vmax for histogram plots
    vmin, vmax = {p: float("inf") for p in peptides}, {p: float("-inf") for p in peptides}
    for peptide_data in chain(iid_samples.values(), langevin_samples.values(), [reference]):
        for peptide, data in peptide_data.items():
            phi, psi = dataset.get_2d_features(data)
            H, _, _ = jnp.histogram2d(phi, psi, bins=bins)
            # Update min/max values directly
            nonzero_min = H[H > 0].min() if jnp.any(H > 0) else vmin[peptide]
            vmin[peptide] = min(vmin[peptide], nonzero_min)
            vmax[peptide] = max(vmax[peptide], H.max())

    plot_samples(data_sources, iid_samples, reference, "iid", vmin, vmax)
    # Filter data sources that support langevin samples
    langevin_data_sources = {name: info for name, info in data_sources.items() if info.supports_langevin}
    # Filter langevin samples for those data sources
    filtered_langevin_samples = {
        name: samples for name, samples in langevin_samples.items() if name in langevin_data_sources
    }
    plot_samples(langevin_data_sources, filtered_langevin_samples, reference, "langevin", vmin, vmax)

    iid_metrics = {}
    langevin_metrics = {}

    for name, data_info in data_sources.items():
        iid_metrics[name] = compute_metrics(iid_samples[name], reference)

    for name, data_info in langevin_data_sources.items():
        langevin_metrics[name] = compute_metrics(langevin_samples[name], reference)

    print("IID")
    print(json.dumps(iid_metrics, indent=2))
    print("Langevin")
    print(json.dumps(langevin_metrics, indent=2))

    print("IID")
    mean_iid_metrics, std_iid_metrics = print_metrics(iid_metrics)
    print("Langevin")
    mean_langevin_metrics, std_langevin_metrics = print_metrics(langevin_metrics)

    for peptide in peptides:
        plt.figure(clear=True)
        plt.bar(iid_metrics.keys(), [iid_metrics[name][peptide]["fe_sq"] for name in iid_metrics.keys()])
        plt.savefig(f"{out_dir}/{peptide}/fe_sq_comparison.png", bbox_inches="tight")
        plt.close()

        plt.figure(clear=True)
        plt.bar(langevin_metrics.keys(), [langevin_metrics[name][peptide]["fe_sq"] for name in langevin_metrics.keys()])
        plt.savefig(f"{out_dir}/{peptide}/langevin_fe_sq_comparison.png", bbox_inches="tight")
        plt.close()

    # print tabular
    print("----")
    for name in data_sources.keys():
        line = f"{name}\t & {mean_iid_metrics[name]['js']:.4f} $\\pm$ {std_iid_metrics[name]['js']:.4f} & "
        if data_sources[name].supports_langevin:
            line += f"{mean_langevin_metrics[name]['js']:.4f} $\\pm$ {std_langevin_metrics[name]['js']:.4f} & "
        else:
            line += " - & "
        line += f"{mean_iid_metrics[name]['fe_sq']:.3f} $\\pm$ {std_iid_metrics[name]['fe_sq']:.3f} & "
        if data_sources[name].supports_langevin:
            line += f"{mean_langevin_metrics[name]['fe_sq']:.3f} $\\pm$ {std_langevin_metrics[name]['fe_sq']:.3f}"
        else:
            line += " -"
        line += " \\\\"

        print(line)

    # plot fp error that we have across all peptides
    plt.figure(clear=True)
    for name, error in scalar_fp_error.items():
        error = jnp.where(jnp.abs(error) < 1e-5, jnp.nan, error)
        plt.plot(jnp.linspace(0, 1, len(error) - 2), error[1:-1], label=name, linewidth=3, linestyle="--")
    plt.legend()
    plt.yscale("log")
    plt.xlabel(r"Diffusion time $t$")
    plt.ylabel("Fokker-Planck Error")
    plt.xlim(0, 1)
    plt.savefig(f"{out_dir}/scalar_fp_error.pdf", bbox_inches="tight")
    plt.close()
