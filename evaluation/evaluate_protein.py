from dataclasses import dataclass
from typing import Literal
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scoremd.data.dataset.protein import SingleProteinDataset
from scoremd.utils.evaluation import helper_metrics_2d, js_divergence
from scoremd.utils.plots import plot_fes as plot_fes_base
from scoremd.utils.contact import compute_contact_map
import jax.numpy as jnp
import numpy as onp
import jax
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from functools import partial
from deeptime.clustering import MiniBatchKMeans
from deeptime.markov import TransitionCountEstimator
from sklearn.preprocessing import normalize
import seaborn as sns

SystemType = Literal["chignolin", "bba"]
system: SystemType = "bba"

out_dir = os.path.expanduser(f"./evaluation/protein/{system}")
os.makedirs(out_dir, exist_ok=True)

plt.style.use("tableau-colorblind10")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bins = 100

base_path = "./"

MAX_SEED = 3
extension = "pdf"


@dataclass
class DataSource:
    path: str
    file_name_prefix: str

    def __post_init__(self):
        self.path = os.path.expanduser(os.path.join(base_path, self.path))


data_sources_chignolin: dict[str, DataSource] = {
    "Diffusion": DataSource(
        path="multirun/2025-07-19/18-17-50",
        file_name_prefix="baseline",
    ),
    "Two For One": DataSource(
        path="multirun/2025-07-19/17-27-00",
        file_name_prefix="two_for_one",
    ),
    "Mixture": DataSource(
        path="multirun/2025-07-19/16-39-35",
        file_name_prefix="mixture",
    ),
    "Fokker-Planck": DataSource(
        path="multirun/2025-07-26/16-25-46",
        file_name_prefix="fp",
    ),
    "Both": DataSource(
        path="multirun/2025-07-21/16-49-36",
        file_name_prefix="both",
    ),
}

data_sources_bba: dict[str, DataSource] = {
    "Diffusion": DataSource(
        path="multirun/2025-07-21/19-48-12",
        file_name_prefix="baseline",
    ),
    "Two For One": DataSource(
        path="multirun/2025-07-21/21-34-46",
        file_name_prefix="two_for_one",
    ),
    "Mixture": DataSource(
        path="multirun/2025-09-06/17-22-18",
        file_name_prefix="mixture",
    ),
    "Fokker-Planck": DataSource(
        path="multirun/2025-08-07/12-39-26",
        file_name_prefix="fp",
    ),
    "Both": DataSource(
        path="multirun/2025-09-18/22-00-04",
        file_name_prefix="both",
    ),
}

# The centers are taken from https://github.com/microsoft/two-for-one-diffusion/blob/94e46061e24419d972be6bbc3861d7238c652856/evaluate/evaluate_fastfolders.ipynb
cluster_centers = {
    "chignolin": jnp.array([[0.69400153, -0.34598462], [-0.48732213, 0.00642035], [1.87483537, 0.06285344]]),
    "bba": jnp.array(
        [[-0.5756589, -0.60663654], [1.7861676, -0.87717611], [0.91295128, 1.07518898], [-0.49210152, 0.40313689]]
    ),
}

max_free_energy = {
    "chignolin": (60.0, 50.0),
    "bba": (30.0, 30.0),
}

if system == "chignolin":
    dataset = SingleProteinDataset(
        paths="./storage/deshaw/chignolin-0_ca.h5",
        tica_path="./storage/deshaw/chignolin_tica.pic",
        topology_path="./storage/deshaw/chignolin.pdb",
    )
    data_sources = data_sources_chignolin
elif system == "bba":
    dataset = SingleProteinDataset(
        paths=["./storage/deshaw/bba-0_ca.h5", "./storage/deshaw/bba-1_ca.h5"],
        tica_path="./storage/deshaw/bba_tica.pic",
        topology_path="./storage/deshaw/bba.pdb",
    )
    data_sources = data_sources_bba
else:
    raise ValueError(f"Unknown system: {system}")

@partial(jax.jit, static_argnums=1)
def get_pwd(x, offset):
    assert x.ndim == 2, "Please reshape to BS, num_beads * 3"
    x = x.reshape(x.shape[0], -1, 3)
    pwd = jnp.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)
    assert pwd.shape[-2] == pwd.shape[-1], "PWD matrix must be square"
    triu_ind = jnp.triu_indices(pwd.shape[-2], k=offset, m=pwd.shape[-1])
    return pwd[:, triu_ind[0], triu_ind[1]]


def compute_pwd_js_div(reference, samples, offset=3, resolution=0.1):
    def compute_histogram(distances, resolution, maxval):
        nbins = int(jnp.floor(maxval / resolution)) + 1
        hist, _ = jnp.histogram(distances, range=(0.0, resolution * nbins), bins=nbins)
        hist = hist.astype(jnp.float32)
        if hist.sum() > 0:
            hist /= hist.sum()  # Normalize to get probability distribution
        return hist + 1e-10

    assert reference.shape == samples.shape
    assert reference.ndim == 2

    pwd_reference = get_pwd(reference, offset=offset)
    pwd_samples = get_pwd(samples, offset=offset)

    num_atoms = pwd_reference.shape[1]

    js_div = 0
    for i in range(num_atoms):
        ref_i, samp_i = pwd_reference[:, i], pwd_samples[:, i]
        maxval = float(jnp.maximum(ref_i.max(), samp_i.max()))

        hist_ref = compute_histogram(ref_i, resolution, maxval)
        hist_samp = compute_histogram(samp_i, resolution, maxval)
        assert hist_ref.shape == hist_samp.shape

        js_div += jensenshannon(hist_ref, hist_samp) ** 2 / num_atoms

    return js_div


def plot_assignments(data, assignments):
    plt.scatter(data[:, 0], data[:, 1], s=0.6, c=assignments, rasterized=True)
    plt.xlim(dataset.range[0])
    plt.ylim(dataset.range[1])
    plt.gca().set_box_aspect(1)
    plt.xlabel("TIC 0")
    plt.ylabel("TIC 1")

    plt.xticks([])
    plt.yticks([])


def plot_transition_matrix(count_matrix, method_name):
    n_clusters = len(cluster_centers[system])
    plt.figure(figsize=(4.8, 4.8))
    plt.title(method_name, fontsize=22, pad=10)
    sns.heatmap(
        count_matrix,
        annot=True,
        square=True,
        cmap="Greens",
        fmt=".3f",
        xticklabels=onp.arange(1, n_clusters + 1),
        yticklabels=onp.arange(1, n_clusters + 1),
        vmin=0,
        vmax=1,
        cbar=False,
        annot_kws={"fontsize": 18},
    )
    plt.xlabel("End State", fontsize=18)
    plt.ylabel("Start State", fontsize=18)


def get_kmeans_count_matrix(tic_data):
    assignments = kmeans.transform(onp.array(tic_data))
    count_matrix = TransitionCountEstimator.count(count_mode="sliding", dtrajs=[assignments.astype("int")], lagtime=1)

    count_matrix = normalize(count_matrix, axis=1, norm="l1")
    return assignments, count_matrix


def plot_fes0(x, **kwargs):
    plt.xlabel("TIC 0")
    plt.title("Free energy projection of TIC 0")
    return plot_fes_base(
        x, dataset.kbT, jnp.linspace(dataset.range[0][0], dataset.range[0][1], 100), None, bw_method=0.05, **kwargs
    )


def plot_fes1(x, **kwargs):
    plt.xlabel("TIC 1")
    plt.title("Free energy projection of TIC 1")
    return plot_fes_base(
        x, dataset.kbT, jnp.linspace(dataset.range[1][0], dataset.range[1][1], 100), None, bw_method=0.2, **kwargs
    )


def load_numpy(path):
    loaded = onp.load(path)

    if path.endswith(".npz"):
        loaded = loaded["samples_np"]

    if system == "chignolin":
        return loaded.reshape(-1, 30)
    elif system == "bba":
        return loaded.reshape(-1, 84)
    else:
        raise ValueError(f"Unknown system: {system}")


def get_dataset_paths(data_source, prefix: Literal["iid", "langevin"]):
    identifier = "samples" if prefix == "iid" else "trajectories"
    if os.path.exists(f"{data_source.path}/0"):  # we have a multirun, so we need to load multiple seeds
        return [f"{data_source.path}/{i}/out/{system}_{prefix}_{identifier}.npy" for i in range(MAX_SEED)]
    else:
        return [f"{data_source.path}/out/{system}_{prefix}_{identifier}.npy"]


def plot_samples(data_sources, reference, prefix: Literal["iid", "langevin"]):
    metrics = {}

    all_tica_samples = {}
    contact_maps = {}
    for name, data_source in tqdm(data_sources.items(), desc="Loading samples"):
        paths = get_dataset_paths(data_source, prefix)

        if len(paths) > 1:
            paths = tqdm(paths, desc="Loading samples from multirun", leave=False)
        for path in paths:
            samples = load_numpy(path)

            if samples.shape[0] < reference.shape[0]:
                print("!!!!!WARNING: samples are less than reference, throw an error here")

            # for the langevin samples we take everything
            if prefix == "langevin":
                if name not in all_tica_samples:  # just do this once
                    # reduce to only every 20th sample
                    tic0, tic1 = dataset.get_2d_features(samples[::20])
                    tics = jnp.concatenate([tic0[:, None], tic1[:, None]], axis=1)
                    assignments, count_matrix = get_kmeans_count_matrix(tics)

                    plt.figure(clear=True)
                    plt.title(name)
                    plot_assignments(tics, assignments)
                    plt.savefig(
                        f"{out_dir}/{data_source.file_name_prefix}_{prefix}_kmeans_assignments.pdf", bbox_inches="tight"
                    )
                    plt.close()

                    plt.figure(clear=True)
                    plot_transition_matrix(count_matrix, name)
                    plt.savefig(
                        f"{out_dir}/{data_source.file_name_prefix}_{prefix}_kmeans_transition_matrix.pdf",
                        bbox_inches="tight",
                    )

            # pick random samples so that samples is the same length as reference
            samples = jax.random.permutation(jax.random.PRNGKey(0), samples)[: reference.shape[0]]

            if name not in all_tica_samples:
                # now we do the contact map, then we can reduce
                contact_maps[name] = compute_contact_map(samples, threshold=1)

            pwd_js_div = compute_pwd_js_div(dataset.train.data, samples, offset=3)

            tic0, tic1 = dataset.get_2d_features(samples)
            del samples
            current_tica_samples = jnp.concatenate([tic0[:, None], tic1[:, None]], axis=1)
            if name not in all_tica_samples:
                onp.save(f"{out_dir}/{data_source.file_name_prefix}_{prefix}_tica.npy", current_tica_samples)
                all_tica_samples[name] = current_tica_samples

            # evaluate js divergence
            js_div = js_divergence(
                reference,
                current_tica_samples,
                bins=bins,
                limits=dataset.range,
            )

            rms_fe_sq_error, rms_mjs_error = helper_metrics_2d(
                reference[:, 0],
                reference[:, 1],
                current_tica_samples[:, 0],
                current_tica_samples[:, 1],
                dataset.range[0],
                dataset.range[1],
                n_bins=bins,
            )
            if name not in metrics:
                metrics[name] = {
                    "js": jnp.array([js_div]),
                    "pwd_js": jnp.array([pwd_js_div]),
                    "fe_sq": jnp.array([rms_fe_sq_error]),
                    "mjs": jnp.array([rms_mjs_error]),
                }
                if prefix == "langevin":
                    metrics[name]["count_matrix"] = jnp.array(count_matrix)
            else:
                metrics[name]["js"] = jnp.concatenate([metrics[name]["js"], jnp.array([js_div])])
                metrics[name]["pwd_js"] = jnp.concatenate([metrics[name]["pwd_js"], jnp.array([pwd_js_div])])
                metrics[name]["fe_sq"] = jnp.concatenate([metrics[name]["fe_sq"], jnp.array([rms_fe_sq_error])])
                metrics[name]["mjs"] = jnp.concatenate([metrics[name]["mjs"], jnp.array([rms_mjs_error])])

    return metrics, all_tica_samples, contact_maps


def get_method_title(method_name):
    if method_name == "iid":
        return "iid"
    elif method_name == "langevin":
        return "sim"

    raise NotImplementedError(f"Unknown method name: {method_name}")


kmeans = None

if __name__ == "__main__":
    mpl.rcParams.update({"font.size": 14, "axes.titlesize": 22, "axes.labelsize": 18})

    # for big figure
    # mpl.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 24})
    # MAX_SEED = 1
    # extension = "svg"

    print(f"Evaluating {system}...")

    tic0, tic1 = dataset.get_2d_features(dataset.train.data)
    metrics = {}

    all_tica_samples = {}
    contact_maps = {}
    reference_tica = jnp.concatenate([tic0[:, None], tic1[:, None]], axis=1)
    tic0_langevin, tic1_langevin = dataset.get_2d_features(
        jnp.array(dataset._dataset.xyz)
    )  # Otherwise the dataset is shuffled
    reference_tica_langevin = jnp.concatenate([tic0_langevin[:, None], tic1_langevin[:, None]], axis=1)

    print("Fitting kmeans to the available data ...")
    kmeans = MiniBatchKMeans(
        n_clusters=len(cluster_centers[system]),
        max_iter=0,
        batch_size=64,
        init_strategy="kmeans++",
        n_jobs=16,
        tolerance=1e-7,
        initial_centers=cluster_centers[system],
    ).fit(onp.array(reference_tica_langevin))

    reference_assignments, reference_count_matrix = get_kmeans_count_matrix(reference_tica_langevin)
    plt.figure(clear=True)
    plt.title("Reference")
    plot_assignments(reference_tica_langevin, reference_assignments)
    plt.savefig(f"{out_dir}/reference_kmeans_assignments.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(clear=True)
    plot_transition_matrix(reference_count_matrix, "Reference")
    plt.savefig(f"{out_dir}/reference_kmeans_transition_matrix.pdf", bbox_inches="tight")
    plt.close()

    print("Loading iid samples...")
    metrics["iid"], all_tica_samples["iid"], contact_maps["iid"] = plot_samples(data_sources, reference_tica, "iid")
    print("Loading langevin samples...")
    metrics["langevin"], all_tica_samples["langevin"], contact_maps["langevin"] = plot_samples(
        data_sources, reference_tica, "langevin"
    )

    # figure out vmin and vmax for contact maps
    vmin = float("inf")
    for prefix in contact_maps.keys():
        for contact_map in contact_maps[prefix].values():
            vmin = min(vmin, contact_map.min() if jnp.any(contact_map > 0) else vmin)
    print(f"Determined vmin: {vmin} for contact maps")

    # plot all contact maps
    plt.figure(clear=True)
    plt.title("Reference")
    dataset.plot_contact_map(dataset.train.data, vmin=vmin, vmax=0.0)
    plt.savefig(f"{out_dir}/reference_contact_map.pdf", bbox_inches="tight")
    plt.close()

    for prefix in contact_maps.keys():
        for name, contact_map in contact_maps[prefix].items():
            plt.figure(clear=True)
            plt.title(name)
            dataset.plot_contact_map(None, contact_map=contact_map, vmin=vmin, vmax=0.0)
            plt.savefig(
                f"{out_dir}/{data_sources[name].file_name_prefix}_{prefix}_contact_map.pdf", bbox_inches="tight"
            )
            plt.close()

    # now we do the plotting of all tica stuff
    for prefix in all_tica_samples.keys():
        for fes, plot_fn in zip([0, 1], [plot_fes0, plot_fes1]):
            plt.figure(clear=True)
            plot_fn(reference_tica[:, fes], label="Reference", color="black", linewidth=5)

            for (name, tica_samples), color in zip(all_tica_samples[prefix].items(), colors):
                plot_fn(tica_samples[:, fes], label=name, linestyle="--", color=color)

            plt.ylim(0, max_free_energy[system][fes])

            if fes == 0:
                plt.legend()
            plt.savefig(f"{out_dir}/{prefix}_tic{fes}_free_energy.pdf", bbox_inches="tight")
            plt.close()

    # figure out vmin and vmax for free energy plots
    vmin, vmax = float("inf"), float("-inf")

    # initialize vmin and vmax with the reference
    H, _, _ = jnp.histogram2d(reference_tica[:, 0], reference_tica[:, 1], range=dataset.range, bins=bins)
    vmin = min(vmin, H[H > 0].min() if jnp.any(H > 0) else vmin)
    vmax = max(vmax, H.max())
    print(f"Reference vmin: {vmin}, vmax: {vmax}")

    for method in all_tica_samples.keys():
        for tica_samples in all_tica_samples[method].values():
            H, _, _ = jnp.histogram2d(tica_samples[:, 0], tica_samples[:, 1], range=dataset.range, bins=bins)
            vmin = min(vmin, H[H > 0].min() if jnp.any(H > 0) else vmin)
            vmax = max(vmax, H.max())

    print(f"Determined vmin: {vmin}, vmax: {vmax}")

    plt.figure(clear=True)
    dataset.plot_2d(dataset.train.data, bins=bins, title="Reference", vmin=vmin, vmax=vmax, free_energy_bar=False)
    plt.savefig(f"{out_dir}/reference_tica.{extension}", bbox_inches="tight")
    plt.close()
    print(all_tica_samples.keys())
    # now that we have a consistent vmin and vmax for tica, we can plot the 2d fes
    for method in all_tica_samples.keys():
        print(all_tica_samples[method].keys())
        for name, tica_samples in all_tica_samples[method].items():
            plt.figure(clear=True)
            dataset.plot_2d(tica_samples, bins=bins, title=f"{name} ({get_method_title(method)})", vmin=vmin, vmax=vmax)
            plt.savefig(
                f"{out_dir}/{data_sources[name].file_name_prefix}_{method}_tica.{extension}", bbox_inches="tight"
            )
            plt.close()

    # print the metrics
    print("Method | IID JS | Langevin JS | IID PMF | Langevin PMF")
    for name, data_source in data_sources.items():
        print(f"{name}", end="")

        for method in ["iid", "langevin"]:
            print(" & ", end="")
            print(f"{jnp.mean(metrics[method][name]['js']):.4f}", end="")
            if len(metrics[method][name]["js"]) > 1:
                print(f" ± {jnp.std(metrics[method][name]['js']):.4f}", end="")

        for method in ["iid", "langevin"]:
            print(" & ", end="")
            print(f"{jnp.mean(metrics[method][name]['fe_sq']):.3f}", end="")
            if len(metrics[method][name]["fe_sq"]) > 1:
                print(f" ± {jnp.std(metrics[method][name]['fe_sq']):.3f}", end="")

        if name != list(data_sources.keys())[-1]:
            print(" \\\\")
        else:
            print("")

    print("Method | IID PWD JS | Langevin PWD JS")
    for name, data_source in data_sources.items():
        print(f"{name}", end="")

        for method in ["iid", "langevin"]:
            print(" & ", end="")
            print(f"{jnp.mean(metrics[method][name]['pwd_js']):.4f}", end="")
            if len(metrics[method][name]["pwd_js"]) > 1:
                print(f" ± {jnp.std(metrics[method][name]['pwd_js']):.4f}", end="")

        if name != list(data_sources.keys())[-1]:
            print(" \\\\")
        else:
            print("")

    # print count matrices
    for name, data_source in data_sources.items():
        print(f"{name} count matrix JS divergence:")
        current_div = (
            jensenshannon(metrics["langevin"][name]["count_matrix"].flatten(), reference_count_matrix.flatten()) ** 2
        )
        print(f"{current_div:.1e}")
