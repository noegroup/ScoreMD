from dataclasses import dataclass
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
import os
from tqdm import tqdm
import jax
import numpy as onp
from scoremd.utils.evaluation import pairwise_distances, phi_psi_metrics, js_divergence
from scipy.stats import wasserstein_distance
import numpy as np
import jax.numpy as jnp


@dataclass
class DataSource:
    path: str

    def __post_init__(self):
        self.path = os.path.expanduser(self.path)


data_sources: dict[str, DataSource] = {
    "Diffusion": DataSource(path="./multirun/2025-05-01/11-15-15"),
    "Two For One": DataSource(path="./multirun/2025-05-01/11-14-58"),
    "Mixture": DataSource(path="./multirun/2025-05-01/11-18-52"),
    "Fokker-Planck": DataSource(path="./multirun/2025-06-19/09-24-45"),
    "Both": DataSource(path="./multirun/2025-07-02/11-19-08"),
}


iid_path = "out/aldp_iid_samples.npy"
langevin_path = "out/aldp_langevin_trajectories.npy"
bins = 60
bw_method = 0.05
num_seeds = 3

dataset = ALDPDataset(coarse_graining_level=CoarseGrainingLevel.FULL, limit_samples=50_000, validation=False)
target_phi, target_psi = dataset.get_2d_features(dataset.train.data)


def compute_metrics(samples):
    metrics = {}
    # Compute phi/psi metrics
    for name in samples.keys():
        phi_model, psi_model = dataset.get_2d_features(samples[name])
        rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(target_phi, target_psi, phi_model, psi_model)
        js_div = js_divergence(
            jnp.concatenate([target_phi[None], target_psi[None]]).T,
            jnp.concatenate([phi_model[None], psi_model[None]]).T,
            bins=64,
        )
        metrics[name] = {"fe_sq": rms_fe_sq_error, "mjs": rms_mjs_error, "js": js_div, "bonds": {}}

    # Compute bond metrics
    for a0, a1 in dataset.bonds:
        a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
        ground_truth_bonds = pairwise_distances(dataset.train.data)[:, a0, a1]

        for name in data_sources.keys():
            current_bonds = pairwise_distances(samples[name])[:, a0, a1]
            w1 = wasserstein_distance(ground_truth_bonds, current_bonds)
            metrics[name]["bonds"][f"{a0_name}-{a1_name}"] = w1

    return metrics


def print_metrics(metrics_list):
    # Get all model names
    model_names = list(metrics_list[0].keys())

    # Print phi/psi metrics
    for name in model_names:
        fe_sq_values = [m[name]["fe_sq"] for m in metrics_list]
        # mjs_values = [m[name]["mjs"] for m in metrics_list]
        js_values = [m[name]["js"] for m in metrics_list]
        print(f"{name}:")
        print(f"  PMF: {np.mean(fe_sq_values):.3f} ± {np.std(fe_sq_values):.3f}")
        # print(f"  MJS: {np.mean(mjs_values):.4f} ± {np.std(mjs_values):.4f}")
        print(f"  JS: {np.mean(js_values):.4f} ± {np.std(js_values):.4f}")

    # Print bond metrics
    for a0, a1 in dataset.bonds:
        if "Both" in model_names:
            a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
            bond_name = f"{a0_name}-{a1_name}"
            print(f"\nBond {bond_name}:")
            # base: diffusion
            bond_values_best = np.array([m["Both"]["bonds"][bond_name] for m in metrics_list])
            for name in model_names:
                bond_values = np.array([m[name]["bonds"][bond_name] for m in metrics_list])
                print(
                    f"  {name}: {np.mean(bond_values):.4f} ± {np.std(bond_values):.4f}, relative; {np.mean(bond_values / bond_values_best):.2f} ± {np.std(bond_values / bond_values_best):.2f}"
                )


if __name__ == "__main__":
    iid_metrics_list = []
    langevin_metrics_list = []

    for seed in range(num_seeds):
        iid_samples = {}
        langevin_samples = {}

        for name, data_info in tqdm(data_sources.items(), desc=f"Loading data for seed {seed}"):
            seed_path = os.path.join(data_info.path, str(seed))
            data = onp.load(os.path.join(seed_path, iid_path))
            iid_samples[name] = data
            num_iid_samples = data.shape[0]

            data = onp.load(os.path.join(seed_path, langevin_path))
            data = data.reshape(-1, *dataset.sample_shape)
            data = jax.random.permutation(jax.random.PRNGKey(0), data)[:num_iid_samples]
            langevin_samples[name] = data

        # Compute metrics for this seed
        iid_metrics_list.append(compute_metrics(iid_samples))
        langevin_metrics_list.append(compute_metrics(langevin_samples))

    # Print results
    print("IID")
    print_metrics(iid_metrics_list)
    print("Langevin")
    print_metrics(langevin_metrics_list)
