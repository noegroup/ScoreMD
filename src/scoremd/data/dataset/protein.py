from dataclasses import dataclass
from os import PathLike
import os
from typing import Callable, Optional, Sequence, Tuple
import jax
from scoremd.data.dataset.base import Datapoints, Dataset
import jax.numpy as jnp
import openmm.unit as unit
import logging
import mdtraj as md
from scoremd.rmsd import kabsch_align_many
from scoremd.simulation import create_langevin_step_function
import scoremd.utils.plots as plots
from scoremd.utils.contact import compute_contact_map
import pickle
import matplotlib.pyplot as plt
from functools import partial
from scoremd.data.dataset.angles import dihedral
from deeptime.decomposition import CovarianceKoopmanModel
from scoremd.data.dataset.utils import write_animation_with_topology
from openmm.app import PDBFile

log = logging.getLogger(__name__)


@dataclass
class SingleProteinDataset(Dataset):
    train_split: Tuple[float, float] = 0.8
    limit_samples: Optional[int] = None
    validation: bool = False
    seed: int = 0

    def __init__(
        self,
        paths: PathLike | Sequence[PathLike],
        tica_path: PathLike,
        topology_path: PathLike,
        train_split: float = 0.8,
        limit_samples: Optional[int] = None,
        validation: bool = True,
        seed: int = 0,
        name="protein",
    ):
        dataset_name = os.path.basename(tica_path).split("_")[0]
        log.info(f"Loading dataset: {dataset_name}")

        self.train_split = train_split
        self.limit_samples = limit_samples
        self.validation = validation
        self.seed = seed

        self.range = None
        if dataset_name == "chignolin":
            T = 340
            self.range = [[-0.9, 2.81], [-1.41, 0.56]]
        elif dataset_name == "trpcage":
            T = 290
            self.range = [[-2.42, 1.16], [-1.55, 3.17]]
        elif dataset_name == "bba":
            T = 325
            self.range = [[-1.23, 2.05], [-3.13, 3.10]]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self._dataset = md.load(paths)
        tica_model = pickle.load(open(tica_path, "rb"))
        self._tica_transform = SingleProteinDataset.build_tica_transform_jax(tica_model)

        self.mass = jnp.array([atom.element.mass for atom in self._dataset.topology.atoms]).reshape(-1, 1)

        self.topology = PDBFile(topology_path).topology

        super().__init__(
            name=name,
            sample_shape=(self.mass.shape[0], 3),
            kbT=(unit.MOLAR_GAS_CONSTANT_R * T * unit.kelvin).value_in_unit(unit.kilojoules_per_mole),
        )

    def _split_data(self, data: jnp.ndarray, key) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        if key is not None:
            data = jax.random.permutation(key, data)

        if self.validation:
            train_set = round(len(data) * self.train_split)
            assert train_set > 0, "No training samples."

            train, val = data[:train_set], data[train_set:]
        else:
            train, val = data, data[:0]

        if self.limit_samples:
            new_train_set_size = min(train.shape[0], self.limit_samples)
            new_val_set_size = int(jnp.ceil(new_train_set_size / self.train_split * (1 - self.train_split)))

            log.info(
                f"Limit samples has been set to: {self.limit_samples}. "
                f"Limiting training dataset to {new_train_set_size} and validation set to {new_val_set_size} sample(s)."
            )
            train = train[:new_train_set_size]
            val = val[:new_val_set_size]

        return train, val if val.shape[0] > 0 else None

    def _get_data(self) -> Tuple[Datapoints, Optional[Datapoints], Optional[Datapoints]]:
        data = jnp.array(self._dataset.xyz)
        data = data.reshape(data.shape[0], -1)
        train, val = self._split_data(data, jax.random.PRNGKey(self.seed))

        train, _ = kabsch_align_many(train, train[0])
        if val is not None:
            val, _ = kabsch_align_many(val, train[0])

        return Datapoints(train, None), Datapoints(val, None) if val is not None else None, None

    @property
    def bonds(self) -> set[Tuple[int, int]]:
        return [(i, i + 1) for i in range(self.mass.shape[0] - 1)]

    @property
    def atom_names(self):
        return [atom.name for atom in self._dataset.topology.atoms]

    @staticmethod
    def _distances(xyz):
        # this function computes the upper triangular part of the distance matrix
        # however, it must be implemented in a way that does not compute the distance between the same atom
        # this is because the distance between the same atom is 0, which destroys grad
        # xyz: (B, N, 3)
        N = xyz.shape[1]
        i, j = jnp.triu_indices(N, k=1)
        diffs = xyz[:, i, :] - xyz[:, j, :]  # (B, M, 3), M = N*(N-1)//2
        # print(diffs)
        # assert jnp.abs(diffs) > 0.01
        return jnp.linalg.norm(diffs, axis=-1)  # (B, M)

    @staticmethod
    def _dihedrals(xyz):
        n_atoms = xyz.shape[1]
        dihedral_indices = jnp.array([list(range(i, i + 4)) for i in range(n_atoms - 3)])

        @partial(jax.vmap, in_axes=(None, 0))
        def _compute_dihedrals(xyz, dihedral_indices):
            return dihedral(xyz[:, dihedral_indices])

        return _compute_dihedrals(xyz, dihedral_indices).T

    @staticmethod
    def _tica_features(xyz):
        return jnp.concatenate([SingleProteinDataset._distances(xyz), SingleProteinDataset._dihedrals(xyz)], axis=-1)

    @staticmethod
    def build_tica_transform_jax(model: CovarianceKoopmanModel):
        # constants (host->device once)
        k = int(model.output_dimension)
        U = jnp.array(jnp.asarray(model.instantaneous_coefficients[:, :k]))  # (n, k)

        # center if training removed the mean; otherwise mean is None -> skip
        mean0 = getattr(model.cov, "mean_0", None)
        if mean0 is not None:
            mean0 = jnp.array(jnp.asarray(mean0))
        # else keep as None

        @jax.jit
        def transform(x):  # x: (T, n) or (n,)
            x = jnp.asarray(x)
            if mean0 is not None:
                x = x - mean0
            return x @ U  # -> (T, k) or (k,)

        return transform

    def get_2d_features(self, samples):
        samples = jnp.array(samples.reshape(-1, *self.sample_shape))
        tica_features = SingleProteinDataset._tica_features(samples)
        samples = self._tica_transform(tica_features)
        return samples[:, 0], samples[:, 1]

    def plot_2d(
        self, samples, weights=None, range=None, title=None, bins=100, cmap="turbo_r", vmin=None, vmax=None, **kwargs
    ):
        """Plot sample histogram in 2D"""
        if samples.shape[1] == 2:  # we are already in 2D
            x, y = samples[:, 0], samples[:, 1]
        else:
            x, y = self.get_2d_features(samples)

        plots.plot_2d(
            x,
            y,
            title=title,
            range=range if range is not None else self.range,
            weights=weights,
            bins=bins,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        plt.xlabel("TIC 0")
        plt.ylabel("TIC 1")

        plt.xticks([])
        plt.yticks([])

        return x, y

    def langevin_step_function(
        self,
        num_steps,
        force: Callable[[jnp.ndarray], jnp.ndarray],
        dt: Optional[float] = None,
    ) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        This function returns a Langevin step function that takes nm coordinates and does num_steps.
        See the `create_langevin_step_function` function for more details.
        """
        return create_langevin_step_function(
            force=force,
            mass=self.mass,
            gamma=1.0,
            num_steps=num_steps,
            dt=dt if dt is not None else 2e-3,  # 2 fs is the step size used in the simulation
            kbT=self.kbT,
        )

    def plot_contact_map(self, samples, threshold=1, cbar=True, vmin=0.0, vmax=None, contact_map=None, **kwargs):
        if contact_map is None:
            contact_map = compute_contact_map(samples, threshold)
        plt.imshow(contact_map, cmap="viridis_r", vmin=vmin, vmax=vmax, **kwargs)
        if cbar:
            plt.colorbar()

        plt.xticks([])
        plt.yticks([])

        return contact_map

    def write_animation(self, trajectory: jnp.ndarray, out: PathLike):
        write_animation_with_topology(trajectory, self.topology, out)


if __name__ == "__main__":
    # dataset = SingleProteinDataset(
    #     paths="./storage/deshaw/chignolin-0_ca.h5", tica_path="./storage/deshaw/chignolin_tica.pic", topology_path="./storage/deshaw/chignolin.pdb"
    # )

    # dataset = SingleProteinDataset(
    #     paths="./storage/deshaw/trpcage-0_ca.h5", tica_path="./storage/deshaw/trpcage_tica.pic", topology_path="./storage/deshaw/trpcage.pdb"
    # )

    dataset = SingleProteinDataset(
        paths=["./storage/deshaw/bba-0_ca.h5", "./storage/deshaw/bba-1_ca.h5"],
        tica_path="./storage/deshaw/bba_tica.pic",
        topology_path="./storage/deshaw/bba.pdb",
    )

    limit = 100_000

    print(dataset.train.data.shape)
    print(dataset.atom_names)

    data = dataset.train.data
    if limit is not None and limit > 0:
        if limit > data.shape[0]:
            log.warning(f"Limit is greater than the number of samples. Using all {data.shape[0]} samples.")
        data = jax.random.permutation(jax.random.PRNGKey(0), data)[:limit]

    dataset.plot_2d(data)
    print("xlim", plt.xlim(), "ylim", plt.ylim())
    plt.show()

    dataset.plot_contact_map(dataset.train.data)
    plt.show()
