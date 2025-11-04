from dataclasses import dataclass
from os import PathLike
import os
from typing import Callable, Optional, Sequence, Tuple
from flax import struct
import jax
from scoremd.data.dataset.angles import dihedral
from scoremd.data.dataset.base import Datapoints, Dataset
import numpy as onp
import jax.numpy as jnp
import openmm.unit as unit
import logging
from tqdm import tqdm
import mdtraj as md
from scoremd.data.dataset.utils import write_animation_with_topology
from scoremd.rmsd import kabsch_align_many
from scoremd.simulation import create_langevin_step_function
import scoremd.utils.plots as plots

log = logging.getLogger(__name__)

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLY",
    "GLU",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

ATOMS_TO_KEEP = ["N", "CA", "CB", "C", "O"]  # do NOT change this order
POSSIBLE_ELEMENTS = ["C", "N", "O"]


class PeptideDatapoints(Datapoints):
    peptides: Sequence[str] = struct.field(pytree_node=False)  # the peptide sequence (e.g., AA, or MP)
    peptide_lengths: Sequence[int] = struct.field(pytree_node=False)  # how many samples there are for each peptide

    def __post_init__(self):
        super().__post_init__()

        assert len(self.peptides) == len(self.peptide_lengths)


@dataclass
class CGMinipeptideDataset(Dataset):
    """
    Generated with:
    pdb = openmm.app.PDBFile(pdb_path)
    forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")

    system = forcefield.createSystem(pdb.topology, nonbondedMethod=openmm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0*openmm.unit.nanometer, constraints=None)
    integrator = openmm.LangevinMiddleIntegrator(310*openmm.unit.kelvin, 0.3/openmm.unit.picosecond, 0.5*openmm.unit.femtosecond)
    openmm_energy = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

    The dataset is stored with a scaling of 30.

    """

    pdb_directory: PathLike
    train_path: PathLike
    max_z: Sequence[int]
    limit_samples: Optional[int] = None
    limit_peptides: Optional[Sequence[str]] = None
    val_path: Optional[PathLike] = None
    test_path: Optional[PathLike] = None

    def __init__(
        self,
        pdb_directory: PathLike,
        train_path: PathLike,
        val_path: Optional[PathLike] = None,
        test_path: Optional[PathLike] = None,
        limit_samples: Optional[int] = None,
        limit_peptides: Optional[Sequence[str]] = None,
        name: str = "minipeptides",
    ):
        self.pdb_directory = pdb_directory
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.limit_peptides = limit_peptides
        self.limit_samples = limit_samples
        peptide_length = 2  # number of amino acids in the peptides
        self.max_z = [len(AMINO_ACIDS), len(ATOMS_TO_KEEP), len(POSSIBLE_ELEMENTS), peptide_length]

        kbT = (unit.MOLAR_GAS_CONSTANT_R * 310 * unit.kelvin).value_in_unit(unit.kilojoules_per_mole)
        sample_shape = (len(ATOMS_TO_KEEP) * peptide_length, 3)  # We keep the atoms for two amino acids

        # [atom.element.mass for atom in topology.atoms]
        self.mass = jnp.array(
            [
                [14.00672],
                [12.01078],
                [12.01078],
                [12.01078],
                [15.99943],
                [14.00672],
                [12.01078],
                [12.01078],
                [12.01078],
                [15.99943],
            ]
        )
        super().__init__(name, sample_shape, kbT)

    def _get_mdtraj_topology(self, peptide: str) -> md.Topology:
        pdb_path = os.path.join(self.pdb_directory, f"{peptide}.pdb")
        topology = md.load_pdb(pdb_path).topology
        indices = topology.select(f"name {' or name '.join(ATOMS_TO_KEEP)}")
        return topology.subset(indices), indices

    def _get_phi_psi_indices(self) -> Tuple[Sequence[int], Sequence[int]]:
        return [3, 5, 6, 8], [0, 1, 3, 5]

    def _get_features(self, topology: md.Topology) -> jnp.ndarray:
        # get for each atom the corresponding amino acid
        amino_acids = [atom.residue.name for atom in topology.atoms]
        # get an index for each amino acid
        amino_acids_indices = [AMINO_ACIDS.index(aa) for aa in amino_acids]

        # let's do the same for the name of the atom
        atoms = [atom.name for atom in topology.atoms]
        atoms_indices = [ATOMS_TO_KEEP.index(atom) for atom in atoms]

        elements = [atom.element.symbol for atom in topology.atoms]
        elements_indices = [POSSIBLE_ELEMENTS.index(element) for element in elements]

        residue_index = [atom.residue.index for atom in topology.atoms]

        return jnp.stack(
            [
                jnp.array(amino_acids_indices),
                jnp.array(atoms_indices),
                jnp.array(elements_indices),
                jnp.array(residue_index),
            ],
            axis=1,
            dtype=jnp.int32,
        )

    def _reorder_atoms(
        self,
        peptide_1_atoms: Sequence[str],
        peptide_2_atoms: Sequence[str],
        target_peptide_1_atoms: Sequence[str],
        target_peptide_2_atoms: Sequence[str],
    ) -> Sequence[int]:
        # first, we ensure that all atoms are present
        assert set(target_peptide_1_atoms).issubset(set(peptide_1_atoms))
        assert set(target_peptide_2_atoms).issubset(set(peptide_2_atoms))

        # now, we sort the atoms and remember the order
        peptide_1_atoms_order = [peptide_1_atoms.index(atom) for atom in target_peptide_1_atoms]
        # we account for the offset in the indices
        peptide_2_atoms_order = [peptide_2_atoms.index(atom) + len(peptide_1_atoms) for atom in target_peptide_2_atoms]

        return peptide_1_atoms_order + peptide_2_atoms_order

    def _process_peptide(self, peptide: str, data: onp.ndarray) -> Datapoints:
        data = jnp.array(data).reshape(data.shape[0], -1, 3)  # the data is stored in nanometers

        # load corresponding pdb file
        topology, indices_to_keep = self._get_mdtraj_topology(peptide)
        assert len(indices_to_keep) == 10  # We assume for now that we keep all 5 atoms
        data = data[:, indices_to_keep, :]

        # center and pre-process data
        data, _ = kabsch_align_many(data, data[0])

        features = self._get_features(topology)
        # repeat the features for each batch element
        features = jnp.broadcast_to(features[None], (data.shape[0], *features.shape))
        assert features.shape == (data.shape[0], len(ATOMS_TO_KEEP) * 2, len(self.max_z))

        # reorder atoms so that they are consinstent across peptides
        peptide_1, peptide_2 = topology.residue(0), topology.residue(1)
        new_order = self._reorder_atoms(
            [atom.name for atom in peptide_1.atoms],
            [atom.name for atom in peptide_2.atoms],
            ATOMS_TO_KEEP,
            ATOMS_TO_KEEP,
        )
        data = data[:, new_order, :]
        features = features[:, new_order, :]
        return Datapoints(data.reshape(data.shape[0], -1), features)

    def _load_dataset(self, path: Optional[PathLike]) -> Optional[PeptideDatapoints]:
        if self.limit_samples is not None:
            log.warning(f"Limiting samples to {self.limit_samples}")

        if path is None:
            return None
        log.info(f"Loading dataset from {path}")
        filename = os.path.basename(path)
        data = onp.load(path, allow_pickle=True).item()

        processed_data = []
        peptides = []
        peptide_lengths = []
        with tqdm(data.items(), desc=f"Processing {filename}") as pbar:
            for peptide, peptide_data in pbar:
                pbar.set_postfix(peptide=peptide)
                if "G" in peptide:  # Removing Glycine for now (it does not have a CB atom)!
                    continue
                if self.limit_peptides is None or peptide in self.limit_peptides:
                    if self.limit_samples is not None:
                        # shuffle peptide_data with jax random permutation
                        key = jax.random.PRNGKey(0)
                        peptide_data = peptide_data[jax.random.permutation(key, len(peptide_data))]
                        peptide_data = peptide_data[: self.limit_samples]

                    processed_data.append(self._process_peptide(peptide, peptide_data))
                    peptides.append(peptide)
                    peptide_lengths.append(peptide_data.shape[0])
        data = processed_data

        if len(data) == 0:
            return None

        # build a dataset with all the data
        all_data = jnp.concatenate([datapoint.data for datapoint in data], axis=0)
        all_features = jnp.concatenate([datapoint.features for datapoint in data], axis=0)
        return PeptideDatapoints(all_data, all_features, peptides, peptide_lengths)

    @property
    def bonds(self) -> set[Tuple[int, int]]:
        # ["0,N", "1,CA", "2,CB", "3,C", "4,O"]
        # ["5,N", "6,CA", "7,CB", "8,C", "9,O"]
        # see topology.bonds(), but they have been reordered
        # Not sure if this is correct across all samples, but this is only for plotting bond distances,
        # so it is not too critical that it is exactly correct for all samples.
        # In the worst case we compute less meaningful bonds.
        return {(1, 3), (3, 4), (1, 2), (0, 1), (3, 5), (6, 8), (8, 9), (6, 7), (5, 6)}

    @property
    def atom_names(self):
        return ATOMS_TO_KEEP + ATOMS_TO_KEEP

    def get_2d_features(self, trajectory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        phi_indices, psi_indices = self._get_phi_psi_indices()

        # we jit it to save on GPU memory when reshaping
        @jax.jit
        def _get_2d_features(trajectory):
            trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)
            return dihedral(trajectory[:, phi_indices]), dihedral(trajectory[:, psi_indices])

        return _get_2d_features(trajectory)

    def _get_data(self) -> Tuple[PeptideDatapoints, Optional[PeptideDatapoints], Optional[PeptideDatapoints]]:
        log.warning("Removing Glycine for now (it does not have a CB atom)!")

        return (
            self._load_dataset(self.train_path),
            self._load_dataset(self.val_path),
            self._load_dataset(self.test_path),
        )

    def plot_2d(
        self, data, title="Histogram", highlight=None, bins=60, vmin=None, vmax=None, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        phi, psi = self.get_2d_features(data)

        plots.phi_psi(phi, psi, title, highlight, bins, vmin=vmin, vmax=vmax, **kwargs)
        return phi, psi

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
            gamma=1.0,  # simulation has been performed with 0.3 ps^-1, but this is not important for the sampling
            num_steps=num_steps,
            dt=dt if dt is not None else 5e-4,  # 5e-4 ps is the step size used in the simulation
            kbT=self.kbT,
        )

    def write_animation(self, trajectory: jnp.ndarray, peptide: str, out: PathLike):
        trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)
        topology, _ = self._get_mdtraj_topology(peptide)

        # we reorder our data back to the original order
        peptide_1, peptide_2 = topology.residue(0), topology.residue(1)
        new_order = self._reorder_atoms(
            ATOMS_TO_KEEP,
            ATOMS_TO_KEEP,
            [atom.name for atom in peptide_1.atoms],
            [atom.name for atom in peptide_2.atoms],
        )
        trajectory = trajectory[:, new_order, :]

        write_animation_with_topology(trajectory, topology.to_openmm(), out)
