import os
from os import PathLike
import jax.numpy as jnp
import openmm.app as app


def write_animation_with_topology(trajectory: jnp.ndarray, topology: app.Topology, out: PathLike):
    """Write a trajectory to a PDB file. The trajectory is in nanometers."""
    with open(os.path.expanduser(out), "w") as pdbfile:
        app.PDBFile.writeHeader(topology, pdbfile)
        for i, xyz in enumerate(trajectory):
            positions = xyz.reshape(-1, 3) * 10  # in Angstrom
            app.PDBFile.writeModel(topology, positions, pdbfile, modelIndex=i + 1)
        app.PDBFile.writeFooter(topology, pdbfile)
