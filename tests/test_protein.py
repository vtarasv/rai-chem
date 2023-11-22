from pathlib import Path

import numpy as np
import pandas as pd

from rai_chem.protein import PDBParser, Protein


def test_atoms():
    prot_path = Path(__file__).parents[0] / "data/2hvc_hs.pdb"
    with open(prot_path, "r") as f:
        pdb_lines = f.readlines()
    prot = PDBParser("test", pdb_lines, remove_hs=False)
    prot = Protein("test", prot.atoms)

    assert prot.atoms.shape[0] == 2043
    assert np.isclose(prot.atoms["Coord"].mean(axis=0), np.array([7.2535925, 26.618773, 11.552467])).all()
    prot_props = \
        pd.DataFrame(prot.atoms[['TotalDegree', 'TotalNumHs', 'IsHydrophobe', 'IsHDonor', 'IsWeakHDonor',
                                 'IsHAcceptor', 'IsPositive', 'IsNegative', 'InAromatic', 'InAmide',
                                 'ResidueIdx']]).mean().to_numpy()
    prot_props_expected = \
        np.array([2.81595693e+00, 1.00978953e+00, 3.28438571e-01, 1.80616740e-01,
                  4.64512971e-01, 1.79637788e-01, 3.52422907e-02, 2.25159080e-02,
                  1.07684777e-01, 4.02349486e-01, 7.94876652e+02])
    assert np.isclose(prot_props, prot_props_expected).all()


def test_aromatics():
    prot_path = Path(__file__).parents[0] / "data/2hvc_hs.pdb"
    with open(prot_path, "r") as f:
        pdb_lines = f.readlines()
    prot = PDBParser("test", pdb_lines, remove_hs=False)
    prot = Protein("test", prot.atoms)

    assert prot.aromatics.shape[0] == 40
    assert np.isclose(prot.aromatics["Centroid"].mean(axis=0), np.array([8.068929, 27.613358, 10.354627])).all()


def test_amides():
    prot_path = Path(__file__).parents[0] / "data/2hvc_hs.pdb"
    with open(prot_path, "r") as f:
        pdb_lines = f.readlines()
    prot = PDBParser("test", pdb_lines, remove_hs=False)
    prot = Protein("test", prot.atoms)

    assert prot.amides.shape[0] == 274
    assert np.isclose(prot.amides["Centroid"].mean(axis=0), np.array([7.104117, 26.31128, 11.474164])).all()
