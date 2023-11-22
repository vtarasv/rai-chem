from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from rai_chem.ligand import Ligand


def test_atoms():
    mol_path = Path(__file__).parents[0] / "data/2hvc_B_LGD.sdf"
    mol = next(Chem.SDMolSupplier(str(mol_path)))
    mol = Chem.AddHs(mol, addCoords=True)
    lig = Ligand("test", mol)

    assert lig.atoms.shape[0] == 26
    assert np.isclose(lig.atoms["Coord"].mean(axis=0), np.array([1.2599616, 29.772152, 4.949961])).all()
    lig_props = \
        pd.DataFrame(lig.atoms[['AtomicNum', 'TotalDegree', 'TotalValence', 'TotalNumHs', 'Mass',
                                'IsHydrophobe', 'IsHDonor', 'IsWeakHDonor', 'IsXDonor', 'IsHAcceptor',
                                'IsPositive', 'IsNegative', 'InAromatic', 'InAmide',
                                'IsHalogen', 'IsMetal']]).mean().to_numpy()
    lig_props_expected = \
        np.array([7.19230769, 2.42307692, 2.80769231, 0.34615385, 14.73649788, 0.57692308, 0.03846154, 0.23076923,
                  0., 0.07692308, 0., 0., 0.38461538, 0., 0.34615385, 0.])
    assert np.isclose(lig_props, lig_props_expected).all()


def test_aromatics():
    mol_path = Path(__file__).parents[0] / "data/2hvc_B_LGD.sdf"
    mol = next(Chem.SDMolSupplier(str(mol_path)))
    mol = Chem.AddHs(mol, addCoords=True)
    lig = Ligand("test", mol)

    assert lig.aromatics.shape[0] == 2
    assert np.isclose(lig.aromatics["Centroid"].mean(axis=0), np.array([1.7994165, 27.72375, 4.8204165])).all()


def test_amides():
    mol = Chem.MolFromSmiles("CC(C(=O)NC(CCC(=O)O)C(=O)N)NC(=O)C(C)OC1C(C(OC(C1O)CO)O)NC(=O)C")
    lig = Ligand("test", mol)

    assert lig.amides.shape[0] == 4
