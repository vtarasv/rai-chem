from pathlib import Path

import numpy as np
from rdkit import Chem

from rai_chem.ligand import Ligand
from rai_chem.protein import PDBParser, Protein
from rai_chem.interaction import ProtLigInteractor


def test_interaction():
    mol_path = Path(__file__).parents[0] / "data/2hvc_B_LGD.sdf"
    mol = next(Chem.SDMolSupplier(str(mol_path)))
    mol = Chem.AddHs(mol, addCoords=True)
    lig = Ligand("test", mol)

    prot_path = Path(__file__).parents[0] / "data/2hvc_hs.pdb"
    with open(prot_path, "r") as f:
        pdb_lines = f.readlines()
    prot = PDBParser("test", pdb_lines, remove_hs=False)
    prot = Protein("test", prot.atoms)

    pli = ProtLigInteractor(prot, lig)
    df = pli.to_table()

    assert df.shape[0] == 52
    inter_props = df[["Distance", 'Theta', 'D-H-A', 'H-A-Y', 'X-D-A', 'D-A-Y']].mean().to_numpy()
    inter_props_expected = np.array([3.80268908, 82.59436035, 141.04113457, 130.6426699, 125.64221407, 131.55172335])
    assert np.isclose(inter_props, inter_props_expected).all()
