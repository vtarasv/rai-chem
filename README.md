[test-image]: https://github.com/vtarasv/rai-chem/actions/workflows/test.yml/badge.svg
[test-url]: https://github.com/vtarasv/rai-chem/actions/workflows/test.yml

### Deployment
[![Test Status][test-image]][test-url]

### Installation
`pip install rai-chem`

### Usage
This package can be utilised to: 
- extract useful atom/subgroup features from small molecules and proteins
  (e.g. hybridization, valence, hydrophobicity, donor/acceptor properties, electrostatic properties)
- detect intermolecular non-covalent interactions 
  (e.g. hydrophobic, electrostatic, hydrogen, halogen bonds, pi stacking, amide-pi, cation-pi)
- detect unfavorable atom-atom contacts 
  (e.g. steric clashes, donors/acceptors/cations/anions in proximity)

We recommend to add the hydrogens to protein PDB files as it is vital for the hydrogen bonds detection.
If your input files don't include H atoms, you may use [reduce](https://github.com/rlabduke/reduce) package or any other tool of your choice.

To extract and store features of a small molecule atoms (`lig.atoms`), aromatic rings (`lig.aromatics`), and amide groups (`lig.amides`):
```
from rdkit import Chem
from rai_chem.ligand import Ligand

mol = next(Chem.SDMolSupplier('my_ligand.sdf', removeHs=False))
mol = Chem.AddHs(mol, addCoords=True)  # if SDF file doesn't contain Hs

lig = Ligand('my_ligand', mol)
```

To extract and store features of a protein atoms (`prot.atoms`), aromatic rings (`prot.aromatics`), and amide groups (`prot.amides`):
```
from rai_chem.protein import PDBParser, Protein

with open('my_protein.pdb', "r") as f:
    pdb_lines = f.readlines()
prot_table = PDBParser('my_protein', pdb_lines, remove_hs=False)
prot = Protein('my_protein', prot_table.atoms)
```

To detect protein-ligand non-covalent interactions (`pli.interactions`, `pli.to_table()`) and unfavorable contacts (`pli.unfavorable`):
```
from rai_chem.interaction import ProtLigInteractor

pli = ProtLigInteractor(prot, lig)
df_interactions = pli.to_table()
```

The scoring of small molecule poses within protein binding site can be performed with:

`python -m rai_chem.score --pdb my_protein.pdb --sdf my_ligands.sdf --save_path ligands_pose_score.sdf`

The scores in the output SDF file include:
- FavorableRate: total number of favorable interactions normalised by the number of ligand heavy atoms
  (the contribution of the hydrophobic interactions was accounted for with a weight of 1/6
  due to their high relative abundance);
- FavorableRateUniq: fraction of the heavy ligand atoms participating in at least one non-bond interaction;
- UnfavorableRate: total number of unfavorable interactions normalised by the number of ligand heavy atoms;
- UnfavorableRateUniq: fraction of the ligand heavy atoms participating in at least one unfavorable interaction;
- VDWClash: sum of all clash distances, which are below the steric clashes threshold;
- FitScore: FavorableRate + FavorableRateUniq - 2*UnfavorableRate - 2*UnfavorableRateUniq - 10*VDWClash

### Cite
This package was created and updated as part of the following works within [RECEPTOR.AI](https://receptor.ai/):

- Boosting performance of generative diffusion model for molecular docking by training on artificial binding pockets 
  (To be added)
  
- 3DProtDTA: a deep learning model for drug-target affinity prediction based on residue-level protein graphs
  (DOI: [doi.org/10.1039/D3RA00281K](https://doi.org/10.1039/D3RA00281K))

Please cite the above-mentioned preprints/articles if you use this code in your own work.

### Acknowledgements
This package is inspired by the [Open Drug Discovery Toolkit](https://github.com/oddt/oddt.git) and based on the [RDKit](https://github.com/rdkit/rdkit.git)