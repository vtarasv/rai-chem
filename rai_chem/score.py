from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
from rdkit.Chem import PandasTools

from .protein import PDBParser, Protein
from .interaction import get_fit_score


def main(args):
    with open(args.pdb, "r") as f:
        pdb_lines = f.readlines()
    prot = PDBParser(args.pdb, pdb_lines, remove_hs=False)
    prot = Protein(args.pdb, prot.atoms)

    df = PandasTools.LoadSDF(args.sdf, idName="ID", molColName="ROMol")
    df = df[~df["ROMol"].isna()]
    df = df[df["ROMol"].apply(lambda x: x.GetNumAtoms()) > 2]
    df.apply(lambda x: x["ROMol"].SetProp("_Name", x["ID"]), axis=1)

    fit_scores = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        mol = row["ROMol"]
        fscores = get_fit_score(mol, prot)
        fit_scores.append(fscores)

    df = pd.concat([df, pd.DataFrame(fit_scores)], axis=1)
    PandasTools.WriteSDF(df, str(args.save_path), molColName="ROMol", idName="ID",
                         properties={c: c for c in df.columns if c not in ["ROMol", "ID"]})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True, help="Path to input protein in PDB format")
    parser.add_argument("--sdf", type=str, required=True, help="Path to input ligands in SDF format")
    parser.add_argument("-s", "--save_path", type=str, required=True, help="Save path of output SDF file")

    main(parser.parse_args())
