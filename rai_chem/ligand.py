from typing import Tuple
from collections import deque
import copy

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from .utils import logger, rot_around_vec


atom_prop_to_smarts = {
    "IsHydrophobe":
        {"hydrophobe": Chem.MolFromSmarts("[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,Cl+0,Br+0,I+0,F+0$(F-C-[F,Cl,Br,I])]")},
    "IsHDonor":
        {"donor": Chem.MolFromSmarts('['
                                     '$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0]),'
                                     '$([$(n[n;H1]),$(nc[n;H1])]),'
                                     '$([Nv3;$([N;H0]=C-[N;!H0])]),'
                                     '$([O,S;H1;+0;!$(O-[C;v4]=[O;H0])])'
                                     ']')},
    "IsWeakHDonor":
        {"donor": Chem.MolFromSmarts("[#6!H0]")},
    "IsXDonor":
        {"donor": Chem.MolFromSmarts("[Cl,Br,I;X1;$([Cl,Br,I]-[#6])]")},
    "IsHAcceptor":
        {"acceptor": Chem.MolFromSmarts('['
                                        '$([O;H1;v2]),'
                                        '$([O;H0;v2;!$(O=N-*)]),'
                                        '$([O;-;!$(*-N=O)]),'
                                        '$([o;+0]),'
                                        '$([n;+0;!X3]),'
                                        '$([n;+0;X3;H1;$([n;H1][n;H0]),$([n;H1]c[n;H0])]),'
                                        '$([N;H0;$(N#[C&v4])]),'
                                        '$([N&v3;H0;$(Nc)]),'
                                        '$([N&v3;H0;$(N=C);!$(N=C-[N;!H0])]),'
                                        '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])'
                                        ']')},
    "IsPositive":
        {"rdkit basic group": Chem.MolFromSmarts("["
                                                 "$([Nv3;$([N;!H0]-C=N),$([N;H0]=C-[N;!H0]),$([N;H1]=C-N)]),"
                                                 "$([N;H2&+0][C;!$(C=*)]),"
                                                 "$([N;H1&+0]([C;!$(C=*)])[C;!$(C=*)]);"
                                                 "!$(N[a])"
                                                 "]"),
         "rdkit posn": Chem.MolFromSmarts("[#7;+;!$([N+]-[O-])]"),
         "cations": Chem.MolFromSmarts("[$([*+1,*+2,*+3]);!$([N+]-[O-])]"),
         "metals": Chem.MolFromSmarts("[Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,"
                                      "Pd,Ag,Cd,In,Sn,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,"
                                      "Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf]")},
    "IsNegative":
        {"O acidic group": Chem.MolFromSmarts("["
                                              "$([OH,O-]-[C,S,N,P,Cl,Br,I]=O),"
                                              "$(O=[C,S,N,P,Cl,Br,I]-[OH,O-])"
                                              "]"),
         "anions": Chem.MolFromSmarts("[*-1,*-2]")},
}

amide_smart = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
roth_60_smart = Chem.MolFromSmarts("[$([O;v2;H1]),$([N;v3;H2]),$([N;v4;H3]);!$(N-*=*)]")
roth_180_smart = Chem.MolFromSmarts("[$([N;v3;H1]),$([N;v4;H2]);$(N=*)]")

metal_anums = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
               50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
               87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
               102, 103]
halogens_anums = [9, 17, 35, 53]

params_3d = AllChem.ETKDGv2()


def slow_3d(mol: rdkit.Chem.rdchem.Mol, max_optim_iters: int = 1000) -> Tuple[rdkit.Chem.rdchem.Mol, int, int]:
    """Common implementation of the molecular conformer generation"""
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    status_3d = AllChem.EmbedMolecule(mol, params_3d)
    status_optim = AllChem.UFFOptimizeMolecule(mol, maxIters=max_optim_iters)
    return mol, status_3d, status_optim


def fast_3d(mol: rdkit.Chem.rdchem.Mol) -> Tuple[rdkit.Chem.rdchem.Mol, int]:
    """Fast implementation of the molecular conformer generation"""
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    mol = Chem.RemoveHs(mol)
    status_3d = AllChem.EmbedMolecule(mol, params_3d)
    mol = Chem.AddHs(mol, addCoords=True)
    return mol, status_3d


class Ligand:
    """
    A class to extract and store features of a small molecule atoms, aromatic rings, and amide groups

    Parameters
    ----------
    id_ : str
        identifier of the instance to use in logging
    mol : rdkit.Chem.rdchem.Mol
        input molecule to extract and store features

    Attributes
    ----------
    id : str
        stores id_ parameter
    mol : rdkit.Chem.rdchem.Mol
        stores mol parameter
    atoms : np.ndarray
        structured numpy array with heavy atoms index (Idx) and features
    idx_to_hcoords : dict
        maps heavy atoms index (Idx) to the coordinates of covalently bound hydrogens
    idx_to_ngbcoords : dict
        maps heavy atoms index (Idx) to the coordinates of covalently bound heavy atoms (neighbors)
    aromatics : np.ndarray
        structured numpy array with aromatic rings features
    amides : np.ndarray
        structured numpy array with amide groups features
    atom_dtypes : list
        stores array names and data types for the atom structured numpy array
    pi_dtypes : list
        stores array names and data types for the aromatic ring / amide group structured numpy array
    prop_to_draw : dict
        maps atoms features to the short representation in the PNG files
    """

    atom_dtypes = [
        ("Idx", "uint32"),
        ("Symbol", "U3"),
        ("AtomicNum", "uint8"),
        ("Hybridization", "U5"),
        ("TotalDegree", "uint8"),
        ("TotalValence", "uint8"),
        ("TotalNumHs", "uint8"),
        ("Mass", "float32"),
        ("IsHydrophobe", "b"),
        ("IsHDonor", "b"),
        ("IsWeakHDonor", "b"),
        ("IsXDonor", "b"),
        ("IsHAcceptor", "b"),
        ("IsPositive", "b"),
        ("IsNegative", "b"),
        ("InAromatic", "b"),
        ("InAmide", "b"),
        ("IsHalogen", "b"),
        ("IsMetal", "b"),
        ("Coord", "float32", 3),
    ]
    pi_dtypes = [
        ("Centroid", "float32", 3),
        ("Normal", "float32", 3),
        ("AtomIdxs", "O"),
    ]
    prop_to_draw = \
        {"IsHydrophobe": "Hp", "IsHDonor": "HD", "IsWeakHDonor": "WHD", "IsXDonor": "XD",
         "IsHAcceptor": "HA", "IsPositive": "+", "IsNegative": "-", "InAromatic": "Ar", "InAmide": "Am"}

    def __init__(self, id_: str, mol: rdkit.Chem.rdchem.Mol):
        self.id = id_
        self.mol = mol
        try:
            self.mol.GetConformer()
        except ValueError:
            self.mol, status_3d, status_optim = slow_3d(self.mol)
            if status_3d != 0:
                logger.error(f"id: {self.id}; failed to generate conformer")
                raise NotImplementedError
            if status_optim != 0:
                logger.warning(f"id: {self.id}; failed to optimize conformer")

        self.atoms = None
        self.idx_to_hcoords = {}
        self.idx_to_ngbcoords = {}
        self.aromatics = None
        self.amides = None

        self.get_atoms()
        self.get_aromatics()
        self.get_amides()

    def get_atoms(self):
        """Extracts and stores atom features"""

        conf = self.mol.GetConformer()

        mol_patterns = {
            prop: np.unique(np.concatenate([
                np.array(self.mol.GetSubstructMatches(v)).flatten()
                for k, v in smarts.items()]))
            for prop, smarts in atom_prop_to_smarts.items()
        }
        roth_60_idxs = np.array(self.mol.GetSubstructMatches(roth_60_smart)).flatten()
        roth_180_idxs = np.array(self.mol.GetSubstructMatches(roth_180_smart)).flatten()

        props = []
        for atom in self.mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum == 1:
                continue
            idx = atom.GetIdx()
            formal_charge = atom.GetFormalCharge()
            coord = conf.GetAtomPosition(idx)

            ngbcoords, hcoords = [], []
            for aneighbor in atom.GetNeighbors():
                _coord = conf.GetAtomPosition(aneighbor.GetIdx())
                if aneighbor.GetAtomicNum() != 1:
                    ngbcoords.append(_coord)
                else:
                    hcoords.append(_coord)
            ngbcoords, hcoords = np.array(ngbcoords), np.array(hcoords)

            atom_props = (
                idx,
                atom.GetSymbol(),
                anum,
                str(atom.GetHybridization()),
                atom.GetTotalDegree(),
                atom.GetTotalValence(),
                hcoords.shape[0],
                atom.GetMass(),

                idx in mol_patterns["IsHydrophobe"],
                idx in mol_patterns["IsHDonor"],
                idx in mol_patterns["IsWeakHDonor"],
                idx in mol_patterns["IsXDonor"],
                idx in mol_patterns["IsHAcceptor"],
                idx in mol_patterns["IsPositive"] or formal_charge >= 1,
                idx in mol_patterns["IsNegative"] or formal_charge <= -1,
                atom.GetIsAromatic(),  # idx in mol_patterns["InAromatic"]
                0,  # InAmide is filled later
                anum in halogens_anums,
                anum in metal_anums,

                tuple(coord),
            )
            props.append(atom_props)

            if (idx in roth_60_idxs or idx in roth_180_idxs) and hcoords.shape[0]:
                vect_st = np.array([list(conf.GetAtomPosition(aneighbor.GetIdx())) for aneighbor in
                                    atom.GetNeighbors() if aneighbor.GetAtomicNum() != 1])
                assert vect_st.shape[0] == 1
                angles = [60, 120, 180, 240, 300] if idx in roth_60_idxs else [180]
                hcoords = np.concatenate([hcoords] + [rot_around_vec(hcoords, coord - vect_st, coord, angle)
                                                      for angle in angles]).astype(np.float32)
                logger.debug(f"id: {self.id}; atom {atom.GetSymbol() + str(idx)}; "
                             f"added {len(angles)} alternative H coordinates per H")

            self.idx_to_hcoords[idx] = hcoords
            self.idx_to_ngbcoords[idx] = ngbcoords

        self.atoms = np.array(props, dtype=self.atom_dtypes)

    def get_aromatics(self):
        """Extracts and stores aromatic rings features"""

        ri = self.mol.GetRingInfo()
        rings_props = []
        for ring_atoms_idx in ri.AtomRings():
            ring_atoms_idx = self.canonize_ring_path(np.array(ring_atoms_idx))
            ring_atoms = self.atoms[np.in1d(self.atoms["Idx"], ring_atoms_idx)]
            aromatic = (ring_atoms["InAromatic"] == 1).all()
            if not aromatic:
                continue
            ring_order = np.vectorize({k: v for v, k in enumerate(ring_atoms_idx)}.get)(ring_atoms["Idx"])
            ring_atoms = ring_atoms[np.argsort(ring_order)]
            centroid = ring_atoms["Coord"].mean(axis=0)
            ring_vectors = ring_atoms["Coord"] - centroid
            normal = np.cross(ring_vectors, np.roll(ring_vectors, shift=-1, axis=0)).mean(axis=0)
            normal /= np.linalg.norm(normal)
            ring_props = (centroid, normal, ring_atoms["Idx"])
            rings_props.append(ring_props)

        self.aromatics = np.array(rings_props, dtype=self.pi_dtypes)

    def get_amides(self):
        """Extracts and stores amide groups features"""

        amides_props = []
        for amide_atoms_idx in self.mol.GetSubstructMatches(amide_smart):
            amide_atoms_idx = np.array(amide_atoms_idx)
            self.atoms["InAmide"][np.in1d(self.atoms["Idx"], amide_atoms_idx)] = 1
            amide_atoms = np.sort(self.atoms[np.in1d(self.atoms["Idx"], amide_atoms_idx)], order="Symbol")
            centroid = amide_atoms["Coord"].mean(axis=0)
            normal = np.cross(amide_atoms["Coord"][1] - amide_atoms["Coord"][0],
                              amide_atoms["Coord"][2] - amide_atoms["Coord"][0])
            normal /= np.linalg.norm(normal)
            amide_props = (centroid, normal, amide_atoms["Idx"])
            amides_props.append(amide_props)

        self.amides = np.array(amides_props, dtype=self.pi_dtypes)

    @staticmethod
    def canonize_ring_path(path: np.ndarray) -> list:
        path = list(path)
        path_deque = deque(path)
        path_deque.rotate(-path.index(min(path)))
        if path_deque[1] - path_deque[0] > path_deque[-1] - path_deque[0]:
            path_deque.reverse()
            path_deque.rotate(1)
        return list(path_deque)

    def to_table(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converts the extracted data to the convenient table format"""

        df_atoms = pd.DataFrame([list(i) for i in self.atoms], columns=[i[0] for i in self.atom_dtypes])
        df_atoms["x"], df_atoms["y"], df_atoms["z"] = df_atoms["Coord"].map(lambda x: x[0]), \
            df_atoms["Coord"].map(lambda x: x[1]), df_atoms["Coord"].map(lambda x: x[2])
        df_atoms.drop(["Coord"], axis=1, inplace=True)

        df_aromatics = pd.DataFrame([list(i) for i in self.aromatics], columns=[i[0] for i in self.pi_dtypes])
        df_amides = pd.DataFrame([list(i) for i in self.amides], columns=[i[0] for i in self.pi_dtypes])
        df_aromatics["Type"], df_amides["Type"] = "Aromatic", "Amide"
        df_pi = pd.concat([df_aromatics, df_amides], ignore_index=True)
        df_pi["x_centroid"], df_pi["y_centroid"], df_pi["z_centroid"] = df_pi["Centroid"].map(lambda x: x[0]), \
            df_pi["Centroid"].map(lambda x: x[1]), df_pi["Centroid"].map(lambda x: x[2])
        df_pi["x_normal"], df_pi["y_normal"], df_pi["z_normal"] = df_pi["Normal"].map(lambda x: x[0]), \
            df_pi["Normal"].map(lambda x: x[1]), df_pi["Normal"].map(lambda x: x[2])
        df_pi["AtomIdxs"] = df_pi["AtomIdxs"].map(lambda x: "|".join([str(i) for i in x]))
        df_pi.drop(["Centroid", "Normal"], axis=1, inplace=True)

        return df_atoms, df_pi

    def draw(self, save_path):
        """Saves PNG file with the visualised atom features"""

        mol = copy.deepcopy(self.mol)
        mol.RemoveAllConformers()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            if atom.GetAtomicNum() == 6:
                atom.SetProp("atomLabel", "")
            props = self.atoms[self.atoms["Idx"] == atom.GetIdx()]
            atom_note = list()
            atom_note.append(props["Hybridization"].item())
            for prop, draw in self.prop_to_draw.items():
                if props[prop].item() > 0:
                    atom_note.append(draw)
            atom.SetProp("atomNote", "|".join(atom_note))

        mol = Chem.RemoveAllHs(mol)
        d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
        d.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        d.WriteDrawingText(str(save_path))
