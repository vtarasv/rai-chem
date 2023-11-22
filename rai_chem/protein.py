from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from .utils import logger, get_close_coords, rot_around_vec


class PDBParser:
    """
    A class to parse lines from Protein Data Bank (PDB) format and represent in table format

    Parameters
    ----------
    id_ : str
        identifier of the instance to use in logging
    lines : list
        lines from PDB file to parse
    first_model : bool
        if PDB file contains more than one model, use the first one
    model_id : str
        if PDB file contains more than one model, use the one specified in the parameter
    remove_hs : bool
        ignores all the hydrogens in the PDB file

    Attributes
    ----------
    id : str
        stores id_ parameter
    lines : list
        stores lines parameter
    first_model : bool
        stores first_model parameter
    model_id : str
        stores model_id parameter
    remove_hs : bool
        stores remove_hs parameter
    resolution : str
        PDB structure resolution (if applicable)
    resolution_lidx : int
        line index in the lines that contains resolution information
    atoms : pd.DataFrame
        parsed ATOM records
    hetatms : pd.DataFrame
        parsed HETATM records
    atom_fields : list
        name of the fields in ATOM / HETATM records
    conventional_atoms : list
        atom symbols present in 20 standard amino acids
    d3to1 : dict
        maps amino acid 3-letter code to the 1-letter code
    """

    atom_fields = ["serial", "name", "altLoc", "resName", "chainID", "resSeq", "iCode",
                   "x", "y", "z", "occupancy", "tempFactor", "element", "charge"]
    conventional_atoms = ["C", "N", "O", "S", "H"]
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    def __init__(self, id_: str, lines: List[str], *, first_model: bool = True, model_id: str = None,
                 remove_hs: bool = True):
        if first_model and model_id is not None:
            raise NotImplementedError("Use either 'first_model' or 'model_id'")

        self.id = id_
        self.lines = lines
        self.first_model = first_model
        self.model_id = str(model_id)
        self.remove_hs = remove_hs

        self.resolution = None
        self.resolution_lidx = None
        self.atoms = None
        self.hetatms = None

        self.parse_lines()

    def parse_lines(self):
        """Parses the PDB file lines and stores in table format"""

        model_id_curr = None
        collect_atoms = True
        atoms, hetatms = [], []
        for l_idx, l_str in enumerate(self.lines):
            if l_str.startswith("REMARK   2 RESOLUTION."):
                assert self.resolution is None, "Multiple resolutions found"
                self.resolution = self.parse_resolution(l_str)
                self.resolution_lidx = l_idx
            elif l_str.startswith("MODEL "):
                model_id_curr = l_str[10:14].strip()
                if self.model_id is not None:
                    if model_id_curr == self.model_id:
                        collect_atoms = True
                    else:
                        collect_atoms = False
            elif l_str.startswith("ATOM  "):
                if not collect_atoms:
                    continue
                d = self.parse_atom(l_str)
                if d["element"] == "H" and self.remove_hs:
                    continue
                d["l_idx"] = l_idx
                atoms.append(d)
            elif l_str.startswith("HETATM"):
                if not collect_atoms:
                    continue
                d = self.parse_atom(l_str)
                if d["element"] == "H" and self.remove_hs:
                    continue
                d["l_idx"] = l_idx
                hetatms.append(d)
            elif l_str.startswith("ENDMDL"):
                if self.first_model or ((self.model_id is not None) and (model_id_curr == self.model_id)):
                    break
            elif l_str.startswith("END   "):
                break

        if atoms:
            self.atoms = pd.DataFrame.from_records(atoms).astype(
                {**{col: "string" for col in self.atom_fields}, **{"l_idx": np.uint64}})
        if hetatms:
            self.hetatms = pd.DataFrame.from_records(hetatms).astype(
                {**{col: "string" for col in self.atom_fields}, **{"l_idx": np.uint64}})

    @staticmethod
    def parse_resolution(l_str: str) -> str:
        """Parses the resolution record"""
        assert l_str.startswith("REMARK   2 RESOLUTION.")
        res = ""
        if l_str[31:41].strip() == "ANGSTROMS.":
            res = l_str[23:30].strip()
        elif "NOT APPLICABLE" in l_str[11:38].strip():
            res = "NOT APPLICABLE"
        return res

    @staticmethod
    def parse_atom(l_str: str) -> dict:
        """
        Parses the ATOM / HETATM record
        https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
        """

        assert l_str.startswith("ATOM  ") or l_str.startswith("HETATM")
        return {
            "serial": l_str[6:11].strip(),  # Atom  serial number.
            "name": l_str[12:16].strip(),  # Atom name.
            "altLoc": l_str[16].strip(),  # Alternate location indicator.
            "resName": l_str[17:20].strip(),  # Residue name.
            "chainID": l_str[21].strip(),  # Chain identifier.
            "resSeq": l_str[22:26].strip(),  # Residue sequence number.
            "iCode": l_str[26].strip(),  # Code for insertion of residues.
            "x": l_str[30:38].strip(),  # Orthogonal coordinates for X in Angstroms.
            "y": l_str[38:46].strip(),  # Orthogonal coordinates for Y in Angstroms.
            "z": l_str[46:54].strip(),  # Orthogonal coordinates for Z in Angstroms.
            "occupancy": l_str[54:60].strip(),  # Occupancy.
            "tempFactor": l_str[60:66].strip(),  # Temperature  factor.
            "element": l_str[76:78].strip(),  # Element symbol, right-justified.
            "charge": l_str[78:80].strip(),  # Charge  on the atom.
        }


class BaseResidue:
    """
    A class to store basic residue data

    Attributes
    ----------
    df_res : pd.DataFrame
        table with the features of each atom of each residue (amino acid)
    resname_to_dfp : dict
        maps residue name to the atoms features table
    resname_to_size : dict
        maps residue name to the size (number of heavy atoms)
    resname_to_rings : dict
        maps residue name with aromatic ring(s) to the number of atoms in the ring
        and conventional order of each atom name
    resname_to_amides : dict
        maps residue name with side chain amide group to the atom names participating in the group
    resname_to_flip : dict
        maps residue name with potential group flips (amide, imidazole) to atom names of rotation vector
        and the group to flip
    resname_to_alths : dict
        maps residue name with potential alternative hydrogen positions to atom names of rotation vector
    resatom_to_hs : dict
        maps residue name to the list of expected hydrogen names
    resatom_to_nngb : dict
        maps residue name to the expected number of covalently bound heavy atoms (neighbors)
    """

    df_res = pd.read_csv(Path(__file__).parents[0] / "data/config/residues.csv")
    df_res.index = df_res["ResidueName"] + df_res["AtomName"]
    df_res["HNames"] = df_res["HNames"].fillna("")
    res_groups = df_res.groupby("ResidueName")
    resname_to_dfp = {k: v for k, v in res_groups}
    resname_to_size = res_groups.size().to_dict()
    resname_to_rings = {"PHE": {6: {"CG": 0, "CD1": 1, "CE1": 2, "CZ": 3, "CE2": 4, "CD2": 5}},
                        "TYR": {6: {"CG": 0, "CD1": 1, "CE1": 2, "CZ": 3, "CE2": 4, "CD2": 5}},
                        "TRP": {5: {"CG": 0, "CD1": 1, "NE1": 2, "CE2": 3, "CD2": 4},
                                6: {"CE2": 0, "CD2": 1, "CE3": 2, "CZ3": 3, "CH2": 4, "CZ2": 5}},
                        "HIS": {5: {"CG": 0, "ND1": 1, "CE1": 2, "NE2": 3, "CD2": 4}}}
    resname_to_amides = {"ASN": ["CG", "OD1", "ND2"], "GLN": ["CD", "OE1", "NE2"]}
    resname_to_flip = {"ASN": {"vect_st": "CB", "vect_en": "CG", "points": ["OD1", "ND2"]},
                       "GLN": {"vect_st": "CG", "vect_en": "CD", "points": ["OE1", "NE2"]},
                       "HIS": {"vect_st": "CB", "vect_en": "CG", "points": ["ND1", "CE1", "NE2", "CD2"]}}
    resname_to_alths = {"LYS": {"vect_st": "CE", "vect_en": "NZ"},
                        "SER": {"vect_st": "CB", "vect_en": "OG"},
                        "THR": {"vect_st": "CB", "vect_en": "OG1"},
                        "TYR": {"vect_st": "CZ", "vect_en": "OH"}}
    resatom_to_hs = df_res["HNames"].str.split("|").to_dict()
    resatom_to_hs = {k: [i for i in v if i] for k, v in resatom_to_hs.items()}
    resatom_to_nngb = df_res["NHeavyNeighbors"].to_dict()


class Protein(BaseResidue):
    """
    A class to extract and store features of a protein atoms, aromatic rings, and amide groups

    Parameters
    ----------
    id_ : str
        identifier of the instance to use in logging
    df_atoms : pd.DataFrame
        table with the parsed atom features obtained with PDBParser class

    Attributes
    ----------
    id : str
        stores id_ parameter
    df_atoms : pd.DataFrame
        stores df_atoms parameter
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
    atoms_flip : np.ndarray
        structured numpy array with heavy atoms index (Idx) and features after a group flip
    idx_to_hcoords_flip : dict
        maps heavy atoms index (Idx) to the coordinates of covalently bound hydrogens after a group flip
    """

    atom_dtypes = [
        ("Idx", "uint32"),
        ("Symbol", "U3"),
        ("ResidueID", "U20"),
        ("AtomName", "U3"),
        ("Hybridization", "U5"),
        ("TotalDegree", "uint8"),
        ("TotalNumHs", "uint8"),
        ("IsHydrophobe", "b"),
        ("IsHDonor", "b"),
        ("IsWeakHDonor", "b"),
        ("IsHAcceptor", "b"),
        ("IsPositive", "b"),
        ("IsNegative", "b"),
        ("InAromatic", "b"),
        ("InAmide", "b"),
        ("Coord", "float32", 3),
        ("ChainID", "U3"),
        ("ResidueIdx", "uint16"),
        ("ResidueName", "U3"),
        ("ICode", "U3"),
    ]
    pi_dtypes = [
        ("Centroid", "float32", 3),
        ("Normal", "float32", 3),
        ("ResidueID", "U20"),
        ("AtomIdxs", "O"),
    ]

    def __init__(self, id_: str, df_atoms: pd.DataFrame):
        """
        Parameters
        ----------
            df_atoms : pd.DataFrame
                pd.DataFrame obtained by PDBParser, each row - one atom
        """
        self.id = id_

        df_atoms = df_atoms.copy()
        df_atoms.loc[df_atoms["resName"] == "HID", "resName"] = "HIS"
        df_atoms.loc[df_atoms["resName"] == "HIE", "resName"] = "HIS"
        df_atoms.index = df_atoms["resName"] + df_atoms["name"]
        self.df_atoms = df_atoms

        self.atoms = None
        self.idx_to_hcoords = {}
        self.idx_to_ngbcoords = {}
        self._rings_res_size_idx = []
        self._amides_idx = []
        self.aromatics = None
        self.amides = None

        self._flips_res_idx = []
        self._alths_res_idx = []
        self.atoms_flip = None
        self.idx_to_hcoords_flip = {}

        self.get_atoms()
        self.get_aromatics()
        self.get_amides()
        self.get_alths()
        self.flip()
        self.get_ngb_coords()

    def get_atoms(self):
        """Extracts and stores atom features"""

        self.df_atoms["ResidueID"] = self.df_atoms["chainID"] + self.df_atoms["resSeq"].astype("string") + \
            self.df_atoms["iCode"] + self.df_atoms["resName"]
        self.df_atoms["serial"] = self.df_atoms["serial"].astype(np.uint32)
        self.df_atoms["TotalDegree"] = 0  # TotalDegree is filled later
        self.df_atoms["TotalNumHs"] = 0  # TotalNumHs is filled later
        self.df_atoms["IsAmide"] = 0  # InAmide is filled later
        self.df_atoms["Coord"] = self.df_atoms[["x", "y", "z"]].to_numpy().astype(np.float32).tolist()
        dfgs = self.df_atoms.groupby(["chainID", "resSeq", "resName", "iCode"], sort=False)

        props = []
        for (chainid, resseq, resname, icode), dfg in dfgs:
            if resname not in self.resname_to_dfp:
                logger.warning(f"id {self.id}; unrecognized residue name {resname}, skipping")
                continue
            df_props = self.resname_to_dfp[resname]
            exp_n_atoms = self.resname_to_size[resname]
            df_heavy = dfg[dfg["element"] != "H"]
            df_hs = dfg[dfg["element"] == "H"]
            res_hcoords = df_hs[["x", "y", "z"]].to_numpy().astype(np.float32)

            df = pd.merge(df_heavy, df_props, left_index=True, right_index=True)
            c_term = "OXT" in df["AtomName"].values
            if df.shape[0] != exp_n_atoms-int(not c_term):
                logger.warning(f"id {self.id}; chain {chainid}, residue {resname+resseq+icode} contains {df.shape[0]} "
                               f"heavy atoms instead of {exp_n_atoms} and will be skipped")
                continue

            if resname in self.resname_to_rings:
                rings_atoms = self.resname_to_rings[resname]
                for ring_size, ring_atoms in rings_atoms.items():
                    ring_atoms_idx = df[df["AtomName"].isin(ring_atoms)]["serial"].tolist()
                    self._rings_res_size_idx.append([resname, ring_size, ring_atoms_idx])
            if resname in self.resname_to_amides:
                amide_atoms = self.resname_to_amides[resname]
                amide_atoms_idx = df[df["AtomName"].isin(amide_atoms)]["serial"].tolist()
                self._amides_idx.append(amide_atoms_idx)
            if resname in self.resname_to_flip:
                flip = self.resname_to_flip[resname]
                flip_atoms = [flip["vect_st"], flip["vect_en"]] + flip["points"]
                flip_atoms_idx = df[df["AtomName"].isin(flip_atoms)]["serial"].tolist()
                self._flips_res_idx.append([resname, flip_atoms_idx])
            if resname in self.resname_to_alths:
                alth_config = self.resname_to_alths[resname]
                alth_atoms = [alth_config["vect_st"], alth_config["vect_en"]]
                alth_atoms_idx = df[df["AtomName"].isin(alth_atoms)]["serial"].tolist()
                self._alths_res_idx.append([resname, alth_atoms_idx])

            df = df[["serial", "element", "ResidueID", "AtomName", "Hybridization", "TotalDegree", "TotalNumHs",
                     "IsHydrophobe", "IsHDonor", "IsWeakHDonor", "IsHAcceptor", "IsPositive", "IsNegative",
                     "IsAromatic", "IsAmide", "Coord", "chainID", "resSeq", "ResidueName", "iCode"]]

            res_coords = np.vstack(df["Coord"]).astype(np.float32)
            res_dm = distance_matrix(res_coords, res_coords)
            ngb_mask = (res_dm < 1.95) & (res_dm > 0.1)
            for rown, (idx, row) in enumerate(df.iterrows()):
                ngbcoords = res_coords[ngb_mask[rown]]
                nngb_exp = self.resatom_to_nngb[idx]
                if ngbcoords.shape[0] != nngb_exp and row["AtomName"] not in ["C", "N"]:  # CYS-CYS is > 1.9A
                    logger.warning(f"id {self.id}; number of heavy neighbors is {ngbcoords.shape[0]}, "
                                   f"while {nngb_exp} expected; chain {chainid}, residue {resname + resseq + icode}, "
                                   f"atom {row['AtomName']}")
                self.idx_to_ngbcoords[row["serial"]] = ngbcoords

                currcoord = np.array(row["Coord"]).reshape(1, -1)
                _, h_mask = get_close_coords(currcoord, res_hcoords, 1.25)
                hcoords = res_hcoords[h_mask]

                row["TotalDegree"] = ngbcoords.shape[0] + hcoords.shape[0]
                row["TotalNumHs"] = hcoords.shape[0]
                nh_exp = len(self.resatom_to_hs[idx])

                if c_term and row["AtomName"] == "O":
                    row["IsNegative"] = 1

                if hcoords.shape[0] == nh_exp:
                    self.idx_to_hcoords[row["serial"]] = hcoords
                    props.append(tuple(row.to_list()))
                    continue

                if row["AtomName"] == "N":
                    if hcoords.shape[0] == 3 and row["ResidueName"] != "PRO":
                        row["IsPositive"] = 1
                    elif hcoords.shape[0] == 2 and row["ResidueName"] == "PRO":
                        row["IsPositive"] = 1
                    else:
                        logger.warning(f"id {self.id}; number of Hs is {hcoords.shape[0]}, while {nh_exp} expected; "
                                       f"chain {chainid}, residue {resname+resseq+icode}, atom {row['AtomName']}")
                elif (resname == "HIS" and row["AtomName"] in ["ND1", "NE2"]) or \
                        (resname == "CYS" and row["AtomName"] == "SG"):
                    # HIS might not include H on either of aromatic N; CYS SG won't have H in CYS-CYS
                    if row["AtomName"] == "SG":
                        row["IsWeakHDonor"] = 0  # remove donor from SG
                        row["IsHydrophobe"] = 1  # make S-S bridge hydrophobic
                    logger.debug(f"id {self.id}; number of Hs is {hcoords.shape[0]}, while {nh_exp} expected; "
                                 f"chain {chainid}, residue {resname+resseq+icode}, atom {row['AtomName']}")
                else:
                    logger.warning(f"id {self.id}; number of Hs is {hcoords.shape[0]}, while {nh_exp} expected; "
                                   f"chain {chainid}, residue {resname+resseq+icode}, atom {row['AtomName']}")

                self.idx_to_hcoords[row["serial"]] = hcoords
                props.append(tuple(row.to_list()))

        self.atoms = np.array(props, dtype=self.atom_dtypes)

    def get_aromatics(self):
        """Extracts and stores aromatic rings features"""

        rings_props = []
        for resname, ring_size, ring_atoms_idx in self._rings_res_size_idx:
            ring_atoms = self.atoms[np.in1d(self.atoms["Idx"], ring_atoms_idx)]
            ring_order = np.vectorize(self.resname_to_rings[resname][ring_size].get)(ring_atoms["AtomName"])
            ring_atoms = ring_atoms[np.argsort(ring_order)]
            centroid = ring_atoms["Coord"].mean(axis=0)
            ring_vectors = ring_atoms["Coord"] - centroid
            normal = np.cross(ring_vectors, np.roll(ring_vectors, shift=-1, axis=0)).mean(axis=0)
            normal /= np.linalg.norm(normal)
            ring_props = (centroid, normal, ring_atoms["ResidueID"][0], ring_atoms["Idx"])
            rings_props.append(ring_props)

        self.aromatics = np.array(rings_props, dtype=self.pi_dtypes)

    def get_amides(self):
        """Extracts and stores amide groups features"""

        back_c = self.atoms[self.atoms["AtomName"] == "C"]
        back_n = self.atoms[self.atoms["AtomName"] == "N"]
        back_o = self.atoms[self.atoms["AtomName"] == "O"]

        same_chainid = (back_c["ChainID"] == back_o["ChainID"]).all()
        same_residx = (back_c["ResidueIdx"] == back_o["ResidueIdx"]).all()
        same_icode = (back_c["ICode"] == back_o["ICode"]).all()
        if sum([same_chainid, same_residx, same_icode]) != 3:
            logger.error(f"id {self.id}; mixed order of backbone C and N or invalid residues")
            raise NotImplementedError

        c_mask, n_mask = get_close_coords(back_c["Coord"], back_n["Coord"], 1.5)
        close_c, close_o, close_n = back_c[c_mask], back_o[c_mask], back_n[n_mask]
        amides_atoms_idx = np.concatenate(
            [close_c["Idx"].reshape(-1, 1), close_o["Idx"].reshape(-1, 1), close_n["Idx"].reshape(-1, 1)], axis=1)
        if len(self._amides_idx):
            amides_atoms_idx = np.concatenate([np.array(self._amides_idx, dtype=np.uint32), amides_atoms_idx])
        self.atoms["InAmide"][np.in1d(self.atoms["Idx"], amides_atoms_idx.flatten())] = 1

        amides_props = []
        for amide_atoms_idx in amides_atoms_idx:
            amide_atoms = np.sort(self.atoms[np.in1d(self.atoms["Idx"], amide_atoms_idx)], order="Symbol")
            centroid = amide_atoms["Coord"].mean(axis=0)
            normal = np.cross(amide_atoms["Coord"][1] - amide_atoms["Coord"][0],
                              amide_atoms["Coord"][2] - amide_atoms["Coord"][0])
            normal /= np.linalg.norm(normal)

            #  ResidueID will be assigned to the residue with C=O
            amide_props = (centroid, normal, amide_atoms["ResidueID"][0], amide_atoms["Idx"])
            amides_props.append(amide_props)

        self.amides = np.array(amides_props, dtype=self.pi_dtypes)

    def flip(self):
        """Flips coordinates of side chain amides / imidazoles"""

        props = []
        for resname, flip_atoms_idx in self._flips_res_idx:
            flip_config = self.resname_to_flip[resname]
            _atoms = self.atoms[np.in1d(self.atoms["Idx"], flip_atoms_idx)]
            vect_st = _atoms[_atoms["AtomName"] == flip_config["vect_st"]]["Coord"]
            vect_en = _atoms[_atoms["AtomName"] == flip_config["vect_en"]]["Coord"]
            flip_atoms = _atoms[np.in1d(_atoms["AtomName"], flip_config["points"])]
            flip_hs_coord = np.concatenate([self.idx_to_hcoords[idx] for idx in flip_atoms["Idx"]])

            points = np.concatenate([flip_atoms["Coord"], flip_hs_coord])
            flipped_points = rot_around_vec(points, vect_en-vect_st, vect_en, 180).astype(np.float32)

            n_heavy = flip_atoms.shape[0]
            flip_atoms["Coord"] = flipped_points[:n_heavy]
            props.extend(flip_atoms.tolist())

            shift = 0
            for idx, n_hs in flip_atoms[["Idx", "TotalNumHs"]]:
                self.idx_to_hcoords_flip[idx] = flipped_points[n_heavy+shift:n_heavy+shift+n_hs]
                shift += n_hs

        self.atoms_flip = np.array(props, dtype=self.atom_dtypes)

    def get_alths(self):
        """Appends alternative hydrogen locations to the original"""
        for resname, alth_atoms_idx in self._alths_res_idx:
            alth_config = self.resname_to_alths[resname]
            _atoms = self.atoms[np.in1d(self.atoms["Idx"], alth_atoms_idx)]
            vect_st = _atoms[_atoms["AtomName"] == alth_config["vect_st"]]["Coord"]
            donor = _atoms[_atoms["AtomName"] == alth_config["vect_en"]]
            vect_en = donor["Coord"]
            idx = donor["Idx"].item()
            hcoords = self.idx_to_hcoords[idx]
            hcoords = np.concatenate([hcoords] + [rot_around_vec(hcoords, vect_en - vect_st, vect_en, angle)
                                                  for angle in [60, 120, 180, 240, 300]]).astype(np.float32)
            self.idx_to_hcoords[idx] = hcoords

    def get_ngb_coords(self):
        """Appends inter-residue covalent bonds to the existing intra-residue bonds"""
        back_c = self.atoms[self.atoms["AtomName"] == "C"]
        back_n = self.atoms[self.atoms["AtomName"] == "N"]
        cys_s = self.atoms[(self.atoms["ResidueName"] == "CYS") & (self.atoms["AtomName"] == "SG")]

        c_mask, n_mask = get_close_coords(back_c["Coord"], back_n["Coord"], 1.5)
        s1_mask, s2_mask = get_close_coords(cys_s["Coord"], cys_s["Coord"], 2.2)

        for c, n in zip(back_c[c_mask], back_n[n_mask]):
            c_ngbs, n_ngbs = self.idx_to_ngbcoords[c["Idx"]], self.idx_to_ngbcoords[n["Idx"]]
            self.idx_to_ngbcoords[c["Idx"]] = np.concatenate([c_ngbs, n["Coord"].reshape(1, -1)])
            self.idx_to_ngbcoords[n["Idx"]] = np.concatenate([n_ngbs, c["Coord"].reshape(1, -1)])

        for s1, s2 in zip(cys_s[s1_mask], cys_s[s2_mask]):
            if s1["Idx"] == s2["Idx"]:
                continue
            s1_ngbs, s2_ngbs = self.idx_to_ngbcoords[s1["Idx"]], self.idx_to_ngbcoords[s2["Idx"]]
            if s1_ngbs.shape[0] > 1:
                continue
            self.idx_to_ngbcoords[s1["Idx"]] = np.concatenate([s1_ngbs, s2["Coord"].reshape(1, -1)])
            self.idx_to_ngbcoords[s2["Idx"]] = np.concatenate([s2_ngbs, s1["Coord"].reshape(1, -1)])
