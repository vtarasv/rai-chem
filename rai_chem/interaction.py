import json
from typing import Tuple, Dict
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem

from .protein import Protein
from .ligand import Ligand
from .utils import logger, get_close_coords, get_angle


class Interactor:
    """
    A class to store non-covalent interactions / unfavorable contacts configuration and methods for their detection

    The non-covalent interactions include:
        - interactions between hydrophobic C and S atoms (hydrophobecs_hydrophobecs method);
        - interactions between hydrophobic C/S and Cl/Br/I atoms (hydrophobecs_hydrophobeclbri method);
        - interactions between hydrophobic C/S and F atoms (hydrophobecs_hydrophobef method);
        - pi stacking (aromatic_aromatic method);
        - amide-pi (aromatic_amide method);
        - hydrogen bonds (hdonor_hacceptor method);
        - weak hydrogen bonds (weakhdonor_hacceptor method);
        - halogen bonds (clbri_hacceptor method);
        - electrostatic interactions (positive_negative method);
        - cation-pi (positive_aromatic method);

    The unfavorable contacts include:
        - distance between the atoms below the steric clashes threshold
          conditioned on the Van der Waals radii (unfavorable_vdw method);
        - hydrogen donors in proximity (unfavorable_hdonor method);
        - hydrogen acceptors in proximity (unfavorable_hacceptor method);
        - cations in proximity (unfavorable_positive method);
        - anions in proximity (unfavorable_negative method);

    Parameters
    ----------
    config : dict
        min, max distance (in angstrom, Å) and angle (in degrees, °) constraints
        for non-covalent interactions / unfavorable contacts

    Attributes
    ----------
    config : dict
        stores config parameter
    max_dist : float
        max potential interaction distance to filter out distal atoms / substructures pairs
    """

    max_dist = 6.0

    def __init__(self, config: dict):
        self.config = config
        periodic_table = Chem.GetPeriodicTable()
        self.get_rvdw = np.vectorize(lambda x: periodic_table.GetRvdw(periodic_table.GetAtomicNumber(x)))

    def hydrophobecs_hydrophobecs(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "HydrophobeCS-HydrophobeCS"
        config = self.config[key]
        hydrophobe1 = atoms1[(atoms1["IsHydrophobe"] == 1) & np.in1d(atoms1["Symbol"], ["C", "S"])]
        hydrophobe2 = atoms2[(atoms2["IsHydrophobe"] == 1) & np.in1d(atoms2["Symbol"], ["C", "S"])]
        mask1, mask2 = get_close_coords(hydrophobe1["Coord"], hydrophobe2["Coord"], config["max"])
        return key, hydrophobe1[mask1], hydrophobe2[mask2]

    def hydrophobecs_hydrophobeclbri(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "HydrophobeCS-HydrophobeClBrI"
        config = self.config[key]
        hydrophobe1 = atoms1[(atoms1["IsHydrophobe"] == 1) & np.in1d(atoms1["Symbol"], ["C", "S"])]
        hydrophobe2 = atoms2[(atoms2["IsHydrophobe"] == 1) & np.in1d(atoms2["Symbol"], ["Cl", "Br", "I"])]
        mask1, mask2 = get_close_coords(hydrophobe1["Coord"], hydrophobe2["Coord"], config["max"])
        return key, hydrophobe1[mask1], hydrophobe2[mask2]

    def hydrophobecs_hydrophobef(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "HydrophobeCS-HydrophobeF"
        config = self.config[key]
        hydrophobe1 = atoms1[(atoms1["IsHydrophobe"] == 1) & np.in1d(atoms1["Symbol"], ["C", "S"])]
        hydrophobe2 = atoms2[(atoms2["IsHydrophobe"] == 1) & (atoms2["Symbol"] == "F")]
        mask1, mask2 = get_close_coords(hydrophobe1["Coord"], hydrophobe2["Coord"], config["max"])
        return key, hydrophobe1[mask1], hydrophobe2[mask2]

    def aromatic_aromatic(self, aromatics1: np.ndarray, aromatics2: np.ndarray):
        key = "Aromatic-Aromatic"
        config = self.config[key]

        max_ftf = config["face-to-face"]["max"]
        mask1_ftf, mask2_ftf = get_close_coords(aromatics1["Centroid"], aromatics2["Centroid"], max_ftf)
        aromatics1_ftf, aromatics2_ftf = aromatics1[mask1_ftf], aromatics2[mask2_ftf]
        thetas_ftf = get_angle(aromatics1_ftf["Normal"], aromatics2_ftf["Normal"])
        tol_ftf = config["face-to-face"]["tolerance"]
        ftf_mask = (thetas_ftf <= 0+tol_ftf) | (thetas_ftf >= 180-tol_ftf)

        max_etf = config["edge-to-face"]["max"]
        mask1_etf, mask2_etf = get_close_coords(aromatics1["Centroid"], aromatics2["Centroid"], max_etf)
        aromatics1_etf, aromatics2_etf = aromatics1[mask1_etf], aromatics2[mask2_etf]
        thetas_etf = get_angle(aromatics1_etf["Normal"], aromatics2_etf["Normal"])
        tol_etf = config["edge-to-face"]["tolerance"]
        etf_mask = (thetas_etf >= 90-tol_etf) & (thetas_etf <= 90+tol_etf)

        return key, aromatics1_ftf[ftf_mask], aromatics2_ftf[ftf_mask], thetas_ftf[ftf_mask], \
            aromatics1_etf[etf_mask], aromatics2_etf[etf_mask], thetas_etf[etf_mask]

    def aromatic_amide(self, aromatics: np.ndarray, amides: np.ndarray):
        key = "Aromatic-Amide"
        config = self.config[key]

        max_ftf = config["face-to-face"]["max"]
        mask1_ftf, mask2_ftf = get_close_coords(aromatics["Centroid"], amides["Centroid"], max_ftf)
        aromatics_ftf, amides_ftf = aromatics[mask1_ftf], amides[mask2_ftf]
        thetas_ftf = get_angle(aromatics_ftf["Normal"], amides_ftf["Normal"])
        tol_ftf = config["face-to-face"]["tolerance"]
        ftf_mask = (thetas_ftf <= 0 + tol_ftf) | (thetas_ftf >= 180 - tol_ftf)

        max_etf = config["edge-to-face"]["max"]
        mask1_etf, mask2_etf = get_close_coords(aromatics["Centroid"], amides["Centroid"], max_etf)
        aromatics_etf, amides_etf = aromatics[mask1_etf], amides[mask2_etf]
        thetas_etf = get_angle(aromatics_etf["Normal"], amides_etf["Normal"])
        tol_etf = config["edge-to-face"]["tolerance"]
        etf_mask = (thetas_etf >= 90 - tol_etf) & (thetas_etf <= 90 + tol_etf)

        return key, aromatics_ftf[ftf_mask], amides_ftf[ftf_mask], thetas_ftf[ftf_mask], \
            aromatics_etf[etf_mask], amides_etf[etf_mask], thetas_etf[etf_mask]

    def hdonor_hacceptor(self, atoms1: np.ndarray, atoms2: np.ndarray, atoms1_hs: dict, atoms1_id: str,
                         atoms1_ngbs: dict, atoms2_ngbs: dict):
        key = "HDonor-HAcceptor"
        config = self.config[key]
        donors = atoms1[atoms1["IsHDonor"] == 1]
        acceptors = atoms2[atoms2["IsHAcceptor"] == 1]
        mask1, mask2 = get_close_coords(donors["Coord"], acceptors["Coord"], config["max"])
        donors, acceptors = donors[mask1], acceptors[mask2]

        dha, hay, xda, day = self._get_hbond_angles(donors, acceptors, atoms1_hs, atoms1_id, atoms1_ngbs, atoms2_ngbs)
        mask = (dha >= config["D-H-A"]) & (hay >= config["H-A-Y"])

        return key, donors[mask], acceptors[mask], \
            {"D-H-A": dha[mask], "H-A-Y": hay[mask], "X-D-A": xda[mask], "D-A-Y": day[mask]}

    def weakhdonor_hacceptor(self, atoms1: np.ndarray, atoms2: np.ndarray, atoms1_hs: dict, atoms1_id: str,
                             atoms1_ngbs: dict, atoms2_ngbs: dict):
        key = "WeakHDonor-HAcceptor"
        config = self.config[key]
        donors = atoms1[atoms1["IsWeakHDonor"] == 1]
        acceptors = atoms2[atoms2["IsHAcceptor"] == 1]
        mask1, mask2 = get_close_coords(donors["Coord"], acceptors["Coord"], config["max"])
        donors, acceptors = donors[mask1], acceptors[mask2]

        dha, hay, xda, day = self._get_hbond_angles(donors, acceptors, atoms1_hs, atoms1_id, atoms1_ngbs, atoms2_ngbs)
        mask = (dha >= config["D-H-A"]) & (hay >= config["H-A-Y"])

        return key, donors[mask], acceptors[mask], \
            {"D-H-A": dha[mask], "H-A-Y": hay[mask], "X-D-A": xda[mask], "D-A-Y": day[mask]}

    def clbri_hacceptor(self, atoms1: np.ndarray, atoms2: np.ndarray, atoms1_ngbs: dict, atoms2_ngbs: dict):
        key = "ClBrI-HAcceptor"
        config = self.config[key]
        donors = atoms1[atoms1["IsXDonor"] == 1]
        acceptors = atoms2[atoms2["IsHAcceptor"] == 1]
        mask1, mask2 = get_close_coords(donors["Coord"], acceptors["Coord"], config["max"])
        donors, acceptors = donors[mask1], acceptors[mask2]

        dxa, xay = self._get_xbond_angles(donors, acceptors, atoms1_ngbs, atoms2_ngbs)
        mask = (dxa >= config["D-X-A"]) & (xay >= config["X-A-Y"])

        return key, donors[mask], acceptors[mask], {"D-H-A": dxa[mask], "H-A-Y": xay[mask]}

    def positive_negative(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "Positive-Negative"
        config = self.config[key]
        positives = atoms1[atoms1["IsPositive"] == 1]
        negatives = atoms2[atoms2["IsNegative"] == 1]
        mask1, mask2 = get_close_coords(positives["Coord"], negatives["Coord"], config["max"])
        positives, negatives = positives[mask1], negatives[mask2]
        min_mask = np.linalg.norm(positives["Coord"] - negatives["Coord"], axis=1) > config["min"]
        return key, positives[min_mask], negatives[min_mask]

    def positive_aromatic(self, atoms: np.ndarray, aromatics: np.ndarray):
        key = "Positive-Aromatic"
        config = self.config[key]
        positives = atoms[atoms["IsPositive"] == 1]

        max_ftf = config["face-to-face"]["max"]
        mask1_ftf, mask2_ftf = get_close_coords(positives["Coord"], aromatics["Centroid"], max_ftf)
        positives, aromatics = positives[mask1_ftf], aromatics[mask2_ftf]
        thetas_ftf = get_angle(positives["Coord"] - aromatics["Centroid"], aromatics["Normal"])
        tol_ftf = config["face-to-face"]["tolerance"]
        ftf_mask = (thetas_ftf <= 0 + tol_ftf) | (thetas_ftf >= 180 - tol_ftf)

        positives, aromatics, thetas_ftf = positives[ftf_mask], aromatics[ftf_mask], thetas_ftf[ftf_mask]
        min_mask = np.linalg.norm(positives["Coord"] - aromatics["Centroid"], axis=1) > config["face-to-face"]["min"]

        return key, positives[min_mask], aromatics[min_mask], thetas_ftf[min_mask]

    def unfavorable_steric(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "StericBump-StericBump"
        config = self.config[key]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], config["max"])
        atoms1, atoms2 = atoms1[mask1], atoms2[mask2]

        da_min_mask = np.linalg.norm(atoms1["Coord"] - atoms2["Coord"], axis=1) > self.config["HDonor-HAcceptor"]["min"]
        da_mask = (atoms1["IsHDonor"] == 1) & (atoms2["IsHAcceptor"] == 1)
        ad_mask = (atoms1["IsHAcceptor"] == 1) & (atoms2["IsHDonor"] == 1)
        da_mask_full = da_min_mask & (da_mask | ad_mask)

        return key, atoms1[~da_mask_full], atoms2[~da_mask_full]

    def unfavorable_vdw(self, atoms1: np.ndarray, atoms2: np.ndarray, vdw_frac: float = None):
        key = "VDWClash"
        if vdw_frac is None:
            vdw_frac = self.config[key]["fraction"]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], 5*vdw_frac)
        atoms1, atoms2 = atoms1[mask1], atoms2[mask2]
        if not atoms1.shape[0]:
            return key, np.array([]), np.array([]), np.array([])

        vdw1, vdw2 = self.get_rvdw(atoms1["Symbol"]), self.get_rvdw(atoms2["Symbol"])
        dists = np.linalg.norm(atoms1["Coord"] - atoms2["Coord"], axis=1)
        dists_clash = (vdw1 + vdw2) * vdw_frac
        clash_mask = dists < dists_clash
        clash_size = dists_clash - dists

        return key, atoms1[clash_mask], atoms2[clash_mask], clash_size[clash_mask]

    def unfavorable_hdonor(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "HDonor-HDonor"
        config = self.config[key]
        atoms1 = atoms1[atoms1["IsHDonor"] == 1]
        atoms2 = atoms2[atoms2["IsHDonor"] == 1]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], config["max"])
        return key, atoms1[mask1], atoms2[mask2]

    def unfavorable_hacceptor(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "HAcceptor-HAcceptor"
        config = self.config[key]
        atoms1 = atoms1[atoms1["IsHAcceptor"] == 1]
        atoms2 = atoms2[atoms2["IsHAcceptor"] == 1]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], config["max"])
        return key, atoms1[mask1], atoms2[mask2]

    def unfavorable_positive(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "Positive-Positive"
        config = self.config[key]
        atoms1 = atoms1[atoms1["IsPositive"] == 1]
        atoms2 = atoms2[atoms2["IsPositive"] == 1]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], config["max"])
        return key, atoms1[mask1], atoms2[mask2]

    def unfavorable_negative(self, atoms1: np.ndarray, atoms2: np.ndarray):
        key = "Negative-Negative"
        config = self.config[key]
        atoms1 = atoms1[atoms1["IsNegative"] == 1]
        atoms2 = atoms2[atoms2["IsNegative"] == 1]
        mask1, mask2 = get_close_coords(atoms1["Coord"], atoms2["Coord"], config["max"])
        return key, atoms1[mask1], atoms2[mask2]

    @staticmethod
    def _get_hbond_angles(donors: np.ndarray, acceptors: np.ndarray,
                          donor_hs: dict, donor_id: str,
                          donor_ngbs: dict, acceptor_ngbs: dict) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts D-H-A, H-A-Y, X-D-A, D-A-Y angles for the hydrogen bond constraints check.
        D is the donor atom covalently bound to the hydrogen;
        H is a hydrogen atom;
        A is an acceptor atom;
        Y is a heavy covalently bound neighbor of the acceptor atom;
        X is a heavy covalently bound neighbor of the donor atom.
        """

        dha, hay, xda, day = [], [], [], []
        for donor, acceptor in zip(donors, acceptors):
            h_coords = donor_hs[donor["Idx"]]
            d_coord, a_coord = donor["Coord"].reshape(1, -1), acceptor["Coord"].reshape(1, -1)
            dngb_coords = donor_ngbs[donor["Idx"]]
            angb_coords = acceptor_ngbs[acceptor["Idx"]]

            if h_coords.shape[0]:
                ah_dists = np.linalg.norm(acceptor["Coord"] - h_coords, axis=1)
                h_coord = h_coords[np.argmin(ah_dists)].reshape(1, -1)

                dha_angle = get_angle(d_coord - h_coord, a_coord - h_coord).max()
                dha.append(dha_angle)
                hay_angle = get_angle(h_coord - a_coord, angb_coords - a_coord).min()
                hay.append(hay_angle)
            else:
                logger.debug(f"id {donor_id}; no H atoms found for donor with Idx {donor['Idx']}, skipping")
                dha.append(-1)
                hay.append(-1)

            xda_angle = get_angle(dngb_coords - d_coord, a_coord - d_coord).min()
            xda.append(xda_angle)
            day_angle = get_angle(d_coord - a_coord, angb_coords - a_coord).min()
            day.append(day_angle)

        return np.array(dha), np.array(hay), np.array(xda), np.array(day)

    @staticmethod
    def _get_xbond_angles(halogens: np.ndarray, acceptors: np.ndarray,
                          halogen_ngbs: dict, acceptor_ngbs: dict) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Extracts D-X-A, X-A-Y angles for the halogen bond constraints check.
        D is the donor atom covalently bound to the halogen;
        X is a halogen atom;
        A is the halogen bond acceptor atom;
        Y is a heavy covalently bound neighbor of the acceptor atom.
        """

        dxa, xay = [], []
        for halogen, acceptor in zip(halogens, acceptors):
            x_coord, a_coord = halogen["Coord"].reshape(1, -1), acceptor["Coord"].reshape(1, -1)
            xngb_coords = halogen_ngbs[halogen["Idx"]]
            angb_coords = acceptor_ngbs[acceptor["Idx"]]

            dxa_angle = get_angle(xngb_coords - x_coord, a_coord - x_coord).max()
            dxa.append(dxa_angle)
            xay_angle = get_angle(x_coord - a_coord, angb_coords - a_coord).min()
            xay.append(xay_angle)

        return np.array(dxa), np.array(xay)


class ProtLigInteractor(Interactor):
    """
    A class to detect protein-ligand non-covalent interactions and unfavorable contacts

    Parameters
    ----------
    prot : Protein
        instance of the protein.Protein class
    lig: Ligand
        instance of the ligand.Ligand class
    config : dict
        min, max distance (in angstrom, Å) and angle (in degrees, °) constraints
        for non-covalent interactions / unfavorable contacts

    Attributes
    ----------
    lig_id : str
        identifier of the ligand instance to use in logging
    lig_atoms : np.ndarray
        structured numpy array with ligand heavy atoms index (Idx) and features
    lig_aromatics : np.ndarray
        structured numpy array with ligand aromatic rings features
    lig_amides : np.ndarray
        structured numpy array with ligand amide groups features
    lig_hs : dict
        maps ligand heavy atoms index (Idx) to the coordinates of covalently bound hydrogens
    lig_ngbs : dict
        maps ligand heavy atoms index (Idx) to the coordinates of covalently bound heavy atoms (neighbors)

    prot_id : str
        identifier of the protein instance to use in logging
    prot_atoms : np.ndarray
        structured numpy array with protein heavy atoms index (Idx) and features
    prot_aromatics : np.ndarray
        structured numpy array with protein aromatic rings features
    prot_amides : np.ndarray
        structured numpy array with protein amide groups features
    prot_hs : dict
        maps protein heavy atoms index (Idx) to the coordinates of covalently bound hydrogens
    prot_ngbs : dict
        maps protein heavy atoms index (Idx) to the coordinates of covalently bound heavy atoms (neighbors)

    interactions : dict
        maps detected non-covalent interactions to the protein-ligand atoms / substructures
        pairs (stored as structured numpy arrays with keys 'prot' and 'lig')
        the arrays has one-to-one correspondence
    unfavorable : dict
        maps detected unfavorable contacts to the protein-ligand atoms
        pairs (stored as structured numpy arrays with keys 'prot' and 'lig');
        the arrays has one-to-one correspondence
    """

    def __init__(self, prot: Protein, lig: Ligand, config: dict = None):
        if config is None:
            with open(Path(__file__).parents[0] / "data/config/interaction_base.json", "r") as f:
                config = json.load(f)
        super().__init__(config)

        self.lig_id = lig.id
        self.lig_atoms = lig.atoms
        self.lig_aromatics = lig.aromatics
        self.lig_amides = lig.amides
        self.lig_hs = lig.idx_to_hcoords
        self.lig_ngbs = lig.idx_to_ngbcoords

        prot_atom_mask, _ = get_close_coords(prot.atoms["Coord"], lig.atoms["Coord"], self.max_dist)
        close_prot_atoms = prot.atoms[np.unique(prot_atom_mask)]
        prot_aramatics_mask = np.in1d(prot.aromatics["ResidueID"], close_prot_atoms["ResidueID"])
        prot_amides_mask = np.in1d(prot.amides["ResidueID"], close_prot_atoms["ResidueID"])

        self.prot_id = prot.id
        self.prot_atoms = close_prot_atoms
        self.prot_aromatics = prot.aromatics[prot_aramatics_mask]
        self.prot_amides = prot.amides[prot_amides_mask]
        self.prot_hs = {idx: prot.idx_to_hcoords[idx] for idx in self.prot_atoms["Idx"]}
        self.prot_ngbs = {idx: prot.idx_to_ngbcoords[idx] for idx in self.prot_atoms["Idx"]}

        self.interactions = self.get_interactions()
        self.unfavorable = self.get_unfavorable()

    def get_interactions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Detects non-covalent interactions"""

        interactions = defaultdict(dict)

        key, i1, i2 = self.hydrophobecs_hydrophobecs(self.prot_atoms, self.lig_atoms)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2

        key, i1, i2 = self.hydrophobecs_hydrophobeclbri(self.prot_atoms, self.lig_atoms)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2

        key, i1, i2 = self.hydrophobecs_hydrophobef(self.prot_atoms, self.lig_atoms)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2

        key, i1_ftf, i2_ftf, thetas_ftf, i1_etf, i2_etf, thetas_etf = \
            self.aromatic_aromatic(self.prot_aromatics, self.lig_aromatics)
        interactions[key+" face-to-face"]["prot"], interactions[key+" face-to-face"]["lig"] = i1_ftf, i2_ftf
        interactions[key+" face-to-face"]["Theta"] = thetas_ftf
        interactions[key+" edge-to-face"]["prot"], interactions[key+" edge-to-face"]["lig"] = i1_etf, i2_etf
        interactions[key+" edge-to-face"]["Theta"] = thetas_etf

        key, i1_ftf, i2_ftf, thetas_ftf, i1_etf, i2_etf, thetas_etf = \
            self.aromatic_amide(self.prot_aromatics, self.lig_amides)
        interactions[key+" face-to-face"]["prot"], interactions[key+" face-to-face"]["lig"] = i1_ftf, i2_ftf
        interactions[key+" face-to-face"]["Theta"] = thetas_ftf
        interactions[key+" edge-to-face"]["prot"], interactions[key+" edge-to-face"]["lig"] = i1_etf, i2_etf
        interactions[key+" edge-to-face"]["Theta"] = thetas_etf
        key, i1_ftf, i2_ftf, thetas_ftf, i1_etf, i2_etf, thetas_etf = \
            self.aromatic_amide(self.lig_aromatics, self.prot_amides)
        key = "-".join(reversed(key.split("-")))
        interactions[key+" face-to-face"]["lig"], interactions[key+" face-to-face"]["prot"] = i1_ftf, i2_ftf
        interactions[key+" face-to-face"]["Theta"] = thetas_ftf
        interactions[key+" edge-to-face"]["lig"], interactions[key+" edge-to-face"]["prot"] = i1_etf, i2_etf
        interactions[key+" edge-to-face"]["Theta"] = thetas_etf

        key, i1, i2, ang = self.hdonor_hacceptor(self.prot_atoms, self.lig_atoms, self.prot_hs, self.prot_id,
                                                 self.prot_ngbs, self.lig_ngbs)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2
        interactions[key] = {**interactions[key], **ang}
        key, i1, i2, ang = self.hdonor_hacceptor(self.lig_atoms, self.prot_atoms, self.lig_hs, self.lig_id,
                                                 self.lig_ngbs, self.prot_ngbs)
        key = "-".join(reversed(key.split("-")))
        interactions[key]["lig"], interactions[key]["prot"] = i1, i2
        interactions[key] = {**interactions[key], **ang}

        key, i1, i2, ang = self.clbri_hacceptor(self.lig_atoms, self.prot_atoms, self.lig_ngbs, self.prot_ngbs)
        key = "-".join(reversed(key.split("-")))
        interactions[key]["lig"], interactions[key]["prot"] = i1, i2
        interactions[key] = {**interactions[key], **ang}

        key, i1, i2 = self.positive_negative(self.prot_atoms, self.lig_atoms)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2
        key, i1, i2 = self.positive_negative(self.lig_atoms, self.prot_atoms)
        key = "-".join(reversed(key.split("-")))
        interactions[key]["lig"], interactions[key]["prot"] = i1, i2

        key, i1, i2, thetas = self.positive_aromatic(self.prot_atoms, self.lig_aromatics)
        interactions[key]["prot"], interactions[key]["lig"] = i1, i2
        interactions[key]["Theta"] = thetas
        key, i1, i2, thetas = self.positive_aromatic(self.lig_atoms, self.prot_aromatics)
        key = "-".join(reversed(key.split("-")))
        interactions[key]["lig"], interactions[key]["prot"] = i1, i2
        interactions[key]["Theta"] = thetas

        return interactions

    def get_unfavorable(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Detects unfavorable contacts"""

        unfavorable = defaultdict(dict)

        key, i1, i2, clash = self.unfavorable_vdw(self.prot_atoms, self.lig_atoms)
        unfavorable[key]["prot"], unfavorable[key]["lig"] = i1, i2
        unfavorable[key]["clash"] = clash

        key, i1, i2 = self.unfavorable_hdonor(self.prot_atoms, self.lig_atoms)
        unfavorable[key]["prot"], unfavorable[key]["lig"] = i1, i2

        key, i1, i2 = self.unfavorable_hacceptor(self.prot_atoms, self.lig_atoms)
        unfavorable[key]["prot"], unfavorable[key]["lig"] = i1, i2

        key, i1, i2 = self.unfavorable_positive(self.prot_atoms, self.lig_atoms)
        unfavorable[key]["prot"], unfavorable[key]["lig"] = i1, i2

        key, i1, i2 = self.unfavorable_negative(self.prot_atoms, self.lig_atoms)
        unfavorable[key]["prot"], unfavorable[key]["lig"] = i1, i2

        return unfavorable

    def to_table(self) -> pd.DataFrame:
        """Converts the extracted data to the convenient table format"""

        cols = ['ProtType', 'LigType', 'Prot', 'Lig', 'Distance', 'Theta', 'D-H-A', 'H-A-Y', 'X-D-A', 'D-A-Y',
                'X_prot', 'Y_prot', 'Z_prot', 'X_lig', 'Y_lig', 'Z_lig']
        dfs = []
        for interaction, ents in self.interactions.items():
            interaction = interaction.split(" ")
            prot_chem, lig_chem = interaction[0].split("-")
            pe, le = ents["prot"], ents["lig"]
            dfc = pd.DataFrame(index=np.arange(pe.shape[0]))
            dfc["ProtType"], dfc["LigType"] = prot_chem, lig_chem
            dfc["ResidueID"] = pe["ResidueID"]
            if "Aromatic" not in prot_chem and "Amide" not in prot_chem:
                dfc["ProtAtom"] = pe["AtomName"]
                prot_coord = pe["Coord"]
            else:
                prot_coord = pe["Centroid"]
            if "Aromatic" not in lig_chem and "Amide" not in lig_chem:
                dfc["LigIdx"], dfc["LigSymbol"] = le["Idx"], le["Symbol"]
                dfc["LigIdx"] = dfc["LigIdx"].astype("string")
                lig_coord = le["Coord"]
            else:
                lig_coord = le["Centroid"]
            dfc["Distance"] = np.linalg.norm(prot_coord - lig_coord, axis=1)
            for k, v in ents.items():
                if k not in ["prot", "lig"]:
                    dfc[k] = v
            dfc["X_prot"], dfc["Y_prot"], dfc["Z_prot"] = prot_coord[:, 0], prot_coord[:, 1], prot_coord[:, 2]
            dfc["X_lig"], dfc["Y_lig"], dfc["Z_lig"] = lig_coord[:, 0], lig_coord[:, 1], lig_coord[:, 2]
            dfs.append(dfc)
        df = pd.concat(dfs, ignore_index=True)
        df["Prot"] = df["ResidueID"].fillna("") + ":" + df["ProtAtom"].fillna("")
        df["Lig"] = df["LigSymbol"].fillna("") + df["LigIdx"].fillna("")
        df = df[cols]
        return df


# noinspection PyTypeChecker
def get_fit_score(pose_mol: rdkit.Chem.rdchem.Mol, prot: Protein) -> Dict[str, float]:
    """
    Calculates small molecule pose scores based on
    detected protein-ligand non-covalent interactions and unfavorable contacts.

    The scores include:
        - FavorableRate: total number of favorable interactions normalised by the number of ligand heavy atoms
          (the contribution of the hydrophobic interactions was accounted for with a weight of 1/6
          due to their high relative abundance);
        - FavorableRateUniq: fraction of the heavy ligand atoms participating in at least one non-bond interaction;
        - UnfavorableRate: total number of unfavorable interactions normalised by the number of ligand heavy atoms;
        - UnfavorableRateUniq: fraction of the ligand heavy atoms participating in at least one unfavorable interaction;
        - VDWClash: sum of all clash distances, which are below the steric clashes threshold;
        - FitScore: FavorableRate + FavorableRateUniq - 2*UnfavorableRate - 2*UnfavorableRateUniq - 10*VDWClash
    """

    _id = pose_mol.GetProp("_Name")
    pose_mol = Chem.RemoveAllHs(pose_mol)
    n_atoms = pose_mol.GetNumAtoms()
    try:
        pose_mol_hs = Chem.AddHs(pose_mol, addCoords=True)
        lig = Ligand(_id, pose_mol_hs)
        pli = ProtLigInteractor(prot, lig)
    except Exception as e:
        logger.warning(f"failed to get interaction data; {_id}; {e}")
        return {"FitScore": -100, "FavorableRate": 0, "FavorableRateUniq": 0,
                "UnfavorableRate": 0, "UnfavorableRateUniq": 0, "BumpScore": 0}
    else:
        favor_atoms, unfavor_atoms, vdw_clash = [], [], 0
        for k, v in pli.interactions.items():
            if not v["lig"].shape[0]:
                continue
            if k in ["HydrophobeCS-HydrophobeCS", "HydrophobeCS-HydrophobeClBrI", "HydrophobeCS-HydrophobeF"]:
                favor_atoms += v["lig"]["Idx"].tolist()
            elif k in ["Aromatic-Aromatic face-to-face", "Aromatic-Aromatic edge-to-face",
                       "Amide-Aromatic face-to-face", "Amide-Aromatic edge-to-face", "Positive-Aromatic"]:
                favor_atoms += np.concatenate(v["lig"]["AtomIdxs"]).tolist()
            elif k in ["Aromatic-Amide face-to-face", "Aromatic-Amide edge-to-face"]:
                favor_atoms += np.concatenate(v["lig"]["AtomIdxs"]).tolist() * 2
            elif k in ["HDonor-HAcceptor", "HAcceptor-HDonor", "HAcceptor-ClBrI",
                       "Positive-Negative", "Negative-Positive", "Aromatic-Positive"]:
                favor_atoms += v["lig"]["Idx"].tolist() * 6
            else:
                raise NotImplementedError(f"Unexpected interaction type: {k}")

        for k, v in pli.unfavorable.items():
            if k == "VDWClash":
                vdw_clash = v["clash"].sum()
            else:
                unfavor_atoms += v["lig"]["Idx"].tolist()

        favor_rate = len(favor_atoms) / (n_atoms * 6)
        favor_rate_uniq = len(set(favor_atoms)) / n_atoms
        unfavor_rate = len(unfavor_atoms) / n_atoms
        unfavor_rate_uniq = len(set(unfavor_atoms)) / n_atoms

        fit_score = favor_rate + favor_rate_uniq - 2*unfavor_rate - 2*unfavor_rate_uniq - 10*vdw_clash
        return {"FitScore": fit_score, "FavorableRate": favor_rate, "FavorableRateUniq": favor_rate_uniq,
                "UnfavorableRate": unfavor_rate, "UnfavorableRateUniq": unfavor_rate_uniq, "VDWClash": vdw_clash}
