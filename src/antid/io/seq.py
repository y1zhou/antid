"""Utility functions for sequence processing."""

from pathlib import Path

import gemmi

AA3TO1 = {
    "ASH": "A",
    "ALA": "A",
    "CYX": "C",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "MSE": "M",
    "ASN": "N",
    "PYL": "O",
    "HYP": "P",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "SEL": "U",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def struct2seq(path: str | Path, **kwargs) -> dict[str, str]:
    """Convert a PDB or mmCIF file to a dictionary of sequences.

    The dictionary keys are chain IDs and the values are sequences.
    Note that only the first model in the structure is considered.

    WARNING: non-standard residue names and non-polymer molecules are silently dropped
        unless they are passed as additional keyword arguments. For example, to map
        "ASH" to "A", you can call this function as `struct2seq(path, ASH="A")`.
        Similarly, you can map water molecules or ligands with
        `struct2seq(path, HOH="o", LIG="l")`.

    Args:
        path: Path to the PDB or mmCIF structure file. The file format is
            automatically detected from the content, and works for both
            text and gzipped files.
        **kwargs: Additional mappings for residue names (3-letter to 1-letter).

    Returns:
        A dictionary where keys are chain IDs and values are sequences.
    """
    st = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()

    # Get all residues
    all_res = {}
    aa_map = AA3TO1.copy() | kwargs
    for chain in st[0]:
        all_res[chain.name] = "".join(aa_map.get(res.name, "") for res in chain)

    return all_res


def fasta2seq(fasta_path: Path) -> dict[str, str]:
    """Convert a FASTA file to a dictionary of sequences."""
    from Bio import SeqIO

    res: dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        res[record.description] = str(record.seq)

    return res
