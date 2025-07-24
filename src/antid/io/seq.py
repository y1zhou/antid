"""Utility functions for sequence processing."""

from pathlib import Path

import gemmi

from antid.utils.constant import AA3TO1, NONSTD_RESIDUES


def struct2seq(
    path: str | Path, substitute_non_standard: bool = False, **kwargs
) -> dict[str, str]:
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
        substitute_non_standard: If True, non-standard residues will be substituted and included.
            Note that this has lower priority than any additional mappings provided.
        **kwargs: Additional mappings for residue names (3-letter to 1-letter).

    Returns:
        A dictionary where keys are chain IDs and values are sequences.
    """
    st = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()

    # Get all residues
    all_res = {}
    aa_map = AA3TO1.copy() | kwargs
    if substitute_non_standard:
        aa_map |= {k: AA3TO1[v] for k, v in NONSTD_RESIDUES.items() if k not in aa_map}
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
