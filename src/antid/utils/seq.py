"""Utility functions for sequence processing."""

from pathlib import Path

import polars as pl

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

    Args:
        path: Path to the PDB or mmCIF structure file.
        **kwargs: Additional mappings for residue names (3-letter to 1-letter).

    Returns:
        A dictionary where keys are chain IDs and values are sequences.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()

    # Get all residues
    all_res = {}
    aa_map = AA3TO1.copy() | kwargs
    for chain in st[0]:
        all_res[chain.name] = "".join(aa_map[res.name] for res in chain)

    return all_res


def fasta2seq(fasta_path: Path) -> dict[str, str]:
    """Convert a FASTA file to a dictionary of sequences."""
    from Bio import SeqIO

    res: dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        res[record.id] = str(record.seq)

    return res


def align_pdb_seq_to_ref(
    ref_seq: str,
    pdb_seqs_df: pl.DataFrame,
    pdb_chain: str,
    chain_col_name: str = "chain_id",
    resi_col_name: str = "resi",
    resn_col_name: str = "resn",
    ref_idx_col_name: str = "ref_idx",
    resn_3to1: bool = False,
) -> pl.DataFrame:
    """Align a PDB sequence to a reference sequence.

    The best use case for this function is to identify missing densities
    in a PDB file. When there are large gaps, the quality of the alignment
    is not guaranteed.

    Args:
        ref_seq: The reference sequence, which should be >= len(pdb_seq).
        pdb_seqs_df: The PDB sequence DataFrame.
        pdb_chain: The chain ID of the PDB file to align the reference sequence to.
        chain_col_name: The name of the column containing chain IDs.
        resi_col_name: The name of the column containing residue indices.
        resn_col_name: The name of the column containing residue names.
        ref_idx_col_name: The name of the column to use for the reference index in the returned DataFrame.
        resn_3to1: Whether to convert 3-letter residue names to 1-letter names.

    Returns:
        A DataFrame with the aligned sequences.

    """
    if resn_3to1:
        pdb_seqs_df = pdb_seqs_df.with_columns(
            pl.col(resn_col_name).replace_strict(AA3TO1)
        )

    # Collect the current sequence in the PDB file
    pdb_seqs = (
        pdb_seqs_df.group_by(pl.col(chain_col_name))
        .agg(pl.col(resn_col_name).alias("seq"))
        .with_columns(pl.col("seq").list.join(""))
    )
    pdb_seq = pdb_seqs.filter(pl.col(chain_col_name) == pl.lit(pdb_chain)).item(
        0, "seq"
    )

    # The alignment is trivial if the two sequences are identical
    if pdb_seq == ref_seq:
        return (
            pdb_seqs_df.filter(pl.col(chain_col_name) == pl.lit(pdb_chain))
            .with_row_index(name=ref_idx_col_name, offset=1)
            .select(
                chain_col_name,
                pl.col(ref_idx_col_name).cast(pl.Utf8),
                resi_col_name,
                resn_col_name,
            )
        )

    from Bio import Align

    aligner = Align.PairwiseAligner(scoring="blastp")
    aln = next(aligner.align(ref_seq, pdb_seq))
    ref_aln: str = aln[0]
    pdb_aln: str = aln[1]

    chain_resi = pdb_seqs_df.filter(pl.col(chain_col_name) == pl.lit(pdb_chain)).select(
        resi_col_name, resn_col_name
    )

    resi_idx = 0
    map_entries = []
    for i, (aa1, aa2) in enumerate(zip(ref_aln, pdb_aln, strict=True)):
        if aa1 == "-":
            raise ValueError(
                f"Reference sequence should not have gaps:\nRef: {ref_aln}\nPDB: {pdb_aln}"
            )

        map_entry = {ref_idx_col_name: i + 1, resi_col_name: None, resn_col_name: aa1}
        if aa2 != "-":
            map_entry[resi_col_name] = chain_resi.item(resi_idx, resi_col_name)
            resi_idx += 1

        map_entries.append(map_entry)
    return pl.from_records(map_entries, orient="row", infer_schema_length=None).select(
        pl.lit(pdb_chain).alias(chain_col_name),
        pl.col(ref_idx_col_name).cast(pl.Utf8),
        resi_col_name,
        resn_col_name,
    )
