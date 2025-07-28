"""Utility functions for dealing with PDB structure files."""

from pathlib import Path

import gemmi
import polars as pl

from antid.io.seq import AA3TO1
from antid.utils import check_path


def align_ref_seq_to_struct(
    ref_seq: str,
    pdb_seqs_df: pl.DataFrame,
    pdb_chain: str,
    chain_col_name: str = "chain",
    resi_col_name: str = "resi",
    insertion_code_col_name: str = "insertion",
    resn_col_name: str = "resn",
    ref_idx_col_name: str = "ref_idx",
    resn_3to1: bool = False,
    **kwargs,
) -> pl.DataFrame:
    """Align a reference sequence to the observed residues in a structure.

    The best use case for this function is to identify missing densities
    in a PDB file. When there are large gaps, the quality of the alignment
    is not guaranteed.

    Args:
        ref_seq: The reference sequence, which should be >= len(pdb_seq).
        pdb_seqs_df: The PDB DataFrame from `antid.io.struct.struct2df`.
        pdb_chain: The chain ID of the PDB file to align the reference sequence to.
        chain_col_name: The name of the column containing chain IDs.
        resi_col_name: The name of the column containing residue indices.
        insertion_code_col_name: The name of the column containing insertion codes.
        resn_col_name: The name of the column containing residue names. Make sure the
            residue names are in 1-letter format. If `resn_3to1` is True, they will be converted.
        ref_idx_col_name: The name of the column to use for the reference index in the returned DataFrame.
        resn_3to1: Whether to convert 3-letter residue names to 1-letter names.
        **kwargs: Additional 3->1 residue mappings.

    Returns:
        A DataFrame with the aligned sequences containing columns:
            - chain_col_name: The chain ID.
            - ref_idx_col_name: The index in the reference sequence.
            - resi_col_name: The residue index (with insertion codes) in the PDB file.
            - insertion_code_col_name: The insertion code in the PDB file.
            - resn_col_name: The residue name represented as a one-letter code.

    """
    pdb_seqs_df_dedup = pdb_seqs_df.unique(
        (chain_col_name, resi_col_name, insertion_code_col_name, resn_col_name),
        maintain_order=True,
        keep="first",
    )

    if resn_3to1:
        aa_map = AA3TO1.copy() | kwargs
        pdb_seqs_df_dedup = pdb_seqs_df_dedup.with_columns(
            pl.col(resn_col_name).replace_strict(aa_map)
        )

    # Collect the current sequence in the PDB file
    pdb_seqs = (
        pdb_seqs_df_dedup.group_by(pl.col(chain_col_name), maintain_order=True)
        .agg(pl.col(resn_col_name).alias("seq"))
        .with_columns(pl.col("seq").list.join(""))
    )
    pdb_seq = pdb_seqs.filter(pl.col(chain_col_name) == pl.lit(pdb_chain)).item(
        0, "seq"
    )

    # The alignment is trivial if the two sequences are identical
    if pdb_seq == ref_seq:
        return (
            pdb_seqs_df_dedup.filter(pl.col(chain_col_name) == pl.lit(pdb_chain))
            .with_row_index(name=ref_idx_col_name, offset=1)
            .select(
                chain_col_name,
                ref_idx_col_name,
                resi_col_name,
                insertion_code_col_name,
                resn_col_name,
            )
        )

    from Bio import Align

    aligner = Align.PairwiseAligner(scoring="blastp")
    aln = next(aligner.align(ref_seq, pdb_seq))
    ref_aln: str = aln[0]
    pdb_aln: str = aln[1]

    chain_resi = pdb_seqs_df_dedup.filter(
        pl.col(chain_col_name) == pl.lit(pdb_chain)
    ).select(resi_col_name, insertion_code_col_name, resn_col_name)

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
        pl.col(ref_idx_col_name).cast(pl.Int64),
        resi_col_name,
        insertion_code_col_name,
        resn_col_name,
    )
