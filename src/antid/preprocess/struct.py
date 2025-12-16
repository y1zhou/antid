"""Utility functions for dealing with PDB structure files."""

from pathlib import Path

import gemmi
import polars as pl

from antid.io.seq import AA3TO1
from antid.numbering.antibody import ValidSchemesType, number_ab_seq
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
    ref_resn_col_name: str = "ref_resn",
    resn_3to1: bool = False,
    **kwargs,
) -> pl.DataFrame:
    """Align a reference sequence to the observed residues in a structure.

    The best use case for this function is to identify missing densities
    in a PDB file. When there are large gaps, the quality of the alignment
    is not guaranteed.

    The sequence extracted from the PDB file is assumed to be a subset of the
    reference sequence. If there are extra residues in the PDB file that are not
    in the reference sequence, results may not be as expected.

    Args:
        ref_seq: The reference sequence, which should be >= len(pdb_seq).
        pdb_seqs_df: The PDB DataFrame from `antid.io.struct.struct2df`.
        pdb_chain: The chain ID of the PDB file to align the reference sequence to.
        chain_col_name: The name of the column containing chain IDs.
        resi_col_name: The name of the column containing residue indices.
        insertion_code_col_name: The name of the column containing insertion codes.
        resn_col_name: The name of the column containing residue names from the PDB.
            Make sure the residue names are in 1-letter format.
            If `resn_3to1` is True, they will be converted.
        ref_idx_col_name: The name of the column to use for the reference index in the returned DataFrame.
        ref_resn_col_name: The name of the column containing residue names from
            the reference sequence.
        resn_3to1: Whether to convert 3-letter residue names to 1-letter names.
        **kwargs: Additional 3->1 residue mappings.

    Returns:
        A DataFrame with the aligned sequences containing columns:
            - chain_col_name: The chain ID.
            - ref_idx_col_name: The index in the reference sequence (1-based).
            - resi_col_name: The residue index (with insertion codes) in the PDB file.
            - insertion_code_col_name: The insertion code in the PDB file.
            - resn_col_name: The residue name represented as a one-letter code.
            - ref_resn_col_name: The residue name from the reference sequence.

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

    if (
        pdb_seqs_df_dedup.filter(
            pl.col(resn_col_name).str.len_bytes() > pl.lit(1)
        ).height
        != 0
    ):
        raise ValueError(
            "Residue names in the PDB DataFrame should all be 1-letter codes. "
            "You may want to set `resn_3to1=True`."
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
                pl.col(resn_col_name).alias(ref_resn_col_name),
            )
        )

    # TODO: replace with https://gemmi.readthedocs.io/en/stable/analysis.html#sequence-alignment
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

        map_entry = {
            ref_idx_col_name: i + 1,
            resi_col_name: None,
            insertion_code_col_name: None,
            resn_col_name: aa2,
            ref_resn_col_name: aa1,
        }
        if aa2 != "-":
            map_entry[resi_col_name] = chain_resi.item(resi_idx, resi_col_name)
            map_entry[insertion_code_col_name] = chain_resi.item(
                resi_idx, insertion_code_col_name
            )
            resi_idx += 1

        map_entries.append(map_entry)
    return pl.from_records(map_entries, orient="row", infer_schema_length=None).select(
        pl.lit(pdb_chain).alias(chain_col_name),
        pl.col(ref_idx_col_name).cast(pl.Int64),
        resi_col_name,
        insertion_code_col_name,
        resn_col_name,
        ref_resn_col_name,
    )


def standardize_struct_file(
    input_file: str | Path,
    output_file: str | Path,
    restart_each_chain: bool = True,
    chain_mapping: dict[str, str] | None = None,
    remove_waters: bool = True,
    remove_ligands_and_waters: bool = False,
    # Antibody-specific
    vh_chain: str | None = None,
    vl_chain: str | None = None,
    numbering_scheme: ValidSchemesType | None = None,
) -> pl.DataFrame:
    """Renumber PDB and mmCIF coordinate files with serial residue indices.

    WARNING: Only the first model in ``input_file`` will be kept in the output.

    NOTE: This somewhat similar to $ROSETTA/tools/protein_tools/scripts/clean_pdb.py.
    For details, see:
    https://docs.rosettacommons.org/docs/latest/rosetta_basics/preparation/preparing-structures#cleaning-pdbs-for-rosetta

    Args:
        input_file: Path to the input PDB file.
        output_file: Path to the output PDB file.
        restart_each_chain: If True, residue indices will restart from 1 for each chain.
            If False, residue indices will be continuous across the entire structure.
            Ignored if `vh_chain` or `vl_chain` is provided.
        chain_mapping: Optional mapping of old chain IDs to new chain IDs. If provided,
            the chain IDs in the output PDB file will be replaced according to this mapping.
            The output file will also *only* contain the chains in the mapping. If not
            provided, all chains in the input file will be kept, and only residue
            renumbering will be performed. If ``vh_chain`` or ``vl_chain`` is provided,
            they should use the new chain IDs after renaming.
        remove_waters: If True, water molecules will be removed from the structure.
        remove_ligands_and_waters: If True, all non-protein residues (including ligands
            and water molecules) will be removed from the structure.
        vh_chain: Heavy chain ID. If provided, and `numbering_scheme` is also provided,
            the heavy chain will be renumbered according to the specified scheme. Residues
            that are not in the Fv region will be removed.
        vl_chain: Light chain ID. If provided, and `numbering_scheme` is also provided,
            the light chain will be renumbered according to the specified scheme. Residues
            that are not in the Fv region will be removed.
        numbering_scheme: Antibody numbering scheme to use for renumbering the heavy
            and light chains.

    Returns:
        A polars DataFrame with:
            - model: The model number.
            - old_chain: The original chain ID.
            - old_resi: The original residue index.
            - old_insertion: The original insertion code.
            - new_chain: The new chain ID after renumbering.
            - new_resi: The new residue index after renumbering.
            - new_insertion: The new insertion code after renumbering.
    """
    input_path = check_path(input_file, exists=True)
    output_path = check_path(output_file, mkdir=True)

    # Read structure file and remove unnecessary stuff
    st = gemmi.read_structure(str(input_path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.standardize_crystal_frame()
    if remove_waters:
        st.remove_waters()
    if remove_ligands_and_waters:
        st.remove_ligands_and_waters()

    # Rename chains if a mapping is provided
    model = st[0]
    all_chains = {c.name for c in model}
    if chain_mapping is not None:
        model_chains = chain_mapping.copy()
        for c in model_chains.keys():
            if c not in all_chains:
                raise ValueError(f"Chain {c} not found in {all_chains}: {input_file}")
    else:
        model_chains: dict[str, str] = {c: c for c in all_chains}
    for old_chain_id, new_chain_id in model_chains.items():
        st.rename_chain(old_chain_id, new_chain_id)

    model_chains_rev = {v: k for k, v in model_chains.items()}

    # Go through the structure and renumber residues
    res_map: list[tuple[int, str, int, str, str, int, str]] = []
    prev_resi = 0

    if (vh_chain is not None) or (vl_chain is not None):
        restart_each_chain = False  # override when using antibody numbering
    chains_to_number = []
    if vh_chain is not None:
        chains_to_number.append(vh_chain)
    if vl_chain is not None:
        chains_to_number.append(vl_chain)

    for chain in model:
        if chain.name not in model_chains_rev:
            model.remove_chain(chain.name)
            continue

        old_chain_id = model_chains_rev[chain.name]

        if chain.name in chains_to_number:
            if numbering_scheme is None:
                raise ValueError(
                    "If `vh_chain` or `vl_chain` is provided, "
                    "`numbering_scheme` must also be provided."
                )
            chain_seq = gemmi.one_letter_code([res.name for res in chain])
            numbered_seq = number_ab_seq(chain_seq, numbering_scheme)
            pos_df = numbered_seq.position.with_columns(
                (pl.col("idx") - pl.lit(1)).alias("idx"),
                pl.col("numbered_pos")
                .str.extract(r"^(\d+)")
                .cast(pl.Int32)
                .alias("resi"),
                pl.col("numbered_pos")
                .str.replace(r"^\d+", "")
                .replace("", " ")
                .alias("icode"),
            )
            res_idx_to_remove = (
                pos_df.filter(pl.col("resi").is_null())
                .get_column("idx")
                .sort(descending=True)
            )

            for res, r in zip(chain, pos_df.iter_rows(named=True), strict=True):
                if r["idx"] in res_idx_to_remove:
                    continue
                res_map.append(
                    (  # pyrefly: ignore[bad-argument-type]
                        model.num,
                        old_chain_id,
                        res.seqid.num,
                        res.seqid.icode,
                        chain.name,
                        r["resi"],
                        r["icode"],
                    )
                )
                res.seqid.num = r["resi"]
                res.seqid.icode = r["icode"]

            for i in res_idx_to_remove:
                del chain[i]

        else:
            resi = 1 if restart_each_chain else prev_resi + 1
            for res in chain:
                res_map.append(
                    (  # pyrefly: ignore[bad-argument-type]
                        model.num,
                        old_chain_id,
                        res.seqid.num,
                        res.seqid.icode,
                        chain.name,
                        resi,
                        " ",
                    )
                )
                res.seqid.num = resi
                res.seqid.icode = " "
                resi += 1

            prev_resi = resi

    st.remove_empty_chains()
    st.assign_serial_numbers(numbered_ter=True)

    # Write the renumbered structure to a new file
    if output_path.suffix == ".pdb":
        st.write_pdb(str(output_path))
    elif output_path.suffix == ".cif":
        st.make_mmcif_document().write_file(str(output_path))
    else:
        raise ValueError(
            f"Unsupported output file format: {output_path.suffix}. "
            "Only .pdb and .cif files are supported."
        )

    return pl.from_records(
        res_map,
        orient="row",
        schema=(
            ("model", pl.Int32),
            ("old_chain", pl.Utf8),
            ("old_resi", pl.Int32),
            ("old_insertion", pl.Utf8),
            ("new_chain", pl.Utf8),
            ("new_resi", pl.Int32),
            ("new_insertion", pl.Utf8),
        ),
    )


def run_pdbfixer(
    pdb_file: str | Path, out_pdb_file: str | Path, add_hydrogens: bool = False
):
    """Run PDBFixer to fix PDB files."""
    out_pdb_path = check_path(out_pdb_file, mkdir=True, is_dir=False)
    if out_pdb_path.exists():
        return out_pdb_path

    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    input_pdb_path = check_path(pdb_file, exists=True)
    fixer = PDBFixer(filename=str(input_pdb_path))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    for k, v in {
        "residues": fixer.missingResidues,
        "atoms": fixer.missingAtoms,
        "terminals": fixer.missingTerminals,
    }.items():
        if v:
            print(f"Missing {k} in {input_pdb_path}: {v}")

    if add_hydrogens:
        fixer.addMissingHydrogens()

    with open(out_pdb_path, "w") as f:
        if fixer.source is not None:
            f.write(f"REMARK   1 PDBFIXER FROM: {fixer.source}\n")
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return out_pdb_path
