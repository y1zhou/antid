"""Scan interactions and calculate atomic SASA using Arpeggia.

For more information, see: https://github.com/y1zhou/arpeggia

TODO: add PyO3 bindings for Arpeggia
"""

from pathlib import Path

import polars as pl

from antid.utils import check_path, command_runner, find_binary
from antid.utils.constant import ARPEGGIA_IDX_COLS


def get_contacts(
    struct_file: str | Path,
    out_dir: str | Path,
    arpeggia_path: str | Path,
    out_file_name: str = "contacts",
    chain_pairs: str = "/",
):
    """Run Arpeggia to calculate contacts between chains in a structure.

    Args:
        struct_file: Path to the PDB or mmCIF structure file.
        out_dir: Directory to save the output file.
        arpeggia_path: Path to the Arpeggia executable.
        out_file_name: Name of the output file (without extension).
        chain_pairs: Chain pairs to calculate contacts for, in the format "A,B/C".
    """
    out_path = check_path(out_dir, mkdir=True, ignore_dots=True)
    out_file = out_path / f"{out_file_name}.parquet"
    if out_file.exists():
        return pl.read_parquet(out_file)

    pdb_path = check_path(struct_file, exists=True)
    arpeggia_bin = find_binary(arpeggia_path)
    cmd = [
        arpeggia_bin,
        "contacts",
        "-j",
        "1",
        "-t",
        "parquet",
        "-i",
        str(pdb_path),
        "-o",
        str(out_path),
        "-n",
        out_file_name,
        "-g",
        chain_pairs,
    ]
    command_runner(cmd, out_dir, log_file="/dev/null")
    return pl.read_parquet(out_file)


def get_atomic_sasa(
    struct_file: str | Path,
    out_dir: str | Path,
    arpeggia_path: str | Path,
    out_file_name: str = "sasa",
):
    """Run Arpeggia to calculate solvent accessible surface area (SASA) for a structure."""
    out_path = check_path(out_dir, mkdir=True, ignore_dots=True)
    out_file = out_path / f"{out_file_name}.parquet"
    if out_file.exists():
        return pl.read_parquet(out_file)

    pdb_path = check_path(struct_file, exists=True)
    arpeggia_bin = find_binary(arpeggia_path)
    cmd = [
        arpeggia_bin,
        "sasa",
        "-j",
        "1",
        "-t",
        "parquet",
        "-i",
        str(pdb_path),
        "-o",
        str(out_path),
        "-n",
        out_file_name,
    ]
    command_runner(cmd, out_dir, log_file="/dev/null")
    return pl.read_parquet(out_file)


def collect_ab_ag_contacts(
    ab_chains: set[str], contact_df: pl.DataFrame, sasa_df: pl.DataFrame
) -> pl.DataFrame:
    """Identify antibody chains from the contact DataFrame.

    Args:
        ab_chains: Set of antibody chain IDs.
        contact_df: DataFrame from ``get_contacts``.
        sasa_df: DataFrame from ``get_atomic_sasa``.

    Returns:
        A DataFrame with headers changed to `ab_*` and `ag_*` groups.
    """
    # TODO: deal with multiple models
    first_model_id = contact_df.item(0, "model")
    return (
        contact_df.filter(
            (pl.col("from_chain") != pl.col("to_chain"))
            & (pl.col("model") == pl.lit(first_model_id))
        )
        .filter(~pl.col("interaction").str.starts_with("Weak"))
        # Normalize order of redidue pairs
        .with_columns(
            pl.when(pl.col("from_chain").is_in(ab_chains))
            .then(pl.col(f"from_{c}"))
            .otherwise(pl.col(f"to_{c}"))
            .alias(f"ab_{c}")
            for c in ARPEGGIA_IDX_COLS
        )
        .with_columns(
            pl.when(pl.col("from_chain").is_in(ab_chains))
            .then(pl.col(f"to_{c}"))
            .otherwise(pl.col(f"from_{c}"))
            .alias(f"ag_{c}")
            for c in ARPEGGIA_IDX_COLS
        )
        .drop(
            *(f"from_{c}" for c in ARPEGGIA_IDX_COLS),
            *(f"to_{c}" for c in ARPEGGIA_IDX_COLS),
        )
        # Add atom SASA for antibody and antigen chains
        .join(
            sasa_df.select(
                pl.exclude("resn", "atomn").name.prefix("ab_"),
            ),
            on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc", "ab_atomi"),
            how="left",
        )
        .join(
            sasa_df.select(
                pl.exclude("resn", "atomn").name.prefix("ag_"),
            ),
            on=("ag_chain", "ag_resi", "ag_insertion", "ag_altloc", "ag_atomi"),
            how="left",
        )
    )


def collect_within_ab_contacts(
    ab_chains: set[str], contact_df: pl.DataFrame, sasa_df: pl.DataFrame
):
    """Collect contacts within the antibody chains and calculate SASA.

    Args:
        ab_chains: Set of antibody chain IDs.
        contact_df: DataFrame from ``get_contacts``.
        sasa_df: DataFrame from ``get_atomic_sasa``.

    Returns:
        A DataFrame with contacts within the antibody chains, including SASA.
    """
    # TODO: deal with multiple models
    first_model_id = contact_df.item(0, "model")

    # Filter contacts to only include those within the antibody
    return (
        contact_df.filter(
            (pl.col("from_chain") == pl.col("to_chain"))
            & pl.col("from_chain").is_in(ab_chains)
            & (pl.col("model") == pl.lit(first_model_id))
        )
        .filter(~pl.col("interaction").str.starts_with("Weak"))
        .sort("distance")
        .join(
            sasa_df.filter(pl.col("chain").is_in(ab_chains)).select(
                pl.exclude("resn", "atomn").name.prefix("from_"),
            ),
            on=(
                "from_chain",
                "from_resi",
                "from_insertion",
                "from_altloc",
                "from_atomi",
            ),
            how="left",
        )
        .join(
            sasa_df.filter(pl.col("chain").is_in(ab_chains)).select(
                pl.exclude("resn", "atomn").name.prefix("to_"),
            ),
            on=("to_chain", "to_resi", "to_insertion", "to_altloc", "to_atomi"),
            how="left",
        )
    )
