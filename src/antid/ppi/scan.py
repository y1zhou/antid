"""Scan interactions and calculate residue SASA using Arpeggia.

For more information, see: https://github.com/y1zhou/arpeggia

TODO: add PyO3 bindings for Arpeggia
"""

from pathlib import Path

import polars as pl

from antid.utils import check_path, command_runner, find_binary


def run_arpeggia_contacts(
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
    cmd = [
        arpeggia_path,
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


def run_arpeggia_sasa(
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
    cmd = [
        arpeggia_path,
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


class InteractionScanner:
    """Class to scan interactions and calculate SASA using Arpeggia."""

    def __init__(self, arpeggia_path: str | Path):
        """Initialize the scanner with the path to the Arpeggia executable."""
        self.arpeggia_path = find_binary(arpeggia_path)

    def scan_contacts(
        self,
        struct_file: str | Path,
        out_dir: str | Path,
        out_file_name: str = "contacts",
        chain_pairs: str = "/",
    ) -> pl.DataFrame:
        """Scan contacts between chains in a structure."""
        return run_arpeggia_contacts(
            struct_file, out_dir, self.arpeggia_path, out_file_name, chain_pairs
        )

    def calculate_sasa(
        self, struct_file: str | Path, out_dir: str | Path, out_file_name: str = "sasa"
    ) -> pl.DataFrame:
        """Calculate solvent accessible surface area (SASA) for a structure."""
        return run_arpeggia_sasa(
            struct_file, out_dir, self.arpeggia_path, out_file_name
        )

    def collect_ab_ag_contacts(
        self, ab_chains: set[str], contact_df: pl.DataFrame, sasa_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Identify antibody chains from the contact DataFrame.

        Returns:
            A DataFrame with headers changed to `ab_*` and `ag_*` groups.
        """
        arpeggia_id_cols = (
            "chain",
            "resn",
            "resi",
            "insertion",
            "altloc",
            "atomn",
            "atomi",
        )
        return (
            contact_df.filter(
                (pl.col("from_chain") != pl.col("to_chain"))
                & (pl.col("model") == pl.lit(0))  # TODO: deal with multiple models
            )
            .filter(~pl.col("interaction").str.starts_with("Weak"))
            # Normalize order of redidue pairs
            .with_columns(
                pl.when(pl.col("from_chain").is_in(ab_chains))
                .then(pl.col(f"from_{c}"))
                .otherwise(pl.col(f"to_{c}"))
                .alias(f"ab_{c}")
                for c in arpeggia_id_cols
            )
            .with_columns(
                pl.when(pl.col("from_chain").is_in(ab_chains))
                .then(pl.col(f"to_{c}"))
                .otherwise(pl.col(f"from_{c}"))
                .alias(f"ag_{c}")
                for c in arpeggia_id_cols
            )
            .drop(
                *(f"from_{c}" for c in arpeggia_id_cols),
                *(f"to_{c}" for c in arpeggia_id_cols),
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
        self, ab_chains: set[str], contact_df: pl.DataFrame, sasa_df: pl.DataFrame
    ):
        """Collect contacts within the antibody chains and calculate SASA."""
        # Filter contacts to only include those within the antibody
        return (
            contact_df.filter(
                (pl.col("from_chain") == pl.col("to_chain"))
                & pl.col("from_chain").is_in(ab_chains)
                & (pl.col("model") == pl.lit(0))  # TODO: deal with multiple models
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

    def find_structural_liabilities(
        self, ab_ag_df: pl.DataFrame, within_ab_df: pl.DataFrame
    ):
        """Find structural liabilities in the antibody-antigen interaction DataFrame."""
        return pl.concat(
            (
                self.find_buried_tyrosine_hydroxyl(ab_ag_df, within_ab_df),
                self.find_sidechain_close_to_backbone(ab_ag_df),
                self.find_buried_polars(ab_ag_df),
                self.find_repelling_charges(ab_ag_df),
            ),
            how="diagonal",
            rechunk=True,
        )

    def find_buried_tyrosine_hydroxyl(
        self, ab_ag_df: pl.DataFrame, within_ab_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Find buried tyrosine hydroxyl groups based on contact and SASA data.

        Args:
            ab_ag_df: DataFrame containing contact and SASA information from Arpeggia.
            within_ab_df: DataFrame containing contacts within the antibody chains.

        Returns:
            DataFrame with tyrosine hydroxyl groups that are buried and have specific interactions.
        """
        interacting_tyr = (
            ab_ag_df.filter(
                pl.col("interaction").str.contains(r"^(CationPi)|(Pi.*Stacking)$")
            )
            .filter(pl.col("ab_resn") == pl.lit("TYR"))
            .select(
                "ab_chain",
                "ab_resi",
                "ab_insertion",
                "ab_altloc",
                "ag_chain",
                "ag_resi",
                "ag_altloc",
                "ag_insertion",
            )
            .unique()
        )

        # We don't want the hydroxyl group to be in proximity to the target and
        # don't form a hydrogen bond (either with the target or within the VHH)
        interacting_tyr_with_hbond = (
            ab_ag_df.join(
                interacting_tyr.select(
                    "ab_chain", "ab_resi", "ab_insertion", "ab_altloc"
                ).unique(),
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
            )
            .filter(pl.col("ab_atomn") == pl.lit("OH"))
            .filter(pl.col("interaction").is_in({"HydrogenBond", "PolarContact"}))
            .sort("distance")
            .unique(keep="first", maintain_order=True)
        )

        interacting_tyr_within_ab = (
            pl.concat(
                (
                    within_ab_df.join(
                        interacting_tyr.select(
                            pl.col(f"ab_{c}").alias(f"from_{c}")
                            for c in ("chain", "resi", "insertion", "altloc")
                        ),
                        on=("from_chain", "from_resi", "from_insertion", "from_altloc"),
                    )
                    .filter(pl.col("from_atomn") == pl.lit("OH"))
                    .select(
                        "interaction",
                        "distance",
                        pl.col("from_chain").alias("tyr_chain"),
                        pl.col("from_resi").alias("tyr_resi"),
                        pl.col("from_insertion").alias("tyr_insertion"),
                        pl.col("from_altloc").alias("tyr_altloc"),
                        pl.col("to_resn").alias("vhh_resn"),
                        pl.col("to_resi").alias("vhh_resi"),
                        pl.col("to_altloc").alias("vhh_altloc"),
                    ),
                    within_ab_df.join(
                        interacting_tyr.select(
                            pl.col(f"ab_{c}").alias(f"to_{c}")
                            for c in ("chain", "resi", "insertion", "altloc")
                        ),
                        on=("to_chain", "to_resi", "to_insertion", "to_altloc"),
                    )
                    .filter(pl.col("to_atomn") == pl.lit("OH"))
                    .select(
                        "interaction",
                        "distance",
                        pl.col("to_chain").alias("tyr_chain"),
                        pl.col("to_resi").alias("tyr_resi"),
                        pl.col("to_insertion").alias("tyr_insertion"),
                        pl.col("to_altloc").alias("tyr_altloc"),
                        pl.col("from_resn").alias("vhh_resn"),
                        pl.col("from_resi").alias("vhh_resi"),
                        pl.col("from_altloc").alias("vhh_altloc"),
                    ),
                ),
                how="vertical",
            )
            .sort("distance")
            .unique(keep="first", maintain_order=True)
            .filter(pl.col("interaction").is_in({"HydrogenBond", "PolarContact"}))
        )

        interacting_tyr_bad_hydroxyl = (
            ab_ag_df.join(
                interacting_tyr.select(
                    "ab_chain", "ab_resi", "ab_insertion", "ab_altloc"
                ).unique(),
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
            )
            .filter(
                (pl.col("ab_atomn") == pl.lit("OH"))
                & (pl.col("interaction") == pl.lit("VanDerWaalsContact"))
            )
            .sort("distance")
            .unique(
                (
                    "ab_chain",
                    "ab_resi",
                    "ab_insertion",
                    "ab_altloc",
                    "ag_resi",
                    "ag_insertion",
                    "ag_altloc",
                ),
                maintain_order=True,
                keep="first",
            )
            # If Y-OH forms hydrogen bond with the target or another residue on the VHH, it's fine
            .join(
                interacting_tyr_with_hbond,
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
                how="anti",
            )
            .join(
                interacting_tyr_within_ab.select(
                    pl.col(f"tyr_{c}").alias(f"ab_{c}")
                    for c in ("chain", "resi", "insertion", "altloc")
                ),
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
                how="anti",
            )
            # If Y-OH has a large SASA for water, it's also okay
            .filter(pl.col("ab_sasa") < pl.lit(10))
            .select(
                pl.lit("BuriedTyrHydroxyl").alias("liability"),
                "ab_chain",
                "ab_resi",
                "ab_insertion",
                "ab_altloc",
                "ab_resn",
            )
        )
        return interacting_tyr_bad_hydroxyl

    def find_sidechain_close_to_backbone(self, ab_ag_df: pl.DataFrame):
        """Avoid Arg/Lys sidechains to be too close to the backbone of the target."""
        # For Arg and Lys forming cation-pi interactions, we don't want the
        # sidechain to be too close to the target
        cation_pi_rk = (
            ab_ag_df.filter(pl.col("interaction") == pl.lit("CationPi"))
            .filter(
                (pl.col("ab_resn") != pl.lit("HIS"))
                & (pl.col("ag_resn") != pl.lit("HIS"))
            )
            .filter(pl.col("ab_resn").is_in({"ARG", "LYS"}))
            .select(
                "ab_chain",
                "ab_resi",
                "ab_insertion",
                "ab_altloc",
                "ag_resi",
                "ag_insertion",
                "ag_altloc",
            )
            .unique()
        )

        # They get a pass if they also form hydrogen bonds with the target
        cation_pi_rk_hbond = ab_ag_df.join(
            cation_pi_rk.select("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
            on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
        ).filter(
            pl.col("interaction").is_in({"HydrogenBond", "PolarContact", "IonicBond"})
        )

        return (
            ab_ag_df.join(
                cation_pi_rk.select(
                    "ab_chain", "ab_resi", "ab_insertion", "ab_altloc"
                ).unique(),
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
            )
            .filter(
                pl.col("interaction").is_in({"VanDerWaalsContact", "IonicRepulsion"})
            )
            .join(
                cation_pi_rk_hbond,
                on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
                how="anti",
            )
            .select(
                pl.lit("SidechainCloseToBackbone").alias("liability"),
                "ab_chain",
                "ab_resi",
                "ab_insertion",
                "ab_altloc",
                "ab_resn",
            )
        )

    def find_buried_polars(self, ab_ag_df: pl.DataFrame):
        """Identify polar residues that are buried by hydrophobics."""
        ag_polars_interaction_count = (
            ab_ag_df.filter(
                pl.col("ag_resn").is_in(
                    {"ARG", "HIS", "LYS", "ASP", "GLU", "ASN", "GLN", "PRO", "CYS"}
                )
            )
            .unique(
                (
                    "ag_chain",
                    "ag_resi",
                    "ag_insertion",
                    "ag_altloc",
                    "ab_chain",
                    "ab_resi",
                    "ab_insertion",
                    "ab_altloc",
                    "interaction",
                )
            )
            .group_by(
                "ag_chain",
                "ag_resi",
                "ag_insertion",
                "ag_altloc",
                "ag_resn",
                "interaction",
                "ag_sasa",
            )
            .len()
            .pivot(on="interaction", values="len", sort_columns=True)
            .fill_null(0)
        )
        for itx in ("HydrophobicContact", "VanDerWaalsContact", "IonicRepulsion"):
            if itx not in ag_polars_interaction_count.columns:
                ag_polars_interaction_count = ag_polars_interaction_count.with_columns(
                    pl.lit(0).alias(itx)
                )
        return (
            ag_polars_interaction_count.filter(
                # No other interactions to these polar residues
                (
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude(
                            "ag_resi",
                            "ag_sasa",
                            "HydrophobicContact",
                            "IonicRepulsion",
                            "VanDerWaalsContact",
                        )
                    )
                    == pl.lit(0)
                )
                & (
                    pl.sum_horizontal("HydrophobicContact", "VanDerWaalsContact")
                    > pl.lit(2)
                )
                & (pl.col("ag_sasa") < pl.lit(10))
            )
            .sort(
                "IonicRepulsion",
                "VanDerWaalsContact",
                "HydrophobicContact",
                descending=True,
            )
            .select(
                pl.lit("BuriedPolars").alias("liability"),
                "ag_chain",
                "ag_resi",
                "ag_insertion",
                "ag_altloc",
                "ag_resn",
                # "HydrophobicContact",
                # "VanDerWaalsContact",
                # "IonicRepulsion",
                # "ag_sasa",
            )
        )

    def find_repelling_charges(self, ab_ag_df: pl.DataFrame, his_charge: float = 0.5):
        """Find residue pairs with repelling charges.

        Args:
            ab_ag_df: DataFrame containing antibody-antigen interaction data.
            his_charge: Charge value for histidine residues (default is 0.5).
        """
        res_pair_cols = (
            "ab_chain",
            "ab_resi",
            "ab_insertion",
            "ab_altloc",
            "ab_resn",
            "ag_chain",
            "ag_resi",
            "ag_insertion",
            "ag_altloc",
            "ag_resn",
        )
        repulsive_charges = (
            ab_ag_df.filter(pl.col("interaction") == pl.lit("IonicRepulsion"))
            .unique(res_pair_cols)
            .join(
                ab_ag_df.filter(
                    pl.col("interaction").is_in({"CationPi", "SaltBridge"})
                ),
                how="anti",
                on=res_pair_cols,
            )
            .with_columns(
                pl.when(
                    (pl.col("ab_resn") == pl.lit("HIS"))
                    | (pl.col("ag_resn") == pl.lit("HIS"))
                )
                .then(pl.lit(his_charge))
                .otherwise(pl.lit(1.0))
                .alias("repulsion_score")
            )
        )
        return repulsive_charges.select(
            pl.lit("ChargeRepulsion").alias("liability"),
            *res_pair_cols,
            "repulsion_score",
        )
        #     .group_by("pdb_path")
        #     .agg(pl.col("repulsion_score").sum())
        #     .sort("repulsion_score", descending=True)
        #     .filter(pl.col("repulsion_score") >= pl.lit(1.0))
        # )
