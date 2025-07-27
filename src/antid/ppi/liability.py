"""Collect structural and sequence liabilities."""

import polars as pl


class LiabilityScanner:
    """Scanner for liabilities in antibody-antigen interactions."""

    resi_cols = ("chain", "resn", "resi", "insertion", "altloc")
    atomi_cols = ("chain", "resn", "resi", "insertion", "altloc", "atomn", "atomi")

    def __init__(self, ab_ag_contacts: pl.DataFrame, within_ab_contacts: pl.DataFrame):
        """Initialize with contact data from ``antid.ppi.scan``."""
        self.ab_ag = ab_ag_contacts
        self.within_ab = within_ab_contacts

    def find_structural_liabilities(self):
        """Find structural liabilities in the antibody-antigen interaction DataFrame."""
        return pl.concat(
            (
                self.find_buried_tyrosine_hydroxyl(),
                self.find_sidechain_close_to_backbone(),
                self.find_buried_polars(),
                self.find_repelling_charges(),
            ),
            how="diagonal",
            rechunk=True,
        )

    def find_buried_tyrosine_hydroxyl(self) -> pl.DataFrame:
        """Find buried tyrosine hydroxyl groups based on contact and SASA data.

        Returns:
            DataFrame with tyrosine hydroxyl groups that are buried and have specific interactions.
        """
        ab_resi_cols: list[str] = [f"ab_{c}" for c in self.resi_cols if c != "resn"]
        interacting_tyr = (
            self.ab_ag.filter(
                pl.col("interaction").str.contains(r"^(CationPi)|(Pi.*Stacking)$")
            )
            .filter(pl.col("ab_resn") == pl.lit("TYR"))
            .select(
                *ab_resi_cols,
                *(f"ag_{c}" for c in self.resi_cols if c != "resn"),
            )
            .unique()
        )

        # We don't want the hydroxyl group to be in proximity to the target and
        # don't form a hydrogen bond (either with the target or within the VHH)
        interacting_tyr_with_hbond = (
            self.ab_ag.join(
                interacting_tyr.select(ab_resi_cols).unique(), on=ab_resi_cols
            )
            .filter(pl.col("ab_atomn") == pl.lit("OH"))
            .filter(pl.col("interaction").is_in({"HydrogenBond", "PolarContact"}))
            .sort("distance")
            .unique(keep="first", maintain_order=True)
        )

        interacting_tyr_within_ab = (
            pl.concat(
                (
                    self.within_ab.join(
                        interacting_tyr.select(
                            pl.col(f"ab_{c}").alias(f"from_{c}") for c in self.resi_cols
                        ),
                        on=[f"from_{c}" for c in self.resi_cols],
                    )
                    .filter(pl.col("from_atomn") == pl.lit("OH"))
                    .select(
                        "interaction",
                        "distance",
                        *(
                            pl.col(f"from_{c}").alias(f"tyr_{c}")
                            for c in self.resi_cols
                        ),
                        *(pl.col(f"to_{c}").alias(f"ab_{c}") for c in self.resi_cols),
                    ),
                    self.within_ab.join(
                        interacting_tyr.select(
                            pl.col(f"ab_{c}").alias(f"to_{c}")
                            for c in ("chain", "resi", "insertion", "altloc")
                        ),
                        on=[f"to_{c}" for c in self.resi_cols],
                    )
                    .filter(pl.col("to_atomn") == pl.lit("OH"))
                    .select(
                        "interaction",
                        "distance",
                        *(pl.col(f"to_{c}").alias(f"tyr_{c}") for c in self.resi_cols),
                        *(pl.col(f"from_{c}").alias(f"ab_{c}") for c in self.resi_cols),
                    ),
                ),
                how="vertical",
            )
            .sort("distance")
            .unique(keep="first", maintain_order=True)
            .filter(pl.col("interaction").is_in({"HydrogenBond", "PolarContact"}))
        )

        interacting_tyr_bad_hydroxyl = (
            self.ab_ag.join(
                interacting_tyr.select(ab_resi_cols).unique(), on=ab_resi_cols
            )
            .filter(
                (pl.col("ab_atomn") == pl.lit("OH"))
                & (pl.col("interaction") == pl.lit("VanDerWaalsContact"))
            )
            .sort("distance")
            .unique(
                (*ab_resi_cols, *(f"ag_{c}" for c in self.resi_cols if c != "resn")),
                maintain_order=True,
                keep="first",
            )
            # If Y-OH forms hydrogen bond with the target or another residue on the VHH, it's fine
            .join(interacting_tyr_with_hbond, on=ab_resi_cols, how="anti")
            .join(
                interacting_tyr_within_ab.select(
                    pl.col(f"tyr_{c}").alias(f"ab_{c}")
                    for c in ("chain", "resi", "insertion", "altloc")
                ),
                on=ab_resi_cols,
                how="anti",
            )
            # If Y-OH has a large SASA for water, it's also okay
            .filter(pl.col("ab_sasa") < pl.lit(10))
            .select(
                pl.lit("BuriedTyrHydroxyl").alias("liability"), *ab_resi_cols, "ab_resn"
            )
        )
        return interacting_tyr_bad_hydroxyl

    def find_sidechain_close_to_backbone(self):
        """Avoid Arg/Lys sidechains to be too close to the backbone of the target."""
        # For Arg and Lys forming cation-pi interactions, we don't want the
        # sidechain to be too close to the target
        cation_pi_rk = (
            self.ab_ag.filter(pl.col("interaction") == pl.lit("CationPi"))
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
        cation_pi_rk_hbond = self.ab_ag.join(
            cation_pi_rk.select("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
            on=("ab_chain", "ab_resi", "ab_insertion", "ab_altloc"),
        ).filter(
            pl.col("interaction").is_in({"HydrogenBond", "PolarContact", "IonicBond"})
        )

        return (
            self.ab_ag.join(
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

    def find_buried_polars(self):
        """Identify polar residues that are buried by hydrophobics."""
        ag_polars_interaction_count = (
            self.ab_ag.filter(
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

    def find_repelling_charges(self, his_charge: float = 0.5):
        """Find residue pairs with repelling charges.

        Args:
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
            self.ab_ag.filter(pl.col("interaction") == pl.lit("IonicRepulsion"))
            .unique(res_pair_cols)
            .join(
                self.ab_ag.filter(
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
