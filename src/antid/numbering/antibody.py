"""Antibody numbering utilities."""

from dataclasses import dataclass
from typing import Literal, overload

import polars as pl
from loguru import logger

from antid.utils.constant import SIMILAR_PAIRS
from antid.utils.patch_antpack import SingleChainAnnotator, VJGeneTool

__all__ = [
    "number_ab_seq",
    "align_ab_seqs",
    "AntibodyAlignment",
    "NumberedAntibody",
    "NumberedAntibodyWithGermline",
    "Germlines",
]

# Type aliases for clarity
ValidSchemesType = Literal["imgt", "martin", "kabat", "aho"]
AssignedSpeciesType = Literal["human", "mouse", "alpaca", "rabbit"]
ValidSpeciesType = AssignedSpeciesType | list[AssignedSpeciesType] | Literal["unknown"]
AntPackAlignmentType = tuple[list[str], float, str, str]


class NumberedAntibody:
    """A class to hold the numbered sequence and its regions.

    Attributes:
        seq: The input antibody sequence.
        scheme: The numbering scheme used ("imgt", "martin", "kabat", or "aho").
        chain_type: Assigned chain type ("H", "L", or "K").
        scheme_aligned_seq: The Fv sequence aligned to the numbering scheme (with gaps).
        position: A DataFrame with the numbered positions and their corresponding amino acids.
        regions: A DataFrame with the start, end, and sequence of each region. Note that
            the indices are 1-based and inclusive.
        fv_seq: The Fv region sequence.
        cdr1_seq, cdr2_seq, cdr3_seq: The CDR sequences.
        cdr1_range, cdr2_range, cdr3_range: The start and end indices of each CDR.
        fr1_seq, fr2_seq, fr3_seq, fr4_seq: The framework region sequences.
        fr1_range, fr2_range, fr3_range, fr4_range: The start and end indices of each FR.
    """

    def __init__(
        self,
        seq: str,
        scheme: ValidSchemesType,
        numbering: list[str],
        percent_identity: float,
        chain_type: str,
        error_message: str,
        region_labels: list[str],
        scheme_aligned_seq: str,
    ):
        """Initialize with the output of ``SingleChainAnnotator.analyze_seq``."""
        self.seq = seq
        self.scheme: ValidSchemesType = scheme
        self.chain_type = chain_type
        self._raw = (numbering, percent_identity, chain_type, error_message)
        self.scheme_aligned_seq = scheme_aligned_seq

        self._process_regions(numbering, region_labels)

    def _process_regions(self, numbering: list[str], region_labels: list[str]):
        """Process the numbering and region labels to create a DataFrame."""
        # Collect numbering results into a DataFrame
        self.position = (
            pl.DataFrame(
                {
                    "region": region_labels,
                    "numbered_pos": numbering,
                    "seq": [aa for aa in self.seq],
                }
            )
            .with_row_index(name="idx", offset=1)
            .with_columns(
                pl.col("region")
                .str.replace("fmwk", "FR", literal=True)
                .str.replace("cdr", "CDR", literal=True)
            )
        )

        self.fv_seq = "".join(
            self.position.filter(pl.col("region") != pl.lit("-"))
            .get_column("seq")
            .to_list()
        )
        if self.fv_seq not in self.seq:
            raise ValueError(
                f"Fv sequence {self.fv_seq} not found in the original sequence {self.seq}. "
                "This may indicate an alignment containing gaps within the Fv region."
            )

        # Get starts, ends, and sequences for each region
        self.regions = (
            self.position.filter(pl.col("region") != pl.lit("-"))
            .group_by("region", maintain_order=True)
            .agg(
                pl.col("idx").min().alias("start"),
                pl.col("idx").max().alias("end"),
                "seq",
            )
            .with_columns(pl.col("seq").list.join(""))
        )
        self.fv_range = (
            self.regions.filter(pl.col("region") == pl.lit("FR1")).item(0, "start"),
            self.regions.filter(pl.col("region") == pl.lit("FR4")).item(0, "end"),
        )
        cdr1_start, cdr1_end, self.cdr1_seq = next(
            self.regions.filter(pl.col("region") == pl.lit("CDR1"))
            .select("start", "end", "seq")
            .iter_rows()
        )
        self.cdr1_range = (cdr1_start, cdr1_end)

        cdr2_start, cdr2_end, self.cdr2_seq = next(
            self.regions.filter(pl.col("region") == pl.lit("CDR2"))
            .select("start", "end", "seq")
            .iter_rows()
        )
        self.cdr2_range = (cdr2_start, cdr2_end)

        cdr3_start, cdr3_end, self.cdr3_seq = next(
            self.regions.filter(pl.col("region") == pl.lit("CDR3"))
            .select("start", "end", "seq")
            .iter_rows()
        )
        self.cdr3_range = (cdr3_start, cdr3_end)

        # Once the CDRs are found, FRs are easy
        self.fr1_range = (self.fv_range[0], self.cdr1_range[0] - 1)
        self.fr2_range = (self.cdr1_range[1] + 1, self.cdr2_range[0] - 1)
        self.fr3_range = (self.cdr2_range[1] + 1, self.cdr3_range[0] - 1)
        self.fr4_range = (self.cdr3_range[1] + 1, self.fv_range[1])

        self.fr1_seq = self.seq[self.fr1_range[0] - 1 : self.fr1_range[1]]
        self.fr2_seq = self.seq[self.fr2_range[0] - 1 : self.fr2_range[1]]
        self.fr3_seq = self.seq[self.fr3_range[0] - 1 : self.fr3_range[1]]
        self.fr4_seq = self.seq[self.fr4_range[0] - 1 : self.fr4_range[1]]

    def format(self, include_non_fv: bool = True, highlight_cdr: bool = True) -> str:
        """Format the numbered sequence with optional CDR highlighting."""
        if include_non_fv:
            n_term_seq = self.seq[: self.fv_range[0] - 1]
            c_term_seq = self.seq[self.fv_range[1] :]

            # Gray out non-Fv regions
            if n_term_seq:
                n_term_seq = f"\033[2;9;30m{n_term_seq}\033[0m"
            if c_term_seq:
                c_term_seq = f"\033[2;9;30m{c_term_seq}\033[0m"
        else:
            n_term_seq, c_term_seq = "", ""

        if highlight_cdr:
            fv_seq = (
                f"{self.fr1_seq}\033[1;4;91m{self.cdr1_seq}\033[0m"
                f"{self.fr2_seq}\033[1;4;91m{self.cdr2_seq}\033[0m"
                f"{self.fr3_seq}\033[1;4;91m{self.cdr3_seq}\033[0m"
                f"{self.fr4_seq}"
            )
        else:
            fv_seq = self.fv_seq

        return f"{n_term_seq if include_non_fv else ''}{fv_seq}{c_term_seq}"

    def __repr__(self) -> str:
        """Return a string representation of the numbered sequence.

        * N/C terminal non-Fv regions: grayed out
        * Framework: regular text
        * CDR: underlined and bold
        """
        return self.format(include_non_fv=True, highlight_cdr=True)

    def __str__(self) -> str:
        """Return the input sequence."""
        return self.seq

    def __iter__(self):
        """Iterate over the sequence."""
        return self.position.iter_rows(named=True)

    def __len__(self) -> int:
        """Return the length of the input sequence."""
        return len(self.seq)

    def __getitem__(self, i: int | str | slice) -> str:
        """Get the sequence for a specific index or numbered position."""
        if isinstance(i, int):
            return self.seq[i]
        elif isinstance(i, str):
            pos = self.position.filter(pl.col("numbered_pos") == pl.lit(i))
            if pos.is_empty():
                raise IndexError(f"Numbered position {i} not found.")
            return pos.item(0, "seq")
        # TODO: support slices
        elif isinstance(i, slice):
            raise NotImplementedError("Slicing is not supported yet.")
        else:
            raise TypeError(
                f"Index must be an integer or a numbered position string, not {type(i)}."
            )


class AntibodyAlignment:
    """A class to hold the numbered alignment of multiple antibody sequences.

    Make sure all sequences are numbered with the same scheme, because the alignment
    is only based on the first sequence's scheme.

    The FR/CDR labels are also assigned based on the first sequence's chain type, so
    it is up to the user to ensure that all sequences are heavy or light chains. It is
    okay to mix kappa and lambda light chains.
    """

    def __init__(self, seqs: list[NumberedAntibody], seq_ids: list[str] | None = None):
        """Initialize with the numbered sequences and their alignment."""
        if not seqs:
            raise ValueError("No sequences provided for alignment.")

        self.scheme = seqs[0].scheme

        region_labels, self.df = self.numbered2df(seqs, self.scheme, seq_ids)
        self.numbered_pos = self.df.drop("seq_id").columns
        self.regions = (
            pl.DataFrame({"region": region_labels, "numbered_pos": self.numbered_pos})
            .group_by("region", maintain_order=True)
            .agg("numbered_pos")
            .select(
                "region",
                pl.col("numbered_pos")
                .list.first()
                .map_elements(
                    lambda x: self.numbered_pos.index(x), return_dtype=pl.UInt32
                )
                .alias("start"),
                pl.col("numbered_pos")
                .list.last()
                .map_elements(
                    lambda x: self.numbered_pos.index(x), return_dtype=pl.UInt32
                )
                .alias("end"),
                "numbered_pos",
            )
        )

    @staticmethod
    def numbered2df(
        seqs: list[NumberedAntibody],
        scheme: ValidSchemesType,
        seq_ids: list[str] | None = None,
    ) -> tuple[list[str], pl.DataFrame]:
        """Align antibody sequences and return a DataFrame with the results.

        Args:
            seqs: List of numbered antibody sequences.
            scheme: The numbering scheme used ("imgt", "martin", "kabat", or "aho").
            seq_ids: Optional list of sequence IDs. If None, will use indices as IDs.
                The result will be sorted by these IDs.

        Returns:
            A tuple containing:

            - A list of region labels (FR1-FR4, CDR1-CDR3)
            - A DataFrame starting with a ``seq_id`` column, followed by numbered
                positions. Each row in the DataFrame corresponds to a sequence.
        """
        # Align antibody sequences with build_msa
        annotator = SingleChainAnnotator(scheme=scheme)
        position_codes, aligned_seqs = annotator.build_msa(
            [s.seq for s in seqs], [s._raw for s in seqs]
        )
        sorted_position_codes = annotator.sort_position_codes(position_codes)
        entries = []

        # Include a row showing FR/CDR regions
        region_map = {
            **{f"fmwk{i}": f"FR{i}" for i in range(1, 5)},
            **{f"cdr{i}": f"CDR{i}" for i in range(1, 4)},
        }
        region_labels = annotator.assign_cdr_labels(
            sorted_position_codes, seqs[0].chain_type
        )
        short_region_labels = [region_map.get(r, "-") for r in region_labels]

        # Build output where each column is a numbered position in the alignment
        if seq_ids is None:
            seq_ids = [str(i) for i in range(len(seqs))]
        elif len(set(seq_ids)) != len(seq_ids):
            raise ValueError("Sequence IDs must be unique.")
        if len(seqs) != len(seq_ids):
            raise ValueError(
                f"Number of sequences ({len(seqs)}) does not match number of IDs ({len(seq_ids)})."
            )
        for seq_id, aligned_seq in zip(seq_ids, aligned_seqs, strict=True):
            entry = [seq_id, *(aa for aa in aligned_seq)]
            entries.append(entry)

        return short_region_labels, (
            pl.from_records(entries, orient="row", schema=["seq_id", *position_codes])
            .select("seq_id", *sorted_position_codes)
            .sort("seq_id")
        )

    def __repr__(self) -> str:
        """Return a string representation of the alignment."""
        return self.format()

    def format(self, highlight_cdr: bool = True, ref_seq_id: str | None = None) -> str:
        """Format the alignment with optional CDR highlighting and germline alignment."""
        aligned_strs = self._build_alignment_str(highlight_cdr, ref_seq_id)
        max_seq_id_len = max(len(seq_id) for seq_id, _, _ in aligned_strs)
        space = " " * (max_seq_id_len + 2)  # +2 for ": "
        return "".join(
            f"\n{space}{aln_indicator}\n{seq_id:>{max_seq_id_len}}: {seq}"
            for seq_id, seq, aln_indicator in aligned_strs
        )

    def _build_alignment_str(
        self, highlight_cdr: bool = True, ref_seq_id: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Build the germline alignment string.

        Args:
            highlight_cdr: If True, format the output with CDR highlighting.
            ref_seq_id: Optional reference sequence to align against. If None, use the
                first sequence in the alignment.

        Returns:
            A list of tuples, each containing:
                - The ID of the sequence.
                - The sequence with gaps (aligned).
                - Alignment indicators of the sequence to the reference sequence.

            The reference sequence will be the first element in the list, with the
            alignment indicators field being an empty string.
        """
        if ref_seq_id is None:
            ref_seq_id: str = self.df.item(0, "seq_id")
        elif ref_seq_id not in self.df.get_column("seq_id").to_list():
            raise KeyError(
                f"Reference sequence ID {ref_seq_id} not found in alignment."
            )

        # Make the alignment rows (|, ., +,  and' ')
        seq_id_order = [ref_seq_id] + (
            self.df.filter(pl.col("seq_id") != pl.lit(ref_seq_id))
            .get_column("seq_id")
            .to_list()
        )
        aligned_seqs = self.df.select(
            "seq_id", pl.concat_str(self.numbered_pos).alias("seq")
        )
        ref_seq: str = aligned_seqs.filter(pl.col("seq_id") == pl.lit(ref_seq_id)).item(
            0, "seq"
        )
        other_seqs_df = aligned_seqs.filter(
            pl.col("seq_id") != pl.lit(ref_seq_id)
        ).sort("seq_id")
        other_seq_ids: list[str] = other_seqs_df.get_column("seq_id").to_list()
        other_seqs: list[str] = other_seqs_df.get_column("seq").to_list()

        aln_indicator = [
            _build_alignment_indicator(ref_seq, other_seq) for other_seq in other_seqs
        ]
        if not highlight_cdr:
            # No need to worry about coloring
            return [(ref_seq_id, ref_seq, "")] + list(
                zip(other_seq_ids, other_seqs, aln_indicator, strict=True)
            )

        # Highlight the CDRs
        highlighted_seqs = (
            self.regions.select(
                "region",
                "start",
                (pl.col("end") - pl.col("start") + pl.lit(1)).alias("region_len"),
            )
            .with_row_index(name="index")
            # Break down the input sequence into regions
            .join(aligned_seqs, how="cross")
            .with_columns(
                pl.col("seq").str.slice(
                    offset=pl.col("start"), length=pl.col("region_len")
                )
            )
            # Highlight the CDRs
            .with_columns(
                pl.when(pl.col("region").str.contains(r"^CDR\d$"))
                .then(
                    pl.concat_str(
                        pl.lit("\033[1;4;91m"), pl.col("seq"), pl.lit("\033[0m")
                    )
                )
                .otherwise(pl.col("seq"))
                .alias("seq"),
            )
            .sort("seq_id", "index")
            .group_by("seq_id", maintain_order=True)
            .agg(pl.col("seq"))
            .with_columns(pl.col("seq").list.join(""))
            .with_columns(pl.col("seq_id").cast(pl.Enum(seq_id_order)))
            .sort("seq_id")
            .with_columns(pl.Series("aln_indicator", [""] + aln_indicator))
        )
        return list(highlighted_seqs.iter_rows(named=False))

    def __getitem__(self, i: str) -> dict[str, str]:
        """Get the sequence for a specific index or numbered position."""
        if not isinstance(i, str):
            raise TypeError(f"Index must be a numbered position string, not {type(i)}.")
        if i not in self.numbered_pos:
            raise KeyError(f"Numbered position {i} not found.")
        return {
            r["seq_id"]: r[i]
            for r in self.df.select("seq_id", pl.col(i)).iter_rows(named=True)
        }


def align_ab_seqs(
    seqs: list[NumberedAntibody], seq_ids: list[str] | None = None
) -> AntibodyAlignment:
    """Align antibody sequences and return a DataFrame with the results."""
    return AntibodyAlignment(seqs, seq_ids)


@dataclass
class Germlines:
    """A class to hold the germline information.

    NOTE: When displaying the germline sequences, only the first J gene is used.
    All V genes are paired with the first J gene for display purposes.

    Attributes:
        v_genes: The V gene names.
        j_genes: The J gene names.
        species: The species assignment for the germline.
        v_gene_seqs: The germline amino acid sequence for the V gene, gapped to be length
            128 consistent with the IMGT numbering. If the ``v_gene`` did not match
            to a germline, this will be ``None``.
        j_gene_seqs: The germline amino acid sequence for the J gene.
    """

    v_genes: list[str]
    j_genes: list[str]
    species: AssignedSpeciesType
    v_gene_seqs: list[str]
    j_gene_seqs: list[str]

    def seq(self, trimmed: bool = False, v_idx: int = 0, j_idx: int = 0) -> str:
        """Return the first merged germline sequence.

        Args:
            trimmed: If True, trim the sequence to remove gaps.
            v_idx: The index of the V gene to use.
            j_idx: The index of the J gene to use.
        """
        merged_seq_aas: list[str] = []
        for v, j in zip(self.v_gene_seqs[v_idx], self.j_gene_seqs[j_idx], strict=True):
            if j == "-":
                if v == "-":
                    merged_seq_aas.append("-")
                else:
                    merged_seq_aas.append(v)
            else:
                merged_seq_aas.append(j)

        return (
            "".join(merged_seq_aas)
            if not trimmed
            else "".join(c for c in merged_seq_aas if c != "-")
        )

    def numbered_seq(self, scheme: ValidSchemesType = "imgt") -> list[NumberedAntibody]:
        """Return the first numbered germline according to the specified scheme."""
        return number_ab_seq(
            [self.seq(trimmed=True, v_idx=i) for i in range(len(self.v_genes))],
            scheme,
            assign_germline=False,
            species=self.species,
        )

    def numbered_seqs(self, scheme: ValidSchemesType = "imgt") -> NumberedAntibody:
        """Return all numbered germlines according to the specified scheme."""
        return number_ab_seq(
            self.seq(trimmed=True), scheme, assign_germline=False, species=self.species
        )

    def __repr__(self) -> str:
        """Return a string representation of the germline."""
        return f"""
IMGT-numbered seq: {self.seq()}
          Species: {self.species}
           V gene: {self.v_genes}
           J gene: {self.j_genes}
""".strip()


class NumberedAntibodyWithGermline(NumberedAntibody):
    """A class to hold the numbered sequence and its regions.

    Attributes:
        closest_germline: The closest germlines assigned to the sequence.
        aligned_germline: A DataFrame with the aligned germline sequence to the numbered sequence.
            Contains columns "region", "numbered_pos", "seq", and "germline".
            The "fv_idx" column is 1-based and indicates the position in the Fv region.

        Other attributes are inherited from ``NumberedAntibody``.
    """

    def __init__(
        self,
        seq: str,
        scheme: ValidSchemesType,
        numbering: list[str],
        percent_identity: float,
        chain_type: str,
        error_message: str,
        region_labels: list[str],
        scheme_aligned_seq: str,
        closest_germline: Germlines,
    ):
        """Initialize with the output of ``SingleChainAnnotator.analyze_seq``."""
        super().__init__(
            seq,
            scheme,
            numbering,
            percent_identity,
            chain_type,
            error_message,
            region_labels,
            scheme_aligned_seq,
        )
        self.closest_germline = closest_germline
        self._aligned_germline = None

    @classmethod
    def from_numbered_antibody(
        cls,
        numbered_ab: NumberedAntibody,
        closest_germline: Germlines,
    ) -> "NumberedAntibodyWithGermline":
        """Create a NumberedAntibodyWithGermline from a NumberedAntibody."""
        return cls(
            numbered_ab.seq,
            numbered_ab.scheme,
            *numbered_ab._raw,
            numbered_ab.position.get_column("region").to_list(),
            numbered_ab.scheme_aligned_seq,
            closest_germline,
        )

    @property
    def aligned_germline(self) -> AntibodyAlignment:
        """Align the germline sequence to the numbered sequence."""
        if self._aligned_germline is not None:
            return self._aligned_germline

        numbered_seqs = [self, *self.closest_germline.numbered_seq(self.scheme)]
        seq_ids = ["Query"] + [
            f"{v}|{self.closest_germline.j_genes[0]}"
            for v in self.closest_germline.v_genes
        ]

        self._aligned_germline = AntibodyAlignment(seqs=numbered_seqs, seq_ids=seq_ids)
        return self._aligned_germline

    @property
    def imputed_seq(self) -> str:
        """Impute gaps in the FR1/FR4 region with the closest germline."""
        position_regions = self.aligned_germline.regions.select(
            "region", "numbered_pos"
        ).explode("numbered_pos")

        # impute seq with the first germline
        keep_germline_id: str = (
            self.aligned_germline.df.filter(pl.col("seq_id") != pl.lit("Query"))
            .get_column("seq_id")
            .item(0)
        )
        aln = (
            self.aligned_germline.df.filter(
                pl.col("seq_id").is_in({"Query", keep_germline_id})
            )
            .with_columns(pl.col("seq_id").replace(keep_germline_id, "Germline"))
            .unpivot(index="seq_id", variable_name="numbered_pos", value_name="aa")
            .join(position_regions, on="numbered_pos")
            .pivot(
                index=("region", "numbered_pos"),
                on="seq_id",
                values="aa",
                aggregate_function=None,
            )
            .with_columns(
                pl.col("numbered_pos").cast(pl.Enum(self.aligned_germline.numbered_pos))
            )
            .sort("numbered_pos")
            .with_row_index(name="fv_idx", offset=1)
            .with_columns(
                pl.when(
                    (pl.col("Query") == pl.lit("-"))
                    & (pl.col("region").is_in({"FR1", "FR4"}))
                )
                .then(pl.col("Germline"))
                .otherwise(pl.col("Query"))
                .alias("imputed_seq")
            )
        )
        imputed_seq = "".join(
            aln.filter(pl.col("imputed_seq") != pl.lit("-"))
            .get_column("imputed_seq")
            .to_list()
        )
        if self.fv_seq not in imputed_seq:
            raise ValueError(
                f"Fv sequence {self.fv_seq} not found in the imputed sequence {imputed_seq}. "
                "This may indicate an alignment containing gaps within the Fv region."
            )
        return imputed_seq

    def format(
        self,
        show_germline: bool = True,
        include_non_fv: bool = True,
        highlight_cdr: bool = True,
    ) -> str:
        """Format the numbered sequence with optional CDR highlighting and germline alignment."""
        if not show_germline:
            return super().format(
                include_non_fv=include_non_fv, highlight_cdr=highlight_cdr
            )

        germline_alignment_str = self.aligned_germline.format(
            highlight_cdr=highlight_cdr, ref_seq_id="Query"
        )
        return (
            super().format(
                include_non_fv=include_non_fv,
                highlight_cdr=highlight_cdr,
            )
            + f"\n\n\033[1m# Closest germline\033[0m\n\n"
            f"Species: {self.closest_germline.species} ({self.chain_type})\n"
            f"V gene: {self.closest_germline.v_genes}\n"
            f"J gene: {self.closest_germline.j_genes}"
            f"{germline_alignment_str}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the numbered sequence with germline."""
        return self.format(show_germline=True)


@overload
def number_ab_seq(
    seq: str,
    scheme: ValidSchemesType,
    assign_germline: Literal[False] = ...,
    species: ValidSpeciesType | None = ...,
) -> NumberedAntibody: ...
@overload
def number_ab_seq(
    seq: str,
    scheme: ValidSchemesType,
    assign_germline: Literal[True] = ...,
    species: ValidSpeciesType | None = ...,
) -> NumberedAntibodyWithGermline: ...
@overload
def number_ab_seq(
    seq: list[str],
    scheme: ValidSchemesType,
    assign_germline: Literal[False] = ...,
    species: ValidSpeciesType | None = ...,
) -> list[NumberedAntibody]: ...
@overload
def number_ab_seq(
    seq: list[str],
    scheme: ValidSchemesType,
    assign_germline: Literal[True] = ...,
    species: ValidSpeciesType | None = ...,
) -> list[NumberedAntibodyWithGermline]: ...
def number_ab_seq(
    seq: str | list[str],
    scheme: ValidSchemesType,
    assign_germline: bool = False,
    species: ValidSpeciesType | None = None,
):
    """Number the sequence according to IMGT numbering scheme.

    Args:
        seq: The antibody sequence to number.
        scheme: The numbering scheme to use. Must be one of "imgt", "martin", "kabat", or "aho".
        assign_germline: If True, assign the closest germline genes to the sequence.
        species: The species to use for germline assignment. Defaults to "unknown", which
            looks for all of human, mouse, alpaca, and rabbit.
    """
    # Antibody numbering
    chain_annotator: SingleChainAnnotator = SingleChainAnnotator(scheme=scheme)

    if assign_germline:
        vj_annotator = VJGeneTool(scheme=scheme)
        if isinstance(seq, str):
            alignment: AntPackAlignmentType = chain_annotator.analyze_seq(seq)
            return _process_alignment_with_germline(
                alignment, seq, chain_annotator, vj_annotator, scheme, species
            )
        elif isinstance(seq, list) and all(isinstance(s, str) for s in seq):
            alignments: list[AntPackAlignmentType] = chain_annotator.analyze_seqs(seq)
            return [
                _process_alignment_with_germline(
                    alignment, s, chain_annotator, vj_annotator, scheme, species
                )
                for alignment, s in zip(alignments, seq, strict=True)
            ]
        else:
            raise TypeError(
                f"Sequence must be a string or a list of strings, not {type(seq)}."
            )
    else:
        if isinstance(seq, str):
            alignment: AntPackAlignmentType = chain_annotator.analyze_seq(seq)
            return _process_alignment(alignment, seq, chain_annotator, scheme)
        elif isinstance(seq, list) and all(isinstance(s, str) for s in seq):
            alignments: list[AntPackAlignmentType] = chain_annotator.analyze_seqs(seq)
            return [
                _process_alignment(alignment, s, chain_annotator, scheme)
                for alignment, s in zip(alignments, seq, strict=True)
            ]
        else:
            raise TypeError(
                f"Sequence must be a string or a list of strings, not {type(seq)}."
            )


def _assign_region_labels(
    alignment: AntPackAlignmentType, chain_annotator: SingleChainAnnotator
) -> list[str]:
    """Assign region labels to the numbered alignment."""
    numbering, percent_identity, chain_type, err = alignment

    if percent_identity < 0.85:
        warning_msg = f"{err}\nPercent identity ({percent_identity:.2%}) is low for assigned chain type {chain_type}"
        # logger.warning(f"{warning_msg}: {seq}")
        if percent_identity < 0.7:
            raise ValueError(warning_msg)
    return chain_annotator.assign_cdr_labels(numbering, chain_type)


def _process_alignment(
    alignment: AntPackAlignmentType,
    seq: str,
    chain_annotator: SingleChainAnnotator,
    scheme: ValidSchemesType,
) -> NumberedAntibody:
    """Process the alignment output to extract relevant information."""
    numbering, percent_identity, chain_type, err = alignment
    region_labels: list[str] = _assign_region_labels(alignment, chain_annotator)

    trimmed_seq, trimmed_aln, *_ = chain_annotator.trim_alignment(seq, alignment)
    _, aligned_seqs = chain_annotator.build_msa(
        [trimmed_seq], [(trimmed_aln, *alignment[1:])], add_unobserved_positions=True
    )

    return NumberedAntibody(
        seq,
        scheme,
        numbering,
        percent_identity,
        chain_type,
        err,
        region_labels,
        aligned_seqs[0],
    )


def _process_alignment_with_germline(
    alignment: AntPackAlignmentType,
    seq: str,
    chain_annotator: SingleChainAnnotator,
    vj_annotator: VJGeneTool,
    scheme: ValidSchemesType,
    species: ValidSpeciesType | None,
) -> NumberedAntibodyWithGermline:
    numbered_ab = _process_alignment(alignment, seq, chain_annotator, scheme)
    region_labels = numbered_ab.position.get_column("region").to_list()

    # Assign VJ germline genes
    query_species = "unknown" if species is None else species
    closest_germline = _assign_closest_germlines(
        alignment, seq, region_labels, vj_annotator=vj_annotator, species=query_species
    )
    return NumberedAntibodyWithGermline.from_numbered_antibody(
        numbered_ab, closest_germline
    )


def _assign_closest_germlines(
    alignment: AntPackAlignmentType,
    seq: str,
    region_labels: list[str],
    vj_annotator: VJGeneTool,
    species: ValidSpeciesType,
) -> Germlines:
    """Assign the closest germline to the sequence.

    NOTE: When multiple V/J genes with identical scores are found,
    all gene names are returned.
    """
    # The assignment works best with IMGT numbering
    # Only use Fv region for VJ gene assignment because of upstream issue:
    # https://github.com/jlparkI/AntPack/issues/31
    full_seq_len = len(seq)
    fv_start = next((i for i, label in enumerate(region_labels) if label != "-"), -1)
    fv_end = next(
        (
            full_seq_len - i
            for i, label in enumerate(reversed(region_labels))
            if label != "-"
        ),
        -1,
    )
    if fv_start == -1 or fv_end == -1:
        raise ValueError(
            "Fv region not found in the sequence. Ensure the sequence is properly formatted."
        )

    fv_seq = seq[fv_start:fv_end]
    fv_alignment = (alignment[0][fv_start:fv_end], *alignment[1:])

    v_genes, j_genes, v_blosum, j_blosum, assigned_species = (
        vj_annotator.assign_vj_genes(
            fv_alignment,
            fv_seq,
            species=species,  # human, mouse, alpaca, rabbit
            mode="evalue",  # better alignment than "identity" but may fail
        )
    )
    if not (v_genes and j_genes):
        # Fallback to identity mode if evalue mode fails
        logger.warning(
            f"Failed to assign V and J genes for sequence: {fv_seq} using evalue mode. "
            "Falling back to identity mode."
        )
        v_genes, j_genes, v_ident, j_ident, assigned_species = (
            vj_annotator.assign_vj_genes(
                fv_alignment, fv_seq, species=species, mode="identity"
            )
        )
    if not (v_genes and j_genes):
        raise ValueError(f"Failed to assign V and J genes for sequence: {seq}.")

    v_gene_names: list[str] = v_genes.split("_")
    j_gene_names: list[str] = j_genes.split("_")
    v_seqs: list[str] = [
        vj_annotator.get_vj_gene_sequence(v, assigned_species) for v in v_gene_names
    ]
    j_seqs: list[str] = [
        vj_annotator.get_vj_gene_sequence(j, assigned_species) for j in j_gene_names
    ]
    return Germlines(v_gene_names, j_gene_names, assigned_species, v_seqs, j_seqs)


def _build_alignment_indicator(seq1: str, seq2: str) -> str:
    """Build an alignment indicator string for two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length for alignment.")

    indicator = []
    for aa1, aa2 in zip(seq1, seq2, strict=True):
        if (aa1 == "-") or (aa2 == "-"):
            indicator.append(" ")
        elif aa1 == aa2:
            indicator.append("|")
        elif f"{aa1}{aa2}" in SIMILAR_PAIRS:
            indicator.append("+")
        else:
            indicator.append(".")
    return "".join(indicator)
