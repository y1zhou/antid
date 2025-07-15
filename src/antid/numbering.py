"""Antibody numbering utilities."""

from dataclasses import dataclass
from typing import Literal, overload

import polars as pl
from loguru import logger

# from antpack import SingleChainAnnotator, VJGeneTool
from antid.utils.patch_antpack import SingleChainAnnotator, VJGeneTool

__all__ = ["number_ab_seq", "align_ab_seqs"]
valid_schemes = Literal["imgt", "martin", "kabat", "aho"]


class NumberedAntibody:
    """A class to hold the numbered sequence and its regions.

    Attributes:
        seq: The input antibody sequence.
        scheme: The numbering scheme used ("imgt", "martin", "kabat", or "aho").
        chain_type: Assigned chain type ("H", "L", or "K").
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
        scheme: valid_schemes,
        numbering: list[str],
        percent_identity: float,
        chain_type: str,
        error_message: str,
        region_labels: list[str],
    ):
        """Initialize with the output of ``SingleChainAnnotator.analyze_seq``."""
        self.seq = seq
        self.scheme: valid_schemes = scheme
        self.chain_type = chain_type
        self._raw = (numbering, percent_identity, chain_type, error_message)

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
            n_term_seq = f"\033[2m{n_term_seq}\033[0m"
            c_term_seq = f"\033[2m{c_term_seq}\033[0m"
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

    def __getitem__(self, i: int | str) -> str:
        """Get the sequence for a specific index or numbered position."""
        if isinstance(i, int):
            return self.seq[i]
        elif isinstance(i, str):
            return self.position.filter(pl.col("numbered_pos") == pl.lit(i)).item(
                0, "seq"
            )
        # TODO: support slices
        else:
            raise TypeError(
                f"Index must be an integer or a numbered position string, not {type(i)}."
            )


@dataclass
class Germline:
    """A class to hold the germline information.

    Attributes:
        v_gene: The V gene name.
        j_gene: The J gene name.
        species: The species assignment for the germline.
        v_gene_seq: The germline amino acid sequence for the V gene, gapped to be length
            128 consistent with the IMGT numbering. If the ``v_gene`` did not match
            to a germline, this will be ``None``.
        j_gene_seq: The germline amino acid sequence for the J gene.
    """

    v_gene: str
    j_gene: str
    species: str
    v_gene_seq: str | None
    j_gene_seq: str | None

    def seq(self, trimmed: bool = False) -> str:
        """Return the merged germline sequence.

        Args:
            trimmed: If True, trim the sequence to remove gaps.
        """
        if not (self.v_gene_seq and self.j_gene_seq):
            raise ValueError(
                "Provided v_gene and j_gene did not both match to a germline."
            )
        merged_seq_aas: list[str] = []
        for v, j in zip(self.v_gene_seq, self.j_gene_seq, strict=True):
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

    def numbered_seq(self, scheme: valid_schemes = "imgt") -> NumberedAntibody:
        """Return the numbered germline according to the specified scheme."""
        return number_ab_seq(
            self.seq(trimmed=True), scheme, assign_germline=False, species=self.species
        )

    def __repr__(self) -> str:
        """Return a string representation of the germline."""
        return f"""
IMGT-numbered seq: {self.seq()}
          Species: {self.species}
           V gene: {self.v_gene}
           J gene: {self.j_gene}
""".strip()


class NumberedAntibodyWithGermline(NumberedAntibody):
    """A class to hold the numbered sequence and its regions.

    Attributes:
        closest_germline: The closest germline assigned to the sequence.
        aligned_germline: A DataFrame with the aligned germline sequence to the numbered sequence.
            Contains columns "region", "numbered_pos", "seq", and "germline".
            The "fv_idx" column is 1-based and indicates the position in the Fv region.

        Other attributes are inherited from ``NumberedAntibody``.
    """

    def __init__(
        self,
        seq: str,
        scheme: valid_schemes,
        numbering: list[str],
        percent_identity: float,
        chain_type: str,
        error_message: str,
        region_labels: list[str],
        closest_germline: Germline,
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
        )
        self.closest_germline = closest_germline
        self._aligned_germline = None

    @property
    def aligned_germline(self) -> pl.DataFrame:
        """Align the germline sequence to the numbered sequence."""
        if self._aligned_germline is not None:
            return self._aligned_germline

        if not self.closest_germline:
            raise ValueError("No closest germline assigned.")

        aligned_germline = align_ab_seqs(
            [self, self.closest_germline.numbered_seq(self.scheme)],
            include_region=True,
        ).with_columns(
            pl.col("seq_id").replace_strict(
                {"region": "region", "0": "self", "1": "germline"}
            )
        )
        aligned_positions = aligned_germline.drop("seq_id").columns
        self._aligned_germline = (
            aligned_germline.drop("seq_id")
            .transpose()
            .select(
                pl.col("column_0").alias("region"),
                pl.Series("numbered_pos", aligned_positions),
                pl.col("column_1").alias("seq"),
                pl.col("column_2").alias("germline"),
            )
            .with_row_index(name="fv_idx", offset=1)
        )
        return self._aligned_germline

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

        self_aligned, aln_indicator, germline_aligned = (
            self._build_germline_alignment_str()
        )
        return (
            super().format(
                include_non_fv=include_non_fv,
                highlight_cdr=highlight_cdr,
            )
            + f"\n\n\033[1m# Closest germline\033[0m\n\n"
            f"Species: {self.closest_germline.species} ({self.chain_type})\n"
            f"V gene: {self.closest_germline.v_gene}\n"
            f"J gene: {self.closest_germline.j_gene}\n\n"
            f"Input seq: {self_aligned}\n"
            f"           {aln_indicator}\n"
            f" Germline: {germline_aligned}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the numbered sequence with germline."""
        return self.format(show_germline=True)

    def _build_germline_alignment_str(
        self, formatted: bool = True
    ) -> tuple[str, str, str]:
        """Build the germline alignment string."""
        self_aligned = self.aligned_germline.get_column("seq").str.join("")[0]
        germline_aligned = self.aligned_germline.get_column("germline").str.join("")[0]

        aln_indicator = _build_alignment_indicator(self_aligned, germline_aligned)
        if not formatted:
            return self_aligned, aln_indicator, germline_aligned

        # Highlight the CDRs
        highlighted_seqs = (
            self.aligned_germline.group_by("region", maintain_order=True)
            .agg("seq", "germline")
            .with_columns(
                pl.when(pl.col("region").str.contains(r"^CDR\d$"))
                .then(
                    pl.concat_str(
                        pl.lit("\033[1;4;91m"),
                        pl.col("seq").list.join(""),
                        pl.lit("\033[0m"),
                    )
                )
                .otherwise(pl.col("seq").list.join(""))
                .alias("seq"),
                pl.when(pl.col("region").str.contains(r"^CDR\d$"))
                .then(
                    pl.concat_str(
                        pl.lit("\033[1;4;91m"),
                        pl.col("germline").list.join(""),
                        pl.lit("\033[0m"),
                    )
                )
                .otherwise(pl.col("germline").list.join(""))
                .alias("germline"),
            )
        )
        self_aligned = highlighted_seqs.get_column("seq").str.join("")[0]
        germline_aligned = highlighted_seqs.get_column("germline").str.join("")[0]

        return self_aligned, aln_indicator, germline_aligned


@overload
def number_ab_seq(
    seq: str, scheme: valid_schemes, assign_germline: Literal[False], species: str
) -> NumberedAntibody: ...
@overload
def number_ab_seq(
    seq: str, scheme: valid_schemes, assign_germline: Literal[True], species: str
) -> NumberedAntibodyWithGermline: ...
@overload
def number_ab_seq(
    seq: list[str], scheme: valid_schemes, assign_germline: Literal[False], species: str
) -> list[NumberedAntibody]: ...
@overload
def number_ab_seq(
    seq: list[str], scheme: valid_schemes, assign_germline: Literal[True], species: str
) -> list[NumberedAntibodyWithGermline]: ...
def number_ab_seq(
    seq: str | list[str],
    scheme: valid_schemes,
    assign_germline: bool = False,
    species: str = "unknown",
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
            alignment: tuple[list[str], float, str, str] = chain_annotator.analyze_seq(
                seq
            )
            return _process_alignment_with_germline(
                alignment, seq, chain_annotator, vj_annotator, scheme, species
            )
        elif isinstance(seq, list):
            alignments: list[tuple[list[str], float, str, str]] = (
                chain_annotator.analyze_seqs(seq)
            )
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
            alignment: tuple[list[str], float, str, str] = chain_annotator.analyze_seq(
                seq
            )
            return _process_alignment(alignment, seq, chain_annotator, scheme)
        elif isinstance(seq, list):
            alignments: list[tuple[list[str], float, str, str]] = (
                chain_annotator.analyze_seqs(seq)
            )
            return [
                _process_alignment(alignment, s, chain_annotator, scheme)
                for alignment, s in zip(alignments, seq, strict=True)
            ]
        else:
            raise TypeError(
                f"Sequence must be a string or a list of strings, not {type(seq)}."
            )


def _assign_region_labels(
    alignment: tuple[list[str], float, str, str], chain_annotator: SingleChainAnnotator
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
    alignment: tuple[list[str], float, str, str],
    seq: str,
    chain_annotator: SingleChainAnnotator,
    scheme: valid_schemes,
) -> NumberedAntibody:
    """Process the alignment output to extract relevant information."""
    numbering, percent_identity, chain_type, err = alignment
    region_labels: list[str] = _assign_region_labels(alignment, chain_annotator)

    return NumberedAntibody(
        seq, scheme, numbering, percent_identity, chain_type, err, region_labels
    )


def _process_alignment_with_germline(
    alignment: tuple[list[str], float, str, str],
    seq: str,
    chain_annotator: SingleChainAnnotator,
    vj_annotator: VJGeneTool,
    scheme: valid_schemes,
    species: str = "unknown",
) -> NumberedAntibodyWithGermline:
    numbering, percent_identity, chain_type, err = alignment
    region_labels: list[str] = _assign_region_labels(alignment, chain_annotator)

    # Assign VJ germline genes
    closest_germline = _assign_closest_germline(
        alignment, seq, region_labels, vj_annotator=vj_annotator, species=species
    )
    return NumberedAntibodyWithGermline(
        seq,
        scheme,
        numbering,
        percent_identity,
        chain_type,
        err,
        region_labels,
        closest_germline,
    )


def _assign_closest_germline(
    alignment: tuple[list[str], float, str, str],
    seq: str,
    region_labels: list[str],
    vj_annotator: VJGeneTool,
    species: str = "unknown",
) -> Germline:
    """Assign the closest germline to the sequence.

    NOTE: When multiple V/J genes with very close scores are found, the first one in the list is returned.
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

    v_gene = v_genes.split("_")[0]
    j_gene = j_genes.split("_")[0]
    v_seq: str | None = vj_annotator.get_vj_gene_sequence(v_gene, assigned_species)
    j_seq: str | None = vj_annotator.get_vj_gene_sequence(j_gene, assigned_species)
    return Germline(v_gene, j_gene, assigned_species, v_seq, j_seq)


def align_ab_seqs(
    seqs: list[NumberedAntibody], include_region: bool = False
) -> pl.DataFrame:
    """Align antibody sequences and return a DataFrame with the results.

    NOTE: The numbering scheme of the first item is used for alignment.
    """
    if not seqs:
        raise ValueError("No sequences provided for alignment.")

    # Align antibody sequences with build_msa
    annotator = SingleChainAnnotator(scheme=seqs[0].scheme)
    position_codes, aligned_seqs = annotator.build_msa(
        [s.seq for s in seqs], [s._raw for s in seqs]
    )
    sorted_position_codes = annotator.sort_position_codes(position_codes)
    entries = []

    # Include a row showing FR/CDR regions
    if include_region:
        region_map = {
            **{f"fmwk{i}": f"FR{i}" for i in range(1, 5)},
            **{f"cdr{i}": f"CDR{i}" for i in range(1, 4)},
        }
        region_labels = annotator.assign_cdr_labels(position_codes, seqs[0].chain_type)
        entries.append(["region", *(region_map.get(r, "-") for r in region_labels)])

    # Build output where each column is a numbered position in the alignment
    for seq_idx, aligned_seq in enumerate(aligned_seqs):
        entry = [str(seq_idx), *(aa for aa in aligned_seq)]
        entries.append(entry)

    return pl.from_records(
        entries, orient="row", schema=["seq_id", *position_codes]
    ).select("seq_id", *sorted_position_codes)


# Based on positive score in Blosum62
# Credit to: https://github.com/prihoda/AbNumber/blob/2cc13f4bafcc74e0c619780aeff018d3b24be3ee/abnumber/common.py#L89
SIMILAR_PAIRS = {
    "AA",
    "AS",
    "CC",
    "DD",
    "DE",
    "DN",
    "ED",
    "EE",
    "EK",
    "EQ",
    "FF",
    "FW",
    "FY",
    "GG",
    "HH",
    "HN",
    "HY",
    "II",
    "IL",
    "IM",
    "IV",
    "KE",
    "KK",
    "KQ",
    "KR",
    "LI",
    "LL",
    "LM",
    "LV",
    "MI",
    "ML",
    "MM",
    "MV",
    "ND",
    "NH",
    "NN",
    "NS",
    "PP",
    "QE",
    "QK",
    "QQ",
    "QR",
    "RK",
    "RQ",
    "RR",
    "SA",
    "SN",
    "SS",
    "ST",
    "TS",
    "TT",
    "VI",
    "VL",
    "VM",
    "VV",
    "WF",
    "WW",
    "WY",
    "YF",
    "YH",
    "YW",
    "YY",
}


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


if __name__ == "__main__":
    # Example usage on Pembrolizumab (5b8c), with additional His-tag on the N-terminus
    # and GS linker on the C-terminus.
    vh = "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    vh_numbered = number_ab_seq(
        "HHHHH" + vh + "GGGGS" * 3, "martin", assign_germline=True, species="unknown"
    )
    print("\n#######\n# VH  #\n#######")
    print(repr(vh_numbered))

    vl = "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIKTSENLYFQ"
    print("\n#######\n# VL  #\n#######")
    vl_numbered = number_ab_seq(
        "HHHHH" + vl, "martin", assign_germline=True, species="unknown"
    )
    print(repr(vl_numbered))
