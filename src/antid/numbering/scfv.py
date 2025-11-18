"""Antibody numbering utilities."""

from typing import overload

from antid.numbering.antibody import (
    AntPackAlignmentType,
    NumberedAntibody,
    NumberedAntibodyWithGermline,
    ValidSchemesType,
    ValidSpeciesType,
    _process_alignment,
    _process_alignment_with_germline,
)
from antid.utils.patch_antpack import PairedChainAnnotator, VJGeneTool

ScFvAlignmentType = tuple[AntPackAlignmentType, AntPackAlignmentType]


class NumberedScFv:
    """Class representing a numbered scFv antibody."""

    def __init__(
        self,
        vh: NumberedAntibody | NumberedAntibodyWithGermline,
        vl: NumberedAntibody | NumberedAntibodyWithGermline,
    ):
        """Initialize with two numbered chains."""
        self.vh = vh
        self.vl = vl

    def __repr__(self) -> str:
        """Return a string representation of the NumberedScFv."""
        return f"""
VH: {self.vh.format()}

VL: {self.vl.format()}
""".strip()


@overload
def number_scfv_seq(
    seq: str,
    scheme: ValidSchemesType,
    assign_germline: bool = ...,
    species: ValidSpeciesType | None = ...,
) -> NumberedScFv: ...
@overload
def number_scfv_seq(
    seq: list[str],
    scheme: ValidSchemesType,
    assign_germline: bool = ...,
    species: ValidSpeciesType | None = ...,
) -> list[NumberedScFv]: ...
def number_scfv_seq(
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
    # scFv numbering
    chain_annotator: PairedChainAnnotator = PairedChainAnnotator(scheme=scheme)

    if assign_germline:
        vj_annotator = VJGeneTool(scheme=scheme)
        if isinstance(seq, str):
            alignments: ScFvAlignmentType = chain_annotator.analyze_seq(seq)
            return _process_scfv_alignment_with_germline(
                alignments, seq, chain_annotator, vj_annotator, scheme, species
            )
        elif isinstance(seq, list) and all(isinstance(s, str) for s in seq):
            alignments: tuple[
                list[AntPackAlignmentType], list[AntPackAlignmentType]
            ] = chain_annotator.analyze_seqs(seq)
            vh_alignments, vl_alignments = alignments
            return [
                _process_scfv_alignment_with_germline(
                    (vh_alignment, vl_alignment),
                    s,
                    chain_annotator,
                    vj_annotator,
                    scheme,
                    species,
                )
                for vh_alignment, vl_alignment, s in zip(
                    vh_alignments, vl_alignments, seq, strict=True
                )
            ]
        else:
            raise TypeError(
                f"Sequence must be a string or a list of strings, not {type(seq)}."
            )
    else:
        if isinstance(seq, str):
            alignments: AntPackAlignmentType = chain_annotator.analyze_seq(seq)
            return _process_scfv_alignment(alignments, seq, chain_annotator, scheme)
        elif isinstance(seq, list) and all(isinstance(s, str) for s in seq):
            alignments: tuple[
                list[AntPackAlignmentType], list[AntPackAlignmentType]
            ] = chain_annotator.analyze_seqs(seq)
            vh_alignments, vl_alignments = alignments
            return [
                _process_scfv_alignment(
                    (vh_alignment, vl_alignment), s, chain_annotator, scheme
                )
                for vh_alignment, vl_alignment, s in zip(
                    vh_alignments, vl_alignments, seq, strict=True
                )
            ]
        else:
            raise TypeError(
                f"Sequence must be a string or a list of strings, not {type(seq)}."
            )


def _process_scfv_alignment(
    alignments: ScFvAlignmentType,
    seq: str,
    chain_annotator: PairedChainAnnotator,
    scheme: ValidSchemesType,
) -> NumberedScFv:
    """Process the scFv alignment and return a NumberedScFv object."""
    vh_alignment, vl_alignment = alignments

    vh_numbered = _process_alignment(vh_alignment, seq, chain_annotator, scheme)
    vl_numbered = _process_alignment(vl_alignment, seq, chain_annotator, scheme)

    return NumberedScFv(vh=vh_numbered, vl=vl_numbered)


def _process_scfv_alignment_with_germline(
    alignments: ScFvAlignmentType,
    seq: str,
    chain_annotator: PairedChainAnnotator,
    vj_annotator: VJGeneTool,
    scheme: ValidSchemesType,
    species: ValidSpeciesType | None,
) -> NumberedScFv:
    """Process the scFv alignment with germline and return a NumberedScFv object."""
    vh_alignment, vl_alignment = alignments

    vh_numbered = _process_alignment_with_germline(
        vh_alignment, seq, chain_annotator, vj_annotator, scheme, species
    )
    vl_numbered = _process_alignment_with_germline(
        vl_alignment, seq, chain_annotator, vj_annotator, scheme, species
    )

    return NumberedScFv(vh=vh_numbered, vl=vl_numbered)
