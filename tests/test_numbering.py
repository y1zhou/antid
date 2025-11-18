"""Tests for antibody numbering."""

import pytest

from antid.numbering import (
    AntibodyAlignment,
    NumberedAntibody,
    NumberedAntibodyWithGermline,
    align_ab_seqs,
    number_ab_seq,
)
from antid.numbering.antibody import _build_alignment_indicator

# Pembrolizumab sequences extracted from PDB ID 5b8c
VH_SEQ = "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
VL_SEQ = "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIKTSENLYFQ"


# ruff: noqa: S101
@pytest.fixture
def vh_martin() -> NumberedAntibody:
    """Return a numbered VH sequence using the Martin scheme."""
    return number_ab_seq(VH_SEQ, "martin", assign_germline=False)


@pytest.fixture
def vl_martin() -> NumberedAntibody:
    """Return a numbered VL sequence using the Martin scheme."""
    return number_ab_seq(VL_SEQ, "martin", assign_germline=False)


@pytest.fixture
def vh_germline_martin() -> NumberedAntibodyWithGermline:
    """Return a numbered VH sequence using the Martin scheme."""
    return number_ab_seq(VH_SEQ, "martin", assign_germline=True)


@pytest.fixture
def vh_germline_imgt() -> NumberedAntibodyWithGermline:
    """Return a numbered VH sequence using the IMGT scheme."""
    return number_ab_seq(VH_SEQ, "imgt", assign_germline=True)


@pytest.fixture
def vl_germline_imgt() -> NumberedAntibodyWithGermline:
    """Return a numbered VL sequence using the IMGT scheme."""
    return number_ab_seq(VL_SEQ, "imgt", assign_germline=True)


def test_number_ab_seq_single_no_germline(vh_martin, vl_martin):
    """Test numbering a single sequence without germline assignment."""
    assert isinstance(vh_martin, NumberedAntibody)
    assert not isinstance(vh_martin, NumberedAntibodyWithGermline)

    # attributes
    assert vh_martin.seq == VH_SEQ
    assert vh_martin.scheme == "martin"
    assert vh_martin.chain_type == "H"

    assert vh_martin.fv_seq == VH_SEQ
    assert vl_martin.fv_seq == VL_SEQ[:111]  # trailing TSENLYFQ is not Fv
    assert vl_martin.chain_type == "K"

    # magic methods
    assert len(vh_martin) == len(VH_SEQ)
    assert str(vh_martin) == VH_SEQ
    assert next(iter(vh_martin)) == {
        "idx": 1,
        "region": "FR1",
        "numbered_pos": "1",
        "seq": "Q",
    }
    assert vh_martin[0] == "Q"
    assert vh_martin["1"] == "Q"
    with pytest.raises(IndexError):
        _ = vh_martin["200"]
    with pytest.raises(IndexError):
        _ = vh_martin[500]
    with pytest.raises(NotImplementedError):
        _ = vh_martin[1:5]
    with pytest.raises(TypeError):
        _ = vh_martin[1.5]  # type: ignore


def test_number_ab_seq_single_with_germline(vh_germline_imgt, vl_germline_imgt):
    """Test numbering a single sequence with germline assignment."""
    assert isinstance(vh_germline_imgt, NumberedAntibodyWithGermline)
    assert vh_germline_imgt.seq == VH_SEQ
    assert vh_germline_imgt.scheme == "imgt"

    assert isinstance(vl_germline_imgt, NumberedAntibodyWithGermline)
    assert vl_germline_imgt.seq == VL_SEQ
    assert vl_germline_imgt.scheme == "imgt"
    assert vl_germline_imgt.chain_type == "K"
    assert vl_germline_imgt.closest_germline is not None
    assert vl_germline_imgt.closest_germline.species == "human"
    assert "IGKV" in vl_germline_imgt.closest_germline.v_gene
    assert "IGKJ" in vl_germline_imgt.closest_germline.j_gene


def test_number_ab_seq_germline_assignment(vh_germline_martin, vh_germline_imgt):
    """Test that germline assignment works correctly."""
    assert vh_germline_martin.closest_germline == vh_germline_imgt.closest_germline


def test_number_ab_seq_list(vh_germline_imgt, vl_germline_imgt):
    """Test numbering a list of sequences."""
    seqs = [VH_SEQ, VL_SEQ]
    numbered_list = number_ab_seq(seqs, "imgt", assign_germline=True)
    assert isinstance(numbered_list, list)
    assert len(numbered_list) == 2
    assert all(isinstance(n, NumberedAntibodyWithGermline) for n in numbered_list)
    assert numbered_list[0].fv_seq == vh_germline_imgt.fv_seq
    assert numbered_list[1].fv_seq == vl_germline_imgt.fv_seq

    numbered_no_germline = number_ab_seq(seqs, "martin", assign_germline=False)
    assert isinstance(numbered_no_germline, list)
    assert len(numbered_no_germline) == 2
    assert all(isinstance(n, NumberedAntibody) for n in numbered_no_germline)
    assert numbered_no_germline[0].fv_seq == vh_germline_imgt.fv_seq
    assert numbered_no_germline[1].fv_seq == vl_germline_imgt.fv_seq

    # Catch type errors
    with pytest.raises(TypeError):
        number_ab_seq(1, "imgt", assign_germline=True)  # type: ignore
    with pytest.raises(TypeError):
        number_ab_seq([VH_SEQ, 2], "imgt")  # type: ignore


def test_numbered_antibody_regions(vh_martin, vh_germline_imgt):
    """Test region assignment of the NumberedAntibody class."""
    assert vh_martin.fv_seq in vh_martin.seq
    assert vh_martin.cdr1_seq == "GYTFTNY"
    assert vh_martin.cdr2_seq == "NPSNGG"
    assert vh_martin.cdr3_seq == "RDYRFDMGFDY"
    assert vh_martin.fr1_seq == "QVQLVQSGVEVKKPGASVKVSCKAS"
    assert vh_martin.fr4_seq == "WGQGTTVTVSS"

    assert vh_germline_imgt.fv_seq in vh_germline_imgt.seq
    assert vh_germline_imgt.cdr1_seq == "GYTFTNYY"
    assert vh_germline_imgt.cdr2_seq == "INPSNGGT"
    assert vh_germline_imgt.cdr3_seq == "ARRDYRFDMGFDY"
    assert vh_germline_imgt.fr1_seq == "QVQLVQSGVEVKKPGASVKVSCKAS"
    assert vh_germline_imgt.fr4_seq == "WGQGTTVTVSS"


def test_numbered_antibody_with_germline_properties(vl_germline_imgt):
    """Test properties of the NumberedAntibodyWithGermline class."""
    assert vl_germline_imgt.closest_germline.v_gene is not None
    assert vl_germline_imgt.closest_germline.j_gene is not None
    aln = vl_germline_imgt.aligned_germline
    assert isinstance(aln, AntibodyAlignment)
    assert aln.df.get_column("seq_id").to_list() == ["Germline", "Query"]


def test_format_methods(vh_martin):
    """Test the format() and __repr__ methods."""
    # Test default format (same as repr)
    numbered = number_ab_seq(f"HHHHH{VH_SEQ}GGGSGGGS", "martin", assign_germline=True)
    repr_str = repr(numbered)
    assert "Closest germline" in repr_str
    assert "V gene" in repr_str
    assert "\033[1;4;91m" in repr_str  # Check for highlighting
    assert "\033[2;9;30m" in repr_str

    # Test format without germline
    format_no_germline = numbered.format(show_germline=False)
    assert "Closest germline" not in format_no_germline
    assert "\033[1;4;91m" in format_no_germline
    assert "\033[2;9;30m" in format_no_germline

    # Test format without CDR highlighting
    format_no_highlight = numbered.format(show_germline=False, highlight_cdr=False)
    assert "\033[1;4;91m" not in format_no_highlight
    assert "\033[2;9;30m" in format_no_highlight

    # Test hiding non-Fv regions
    format_hide_non_fv = numbered.format(show_germline=False, include_non_fv=False)
    assert "HHHHH" not in format_hide_non_fv
    assert "\033[1;4;91m" in format_hide_non_fv
    assert "\033[2;9;30m" not in format_hide_non_fv
    assert format_hide_non_fv == repr(vh_martin)

    # Test germline format
    germline_str = repr(numbered.closest_germline)
    assert "IMGT-numbered seq" in germline_str
    assert len(germline_str.split("\n")) == 4


def test_align_ab_seqs(vh_martin):
    """Test alignment of multiple antibody sequences."""
    # Create a variant with a deletion to test alignment
    vh_variant_seq = VH_SEQ[:30] + VH_SEQ[31:]
    vh_variant_numbered = number_ab_seq(vh_variant_seq, "martin", assign_germline=True)

    seqs_to_align = [vh_martin, vh_variant_numbered]
    seq_ids = ["Query", "Variant"]
    alignment = align_ab_seqs(seqs=seqs_to_align, seq_ids=seq_ids)
    assert alignment.df.height == 2
    assert alignment.df.get_column("seq_id").to_list() == seq_ids
    assert alignment["31"] == {"Query": vh_martin["31"], "Variant": "-"}

    # Test getter methods
    with pytest.raises(
        TypeError, match=r"Index must be a numbered position string, not <class 'int'>"
    ):
        _ = alignment[31]  # type: ignore
    with pytest.raises(KeyError, match=r"Numbered position 999 not found"):
        _ = alignment["999"]

    # Test rearranging alignment strings
    aln_fmt = alignment.format(highlight_cdr=False)
    assert aln_fmt.strip().startswith("Query: ")
    aln_refmt = alignment.format(highlight_cdr=False, ref_seq_id="Variant")
    assert aln_refmt.strip().startswith("Variant: ")
    with pytest.raises(
        KeyError, match=r"Reference sequence ID XX not found in alignment"
    ):
        alignment.format(highlight_cdr=False, ref_seq_id="XX")

    # Test alignment without seq_ids
    alignment_wo_ids = align_ab_seqs(seqs=seqs_to_align)
    assert alignment_wo_ids.df.get_column("seq_id").to_list() == ["0", "1"]

    # Test alignment with wrong seq_ids
    with pytest.raises(ValueError, match=r"Sequence IDs must be unique"):
        align_ab_seqs(seqs=seqs_to_align, seq_ids=["Query", "Query"])
    with pytest.raises(
        ValueError,
        match=r"Number of sequences \(2\) does not match number of IDs \(3\)",
    ):
        align_ab_seqs(seqs=seqs_to_align, seq_ids=["Query", "Variant", "Extra"])


def test_align_ab_seqs_empty_list():
    """Test that align_ab_seqs raises an error for an empty list."""
    with pytest.raises(ValueError, match="No sequences provided for alignment."):
        align_ab_seqs([])


def test_build_alignment_indicator():
    """Test the _build_alignment_indicator helper function."""
    assert _build_alignment_indicator("AEC", "AEC") == "|||"
    assert _build_alignment_indicator("AEC", "ADC") == "|+|"
    assert _build_alignment_indicator("ARC", "ADC") == "|.|"
    assert _build_alignment_indicator("A-C", "ADC") == "| |"
    assert _build_alignment_indicator("KR", "EQ") == "++"  # Test similar pairs
    assert _build_alignment_indicator("K-R", "E-Q") == "+ +"
    assert _build_alignment_indicator("KR", "DW") == ".."  # Test dissimilar pairs
    with pytest.raises(ValueError):
        _build_alignment_indicator("AEC", "AB")


def test_number_ab_seq_partial():
    """Test partial input for the number_ab_seq function."""
    with pytest.raises(
        ValueError,
        match=r"Percent identity \(\d+\.\d+%\) is low for assigned chain type H",
    ):
        number_ab_seq(VH_SEQ[:30], "imgt", assign_germline=True)


def test_impute_missing_res_with_germline():
    """Test that missing residues are imputed with germline sequences."""
    numbered = number_ab_seq(VH_SEQ[1:], "imgt", assign_germline=True)
    assert numbered.imputed_seq == VH_SEQ


def test_number_scfv_seq(vh_martin, vl_martin):
    """Test numbering of scFv sequences."""
    from antid.numbering import NumberedScFv, number_scfv_seq

    g4s_4 = "GGGGS" * 4
    vh_gs_vl = VH_SEQ + g4s_4 + VL_SEQ

    scfv = number_scfv_seq(vh_gs_vl, "martin", assign_germline=False)
    assert isinstance(scfv, NumberedScFv)
    assert scfv.vh.fv_seq == vh_martin.fv_seq
    assert scfv.vl.fv_seq == vl_martin.fv_seq
    assert scfv.vl.fv_range[0] == len(VH_SEQ) + 20 + 1

    # Test with list input
    vl_gs_vh = VL_SEQ + g4s_4 + VH_SEQ
    numbered_list = number_scfv_seq(
        [vh_gs_vl, vl_gs_vh], "martin", assign_germline=False
    )
    assert isinstance(numbered_list, list)
    assert len(numbered_list) == 2
    assert all(isinstance(n, NumberedScFv) for n in numbered_list)

    for scfv in numbered_list:
        assert scfv.vh.fv_seq == vh_martin.fv_seq
        assert scfv.vl.fv_seq == vl_martin.fv_seq


def test_patch_antpack_identical():
    """Test that the patched functions work the same as the original."""
    from antpack import SingleChainAnnotator as original_sc
    from antpack import VJGeneTool as original_vj

    from antid.utils.patch_antpack import SingleChainAnnotator, VJGeneTool

    sc_instance = SingleChainAnnotator(scheme="imgt")
    vj_instance = VJGeneTool(scheme="imgt")

    original_sc_instance = original_sc(scheme="imgt")
    original_vj_instance = original_vj(scheme="imgt")

    # Test that the patched instance behaves like the original
    alignment = sc_instance.analyze_seq(VH_SEQ)
    original_alignment = original_sc_instance.analyze_seq(VH_SEQ)
    assert alignment == original_alignment

    assert vj_instance.retrieve_db_dates() == original_vj_instance.retrieve_db_dates()
    assert vj_instance.assign_vj_genes(
        alignment, VH_SEQ, "unknown", "evalue"
    ) == original_vj_instance.assign_vj_genes(alignment, VH_SEQ, "unknown", "evalue")
