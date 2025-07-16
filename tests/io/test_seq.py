"""Test for I/O functions related to sequence handling."""

from pathlib import Path

import pytest

from antid.io.seq import fasta2seq, struct2seq


@pytest.fixture
def data_dir():
    """Fixture to provide the path to the test data directory."""
    return Path(__file__).resolve().parent.parent / "data"


# ruff: noqa: S101
def test_struct2seq(data_dir):
    """Test collecting protein sequences from various coordinate files."""
    seqs = struct2seq(data_dir / "5b8c.pdb1.gz")
    assert len(seqs) == 3

    assert (
        seqs["A"]
        == "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"
    )
    assert (
        seqs["B"]
        == "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    )
    assert (
        seqs["C"]
        == "SWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDSRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTE"
    )

    mmcif_seqs = struct2seq(data_dir / "5b8c-assembly1.cif.gz")
    assert mmcif_seqs == seqs


def test_struct2seq_with_custom_mappings(data_dir):
    """Test struct2seq with custom residue mappings."""
    seqs = struct2seq(data_dir / "5b8c.pdb1.gz", HOH="o")
    assert len(seqs) == 3

    assert (
        seqs["A"]
        == "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"
        + "o" * 29
    )
    assert (
        seqs["B"]
        == "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
        + "o" * 37
    )
    assert (
        seqs["C"]
        == "SWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDSRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTE"
        + "o" * 24
    )


def test_fasta2seq(data_dir):
    """Test converting a FASTA file to a sequence dictionary."""
    fasta_path = data_dir / "rcsb_pdb_5B8C.fasta"
    seqs = fasta2seq(fasta_path)
    assert seqs == {
        "5B8C_2|Chains B, E, H, K|Pembrolizumab heavy chain variable region (PemVH)|Homo sapiens (9606)": "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
        "5B8C_3|Chains C, F, I, L|Programmed cell death protein 1|Homo sapiens (9606)": "GSWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDSRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPTSENLYFQ",
        "5B8C_1|Chains A, D, G, J|Pembrolizumab light chain variable region (PemVL)|Homo sapiens (9606)": "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIKTSENLYFQ",
    }
