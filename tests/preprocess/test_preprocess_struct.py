"""Test functions for preprocessing structure files."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from antid.io import fasta2seq, struct2df
from antid.preprocess.struct import align_ref_seq_to_struct, standardize_struct_file


@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Fixture to provide the path to the test data directory."""
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def pembro_df(data_dir):
    """PDB DataFrame of 5B8C for testing."""
    return struct2df(data_dir / "5b8c.pdb1.gz").filter(pl.col("resn") != pl.lit("HOH"))


@pytest.fixture
def pembro_seqs(data_dir):
    """Reference sequence for 5B8C."""
    seqs = fasta2seq(data_dir / "rcsb_pdb_5B8C.fasta")
    # A: VL; B: VH; C: antigen
    return {k.split("|")[1].split(" ")[1][0]: v for k, v in seqs.items()}


# ruff: noqa: S101
def test_align_ref_seq_to_struct_identical(pembro_df, pembro_seqs):
    """Test alignment when PDB sequence is identical to reference."""
    result_df = align_ref_seq_to_struct(
        ref_seq=pembro_seqs["B"],
        pdb_seqs_df=pembro_df,
        pdb_chain="B",
        resn_3to1=True,
    )
    assert result_df.height == len(pembro_seqs["B"])
    assert result_df.get_column("ref_idx").max() == len(pembro_seqs["B"])
    aligned_pdb_seq = "".join(result_df.get_column("resn").to_list())
    assert aligned_pdb_seq == pembro_seqs["B"]


def test_align_ref_seq_to_struct_partial(pembro_df, pembro_seqs):
    """Test alignment when PDB sequence is a partial match to reference."""
    result_df = align_ref_seq_to_struct(
        ref_seq=pembro_seqs["A"],
        pdb_seqs_df=pembro_df,
        pdb_chain="A",
        resn_3to1=True,
    )
    assert result_df.height == len(pembro_seqs["A"])
    assert result_df.get_column("ref_idx").max() == len(pembro_seqs["A"])

    # Missing residues in the PDB sequence should be represented as '-'
    missing_res_df = result_df.filter(pl.col("resi").is_null())
    assert missing_res_df.height == 8
    assert result_df.filter(pl.col("resn") == pl.lit("-")).height == 8
    assert (
        "".join(missing_res_df.get_column("ref_resn").to_list())
        == pembro_seqs["A"][-8:]
    )


def test_align_ref_seq_to_struct_wrong_ref(pembro_df, pembro_seqs):
    """Test alignment when PDB sequence is a partial match to reference."""
    vh_seq = pembro_seqs["A"]
    vh_seq_del = vh_seq[:10] + vh_seq[11:]

    with pytest.raises(
        ValueError,
        match=f"Reference sequence should not have gaps:\nRef: {vh_seq[:10]}-",
    ):
        align_ref_seq_to_struct(
            ref_seq=vh_seq_del,
            pdb_seqs_df=pembro_df,
            pdb_chain="A",
            resn_3to1=True,
        )


def test_standardize_struct_file_renum_pdb(tmp_path, data_dir):
    """Test renumbering of PDB files."""
    # 5b8c is serial in the two antibody chains, but in the antigen chain
    # there is a offset of 30.
    resi_map = standardize_struct_file(
        data_dir / "5b8c.pdb1.gz", tmp_path / "5b8c_standardized.pdb"
    )
    old_pdb_df = struct2df(data_dir / "5b8c.pdb1.gz")
    new_pdb_df = struct2df(tmp_path / "5b8c_standardized.pdb")

    assert old_pdb_df.height == new_pdb_df.height
    assert (
        new_pdb_df.unique(
            ("chain", "resi", "insertion"), maintain_order=True, keep="first"
        )
        .with_columns(pl.col("chain").cum_count().over("chain").alias("dummy_idx"))
        .filter(pl.col("dummy_idx") != pl.col("resi"))
        .height
        == 0
    )
    diff_resi = (
        old_pdb_df.filter(pl.col("resn") != pl.lit("HOH"))
        .select("model", "chain", "resi", "insertion")
        .select("model", pl.exclude("model").name.prefix("old_"))
        .join(
            resi_map.filter(
                pl.concat_str("old_resi", "old_insertion")
                != pl.concat_str("new_resi", "new_insertion")
            ),
            on=("model", "old_chain", "old_resi", "old_insertion"),
        )
    )
    assert (
        diff_resi.filter(pl.col("new_resi") != pl.col("old_resi") - pl.lit(30)).height
        == 0
    )

    # Check that the renumbering is correct (atomi could be different)
    old2new = (
        old_pdb_df.with_columns(
            pl.col(c).alias(f"old_{c}") for c in ("chain", "resi", "insertion")
        )
        .join(
            resi_map,
            on=("model", "old_chain", "old_resi", "old_insertion"),
            maintain_order="left",
        )
        .select(pl.col(f"new_{c}").alias(c) for c in ("chain", "resi", "insertion"))
        .with_columns(pl.col("resi").cast(pl.Int64))
        .unique(maintain_order=True, keep="first")
    )
    assert_frame_equal(
        old2new,
        new_pdb_df.select("chain", "resi", "insertion").unique(
            maintain_order=True, keep="first"
        ),
    )


def test_standardize_struct_file_other_formats(tmp_path, data_dir):
    """Test outputting the standardized structure file in CIF format."""
    resi_map = standardize_struct_file(
        data_dir / "5b8c.pdb1.gz", tmp_path / "5b8c_standardized.cif"
    )
    old_pdb_df = struct2df(data_dir / "5b8c.pdb1.gz")
    new_pdb_df = struct2df(tmp_path / "5b8c_standardized.cif")
    assert old_pdb_df.height == new_pdb_df.height
    diff_resi = (
        old_pdb_df.filter(pl.col("resn") != pl.lit("HOH"))
        .select("model", "chain", "resi", "insertion")
        .select("model", pl.exclude("model").name.prefix("old_"))
        .join(
            resi_map.filter(
                pl.concat_str("old_resi", "old_insertion")
                != pl.concat_str("new_resi", "new_insertion")
            ),
            on=("model", "old_chain", "old_resi", "old_insertion"),
        )
    )
    assert (
        diff_resi.filter(pl.col("new_resi") != pl.col("old_resi") - pl.lit(30)).height
        == 0
    )

    with pytest.raises(ValueError, match="Unsupported output file format: .coord."):
        standardize_struct_file(
            data_dir / "5b8c.pdb1.gz", tmp_path / "5b8c_standardized.coord"
        )


def test_standardize_struct_file_subset_chain(tmp_path, data_dir):
    """Test renumbering of PDB files with a subset of chains."""
    resi_map = standardize_struct_file(
        data_dir / "5b8c.pdb1.gz",
        tmp_path / "5b8c_standardized_subset.pdb",
        chain_mapping={"A": "L", "B": "H"},
    )
    old_pdb_df = struct2df(data_dir / "5b8c.pdb1.gz")
    new_pdb_df = struct2df(tmp_path / "5b8c_standardized_subset.pdb")

    assert new_pdb_df.n_unique("chain") == 2
    assert (
        old_pdb_df.filter(pl.col("chain").is_in({"A", "B"})).height == new_pdb_df.height
    )
    diff_resi = (
        old_pdb_df.filter(pl.col("resn") != pl.lit("HOH"))
        .select("model", "chain", "resi", "insertion")
        .select("model", pl.exclude("model").name.prefix("old_"))
        .join(
            resi_map.filter(
                pl.concat_str("old_resi", "old_insertion")
                != pl.concat_str("new_resi", "new_insertion")
            ),
            on=("model", "old_chain", "old_resi", "old_insertion"),
        )
    )
    assert diff_resi.height == 0

    # Test providing chains that are not in the PDB file
    with pytest.raises(ValueError, match="Chain X not found"):
        standardize_struct_file(
            data_dir / "5b8c.pdb1.gz",
            tmp_path / "5b8c_standardized_subset.pdb",
            chain_mapping={"X": "A"},
        )
