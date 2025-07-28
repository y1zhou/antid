"""Test cases for the interaction scanner and Arpeggia utilities in the antid package.

NOTE: Arpeggia is a third-party tool for protein interaction analysis. If not in $PATH,
ensure ARPEGGIA_PATH is set to the correct binary location.
"""

import os
from pathlib import Path

import polars as pl
import pytest

from antid.ppi.scan import (
    collect_ab_ag_contacts,
    collect_within_ab_contacts,
    get_atomic_sasa,
    get_contacts,
)
from antid.utils import find_binary
from antid.utils.constant import ARPEGGIA_IDX_COLS


@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Fixture to provide the path to the test data directory."""
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def arpeggia_bin() -> str:
    """Fixture to provide the path to the Arpeggia binary."""
    # Set $ARPEGGIA_PATH to your binary
    arpeggia_path = os.environ.get("ARPEGGIA_PATH", "arpeggia")
    return find_binary(arpeggia_path)


@pytest.fixture(scope="module")
def arpeggia_contact_cols():
    """Fixture to provide the column names for Arpeggia contacts."""
    return (
        "interaction",
        "distance",
        "sc_centroid_dist",
        "sc_centroid_angle",
        "sc_dihedral",
    )


# ruff: noqa: S101
def test_run_arpeggia_contacts_and_sasa(
    tmp_path: Path,
    data_dir: Path,
    arpeggia_bin: str,
    arpeggia_contact_cols: tuple[str, ...],
):
    """Test running Arpeggia for contacts and SASA calculations."""
    # Test both PDB and mmCIF input
    struct_files: dict[str, Path] = {
        "pdb": data_dir / "5b8c.pdb1.gz",
        "mmcif": data_dir / "5b8c-assembly1.cif.gz",
    }

    contact_dfs: dict[str, pl.DataFrame] = {}
    sasa_dfs: dict[str, pl.DataFrame] = {}

    for file_type, struct_file in struct_files.items():
        contacts = get_contacts(
            struct_file, tmp_path, arpeggia_bin, out_file_name=f"{file_type}-contacts"
        )
        contact_dfs[file_type] = contacts
        assert isinstance(contacts, pl.DataFrame)
        assert contacts.height > 0
        contact_df_cols = contacts.columns
        for col in arpeggia_contact_cols:
            assert col in contact_df_cols
        for col in ARPEGGIA_IDX_COLS:
            assert f"from_{col}" in contact_df_cols
            assert f"to_{col}" in contact_df_cols

        sasa = get_atomic_sasa(
            struct_file, tmp_path, arpeggia_bin, out_file_name="sasa"
        )
        sasa_dfs[file_type] = sasa
        assert isinstance(sasa, pl.DataFrame)
        assert sasa.height > 0
        assert "sasa" in sasa.columns
        for col in ARPEGGIA_IDX_COLS:
            assert col in sasa.columns

        # Test cache: should read from file, not rerun
        contacts = get_contacts(
            struct_file, tmp_path, arpeggia_bin, out_file_name=f"{file_type}-contacts"
        )
        assert contacts.shape == contact_dfs[file_type].shape
        assert (tmp_path / f"{file_type}-contacts.parquet").exists()

        sasa = get_atomic_sasa(
            struct_file, tmp_path, arpeggia_bin, out_file_name=f"{file_type}-sasa"
        )
        assert sasa.shape == sasa_dfs[file_type].shape
        assert (tmp_path / f"{file_type}-sasa.parquet").exists()

    # Running the same command on the two file types should yield the same results
    assert contact_dfs["pdb"].shape == contact_dfs["mmcif"].shape
    assert sasa_dfs["pdb"].shape == sasa_dfs["mmcif"].shape


@pytest.fixture(scope="module")
def pembro_contacts(tmp_path_factory, data_dir: Path, arpeggia_bin: str):
    """Fixture to run Arpeggia contacts for the Pembrolizumab structure."""
    pdb_file = data_dir / "5b8c.pdb1.gz"
    out_path = tmp_path_factory.mktemp("data")

    contacts = get_contacts(pdb_file, out_path, arpeggia_bin)
    assert contacts.height > 0
    return contacts


@pytest.fixture(scope="module")
def pembro_sasa(tmp_path_factory, data_dir: Path, arpeggia_bin: str):
    """Fixture to run Arpeggia SASA for the Pembrolizumab structure."""
    pdb_file = data_dir / "5b8c.pdb1.gz"
    out_path = tmp_path_factory.mktemp("data")
    sasa = get_atomic_sasa(pdb_file, out_path, arpeggia_bin)
    assert sasa.height > 0
    return sasa


def test_collect_ab_ag_contacts_and_within_ab(
    pembro_contacts: pl.DataFrame,
    pembro_sasa: pl.DataFrame,
    arpeggia_contact_cols: tuple[str, ...],
):
    """Test collecting antibody-antigen and within-antibody contacts."""
    ab_chains = {"A", "B"}  # VL and VH chains for 5b8c
    ab_ag_df = collect_ab_ag_contacts(ab_chains, pembro_contacts, pembro_sasa)
    assert isinstance(ab_ag_df, pl.DataFrame)
    assert ab_ag_df.height > 0
    for col in arpeggia_contact_cols:
        assert col in ab_ag_df.columns
    for col in ARPEGGIA_IDX_COLS:
        assert f"ab_{col}" in ab_ag_df.columns
        assert f"ag_{col}" in ab_ag_df.columns

    assert "ab_sasa" in ab_ag_df.columns
    assert "ag_sasa" in ab_ag_df.columns

    within_ab_df = collect_within_ab_contacts(ab_chains, pembro_contacts, pembro_sasa)
    assert isinstance(within_ab_df, pl.DataFrame)
    assert within_ab_df.height > 0
    for col in arpeggia_contact_cols:
        assert col in within_ab_df.columns
    for col in ARPEGGIA_IDX_COLS:
        assert f"from_{col}" in within_ab_df.columns
        assert f"to_{col}" in within_ab_df.columns

    assert "from_sasa" in within_ab_df.columns
    assert "to_sasa" in within_ab_df.columns
