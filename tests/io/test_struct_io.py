"""Test cases for antid.io.struct utilities using pytest fixtures and PDB/mmCIF files."""

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import requests

from antid.io.struct import RCSBDownloader, df2pdb, gemmi_convert, struct2df


@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Fixture to provide the path to the test data directory."""
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def pdb_gz_path(data_dir: Path) -> Path:
    """Fixture to provide gzipped PDB file path."""
    return data_dir / "5b8c.pdb1.gz"


@pytest.fixture(scope="module")
def cif_gz_path(data_dir: Path) -> Path:
    """Fixture to provide gzipped mmCIF file path."""
    return data_dir / "5b8c-assembly1.cif.gz"


@pytest.fixture(scope="module")
def fasta_path(data_dir: Path) -> Path:
    """Fixture to provide the FASTA file path."""
    return data_dir / "1MFV.fasta"


# ruff: noqa: S101
@pytest.mark.slow
def test_download_pdb_from_rcsb(tmp_path: Path, pdb_gz_path: Path):
    """Test downloading PDB file from RCSB."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    pdb_path = downloader.fetch_pdb("5b8c", fallback_to_cif=False)
    assert pdb_path.exists()
    assert pdb_path.name == "5B8C.pdb.gz"

    # Check if the file is equivalent to the fixture
    with open(pdb_gz_path, "rb") as f1, open(pdb_path, "rb") as f2:
        assert f1.read() == f2.read()

    # Redownloading should be quick and return the same file directly
    pdb_path2 = downloader.fetch_pdb("5B8C", fallback_to_cif=False)
    assert pdb_path2 == pdb_path


@pytest.mark.slow
def test_download_cif_from_rcsb(tmp_path: Path, cif_gz_path: Path):
    """Test downloading PDB file from RCSB."""
    downloader = RCSBDownloader(out_dir=tmp_path)

    # Similar tests for mmCIF
    cif_path = downloader.fetch_mmcif("5b8c")
    assert cif_path.exists()
    assert cif_path.name == "5B8C-assembly1.cif.gz"
    with open(cif_gz_path, "rb") as f1, open(cif_path, "rb") as f2:
        assert f1.read() == f2.read()

    cif_path2 = downloader.fetch_mmcif("5B8C")
    assert cif_path2 == cif_path


@pytest.mark.slow
def test_download_nonexistent_pdb(tmp_path: Path):
    """Test downloading a nonexistent PDB ID."""
    downloader = RCSBDownloader(out_dir=tmp_path, timeout=2)
    with pytest.raises(requests.HTTPError):
        downloader.fetch_pdb("XXXX", fallback_to_cif=False)

    # Falling back to mmCIF would not fix this
    with pytest.raises(requests.HTTPError):
        downloader.fetch_pdb("XXXX", fallback_to_cif=True)


@pytest.mark.slow
def test_download_pdb_fallback_to_cif(tmp_path: Path):
    """Test downloading a PDB ID that does not have a PDB file but has an mmCIF file."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    # For new structures with no PDB files available, falling back would work
    cif_path = downloader.fetch_pdb("9kas", fallback_to_cif=True)
    assert cif_path.exists()
    assert cif_path.name == "9KAS-assembly1.cif.gz"


@pytest.mark.slow
def test_download_fasta(tmp_path: Path, fasta_path: Path):
    """Test downloading a FASTA file."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    fasta_path = downloader.fetch_fasta("1mkv")
    assert fasta_path.exists()
    assert fasta_path.name == "1MKV.fasta"

    with open(fasta_path, "rb") as f1, open(fasta_path, "rb") as f2:
        assert f1.read() == f2.read()

    # Redownloading should return the same file directly
    fasta_path2 = downloader.fetch_fasta("1MKV")
    assert fasta_path2 == fasta_path


def test_fetch_all_fasta_success(tmp_path: Path):
    """Test successful download of the FASTA file."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    mock_response = MagicMock()
    mock_response.status_code = 200
    fasta_content = b">P1;5B8C\nQVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS\n"
    gzipped_content = gzip.compress(fasta_content)
    mock_response.headers = {"Content-Length": str(len(gzipped_content))}
    mock_response.iter_content.return_value = [gzipped_content]
    mock_response.content = gzipped_content

    with patch("requests.Session.get", return_value=mock_response) as mock_get:
        out_path = downloader.fetch_all_fasta()
        mock_get.assert_called_once_with(
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
            stream=True,
        )
        assert out_path == tmp_path / "pdb_seqres.txt.gz"
        assert out_path.exists()
        with gzip.open(out_path, "rb") as f:
            assert f.read() == fasta_content


def test_fetch_all_fasta_already_exists(tmp_path: Path):
    """Test when the FASTA file already exists."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    out_path = tmp_path / "pdb_seqres.txt.gz"
    out_path.touch()

    with patch("requests.Session.get") as mock_get:
        result_path = downloader.fetch_all_fasta()
        mock_get.assert_not_called()
        assert result_path == out_path


def test_fetch_all_fasta_download_error(tmp_path: Path):
    """Test for a runtime error if downloaded file size does not match."""
    downloader = RCSBDownloader(out_dir=tmp_path)
    mock_response = MagicMock()
    mock_response.status_code = 200
    fasta_content = b">P1;5B8C\nSEQUENCE"
    gzipped_content = gzip.compress(fasta_content)
    mock_response.headers = {
        "Content-Length": str(len(gzipped_content) + 10)
    }  # Mismatch
    mock_response.iter_content.return_value = [gzipped_content]

    with patch("requests.Session.get", return_value=mock_response):
        with pytest.raises(
            RuntimeError, match="Downloaded file size does not match expected size."
        ):
            downloader.fetch_all_fasta()


def _test_pembro_shapes(df: pl.DataFrame):
    """Helper function to check shapes of DataFrame."""
    assert len(df.columns) == 15
    assert df.height == 2764
    assert df.n_unique("atomi") == 2764
    assert set(df.get_column("chain").unique().to_list()) == {"A", "B", "C"}
    num_residues = df.filter(pl.col("resn") != pl.lit("HOH")).n_unique(
        ("chain", "resi", "insertion")
    )
    assert num_residues == 347


def test_struct2df_pdb(pdb_gz_path: Path):
    """Test struct2df on gzipped PDB file."""
    df = struct2df(pdb_gz_path)
    _test_pembro_shapes(df)


def test_struct2df_cif(cif_gz_path: Path):
    """Test struct2df on gzipped mmCIF file."""
    df = struct2df(cif_gz_path)
    _test_pembro_shapes(df)


def test_df2pdb(pdb_gz_path: Path, tmp_path):
    """Test roundtrip: struct2df -> df2pdb."""
    df = struct2df(pdb_gz_path)
    out_pdb = tmp_path / "out.pdb"
    df2pdb(df, out_pdb)
    assert out_pdb.exists()

    df_reread = struct2df(out_pdb)
    _test_pembro_shapes(df_reread)


def test_gemmi_convert(pdb_gz_path: Path, tmp_path):
    """Test gemmi_convert for PDB <-> mmCIF conversion."""
    out_cif = tmp_path / "converted.cif"
    result = gemmi_convert(pdb_gz_path, out_cif)
    assert result.exists()
    assert result.suffix == ".cif"
    df = struct2df(out_cif)
    _test_pembro_shapes(df)

    out_pdb = tmp_path / "converted.pdb"
    result2 = gemmi_convert(out_cif, out_pdb)
    assert result2.exists()
    assert result2.suffix == ".pdb"
    df = struct2df(out_pdb)
    _test_pembro_shapes(df)

    # Running the conversion again should skip if the output exists
    result3 = gemmi_convert(out_cif, out_pdb)
    assert result3 == result2
