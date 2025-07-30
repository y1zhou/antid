"""Utility functions for dealing with structure files."""

from collections.abc import Iterable
from pathlib import Path

import gemmi
import polars as pl
import requests
from loguru import logger
from tqdm import tqdm

from antid.utils import check_path
from antid.utils.constant import ARPEGGIA_IDX_COLS


class RCSBDownloader:
    """Download structure files from RCSB."""

    def __init__(
        self,
        out_dir: str | Path,
        req_session: requests.Session | None = None,
        timeout: int = 10,
    ):
        """Initialize the downloader.

        Args:
            out_dir: Directory to save downloaded files.
            req_session: Optional requests.Session object.
            timeout: Timeout for requests in seconds.
        """
        self.out_dir = check_path(out_dir, mkdir=True, is_dir=True)
        self.session = requests.Session() if req_session is None else req_session
        self.timeout = timeout

    def fetch_pdb(self, pdb_id: str, fallback_to_cif: bool = True) -> Path:
        """Download the first biological assembly PDB file from RCSB.

        Args:
            pdb_id: The PDB ID.
            fallback_to_cif: If True, fall back to downloading the mmCIF file if the PDB
            file is not found. For details, see https://www.rcsb.org/docs/general-help/structures-without-legacy-pdb-format-files

        Returns:
            Path to the downloaded file.
        """
        pdb_id = pdb_id.upper()
        gz_pdb_file = self.out_dir / f"{pdb_id}.pdb.gz"
        if gz_pdb_file.exists():
            return gz_pdb_file

        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb1.gz"
        r = self.session.get(pdb_url, timeout=self.timeout)
        if r.status_code == 404 and fallback_to_cif:
            # Try mmCIF file if the PDB is nonexistent
            logger.warning(f"PDB for {pdb_id} not found, falling back to mmCIF.")
            return self.fetch_mmcif(pdb_id)

        r.raise_for_status()
        with open(gz_pdb_file, "wb") as f:
            f.write(r.content)
        return gz_pdb_file

    def fetch_mmcif(self, pdb_id: str) -> Path:
        """Download the first biological assembly mmCIF file from RCSB.

        Args:
            pdb_id: The PDB ID.

        Returns:
            Path to the downloaded file.
        """
        pdb_id = pdb_id.upper()
        gz_cif_file = self.out_dir / f"{pdb_id}-assembly1.cif.gz"
        if gz_cif_file.exists():
            return gz_cif_file

        cif_url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
        r = self.session.get(cif_url, timeout=self.timeout)
        r.raise_for_status()
        with open(gz_cif_file, "wb") as f:
            f.write(r.content)
        return gz_cif_file

    def fetch_fasta(self, pdb_id: str) -> Path:
        """Download the FASTA file for a given PDB ID.

        Args:
            pdb_id: The PDB ID.

        Returns:
            Path to the downloaded FASTA file.
        """
        pdb_id = pdb_id.upper()
        fasta_file = self.out_dir / f"{pdb_id}.fasta"
        if fasta_file.exists():
            return fasta_file

        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
        r = self.session.get(fasta_url, timeout=self.timeout)
        r.raise_for_status()
        with open(fasta_file, "wb") as f:
            f.write(r.content)
        return fasta_file

    def fetch_all_fasta(self) -> Path:
        """Download FASTA file containing sequences for all PDB entries."""
        out_path = self.out_dir / "pdb_seqres.txt.gz"
        if out_path.exists():
            return out_path

        fasta_url = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
        r = self.session.get(fasta_url, stream=True)
        total_size = int(r.headers.get("Content-Length", 0))
        block_size = 1024  # 1 KB
        with (
            open(out_path, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading FASTA"
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))

        if total_size != 0 and bar.n != total_size:
            raise RuntimeError("Downloaded file size does not match expected size.")

        return out_path


def struct2df(path: str | Path) -> pl.DataFrame:
    """Convert coordinates in a structure file to a Polars DataFrame."""
    st: gemmi.Structure = gemmi.read_structure(
        str(path), format=gemmi.CoorFormat.Detect
    )
    st.setup_entities()
    # st.assign_label_seq_id()

    # Get all residues
    all_atoms = [
        (
            model.num,
            cra.chain.name,
            cra.residue.name,
            cra.residue.seqid.num,
            cra.residue.seqid.icode,
            cra.atom.altloc,
            cra.atom.name,
            cra.atom.serial,
            cra.atom.element.name,
            cra.atom.occ,
            cra.atom.b_iso,
            cra.atom.charge,
            *(cra.atom.pos),
        )
        for model in st
        for cra in model.all()
    ]

    return pl.from_records(
        all_atoms,
        orient="row",
        schema=(
            "model",
            *ARPEGGIA_IDX_COLS,
            "element",
            "occupancy",
            "bfactor",
            "charge",
            "x",
            "y",
            "z",
        ),
    )


def df2pdb(df: pl.DataFrame, out_path: str | Path) -> None:
    """Convert a DataFrame with coordinates to a PDB file.

    NOTE: Column names in ``df`` should match the ones in ``struct2df``.
        We also assume there to be only one model in the DataFrame.

    Args:
        df: Polars DataFrame containing coordinates data.
        out_path: Output path for the structure file.
    """
    pdb_atom_rows: list[str] = []
    prev_chain: str = df.item(0, "chain")
    for r in df.iter_rows(named=True):
        if r["chain"] != prev_chain:
            pdb_atom_rows.append("TER\n")
            prev_chain = r["chain"]

        pdb_atom_rows.append(
            f"ATOM  {r['atomi']:>5} {r['atomn']:<4} {r['resn']:>3} "
            f"{r['chain']:<1}{r['resi']:>4}{r['insertion']:<1}   "
            f"{r['x']:>8.3f}{r['y']:>8.3f}{r['z']:>8.3f} "
            f"{r['occupancy']:>6.2f}{r['bfactor']:>6.2f}          "
            f"{r['element']:<2}\n"
        )
    pdb_atom_rows.append("END\n")
    with open(out_path, "w") as f:
        f.writelines(pdb_atom_rows)


def gemmi_convert(
    struct_file1: str | Path,
    struct_file2: str | Path,
    additional_gemmi_args: Iterable[str] = (),
):
    """Convert PDB file to mmCIF format, or vice versa.

    Some common additional args:
        - ``--segment-as-chain``: assign segment ID to label_asym_id
        - ``-s <fasta_file>``: assign best matching sequences to chains
        - ``--rename-chain=OLD:NEW``: rename chain IDs
        - ``--remove-h``: remove all hydrogens
        - ``--remove-waters``: remove all waters
        - ``--remove-lig-wat``: remove ligands and waters

    Args:
        struct_file1: Path to the input structure file.
        struct_file2: Path to the output structure file.
        additional_gemmi_args: Additional arguments to pass to gemmi convert.

    """
    from antid.utils import check_path, command_runner

    struct_path2 = check_path(struct_file2, mkdir=True)
    if struct_path2.exists():
        logger.info(f"Output file {struct_path2} already exists, skipping conversion.")
        return struct_path2

    struct_path1 = check_path(struct_file1, exists=True)
    _ = command_runner(
        cmd=[
            "gemmi",
            "convert",
            *additional_gemmi_args,
            str(struct_path1),
            str(struct_path2),
        ],
        cwd=struct_path2.parent,
        log_file="/dev/null",
    )
    return struct_path2
