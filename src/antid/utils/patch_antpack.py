"""Patch for antpack's VJGeneTool.

This module contains a patched version of the VJGeneTool
class from antpack, which uses a preloaded database of VJ genes
and a BLOSUM matrix for amino acid sequence alignment.
This avoids the need to load the database from disk each time
the class is instantiated, improving performance in applications
where the class is used frequently.
"""

import gzip
from pathlib import Path

import antpack
import numpy as np
from antpack.antpack_cpp_ext import SingleChainAnnotatorCpp, VJMatchCounter
from antpack.numbering_tools.cterm_finder import _load_nterm_kmers

from antid.utils import check_path


def load_vj_gene_consensus_db(database_path: str | Path, database: str = "imgt"):
    """Patch of antpack.utilities.vj_utilities.load_vh_gene_consensus_db.

    Args:
        current_dir: The current user directory.
        database_path: The location of the consensus databases.
        database: The name of the database to load (default is "imgt").
    """
    vj_names, vj_seqs, retrieved_dates = {}, {}, {}

    try:
        db_path = check_path(database_path, exists=True, is_dir=True)
        db_files = [
            f for f in db_path.glob("*.fa.gz") if f.stem.split("_")[1] == database
        ]
        for db_file in db_files:
            fname = db_file.name.split(".")[0]
            species, _, receptor, retrieved_date = fname.split("_")
            if species not in retrieved_dates:
                retrieved_dates[species] = {}

            retrieved_dates[species][receptor] = retrieved_date

            # We avoid using Biopython's SeqIO (since it introduces an additional
            # unnecessary dependency). Since we wrote the db files, we know
            # how they are formatted -- one aa seq for each gene name -- and
            # can use a very simple procedure here which is not applicable to all
            # fasta files.
            with gzip.open(db_file, "rt", encoding="utf-8") as f:
                sequences, names = [], []

                for line in f:
                    if line.startswith(">"):
                        description = line.strip()[1:]
                    else:
                        aa_seq = line.strip()
                        sequences.append(aa_seq)
                        names.append(description)

            key = f"{species}_{receptor}"
            if key in vj_names:
                raise RuntimeError(
                    f"Duplicate db files found for {species}, {receptor}."
                )

            vj_names[key] = names
            vj_seqs[key] = sequences

    except Exception as e:
        raise RuntimeError(
            "The consensus data for the package either has been deleted or "
            "moved or was never properly installed."
        ) from e

    return vj_names, vj_seqs, retrieved_dates


# Global variables to avoid loading the database multiple times
antpack_path = Path(antpack.__file__).parent.resolve()
db_path = antpack_path / "vj_tools" / "consensus_data"
vj_names, vj_seqs, retrieved_dates = load_vj_gene_consensus_db(db_path, "imgt")
ig_aligner_consensus_path = antpack_path / "numbering_tools" / "consensus_data"
blosum_matrix = np.load(
    ig_aligner_consensus_path / "mabs" / "blosum_matrix.npy"
).astype(np.float64)


class VJGeneTool(VJMatchCounter):
    """Patched version of antpack.VJGeneTool."""

    def __init__(self, scheme: str = "imgt"):
        """Class constructor.

        Args:
            scheme (str): One of 'aho, 'imgt', 'kabat' or 'martin'.
                Determines the numbering scheme that will be used
                when assigning vj genes based on alignments.
        """
        self.retrieved_dates = retrieved_dates

        super().__init__(
            vj_names, vj_seqs, blosum_matrix, scheme, str(ig_aligner_consensus_path)
        )

    def retrieve_db_dates(self):
        """Identical method from antpack.VJGeneTool.

        Returns the dates when each VJ gene database
        used for this assignment was last updated by downloading
        from IMGT or OGRDB.
        """
        return self.retrieved_dates


consensus_path = str(antpack_path / "numbering_tools" / "consensus_data")
kmer_dict = _load_nterm_kmers()


class SingleChainAnnotator(SingleChainAnnotatorCpp):
    """Patched version of antpack.SingleChainAnnotator."""

    def __init__(self, chains=None, scheme="imgt"):
        """Patched version of antpack.SingleChainAnnotator with global variables."""
        if chains is None:
            chains = ["H", "K", "L"]
        super().__init__(chains, scheme, consensus_path, kmer_dict)
