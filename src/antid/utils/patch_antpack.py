"""Patch for antpack's VJGeneTool.

This module contains a patched version of the VJGeneTool
class from antpack, which uses a preloaded database of VJ genes
and a BLOSUM matrix for amino acid sequence alignment.
This avoids the need to load the database from disk each time
the class is instantiated, improving performance in applications
where the class is used frequently.
"""

from pathlib import Path

import antpack
import numpy as np
from antpack.antpack_cpp_ext import VJMatchCounter
from antpack.utilities.vj_utilities import load_vj_gene_consensus_db

antpack_path = Path(antpack.__file__).parent.resolve()
db_path = antpack_path / "vj_tools" / "consensus_data"
vj_names, vj_seqs, retrieved_dates = load_vj_gene_consensus_db(
    antpack_path, db_path, "imgt"
)
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
