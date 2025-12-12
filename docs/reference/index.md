# API Reference

This section provides detailed API documentation for all modules in `antid`.

## Modules

### [Numbering](numbering.md)

Number antibody sequences using standardized schemes.

- `number_ab_seq` - Number antibody sequences
- `align_ab_seqs` - Align numbered sequences
- `NumberedAntibody` - Numbered antibody representation
- `AntibodyAlignment` - Sequence alignment results

### [I/O](io.md)

Read and write sequence and structure files.

- `fasta2seq` - Read FASTA files
- `struct2seq` - Extract sequences from structures
- `RCSBDownloader` - Download structures from RCSB
- `df2pdb` - Convert DataFrames to PDB format
- `gemmi_convert` - Convert structure formats
- `struct2df` - Convert structures to DataFrames

### [PPI](ppi.md)

Analyze protein-protein interactions.

- `collect_ab_ag_contacts` - Find antibody-antigen contacts
- `collect_within_ab_contacts` - Find intra-antibody contacts
- `get_contacts` - Get atomic contacts
- `get_atomic_sasa` - Calculate solvent accessible surface area
- `LiabilityScanner` - Scan for sequence liabilities
