# Getting Started

This guide will help you get started with `antid` for antibody sequence analysis.

## Numbering Antibody Sequences

The most common use case is numbering antibody sequences using a standardized scheme.

### Using Python

```python
from antid.numbering import number_ab_seq

# Number a single sequence using IMGT scheme (default)
seq = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR"
numbered = number_ab_seq(seq, scheme="imgt")

# Format the output
print(numbered.format())
```

### Available Numbering Schemes

`antid` supports several numbering schemes:

- `imgt` - IMGT numbering (default)
- `kabat` - Kabat numbering
- `chothia` - Chothia numbering
- `martin` - Martin (Enhanced Chothia) numbering
- `aho` - AHo numbering

### Command Line Interface

You can also use the CLI for quick analysis:

```bash
# Number a sequence
antid number "QVQLVQSGAE..." --scheme imgt

# Align multiple sequences
antid align "SEQ1" "SEQ2" --scheme kabat
```

## Working with scFv Sequences

For single-chain variable fragments (scFv):

```python
from antid.numbering.scfv import number_scfv_seq

scfv_seq = "HEAVY_CHAIN_LINKER_LIGHT_CHAIN"
numbered = number_scfv_seq(scfv_seq, scheme="imgt")
```

## Sequence Alignment

Align multiple antibody sequences:

```python
from antid.numbering import align_ab_seqs, number_ab_seq

seqs = ["SEQ1", "SEQ2", "SEQ3"]
numbered_seqs = number_ab_seq(seqs, scheme="imgt")
alignment = align_ab_seqs(numbered_seqs)
print(alignment.format(ref_seq_id="0"))
```

## Reading Sequence Files

### From FASTA Files

```python
from antid.io import fasta2seq

sequences = fasta2seq("antibodies.fasta")
```

### From Structure Files

Extract sequences from PDB/mmCIF files:

```python
from antid.io import struct2seq

sequences = struct2seq("structure.pdb")
```

## Working with Structures

### Download Structures from RCSB

```python
from antid.io import RCSBDownloader

downloader = RCSBDownloader()
downloader.download("1IGT", output_dir="structures/")
```

### Convert Structure Formats

```python
from antid.io import gemmi_convert

gemmi_convert("input.pdb", "output.cif")
```

## PPI Analysis

Analyze protein-protein interactions in antibody-antigen complexes:

```python
from antid.ppi import collect_ab_ag_contacts, get_contacts

# Get contacts between antibody and antigen
contacts = collect_ab_ag_contacts("complex.pdb")

# Get all atomic contacts
all_contacts = get_contacts("structure.pdb", distance_cutoff=4.0)
```

## Next Steps

- Check the [API Reference](reference/index.md) for detailed documentation
- Explore the [CLI documentation](cli.md) for command-line usage
