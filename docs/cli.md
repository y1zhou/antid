# Command Line Interface

`antid` provides a command-line interface for common antibody analysis tasks.

## Basic Usage

```bash
antid [COMMAND] [OPTIONS] [ARGUMENTS]
```

## Commands

### `number`

Number an antibody sequence.

```bash
antid number SEQUENCE [OPTIONS]
```

**Arguments:**

- `SEQUENCE`: The antibody sequence to number

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--scheme` | `-s` | Numbering scheme (imgt, kabat, chothia, martin, aho) | `imgt` |
| `--assign-germline` | `-g` | Assign germline genes to the sequence | `false` |
| `--species` | `-p` | Species of the antibody sequence | `None` |
| `--scfv` | | Indicates the input is an scFv sequence | `false` |

**Examples:**

```bash
# Number with IMGT scheme
antid number "QVQLVQSGAE..." --scheme imgt

# Number and assign germline
antid number "QVQLVQSGAE..." -g --species human

# Number an scFv sequence
antid number "HEAVY_LINKER_LIGHT" --scfv
```

### `align`

Align multiple antibody sequences.

```bash
antid align SEQUENCE1 SEQUENCE2 [SEQUENCE...] [OPTIONS]
```

**Arguments:**

- `SEQUENCE1, SEQUENCE2, ...`: Antibody sequences to align. All sequences are compared with the first.

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--scheme` | `-s` | Numbering scheme | `imgt` |

**Examples:**

```bash
# Align two sequences
antid align "SEQ1" "SEQ2" --scheme kabat

# Align multiple sequences
antid align "REF_SEQ" "SEQ1" "SEQ2" "SEQ3"
```

## Getting Help

```bash
# Show main help
antid --help

# Show command-specific help
antid number --help
antid align --help
```
