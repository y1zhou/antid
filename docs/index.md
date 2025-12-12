# Antibody Design (antid)

[![PyPI version](https://badge.fury.io/py/antid.svg)](https://badge.fury.io/py/antid)

A Python package containing helper functions for antibody design.

## Overview

`antid` provides a comprehensive toolkit for working with antibody sequences and structures, including:

- **Numbering**: Number antibody sequences using various schemes (IMGT, Kabat, Chothia, etc.)
- **I/O**: Read and write sequence and structure files
- **PPI Analysis**: Analyze protein-protein interactions in antibody-antigen complexes
- **Preprocessing**: Prepare molecular structures for analysis

## Quick Start

### Installation

Install `antid` from PyPI using your preferred package manager:

```bash
# With uv
uv pip install antid

# With pip
pip install antid
```

### Basic Usage

Number an antibody sequence:

```python
from antid.numbering import number_ab_seq

numbered = number_ab_seq("QVQLVQSGAE...", scheme="imgt")
print(numbered.format())
```

Use the command-line interface:

```bash
antid number "QVQLVQSGAE..." --scheme imgt
```

## Documentation

- [Installation](installation.md) - How to install `antid`
- [Getting Started](getting-started.md) - Quick start guide
- [API Reference](reference/index.md) - Full API documentation

## License

This project is licensed under the Apache License 2.0.
