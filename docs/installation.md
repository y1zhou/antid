# Installation

## Requirements

- Python 3.12 or higher

## Installing from PyPI

The recommended way to install `antid` is from PyPI:

=== "uv"

    ```bash
    uv pip install antid
    ```

=== "pip"

    ```bash
    pip install antid
    ```

## Installing from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/y1zhou/antid.git
cd antid

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
uv pip install -e ".[dev]"
```

## Verifying Installation

After installation, verify that `antid` is correctly installed:

```bash
antid --help
```

You should see the available commands and options.
