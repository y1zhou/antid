"""CLI for antid."""

from typing import Annotated

import typer

from antid.numbering import ValidSchemesType, align_ab_seqs, number_ab_seq

app = typer.Typer()


@app.command()
def number(
    seq: Annotated[str, typer.Argument(help="Antibody sequence to number.")],
    scheme: Annotated[
        ValidSchemesType,
        typer.Option(
            "--scheme",
            "-s",
            help="Numbering scheme to use.",
            case_sensitive=False,
        ),
    ] = "imgt",
    assign_germline: Annotated[
        bool,
        typer.Option(
            "--assign-germline",
            "-g",
            help="Assign germline genes to the sequence.",
            is_flag=True,
        ),
    ] = False,
):
    """Number an antibody sequence."""
    # The type hint for species in number_ab_seq is a bit complex,
    # but it can handle a string or a list of strings.
    numbered_seq = number_ab_seq(seq, scheme=scheme, assign_germline=assign_germline)
    print(numbered_seq.format())


@app.command()
def align(
    seqs: Annotated[
        list[str],
        typer.Argument(
            help="Antibody sequences to align. Note that all sequences will be compared with the first."
        ),
    ],
    scheme: Annotated[
        ValidSchemesType,
        typer.Option(
            "--scheme",
            "-s",
            help="Numbering scheme to use.",
            case_sensitive=False,
        ),
    ] = "imgt",
):
    """Align multiple antibody sequences."""
    aligned_seqs = align_ab_seqs(number_ab_seq(seqs, scheme=scheme))
    print(aligned_seqs.format(ref_seq_id="0"))


if __name__ == "__main__":
    app()
