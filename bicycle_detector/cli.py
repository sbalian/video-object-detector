import pathlib
from typing import Annotated

import typer

from . import classifier

app = typer.Typer()


@app.command()
def main(
    input: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    paths = sorted(list(input.glob("*.dav")))
    for path in paths:
        classifier.run_for_video(path, output)
