"""General unitility functions."""

import os
import shutil
import subprocess as sp
from datetime import UTC, datetime
from pathlib import Path


def check_path(
    p: str | Path, mkdir: bool = False, exists: bool = False, ignore_dots: bool = False
) -> Path:
    """Canonical way of dealing with file paths.

    Args:
        p: input file path to check and normalize.
        mkdir: create directory if it doesn't already exist. If ``p`` has a
            suffix, create its parent directory. If ``p`` does not exist, the
            directory is created at ``p`` if it doesn't have a suffix, and at
            its parent if it does have a suffix.
        exists: raise an error if ``p`` does not exist.
        ignore_dots: if True, ignore periods in the path when making directories.
            For example, ``/foo.bar`` will create the directory with the
            name ``foo.bar`` instead of treating it as a file.

    Returns:
        The absolute path to ``p``.
    """
    filepath = Path(p).expanduser().resolve()
    if exists:
        if not filepath.exists():
            raise FileNotFoundError(filepath)
    if mkdir:
        # When the file path doesn't exist
        if filepath.suffix and not ignore_dots:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            filepath.mkdir(parents=True, exist_ok=True)

    return filepath


def find_binary(name: str | Path) -> str:
    """Find a binary executable in the system PATH.

    Args:
        name: Name of the executable to find. Can be a path or just the name.

    Returns:
        The absolute path to the executable if found.
    """
    exe = shutil.which(name)
    if exe is None:
        raise FileNotFoundError(f"Executable '{name}' not found in system PATH.")
    if not os.access(exe, os.X_OK):
        raise PermissionError(f"Executable '{exe}' is not executable.")
    return exe


def command_runner(
    cmd: list[str],
    cwd: str | Path,
    log_file: str | Path,
    verbose: bool = False,
    track_metadata: bool = True,
    **kwargs,
):
    """Run a command and log its output to a file.

    Args:
        cmd: command to run as a list of strings.
        cwd: working directory to run the command in.
        log_file: file to log the output to. Note that this is independent of ``cwd``.
        verbose: if True, print the output to stdout as well.
        track_metadata: if True, log the time and command at the start and end
            of the log file.
        kwargs: additional keyword arguments passed to ``subprocess.Popen``.
    """
    with open(log_file, "a") as f:
        if track_metadata:
            f.write("Time: " + str(datetime.now(UTC)) + "\n")
            f.write("Command: " + " ".join(cmd) + "\n\n")
        with sp.Popen(  # noqa: S603
            cmd,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf-8",
            cwd=cwd,
            **kwargs,
        ) as p:
            while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
                if verbose:
                    print(buffered_output, end="", flush=True)
                f.write(buffered_output)
                f.flush()

        if track_metadata:
            f.write("\nFinish time: " + str(datetime.now(UTC)) + "\n")


def read_n_to_last_line(filename, n=1, decode: bool = True) -> str | bytes:
    """Returns the nth before last line of a file (n=1 gives last line).

    Note that if your last line is empty then it skips it.

    Ref: https://stackoverflow.com/a/73195814
    """
    num_newlines = 0
    with open(filename, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b"\n":
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline()
    if decode:
        return last_line.decode()
    else:
        return last_line
