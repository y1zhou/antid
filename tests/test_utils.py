"""Tests for the utils module."""

import os
import stat
from pathlib import Path

import pytest

from antid.utils import (
    check_path,
    chunks,
    command_runner,
    find_binary,
    read_n_to_last_line,
)


# ruff: noqa: S101
# Tests for check_path
def test_check_path_creates_dir(tmp_path):
    """Test that check_path creates a directory."""
    dir_path = tmp_path / "test_dir"
    check_path(dir_path, mkdir=True)
    assert dir_path.is_dir()
    assert dir_path.exists()

    nested_dir = tmp_path / "test_dir" / "nested"
    check_path(nested_dir, mkdir=True)
    assert nested_dir.is_dir()
    assert nested_dir.exists()


def test_check_path_creates_parent_dir(tmp_path):
    """Test that check_path creates a parent directory for a file."""
    file_path = tmp_path / "test_dir" / "test_file.txt"
    check_path(file_path, mkdir=True)
    assert file_path.parent.is_dir()
    assert file_path.parent.exists()


def test_check_path_ignore_dots(tmp_path):
    """Test that check_path ignores paths with dots."""
    dot_path = tmp_path / "dir.to.create"
    check_path(dot_path, mkdir=True, is_dir=True)
    assert dot_path.is_dir()
    assert dot_path.exists()

    dot_path_with_file = tmp_path / "another.dir.to.create" / "file.txt"
    check_path(dot_path_with_file, mkdir=True, is_dir=False)
    assert dot_path_with_file.parent.is_dir()
    assert dot_path_with_file.parent.exists()


def test_check_path_exists_raises_error(tmp_path):
    """Test that check_path raises an error if a path does not exist."""
    with pytest.raises(FileNotFoundError):
        check_path(tmp_path / "non_existent_file", exists=True)


def test_check_path_resolves_path(tmp_path):
    """Test that check_path returns a resolved path."""
    relative_path = ".."
    resolved_path = check_path(tmp_path / relative_path)
    assert resolved_path == tmp_path.parent.resolve()


# Tests for find_binary
def test_find_binary_found(tmp_path, monkeypatch):
    """Test that find_binary finds an existing executable."""
    # Use a common command that should exist on most systems
    # Create a dummy executable to avoid system dependency
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    dummy_exe_path = bin_dir / "my_test_exe"
    dummy_exe_path.touch()
    dummy_exe_path.chmod(dummy_exe_path.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)

    found_path = find_binary("my_test_exe")
    assert Path(found_path).name == "my_test_exe"
    assert Path(found_path).is_absolute()

    found_path = find_binary(dummy_exe_path)
    assert Path(found_path) == dummy_exe_path.resolve()


def test_find_binary_not_found():
    """Test that find_binary raises FileNotFoundError for a non-existent binary."""
    with pytest.raises(
        FileNotFoundError, match="Executable 'non_existent_binary' not found"
    ):
        find_binary("non_existent_binary")


def test_find_binary_not_executable(tmp_path, monkeypatch):
    """Test that find_binary raises PermissionError for a non-executable file."""
    non_exec_file = tmp_path / "not_executable"
    non_exec_file.touch()
    # Ensure the file is not executable
    non_exec_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)

    with pytest.raises(
        PermissionError,
        match=f"Executable '{non_exec_file.resolve()}' is not executable.",
    ):
        find_binary("not_executable")

    with pytest.raises(
        PermissionError,
        match=f"Executable '{non_exec_file.resolve()}' is not executable.",
    ):
        find_binary(non_exec_file)


# Tests for command_runner
def test_command_runner_logs_output(tmp_path):
    """Test that command_runner logs command output to a file."""
    log_file = tmp_path / "test.log"
    command = ["echo", "hello world"]
    command_runner(command, cwd=tmp_path, log_file=log_file)

    with open(log_file) as f:
        content = f.read()
        assert "hello world" in content
        assert "Command: echo hello world" in content


# Tests for command_runner with verbose output to stdout
def test_command_runner_verbose_output(tmp_path, capsys):
    """Test that command_runner logs command output to a file."""
    log_file = "/dev/null"
    command = ["echo", "hello world"]
    command_runner(
        command, cwd=tmp_path, log_file=log_file, verbose=True, track_metadata=False
    )
    captured = capsys.readouterr()
    # with capsys.disabled():
    #     print(captured)
    assert "hello world\n" == captured.out


# Tests for read_n_to_last_line
def test_read_n_to_last_line(tmp_path):
    """Test that read_n_to_last_line reads the correct line."""
    file_path = tmp_path / "test.txt"
    lines = ["line 1", "line 2", "line 3"]
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    assert read_n_to_last_line(file_path, n=1).strip() == "line 3"
    assert read_n_to_last_line(file_path, n=2).strip() == "line 2"
    assert read_n_to_last_line(file_path, n=3).strip() == "line 1"


def test_read_n_to_last_line_empty_file(tmp_path):
    """Test read_n_to_last_line on an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()
    assert read_n_to_last_line(file_path) == ""


def test_read_n_to_last_line_return_bytes(tmp_path):
    """Test read_n_to_last_line without decoding."""
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("line 1\n")

    assert read_n_to_last_line(file_path, n=1, decode=False) == b"line 1\n"


# Test for helper chunking function
def test_chunking_iterable():
    """Test chunking of an iterable."""
    res = list(chunks([1, 2, 3, 4, 5], 2))
    assert res == [[1, 2], [3, 4], [5]]

    res = list(chunks("abcdefg", 3))
    assert res == [["a", "b", "c"], ["d", "e", "f"], ["g"]]

    res = list(chunks([], 2))
    assert res == []
