"""
Storage utilities using fsspec for local and cloud file access.

Provides abstraction for accessing files locally (files/) with path to cloud migration.
"""

from pathlib import Path
from typing import Optional
import fsspec


def get_filesystem(storage_type: str = "local") -> fsspec.AbstractFileSystem:
    """
    Get fsspec filesystem instance.

    Args:
        storage_type: Type of storage ('local' for now, 'gcs' or 's3' in future)

    Returns:
        fsspec filesystem instance
    """
    if storage_type == "local":
        return fsspec.filesystem("file")
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def get_file_path(filename: str, base_dir: Optional[str] = None) -> str:
    """
    Get the full path to a file in storage.

    Args:
        filename: Name of the file to retrieve
        base_dir: Base directory (defaults to 'files/' at project root)

    Returns:
        Full path string to the file
    """
    if base_dir is None:
        # Default to files/ at project root
        project_root = Path(__file__).parent.parent.parent
        base_dir = str(project_root / "files")

    return f"{base_dir}/{filename}"


def read_file(filename: str, mode: str = "r", base_dir: Optional[str] = None) -> str:
    """
    Read a file from storage using fsspec.

    Args:
        filename: Name of the file to read
        mode: File mode ('r' for text, 'rb' for binary)
        base_dir: Base directory (defaults to 'files/' at project root)

    Returns:
        File contents as string (text mode) or bytes (binary mode)
    """
    fs = get_filesystem("local")
    file_path = get_file_path(filename, base_dir)

    with fs.open(file_path, mode) as f:
        return f.read()


def write_file(filename: str, content: str, mode: str = "w", base_dir: Optional[str] = None) -> str:
    """
    Write a file to storage using fsspec.

    Args:
        filename: Name of the file to write
        content: Content to write to the file
        mode: File mode ('w' for text, 'wb' for binary)
        base_dir: Base directory (defaults to 'files/' at project root)

    Returns:
        Full path to the written file
    """
    fs = get_filesystem("local")
    file_path = get_file_path(filename, base_dir)

    # Ensure parent directory exists
    parent_dir = str(Path(file_path).parent)
    fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(file_path, mode) as f:
        f.write(content)

    return file_path


def list_files(base_dir: Optional[str] = None, pattern: str = "*") -> list[str]:
    """
    List all files in storage directory.

    Args:
        base_dir: Base directory (defaults to 'files/' at project root)
        pattern: Glob pattern for filtering files (default: "*" for all files)

    Returns:
        List of file paths
    """
    if base_dir is None:
        project_root = Path(__file__).parent.parent.parent
        base_dir = str(project_root / "files")

    fs = get_filesystem("local")

    if not fs.exists(base_dir):
        return []

    # fsspec glob returns full paths
    full_pattern = f"{base_dir}/{pattern}"
    return sorted(fs.glob(full_pattern))
