import os
import pathlib

def get_repo_root():
    """Get repo root path"""
    path=os.getcwd()
    current_dir = pathlib.Path(path)
    while not (current_dir / ".git").exists():
        current_dir = current_dir.parent
    return str(current_dir)

def get_abs_path(rel_path):
    """Get abs path"""
    if os.path.isabs(rel_path):
        return rel_path
    abs_path=os.path.join(get_repo_root(),rel_path)
    return abs_path
