import shutil
from pathlib import Path

# Define paths
xeno_dir = Path.home() / ".xeno"
xeno_database_path = xeno_dir / "memories.sqlite"
xeno_files_path = xeno_dir / "memory_files"

# Delete the xeno_db_path file
if xeno_database_path.exists() and xeno_database_path.is_file():
    xeno_database_path.unlink()
    print(f"Deleted database: {xeno_database_path}")

# Delete the xeno_files_path directory and its contents
if xeno_files_path.exists() and xeno_files_path.is_dir():
    shutil.rmtree(xeno_files_path)
    print(f"Deleted files: {xeno_files_path}")
