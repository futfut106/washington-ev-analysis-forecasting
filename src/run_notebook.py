"""Chạy một notebook và ghi kết quả trở lại file. Dùng: python src/run_notebook.py notebooks/hypothesis_tests.ipynb"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
print("Done:", path)
