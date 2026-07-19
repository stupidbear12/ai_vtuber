# -*- coding: utf-8 -*-
import sys, os
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb_data")
os.makedirs(data_dir, exist_ok=True)
from chromadb.cli.cli import app as typer_app
sys.argv = ["chroma", "run", "--host", "0.0.0.0", "--port", "8010", "--path", data_dir]
typer_app()
