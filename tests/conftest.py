"""
pytest configuration — add the source/ directory to sys.path so that
intra-package imports like `from franchise_db import ...` resolve correctly
when tests are run from the repository root.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))
