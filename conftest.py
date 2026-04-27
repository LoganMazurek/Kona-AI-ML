"""
Root conftest — ensures the repository root is on sys.path so that
`import source.web_app`, `import source.auth`, etc. resolve correctly
when running `pytest` from the repo root.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
