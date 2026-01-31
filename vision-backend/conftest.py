"""
Root conftest - shared pytest configuration and fixtures.
Ensures app package is discoverable when running pytest from vision-backend/.
"""
import os
import sys
from pathlib import Path

# Ensure vision-backend root is in path for 'from app...' imports
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
