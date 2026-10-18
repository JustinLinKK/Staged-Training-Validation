"""Let compatibility tests import the checker in standalone and submodule clones."""

import sys
from pathlib import Path
from types import ModuleType

if "Preflight_checker" not in sys.modules:
    package = ModuleType("Preflight_checker")
    package.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["Preflight_checker"] = package
