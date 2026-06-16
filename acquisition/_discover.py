import os
import platform
import sys
from pathlib import Path

_default_base = Path(__file__).resolve().parent.parent / "PLUX-API-Python3"
_major, _minor = platform.python_version_tuple()[:2]
_default_api_dir = _default_base / f"Win64_{_major}{_minor}"
_api_dir = Path(os.environ.get("PLUX_API_DIR", str(_default_api_dir)))
if not _api_dir.is_dir():
    print(f"ERROR: PLUX API not found at {_api_dir}. Set PLUX_API_DIR env var.")
    sys.exit(1)
sys.path.append(str(_api_dir))
import plux

print("=== help(plux.BaseDev.findDevices) ===")
help(plux.BaseDev.findDevices)

print("\n=== findDevices() no args ===")
print(plux.BaseDev.findDevices())

print("\n=== findDevices('BTH') ===")
try:
    print(plux.BaseDev.findDevices("BTH"))
except Exception as e:
    print(type(e).__name__, e)

print("\n=== findDevices('BTH00:07:80:8C:AD:4F') ===")
try:
    print(plux.BaseDev.findDevices("BTH00:07:80:8C:AD:4F"))
except Exception as e:
    print(type(e).__name__, e)
