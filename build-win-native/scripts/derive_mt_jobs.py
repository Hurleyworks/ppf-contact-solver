# File: scripts/derive_mt_jobs.py
# Code: Claude Code
# License: Apache v2.0
#
# Used by build-cuda-static-local.bat. Reads the DLL build's compile-jobs.txt
# and writes a /MT variant into another output directory, dropping stub.cpp
# (its empty print_rust would collide with the Rust one), plus an nvcc/lib.exe
# response file listing the objects that variant produces.
#
#   python derive_mt_jobs.py <dll_out_dir> <mt_out_dir>
import re
import sys
from pathlib import Path

dll_out = Path(sys.argv[1])
mt_out = Path(sys.argv[2])
jobs = []
objects = []
for line in (dll_out / "compile-jobs.txt").read_text().splitlines():
    if not line.strip() or line.startswith("stub.cpp|"):
        continue
    line = line.replace("/MD ", "/MT ")
    # Only the OUTPUT objects move; sources and includes (build\kernelgen)
    # stay where the DLL build rendered them.
    for sub in ("obj", "cuda-entry"):
        line = line.replace(str(dll_out / sub) + "\\", str(mt_out / sub) + "\\")
    match = re.search(r'-o "([^"]+)"\s*$', line)
    if not match:
        sys.exit(f"no output object in job: {line[:80]}")
    objects.append(f'"{match.group(1)}"')
    jobs.append(line)
(mt_out / "compile-jobs.txt").write_text("\n".join(jobs) + "\n")
(mt_out / "objects.rsp").write_text("\n".join(objects) + "\n")
print(f"{len(jobs)} /MT compile jobs")
