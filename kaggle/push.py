"""Bundle a run script with the connector and push it as a private Kaggle kernel.

    python kaggle/push.py smoke

Kaggle script kernels are a single file, so the zeroprefill package and lib.py
are embedded as strings and written to /kaggle/working/zp_code at startup.
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kaggle_user():
    """Whoever is logged in to the Kaggle CLI, so collaborators push to their own account."""
    out = subprocess.run([sys.executable, "-m", "kaggle", "config", "view"],
                         capture_output=True, text=True).stdout
    match = re.search(r"username:\s*(\S+)", out)
    if not match or match.group(1) == "None":
        sys.exit("Kaggle CLI is not logged in: python -m kaggle auth login")
    return match.group(1)


HEADER = '''\
import os, sys
_FILES = {files}
for _rel, _src in _FILES.items():
    _path = os.path.join("/kaggle/working/zp_code", _rel)
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    with open(_path, "w") as _f:
        _f.write(_src)
sys.path.insert(0, "/kaggle/working/zp_code")
'''


def main(run, kernel_sources=()):
    user = kaggle_user()
    files = {"lib.py": open(f"{ROOT}/kaggle/lib.py", encoding="utf-8").read()}
    for name in sorted(os.listdir(f"{ROOT}/zeroprefill")):
        if name.endswith(".py"):
            files[f"zeroprefill/{name}"] = open(f"{ROOT}/zeroprefill/{name}", encoding="utf-8").read()

    slug = f"zeroprefill-{run}"
    out = f"{ROOT}/build/{slug}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{slug}.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER.format(files=repr(files)))
        f.write(open(f"{ROOT}/kaggle/{run}.py", encoding="utf-8").read())
    with open(f"{out}/kernel-metadata.json", "w") as f:
        json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.py",
                   "language": "python", "kernel_type": "script", "is_private": True,
                   "enable_gpu": True, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4",
                   "dataset_sources": [], "competition_sources": [],
                   "kernel_sources": list(kernel_sources), "model_sources": []},
                  f, indent=2)
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", out], check=True)
    print(f"status: python -m kaggle kernels status {user}/{slug}")
    print(f"output: python -m kaggle kernels output {user}/{slug} -p runs/{run}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
