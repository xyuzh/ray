"""Build an ownership-true copy of a cached rootfs inside a mapped user namespace.

Invoked as ``python -m ray.experimental.sandbox._internal.idmap_extract SRC
SIDECAR DEST`` by :func:`ray.experimental.sandbox._internal.image_utils.ensure_idmapped_rootfs`
through ``nsenter`` into a user namespace whose uid/gid maps cover the image's
ids; only there can ``lchown`` give files their true owners. ``cp -a``
reproduces the worker-owned tree (hardlinks included) as mapped root, then
the owners recorded in the ``.ownership.json`` sidecar are applied.
"""

import argparse
import json
import os
import stat
import subprocess
import sys
from typing import Dict, Tuple


def apply_ownership(dest: str, ownership: Dict[str, Tuple[int, int]]) -> None:
    """Chown ``dest`` entries per ``ownership``, restoring the modes chown clears."""
    from ray.experimental.sandbox._internal.image_utils import _lchown_preserving

    for rel, (uid, gid) in ownership.items():
        path = os.path.join(dest, rel)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        _lchown_preserving(path, uid, gid)
        if not stat.S_ISLNK(st.st_mode):
            # chown clears setuid/setgid on files; put the mode back.
            try:
                os.chmod(path, stat.S_IMODE(st.st_mode))
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="worker-owned rootfs to copy")
    parser.add_argument("sidecar", help=".ownership.json recorded at extraction")
    parser.add_argument("dest", help="directory to materialize")
    args = parser.parse_args()

    subprocess.run(["cp", "-a", "--", args.src, args.dest], check=True)
    with open(args.sidecar, encoding="utf-8") as f:
        ownership = {p: (int(ids[0]), int(ids[1])) for p, ids in json.load(f).items()}
    apply_ownership(args.dest, ownership)
    return 0


if __name__ == "__main__":
    sys.exit(main())
