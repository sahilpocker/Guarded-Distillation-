"""Download a model bundle from the release registry (used at image build time to bake the latest model in).

    python -m guarded_distillation.serve.fetch --repo owner/repo --tag model-v2 --out models/current
Exits 0 without output when the registry cannot be reached, so a build never fails on it (the image then uses models/v1).
"""
import argparse
import shutil
import sys

from .rollout import Registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="models/current")
    a = ap.parse_args()
    try:
        reg = Registry(a.repo)
        rels = reg.releases()
        rel = next((r for r in rels if r["tag"] == a.tag), rels[0] if rels else None)
        if rel is None:
            print("no release found; keeping models/v1")
            return
        b = reg.fetch(rel)
        # Registry.fetch unpacked to its cache; copy the bundle directory to --out
        src = next(reg.cache.joinpath(rel["tag"]).rglob("bundle.pkl")).parent
        shutil.rmtree(a.out, ignore_errors=True)
        shutil.copytree(src, a.out)
        print(f"baked {b.version} from {rel['tag']} into {a.out}")
    except Exception as e:
        print(f"registry unavailable ({e}); keeping models/v1", file=sys.stderr)


if __name__ == "__main__":
    main()
