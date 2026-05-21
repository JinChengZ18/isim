"""
Auto-download helpers for benchmark datasets.

Supports G-set (Max-Cut) and TSPLIB95 (TSP). Each dataset has a list of
mirror URL templates that are tried in order; the first mirror that
responds successfully wins. Downloads are idempotent: files already
present on disk are not re-downloaded.

CLI usage
---------
    python fetch_data.py gset    --instances G1 G14 G22 --output ./gset
    python fetch_data.py tsplib  --instances berlin52 eil51 --output ./tsplib
    python fetch_data.py all     --output-root .

Module usage (invoked automatically by the bench drivers)
---------------------------------------------------------
    from fetch_data import ensure_gset, ensure_tsplib
    ensure_gset(['G1', 'G14'], './gset')
    ensure_tsplib(['berlin52'], './tsplib')

Environment overrides
---------------------
    GSET_MIRROR    : single URL template with {name}; if set, used as
                     the sole mirror (overrides defaults)
    TSPLIB_MIRROR  : single URL template with {name}; if set, used as
                     the sole mirror (overrides defaults)
Useful in restricted networks where a known-good mirror is available.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


# Mirror lists. Each URL template must contain the {name} placeholder.
# URLs ending in .gz are transparently decompressed after download.
DEFAULT_GSET_MIRRORS = [
    "https://raw.githubusercontent.com/0816keisuke/Gset-Max-cut/main/Gset/{name}",
    "https://raw.githubusercontent.com/0816keisuke/Gset-Max-cut/master/Gset/{name}",
    "https://web.stanford.edu/~yyye/yyye/Gset/{name}",
]

DEFAULT_TSPLIB_MIRRORS = [
    "https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp",
    "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/{name}.tsp",
    "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz",
]

USER_AGENT = "Mozilla/5.0 (compatible; IsingBenchmarkFetcher/1.1)"
TIMEOUT_SECONDS = 20


def _mirror_list(env_var: str, defaults):
    """If the env var is set, use it as the sole mirror; otherwise use
    defaults. This keeps the env override behavior familiar (single URL)
    while still allowing automatic failover when the user has not set it."""
    override = os.environ.get(env_var)
    if override:
        return [override]
    return list(defaults)


def _urlopen(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)


def _download_bytes(url: str) -> bytes:
    with _urlopen(url) as resp:
        return resp.read()


def _needs_fetch(path: Path) -> bool:
    return (not path.exists()) or path.stat().st_size == 0


def _try_mirrors(name: str, urls, dest: Path, verbose: bool):
    """Try each URL in sequence. Returns (True, None) on success,
    or (False, list_of_error_messages) on total failure."""
    errors = []
    for url_template in urls:
        url = url_template.format(name=name)
        decompress = url.endswith(".gz")
        if verbose:
            print(f"  trying {url}", flush=True)
        try:
            data = _download_bytes(url)
            if decompress:
                data = gzip.decompress(data)
            if len(data) == 0:
                raise RuntimeError("received empty payload")
            # Atomic write: .part then rename.
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            if verbose:
                print(f"  success: {dest} ({len(data)} bytes)", flush=True)
            return True, None
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionResetError, OSError, RuntimeError) as e:
            errors.append(f"{url}\n      reason: {e}")
    return False, errors


def _ensure_generic(dataset_label: str, instances, output_dir, mirrors,
                    dest_suffix: str = "", force: bool = False,
                    verbose: bool = True):
    """Shared implementation for ensure_gset and ensure_tsplib."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}{dest_suffix}" for name in instances]
    to_fetch = [(name, p) for name, p in zip(instances, paths)
                if force or _needs_fetch(p)]
    if not to_fetch:
        return paths

    total = len(to_fetch)
    failures = []
    for idx, (name, dest) in enumerate(to_fetch, 1):
        if verbose:
            print(f"[fetch {dataset_label} {idx}/{total}] {dest.name}", flush=True)
        ok, errs = _try_mirrors(name, mirrors, dest, verbose=verbose)
        if not ok:
            failures.append((name, dest, errs))

    if failures:
        lines = [f"Failed to download {len(failures)} {dataset_label} "
                 f"instance(s) after trying all mirrors:"]
        for name, dest, errs in failures:
            lines.append(f"  {name} -> {dest}")
            for e in errs:
                lines.append(f"    tried: {e}")
        lines.append("")
        lines.append("Options to recover:")
        lines.append("  1. Retry later (transient network/server issues are common).")
        lines.append("  2. Override the mirror with an environment variable, e.g.")
        lines.append("       set GSET_MIRROR=https://your.mirror/Gset/{name}")
        lines.append("       set TSPLIB_MIRROR=https://your.mirror/tsp/{name}.tsp")
        lines.append("     (preserve the literal {name} placeholder).")
        lines.append("  3. Download manually and place the files at the paths above.")
        raise RuntimeError("\n".join(lines))
    return paths


def ensure_gset(instances, output_dir, force: bool = False,
                verbose: bool = True):
    """Ensure G-set files are present locally. Tries each configured
    mirror for each missing instance. Returns a list of local Paths."""
    mirrors = _mirror_list("GSET_MIRROR", DEFAULT_GSET_MIRRORS)
    return _ensure_generic("gset", instances, output_dir, mirrors,
                           dest_suffix="", force=force, verbose=verbose)


def ensure_tsplib(instances, output_dir, force: bool = False,
                  verbose: bool = True):
    """Ensure TSPLIB .tsp files are present locally. Transparently
    decompresses .tsp.gz payloads if the winning mirror returns one.
    Returns a list of local Paths to .tsp files."""
    mirrors = _mirror_list("TSPLIB_MIRROR", DEFAULT_TSPLIB_MIRRORS)
    return _ensure_generic("tsplib", instances, output_dir, mirrors,
                           dest_suffix=".tsp", force=force, verbose=verbose)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets used by Section 3.4.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_g = sub.add_parser("gset", help="Download G-set Max-Cut instances")
    p_g.add_argument("--instances", nargs="+",
                     default=["G1", "G14", "G22"])
    p_g.add_argument("--output", default="./gset")
    p_g.add_argument("--force", action="store_true",
                     help="Re-download even if files already exist")

    p_t = sub.add_parser("tsplib", help="Download TSPLIB TSP instances")
    p_t.add_argument("--instances", nargs="+",
                     default=["berlin52", "eil51"])
    p_t.add_argument("--output", default="./tsplib")
    p_t.add_argument("--force", action="store_true")

    p_a = sub.add_parser("all", help="Download both default datasets")
    p_a.add_argument("--output-root", default=".")
    p_a.add_argument("--gset-instances", nargs="+",
                     default=["G1", "G14", "G22"])
    p_a.add_argument("--tsplib-instances", nargs="+",
                     default=["berlin52", "eil51"])
    p_a.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "gset":
        ensure_gset(args.instances, args.output, force=args.force)
    elif args.cmd == "tsplib":
        ensure_tsplib(args.instances, args.output, force=args.force)
    elif args.cmd == "all":
        root = Path(args.output_root)
        ensure_gset(args.gset_instances, root / "gset", force=args.force)
        ensure_tsplib(args.tsplib_instances, root / "tsplib", force=args.force)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
