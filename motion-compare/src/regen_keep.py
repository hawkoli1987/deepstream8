#!/usr/bin/env python3
"""
regen_keep.py — rebuild the container-only intermediate `keep.json` from the
committed motion_scores.json, instead of re-deduping the 120-frame source.

build_json.py reads <outdir>/keep.json unconditionally; the original keep.json
was container-only scratch that no longer exists, but the keep-list survives
inside motion_scores.json -> video.kept_source_frames.

usage: regen_keep.py <motion_scores.json> <out-keep.json>
"""
import sys, json

src, dst = sys.argv[1], sys.argv[2]
doc = json.load(open(src))
keep = doc["video"]["kept_source_frames"]
assert isinstance(keep, list) and len(keep) == 80, f"unexpected keep list: {len(keep)} entries"
json.dump({"keep": keep, "fps_src": 2.0, "n": len(keep)}, open(dst, "w"))
print(f"wrote {dst}  (n={len(keep)})")