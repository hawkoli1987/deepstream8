#!/usr/bin/env python3
"""Inline a video (base64) and the score JSON into a template -> the deliverable.

usage: build_html.py [template] [video] [json] [out]   (defaults = v1 cam6)
"""
import base64, sys, pathlib

here = pathlib.Path(__file__).resolve().parent.parent
tpl   = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here / "templates/cam6.html"
mp4   = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else here / "runs/cam6/clip/cam6d.mp4"
data  = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else here / "runs/cam6/scores/motion_scores.json"
dest  = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else here / "motion-compare-v1.html"

tpl = tpl.read_text()
data = data.read_text()
mp4 = mp4.read_bytes()

assert "</script>" not in data, "JSON contains </script>"
b64 = base64.b64encode(mp4).decode()

out = tpl.replace("__DATA__", data).replace(
    "__VIDEO__", "data:video/mp4;base64," + b64)

dest.write_text(out)
print(f"{dest}  ({len(out)/1048576:.1f} MB)  video {len(mp4)/1048576:.1f} MB  json {len(data)/1024:.0f} KB")
