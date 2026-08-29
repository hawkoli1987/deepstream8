#!/usr/bin/env python3
"""Inline the video (base64) and score JSON into template.html -> motion-compare.html"""
import base64, sys, pathlib

here = pathlib.Path(__file__).resolve().parent.parent
tpl = (here / "template.html").read_text()
data = (here / "motion_scores.json").read_text()
mp4 = (here / "cam6d.mp4").read_bytes()

assert "</script>" not in data, "JSON contains </script>"
b64 = base64.b64encode(mp4).decode()

out = tpl.replace("__DATA__", data).replace(
    "__VIDEO__", "data:video/mp4;base64," + b64)

dest = here / "motion-compare.html"
dest.write_text(out)
print(f"{dest}  ({len(out)/1048576:.1f} MB)  video {len(mp4)/1048576:.1f} MB  json {len(data)/1024:.0f} KB")
