#!/usr/bin/env python3
"""build_html.py — inline summary.json into templates/report.html.

Replaces the single `__DATA__` token with the summary payload and writes the
workstream deliverable `throughput-test-v1.html` at the workstream root
(next to src/). Pass --out to override.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "report.html")

TOKEN = "__DATA__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        data = json.load(f)
    with open(args.template) as f:
        html = f.read()
    if TOKEN not in html:
        print("ERROR: %s missing in %s" % (TOKEN, args.template), file=sys.stderr)
        return 2
    payload = json.dumps(data, indent=1)
    if "</script" in payload.lower():
        payload = payload.replace("</", "<\\/")
    html = html.replace(TOKEN, payload)
    if args.out is None:
        ws_root = os.path.dirname(HERE)
        args.out = os.path.join(ws_root, "throughput-test-v1.html")
    with open(args.out, "w") as f:
        f.write(html)
    print("html -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())