#!/usr/bin/env python3
"""Build the throughput-test NvDCF tracker ll-config.

Primary path: patch the in-container preset config_tracker_NvDCF_perf.yml
(the three locked keys already match this preset). Fallback: render
templates/tracker_NvDCF_yolo26.yml.tmpl. The used path is printed.
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAINER_PRESET = ("/opt/nvidia/deepstream/deepstream/samples/configs/"
                    "deepstream-app/config_tracker_NvDCF_perf.yml")


def patch(text, overrides):
    """Patch top-level keys only (flush-left lines), preserving structure.

    Matching is done on the un-stripped line so nested same-name keys are
    never touched; replacements keep their original indentation. Keys absent
    from the source are inserted right after the '---' document start so they
    can never land inside a nested block and break the map."""
    out = []
    seen = set()
    pending = dict(overrides)
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        hit = None
        for key in list(pending):
            if re.match(rf"^{re.escape(key)}\s*:", line):
                hit = key
                break
        if hit:
            out.append(f"{indent}{hit}: {pending.pop(hit)}")
            seen.add(hit)
        else:
            out.append(line)
    if pending:
        placed = []
        rest = []
        for line in out:
            placed.append(line)
            if re.match(r"^---\s*$", line):
                for key in sorted(pending):
                    placed.append(f"{key}: {pending[key]}")
                pending.clear()
                break
        rest = out[len(placed):]
        for key in sorted(pending):
            placed.append(f"{key}: {pending[key]}")
        out = placed + rest
    return "\n".join(out) + "\n", seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=CONTAINER_PRESET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--visual-type", type=int, choices=[1, 2], default=2)
    ap.add_argument("--max-shadow", type=int, default=51)
    ap.add_argument("--max-targets", type=int, default=150)
    ap.add_argument("--template",
                    default=os.path.join(HERE, "templates", "tracker_NvDCF_yolo26.yml.tmpl"))
    args = ap.parse_args()

    overrides = {
        "maxTargetsPerStream": str(args.max_targets),
        "maxShadowTrackingAge": str(args.max_shadow),
        "visualTrackerType": str(args.visual_type),
    }
    src = None
    used = "no source"
    if os.path.isfile(args.source):
        with open(args.source) as f:
            src = f.read()
        used = "patched preset " + args.source
    elif os.path.isfile(args.template):
        with open(args.template) as yml_f:
            src = yml_f.read()
        used = "rendered fallback template"
    else:
        raise SystemExit("tracker yml: no preset at %s and no fallback template" % args.source)

    txt, seen = patch(src, overrides)
    with open(args.out, "w") as f:
        f.write(txt)
    print("tracker yml -> %s (%s; set %s)" % (args.out, used, ",".join(sorted(seen)) or "appended"))


if __name__ == "__main__":
    main()
