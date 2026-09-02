#!/usr/bin/env python3
"""Render the PGIE (nvinfer) config for one batch size."""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "pgie_yolo26m.txt.tmpl")
PARSER_LIB = "/work/throughput/lib/libnvdsinfer_custom_impl_Yolo26E2e.so"
LABELS = "/work/throughput/cfg/labels.txt"


def render(txt, vals):
    for k, v in vals.items():
        txt = txt.replace(k, v)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--parse-func", default="NvDsInferParseYolo26E2e")
    ap.add_argument("--parser-lib", default=PARSER_LIB)
    ap.add_argument("--cluster-mode", type=int, default=4)
    args = ap.parse_args()

    with open(args.template) as f:
        txt = f.read()
    vals = {
        "@ONNX@": args.onnx,
        "@ENGINE@": args.engine,
        "@BATCH@": str(args.batch),
        "@LABELS@": args.labels,
        "@PARSE_FUNC@": args.parse_func,
        "@PARSER_LIB@": args.parser_lib,
        "@CLUSTER_MODE@": str(args.cluster_mode),
    }
    txt = render(txt, vals)
    with open(args.out, "w") as f:
        f.write(txt)
    print("pgie config -> %s (batch %d, %s, cluster-mode %d)"
          % (args.out, args.batch, args.parse_func, args.cluster_mode))


if __name__ == "__main__":
    main()
