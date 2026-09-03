#!/usr/bin/env python3
"""
build_study_data.py — assemble the NvOF stillness-study dataset from the v3 run.

Input : runs/street/scores/motion_scores.json  (the nvof lane only)
Output: runs/nvof-study/study_data.json

The study compares score normalizations x threshold rules for detecting truly-static
contiguous segments (so a pipeline can throttle framerate during them), under an
asymmetric cost: a motion frame labelled STATIC is BAD; a static frame labelled
MOTION is a cheap missed OPPortunity.

stdlib only (the Mac has no numpy). All aggregation / normalization / thresholding
maths is re-done in the browser; this script also runs it once (see `verify()`)
to print a report and to embed a `reference` block the HTML asserts against.
"""
import sys, os, json, base64, math, datetime

SRC = "runs/street/scores/motion_scores.json"
OUT = "runs/nvof-study/study_data.json"

# ---- ground truth: hand labels from the contact sheets (1 = motion) ---------
# runs/nvof-study/qa/{sheet_A,sheet_B,strip_*}.png  — adjudicated 2026-09-02.
EVENTS = [
    dict(id="E1", f0=0,   f1=8,   name="pedestrian A",
         note="hooded figure crosses the foreground L->R, occluding most of the frame"),
    dict(id="E2", f0=40,  f1=72,  name="car + pedestrian B",
         note="white car(s) transit the intersection (40-58); a second pedestrian "
              "crosses close (59-68); trailing cars (69-72). One contiguous motion run."),
    dict(id="E3", f0=83,  f1=102, name="teal hatchback",
         note="a dark-teal hatchback drives left across the whole frame"),
    dict(id="E4", f0=155, f1=166, name="approaching headlights",
         note="a car approaches head-on through fog; headlights brighten and enlarge "
              "but lateral optical flow stays near the fog noise floor"),
]
LABELS_NOTE = (
    "Per-frame motion/static labelled by eye from numbered contact sheets, not from the "
    "detector. Four motion events (E1-E4); everything else static. Event boundaries are "
    "soft by +/-1-2 frames. E2 is compound (car, then close pedestrian, then trailing "
    "cars) but one unbroken motion run. Frames 119-131 - a provisional heat-inferred "
    "window - hold a STATIONARY parked car, a standing figure and traffic-light colour "
    "changes plus fog drift: no translational motion, NvOF sits at 0.08-0.35 raw px "
    "(noise floor) -> labelled static. E4 is a real vehicle whose frame-aggregate flow "
    "barely clears the fog noise: affine/raw representations only catch it at a near-zero "
    "cut, but rank normalization lifts its frames to ~p85 so Otsu catches it cleanly - a "
    "case where the normalization choice decides what is detectable at all."
)


def pctl(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_nvof(src):
    d = json.load(open(src))
    nv = [a for a in d["algorithms"] if a["id"] == "nvof"][0]
    F = d["video"]["frame_count"]
    rows, cols = d["grid"]["rows"], d["grid"]["cols"]
    NC = rows * cols
    hb = base64.b64decode(nv["heat_b64"])
    assert len(hb) == F * NC, (len(hb), F * NC)
    p99 = nv["heat_p99"]
    cells = [[hb[f * NC + i] / 255.0 * p99 for i in range(NC)] for f in range(F)]
    return d, nv, F, rows, cols, NC, cells


# ---- the study maths (mirrored in JS) --------------------------------------
def aggregate(cells, valid, kind, raw_px=None):
    """mean uses the canonical full-res frame-mean (raw_px, = v3's series);
    max / top4 use the 40x22 pooled heat grid (all that survives in the JSON)."""
    out = [None] * len(cells)
    for f, ok in enumerate(valid):
        if not ok:
            continue
        c = cells[f]
        if kind == "mean":
            out[f] = raw_px[f] if raw_px is not None else sum(c) / len(c)
        elif kind == "max":
            out[f] = max(c)
        elif kind == "top4":
            out[f] = sum(sorted(c)[-4:]) / 4.0
    return out


def percell_aggregate(cells, valid, NC, kind):
    """Normalize every cell by its own temporal p99 over valid frames, then aggregate."""
    vidx = [f for f, ok in enumerate(valid) if ok]
    c99 = []
    for i in range(NC):
        col = [cells[f][i] for f in vidx]
        p = pctl(col, 0.99) or 1e-12
        c99.append(p)
    out = [None] * len(cells)
    for f in vidx:
        c = [min(1.0, cells[f][i] / c99[i]) for i in range(NC)]
        if kind == "mean":
            out[f] = sum(c) / NC
        elif kind == "max":
            out[f] = max(c)
        elif kind == "top4":
            out[f] = sum(sorted(c)[-4:]) / 4.0
    return out


def transform(series, valid, kind):
    """Monotone/rank transform of an aggregate series (pre display-scaling)."""
    vidx = [f for f, ok in enumerate(valid) if ok]
    v = [series[f] for f in vidx]
    out = [None] * len(series)
    if kind in ("identity", "minmax", "z", "robustz"):
        # all affine -> identical after display-scaling; keep identity
        for f in vidx:
            out[f] = series[f]
    elif kind in ("log1p", "log1p_robustz"):
        for f in vidx:
            out[f] = math.log1p(series[f])
    elif kind == "ecdf":
        sv = sorted(v)
        n = len(sv)
        for f in vidx:
            # fraction of valid frames <= this value
            lo = 0; hi = n
            x = series[f]
            # rank by count <=
            cnt = sum(1 for y in v if y <= x)
            out[f] = (cnt - 1) / (n - 1) if n > 1 else 0.0
    else:
        raise ValueError(kind)
    return out


def display_scale(series, valid):
    """clip((x - p1) / (p99 - p1), 0, 1) over valid frames — the v3 convention."""
    vidx = [f for f, ok in enumerate(valid) if ok]
    v = [series[f] for f in vidx]
    lo, hi = pctl(v, 0.01), pctl(v, 0.99)
    if hi <= lo:
        hi = lo + 1e-12
    S = [None] * len(series)
    for f in vidx:
        S[f] = min(1.0, max(0.0, (series[f] - lo) / (hi - lo)))
    return S, lo, hi


def otsu(S, valid, bins=256):
    vidx = [f for f, ok in enumerate(valid) if ok]
    xs = [S[f] for f in vidx]
    h = [0] * bins
    for x in xs:
        h[min(bins - 1, int(x * bins))] += 1
    tot = len(xs)
    p = [x / tot for x in h]
    centres = [(i + 0.5) / bins for i in range(bins)]
    w0 = cum = 0.0
    muT = sum(p[i] * centres[i] for i in range(bins))
    best_k, best_v = 0, -1.0
    for k in range(bins):
        w0 += p[k]; cum += p[k] * centres[k]
        w1 = 1 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        sb = (muT * w0 - cum) ** 2 / (w0 * w1)
        if sb > best_v:
            best_v, best_k = sb, k
    var = sum(p[i] * (centres[i] - muT) ** 2 for i in range(bins))
    eta = best_v / var if var > 0 else 0.0
    return centres[best_k], eta


def apply_threshold(S, valid, thr):
    return {f: (S[f] >= thr) for f, ok in enumerate(valid) if ok}


def score(pred, labels, valid, cbad=10, copp=1):
    vidx = [f for f, ok in enumerate(valid) if ok]
    bad = sum(1 for f in vidx if labels[f] == 1 and not pred[f])
    opp = sum(1 for f in vidx if labels[f] == 0 and pred[f])
    run = mx = 0
    for f in vidx:
        if labels[f] == 1 and not pred[f]:
            run += 1; mx = max(mx, run)
        else:
            run = 0
    # static segments (contiguous throttle spans) over valid frames
    segs = 0; instatic = False
    for f in vidx:
        s = not pred[f]
        if s and not instatic:
            segs += 1
        instatic = s
    duty = sum(1 for f in vidx if not pred[f]) / len(vidx)
    return dict(bad=bad, opp=opp, longest_bad=mx, segments=segs,
                duty=round(duty, 4), cost=cbad * bad + copp * opp)


def cost_opt_cut(S, valid, labels, cbad=10, copp=1):
    vidx = [f for f, ok in enumerate(valid) if ok]
    cand = sorted(set(S[f] for f in vidx))
    cuts = [0.0] + [(cand[i] + cand[i + 1]) / 2 for i in range(len(cand) - 1)] + [1.0001]
    best_t, best_c = 0.0, None
    for t in cuts:
        m = score(apply_threshold(S, valid, t), labels, valid, cbad, copp)
        if best_c is None or m["cost"] < best_c:
            best_c, best_t = m["cost"], t
    return best_t


def event_recall(pred, valid):
    rec = []
    for e in EVENTS:
        hit = any(valid[f] and pred.get(f, False) for f in range(e["f0"], e["f1"] + 1))
        rec.append(dict(id=e["id"], name=e["name"], hit=bool(hit)))
    return rec


NORMS = ["identity", "minmax", "z", "robustz", "log1p", "log1p_robustz", "ecdf", "percell"]
AGGS = ["mean", "max", "top4"]


def build_reference(cells, valid, NC, labels, v3_norm, v3_thr, raw_px):
    """Run every agg x norm under Otsu and cost-opt; return a compact table."""
    ref = dict(v3=None, grid=[])
    v3pred = {f: (v3_norm[f] >= v3_thr) for f, ok in enumerate(valid) if ok}
    ref["v3"] = dict(score(v3pred, labels, valid),
                     recall=event_recall(v3pred, valid), threshold=v3_thr)
    for ag in AGGS:
        base = aggregate(cells, valid, ag, raw_px)
        for nm in NORMS:
            if nm == "percell":
                series = percell_aggregate(cells, valid, NC, ag)
                S, lo, hi = display_scale(series, valid)
            else:
                series = transform(base, valid, nm)
                S, lo, hi = display_scale(series, valid)
            t_otsu, eta = otsu(S, valid)
            m_otsu = score(apply_threshold(S, valid, t_otsu), labels, valid)
            t_co = cost_opt_cut(S, valid, labels)
            m_co = score(apply_threshold(S, valid, t_co), labels, valid)
            ref["grid"].append(dict(
                agg=ag, norm=nm,
                otsu=dict(threshold=round(t_otsu, 4), eta=round(eta, 3), **m_otsu,
                          recall=event_recall(apply_threshold(S, valid, t_otsu), valid)),
                cost_opt=dict(threshold=round(t_co, 4), **m_co,
                              recall=event_recall(apply_threshold(S, valid, t_co), valid)),
            ))
    return ref


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, SRC)
    out = os.path.join(root, OUT)
    d, nv, F, rows, cols, NC, cells = load_nvof(src)

    labels = [0] * F
    for e in EVENTS:
        for i in range(e["f0"], e["f1"] + 1):
            labels[i] = 1
    valid = nv["valid"]

    reference = build_reference(cells, valid, NC, labels,
                                {f: nv["norm"][f] for f in range(F)}, nv["threshold"],
                                nv["extra"]["raw_px"])

    doc = dict(
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        source_json=SRC,
        objective=("Detect truly-static contiguous segments so a pipeline can throttle "
                   "framerate during them. Asymmetric cost: a motion frame -> STATIC is "
                   "BAD (event dropped); a static frame -> MOTION is a cheap missed "
                   "throttling OPPortunity."),
        video=dict(file=d["video"]["file"], width=d["video"]["width"],
                   height=d["video"]["height"], fps=d["video"]["fps"],
                   frame_count=F, duration_s=d["video"]["duration_s"],
                   diagonal_px=d["video"]["diagonal_px"],
                   source_note=d["video"]["source_note"]),
        grid=dict(rows=rows, cols=cols),
        nvof=dict(
            raw=nv["raw"],
            raw_px=nv["extra"]["raw_px"],
            valid=valid,
            heat_b64=nv["heat_b64"],
            heat_p99=nv["heat_p99"],
            v3=dict(norm=nv["norm"], threshold=nv["threshold"],
                    method=nv["threshold_method"], eta=nv["threshold_separability"],
                    norm_constants=nv["norm_constants"]),
        ),
        labels=labels,
        labels_note=LABELS_NOTE,
        events=EVENTS,
        norms=NORMS,
        aggs=AGGS,
        reference=reference,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(doc, open(out, "w"))
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    verify(reference, labels, valid)


def verify(ref, labels, valid):
    nmot = sum(labels[f] for f, ok in enumerate(valid) if ok)
    nval = sum(1 for ok in valid if ok)
    print(f"\nGROUND TRUTH: {nval} valid frames, {nmot} motion / {nval - nmot} static "
          f"(duty ceiling {(nval - nmot) / nval:.2f})")
    v = ref["v3"]
    print(f"\nv3 operating point (mean/affine/Otsu thr={v['threshold']:.3f}): "
          f"BAD={v['bad']} OPP={v['opp']} longestBAD={v['longest_bad']} "
          f"duty={v['duty']:.2f} cost={v['cost']}  recall={[r['id'] for r in v['recall'] if r['hit']]}")
    print(f"\n{'agg':5s} {'norm':13s} | {'Otsu: thr':>9s} BAD OPP lBAD duty cost | "
          f"{'cost-opt thr':>12s} BAD OPP lBAD duty cost")
    for row in ref["grid"]:
        o, c = row["otsu"], row["cost_opt"]
        print(f"{row['agg']:5s} {row['norm']:13s} | {o['threshold']:9.3f} "
              f"{o['bad']:3d} {o['opp']:3d} {o['longest_bad']:4d} {o['duty']:.2f} {o['cost']:4d} | "
              f"{c['threshold']:12.3f} {c['bad']:3d} {c['opp']:3d} {c['longest_bad']:4d} "
              f"{c['duty']:.2f} {c['cost']:4d}")


if __name__ == "__main__":
    main()
