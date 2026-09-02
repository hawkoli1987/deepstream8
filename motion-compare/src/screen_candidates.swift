// screen_candidates.swift — motion-profile screening tool for candidate clips. No external
// dependencies — decodes via AVFoundation/VideoToolbox. Usage:
//   swift -swift-version 5 src/screen_candidates.swift <dir-with-mp4s> [--frames]
// Samples ≤300 frames (~8 Hz) at ≤128px grey, computes frame-to-frame mean absdiff —
// raw AND global-shift-compensated (±6 px integer shift search) — derives quiet% and
// discrete events from the COMPENSATED series (a panning camera nulls out; scene motion
// can't), prints a table, writes motion_profile.json, and (with --frames) saves PNGs at
// t0 + a quiet frame + event midpoints.

import Foundation
import AVFoundation
import CoreGraphics
import ImageIO

guard CommandLine.arguments.count >= 2 else {
    FileHandle.standardError.write("usage: swift -swift-version 5 screen_candidates.swift <dir-with-mp4s> [--frames]\n".data(using: .utf8)!)
    exit(2)
}
let dirURL = URL(fileURLWithPath: CommandLine.arguments[1])
let saveFrames = CommandLine.arguments.contains("--frames")

func percentile(_ a: [Double], _ p: Double) -> Double {
    let s = a.sorted()
    let i = min(s.count - 1, max(0, Int(p * Double(s.count - 1))))
    return s[i]
}

func savePNG(_ cg: CGImage, _ path: URL) {
    guard let dest = CGImageDestinationCreateWithURL(path as CFURL, "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
}

/// Mean absdiff of b vs a after the best global (dx,dy) integer shift in ±maxShift.
/// A panning/dollying camera nulls out; true scene motion cannot.
func bestShiftDiff(_ a: [UInt8], _ b: [UInt8], w: Int, h: Int, maxShift: Int) -> (dx: Int, dy: Int, diff: Double) {
    var best = Double.greatestFiniteMagnitude, bdx = 0, bdy = 0
    for dy in -maxShift...maxShift {
        for dx in -maxShift...maxShift {
            var s = 0, n = 0
            let x0 = max(0, dx), x1 = min(w, w + dx)
            let y0 = max(0, dy), y1 = min(h, h + dy)
            for y in y0..<y1 {
                let ro = y * w, rn = (y - dy) * w
                for x in x0..<x1 {
                    s += abs(Int(a[ro + x]) - Int(b[rn + x - dx]))
                    n += 1
                }
            }
            let d = Double(s) / Double(n)
            if d < best { best = d; bdx = dx; bdy = dy }
        }
    }
    return (bdx, bdy, best)
}

let fm = FileManager.default
let urls = (try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil))
    .filter { $0.pathExtension.lowercased() == "mp4" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

if urls.isEmpty {
    FileHandle.standardError.write("no mp4 files in \(dirURL.path)\n".data(using: .utf8)!)
    exit(1)
}

var report: [[String: Any]] = []

for url in urls {
    let asset = AVURLAsset(url: url)
    let dur = CMTimeGetSeconds(try await asset.load(.duration))
    let trks = try await asset.load(.tracks)
    guard let trk = trks.first(where: { $0.mediaType == .video }) else {
        FileHandle.standardError.write("no video track: \(url.lastPathComponent)\n".data(using: .utf8)!)
        continue
    }
    let nat = try await trk.load(.naturalSize)

    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    gen.maximumSize = CGSize(width: 128, height: 128)
    gen.requestedTimeToleranceBefore = CMTime(seconds: 0.02, preferredTimescale: 600)
    gen.requestedTimeToleranceAfter = CMTime(seconds: 0.02, preferredTimescale: 600)

    let N = min(300, max(60, Int(dur * 8)))
    struct Sample { var raw: Double; var comp: Double; var dx: Int; var dy: Int }
    var prev: [UInt8]? = nil
    var diffs: [Sample] = []
    var thumb: CGImage? = nil

    for k in 0..<N {
        let t = CMTime(seconds: 0.04 + Double(k) / Double(N - 1) * (dur - 0.12), preferredTimescale: 600)
        do {
            let cg = try gen.copyCGImage(at: t, actualTime: nil)
            let w = cg.width, h = cg.height
            var buf = [UInt8](repeating: 0, count: w * h)
            let cs = CGColorSpaceCreateDeviceGray()
            guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w, space: cs,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                throw NSError(domain: "screen", code: 1)
            }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            if let pv = prev {
                var s = 0
                for i in 0..<buf.count { s += abs(Int(buf[i]) - Int(pv[i])) }
                let raw = Double(s) / Double(buf.count)
                let bs = bestShiftDiff(pv, buf, w: w, h: h, maxShift: 6)
                diffs.append(Sample(raw: raw, comp: bs.diff, dx: bs.dx, dy: bs.dy))
            } else {
                diffs.append(Sample(raw: 0, comp: 0, dx: 0, dy: 0))
                thumb = cg
            }
            prev = buf
        } catch {
            diffs.append(Sample(raw: -1, comp: -1, dx: 0, dy: 0))
        }
    }

    // ---- stats: events/quiet from the COMPENSATED series ----
    let raws = diffs.map { $0.raw }.filter { $0 >= 0 }
    let comps = diffs.map { $0.comp }.filter { $0 >= 0 }
    guard comps.count > 8, diffs[0].raw == 0 else {
        FileHandle.standardError.write("decode failed for \(url.lastPathComponent)\n".data(using: .utf8)!)
        continue
    }
    let rp50 = percentile(raws, 0.5), rp90 = percentile(raws, 0.9), rp99 = percentile(raws, 0.99)
    let p50 = percentile(comps, 0.5), p90 = percentile(comps, 0.9), p99 = percentile(comps, 0.99)
    let quietThr = max(1.5, p50 * 0.6, p90 * 0.35)
    let motionThr = max(3.0, p90 * 0.6)
    let shifts = diffs.map { Double(max(abs($0.dx), abs($0.dy))) }
    let shP90 = percentile(Array(shifts.dropFirst()), 0.9)
    var quietCount = 0
    var isMo: [Bool] = []
    for d in diffs {
        let m = d.comp >= motionThr
        isMo.append(m)
        if d.comp >= 0 && d.comp < quietThr { quietCount += 1 }
    }
    var events: [(s: Int, e: Int)] = []
    var cur: (s: Int, e: Int)? = nil
    for (i, m) in isMo.enumerated() {
        if m {
            if cur != nil { cur!.e = i } else { cur = (i, i) }
        } else if let c = cur { events.append(c); cur = nil }
    }
    if let c = cur { events.append(c) }
    events = events.filter { $0.e - $0.s >= 1 }
    let dt = (dur - 0.16) / Double(N - 1)
    func tOf(_ k: Int) -> Double { 0.04 + Double(k) * dt }
    let quietPct = Int((Double(quietCount) / Double(N) * 100).rounded())
    let verdict = (quietPct >= 30 && events.count >= 1 && events.count <= 6) ? "GOOD CANDIDATE" : "REJECT"

    print("== \(url.lastPathComponent)  \(Int(nat.width))x\(Int(nat.height))  \(String(format: "%.1f", dur))s  N=\(N)")
    print("   raw  p50=\(String(format: "%.2f", rp50)) p90=\(String(format: "%.2f", rp90)) p99=\(String(format: "%.2f", rp99))")
    print("   comp p50=\(String(format: "%.2f", p50)) p90=\(String(format: "%.2f", p90)) p99=\(String(format: "%.2f", p99))  shift_p90=\(String(format: "%.0f", shP90))px")
    print("   quiet=\(quietPct)%  events=\(events.count) [\(events.map { "\(String(format: "%.1f", tOf($0.s)))-\(String(format: "%.1f", tOf($0.e + 1)))s" }.joined(separator: ", "))]  → \(verdict)")

    if saveFrames {
        let vgen = AVAssetImageGenerator(asset: asset)
        vgen.appliesPreferredTrackTransform = true
        vgen.maximumSize = CGSize(width: 640, height: 640)
        vgen.requestedTimeToleranceBefore = CMTime(seconds: 0.05, preferredTimescale: 600)
        vgen.requestedTimeToleranceAfter = CMTime(seconds: 0.05, preferredTimescale: 600)
        var times: [(String, Double)] = [("t0", 0.1)]
        let quietIdx = diffs.firstIndex { $0.comp >= 0 && $0.comp < quietThr && $0.comp > 0 }
        if let qi = quietIdx { times.append(("quiet", tOf(qi))) }
        for (n, ev) in events.prefix(3).enumerated() {
            times.append(("ev\(n + 1)", tOf((ev.s + ev.e) / 2)))
        }
        for (tag, t) in times {
            if let cg = try? vgen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) {
                savePNG(cg, dirURL.appendingPathComponent("\(url.deletingPathExtension().lastPathComponent)_\(tag).png"))
            }
        }
    }

    report.append([
        "file": url.lastPathComponent,
        "res": "\(Int(nat.width))x\(Int(nat.height))",
        "dur_s": Double(String(format: "%.2f", dur))!,
        "samples": N,
        "raw_p50": Double(String(format: "%.3f", rp50))!, "raw_p90": Double(String(format: "%.3f", rp90))!,
        "comp_p50": Double(String(format: "%.3f", p50))!, "comp_p90": Double(String(format: "%.3f", p90))!,
        "quiet_pct": quietPct,
        "events": events.count,
        "event_times_s": events.map { [Double(String(format: "%.1f", tOf($0.s)))!, Double(String(format: "%.1f", tOf($0.e + 1)))!] },
        "verdict": verdict.hasPrefix("GOOD") ? "GOOD" : "REJECT",
    ])
}

let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try data.write(to: dirURL.appendingPathComponent("motion_profile.json"))
print("\nwrote \(dirURL.appendingPathComponent("motion_profile.json").path)")
