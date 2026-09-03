import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import CoreText

// contact_sheet.swift <video> <outPNG> <fps> <nframes> [cols] [tileW] [f0] [f1]
//   Renders frames f0..f1-1 (default 0..nframes) sampled at t=(f+0.5)/fps into one
//   numbered grid PNG. Tile aspect follows the source. Frame index is burned into
//   each tile (top-left, yellow on black).
let a = CommandLine.arguments
guard a.count >= 5 else {
    FileHandle.standardError.write("usage: contact_sheet.swift <video> <outPNG> <fps> <nframes> [cols] [tileW] [f0] [f1]\n".data(using: .utf8)!)
    exit(2)
}
let videoPath = a[1]
let outPath = a[2]
let fps = Double(a[3])!
let nframes = Int(a[4])!
let cols = a.count > 5 ? Int(a[5])! : 8
let tileW = a.count > 6 ? Int(a[6])! : 360
let f0 = a.count > 7 ? Int(a[7])! : 0
let f1 = a.count > 8 ? Int(a[8])! : nframes

let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
guard let track = asset.tracks(withMediaType: .video).first else { exit(3) }
let natSize = track.naturalSize.applying(track.preferredTransform)
let srcW = abs(natSize.width), srcH = abs(natSize.height)
let tileH = Int((Double(tileW) * srcH / srcW).rounded())

let gen = AVAssetImageGenerator(asset: asset)
gen.appliesPreferredTrackTransform = true
gen.maximumSize = CGSize(width: tileW, height: tileH)
gen.requestedTimeToleranceBefore = CMTime(seconds: 0.02, preferredTimescale: 600)
gen.requestedTimeToleranceAfter = CMTime(seconds: 0.02, preferredTimescale: 600)

let count = f1 - f0
let rows = (count + cols - 1) / cols
let pad = 2
let W = cols * (tileW + pad) + pad
let H = rows * (tileH + pad) + pad

let cs = CGColorSpace(name: CGColorSpace.sRGB)!
guard let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                          space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { exit(4) }
ctx.setFillColor(CGColor(red: 0.05, green: 0.05, blue: 0.05, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))

let font = CTFontCreateWithName("Menlo-Bold" as CFString, 18, nil)

for i in 0..<count {
    let f = f0 + i
    let t = (Double(f) + 0.5) / fps
    guard let cg = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) else {
        FileHandle.standardError.write("miss frame \(f)\n".data(using: .utf8)!)
        continue
    }
    let cx = i % cols
    let cy = i / cols
    let x = pad + cx * (tileW + pad)
    // top-left origin: flip row
    let y = H - (pad + (cy + 1) * tileH + cy * pad)
    ctx.draw(cg, in: CGRect(x: x, y: y, width: tileW, height: tileH))

    // label
    let label = "\(f)" as CFString
    let attrs: [CFString: Any] = [
        kCTFontAttributeName: font,
        kCTForegroundColorAttributeName: CGColor(red: 1, green: 0.9, blue: 0.1, alpha: 1)
    ]
    let astr = CFAttributedStringCreate(nil, label, attrs as CFDictionary)!
    let line = CTLineCreateWithAttributedString(astr)
    let lb = CTLineGetImageBounds(line, ctx)
    let bx = CGFloat(x) + 3
    let by = CGFloat(y + tileH) - lb.height - 5
    ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.75))
    ctx.fill(CGRect(x: bx - 2, y: by - 2, width: lb.width + 6, height: lb.height + 6))
    ctx.textPosition = CGPoint(x: bx, y: by)
    CTLineDraw(line, ctx)
}

guard let img = ctx.makeImage(),
      let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outPath) as CFURL, "public.png" as CFString, 1, nil)
else { exit(5) }
CGImageDestinationAddImage(dest, img, nil)
CGImageDestinationFinalize(dest)
print("saved \(outPath)  (\(W)x\(H), \(count) frames, tile \(tileW)x\(tileH))")
