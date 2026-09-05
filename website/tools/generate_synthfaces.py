#!/usr/bin/env python3
"""Generates the synthetic faces dataset used by the "synthfaces" graphical
example on the website (an original, procedurally-drawn replacement for the
MIT-CBCL face database, whose license forbids redistributing its images).

Each face is a simple cartoon head rasterized from two continuous parameters
-- turn and expression -- giving pixel data with a genuine 2D intrinsic
manifold. Rendering and PNG encoding are done from scratch with the Python
standard library only (no Pillow/ImageMagick) so the script has no extra
dependencies. Computing the plot layout requires the tapkee CLI binary
(built with BUILD_EXAMPLES=ON), which is run to produce an LTSA embedding of
the generated images.

Usage:
    python3 website/tools/generate_synthfaces.py \
        --tapkee-cli bin/tapkee \
        --img-dir website/resources/public/img/synthfaces \
        --out-json website/resources/public/data/synthfaces.json
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib

IMAGE_SIZE = 40
SUPERSAMPLE = 4
GRID_N = 18  # 18 x 18 = 324 faces, matching the size of the original demo

BG = (0, 0, 0, 0)
SKIN = (240, 192, 144, 255)
OUTLINE = (138, 90, 48, 255)
EYE = (42, 42, 42, 255)
NOSE = (200, 136, 96, 255)
MOUTH = (122, 48, 48, 255)


class Canvas:
    def __init__(self, w, h, fill=BG):
        self.w, self.h = w, h
        self.px = [list(fill) for _ in range(w * h)]

    def blend(self, x, y, color):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        r, g, b, a = color
        if a <= 0:
            return
        dst = self.px[y * self.w + x]
        a /= 255.0
        inv = 1 - a
        dst[0] = r * a + dst[0] * inv
        dst[1] = g * a + dst[1] * inv
        dst[2] = b * a + dst[2] * inv
        dst[3] = a * 255 + dst[3] * inv

    def fill_ellipse(self, cx, cy, rx, ry, color, samples=SUPERSAMPLE):
        rx = max(rx, 0.5)
        ry = max(ry, 0.5)
        x0, x1 = int(cx - rx - 1), int(cx + rx + 1)
        y0, y1 = int(cy - ry - 1), int(cy + ry + 1)
        step = 1.0 / samples
        for y in range(max(y0, 0), min(y1, self.h - 1) + 1):
            for x in range(max(x0, 0), min(x1, self.w - 1) + 1):
                hits = 0
                for sy in range(samples):
                    fy = y + (sy + 0.5) * step
                    dy = (fy - cy) / ry
                    for sx in range(samples):
                        fx = x + (sx + 0.5) * step
                        dx = (fx - cx) / rx
                        if dx * dx + dy * dy <= 1.0:
                            hits += 1
                if hits:
                    coverage = hits / (samples * samples)
                    r, g, b, a = color
                    self.blend(x, y, (r, g, b, a * coverage))

    def stroke_dot_path(self, points, width, color):
        r = width / 2.0
        for (px, py) in points:
            self.fill_ellipse(px, py, r, r, color)

    def to_bytes_rgba8(self):
        out = bytearray(self.w * self.h * 4)
        i = 0
        for r, g, b, a in self.px:
            out[i] = max(0, min(255, round(r)))
            out[i + 1] = max(0, min(255, round(g)))
            out[i + 2] = max(0, min(255, round(b)))
            out[i + 3] = max(0, min(255, round(a)))
            i += 4
        return bytes(out)


def write_png(path, canvas):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    w, h = canvas.w, canvas.h
    raw = canvas.to_bytes_rgba8()
    stride = w * 4
    scanlines = bytearray()
    for y in range(h):
        scanlines.append(0)  # filter type: None
        scanlines.extend(raw[y * stride:(y + 1) * stride])

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def quadratic_bezier_points(p0, p1, p2, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def render_face(turn, expr, size=IMAGE_SIZE, supersample=SUPERSAMPLE):
    """turn, expr in [-1, 1]: head turn amount and mouth expression."""
    hi = size * supersample
    c = Canvas(hi, hi)
    s = supersample
    cx, cy = hi / 2, hi / 2

    base_rx = size * 0.32 * s
    base_ry = size * 0.40 * s
    head_rx = base_rx * (1 - 0.35 * abs(turn))
    head_shift = turn * size * 0.16 * s

    c.fill_ellipse(cx + head_shift * 0.3, cy, head_rx, base_ry, SKIN)

    eye_dx = size * 0.11 * s * (1 - 0.45 * abs(turn))
    eye_y = cy - size * 0.07 * s
    eye_r = size * (0.032 + 0.014 * expr) * s
    c.fill_ellipse(cx + head_shift - eye_dx, eye_y, eye_r, eye_r, EYE)
    c.fill_ellipse(cx + head_shift + eye_dx, eye_y, eye_r, eye_r, EYE)

    brow_y = eye_y - size * (0.09 + 0.05 * expr) * s
    brow_w = eye_r * 1.6
    brow_tilt = expr * size * 0.045 * s
    for side in (-1, 1):
        bx = cx + head_shift + side * eye_dx
        pts = quadratic_bezier_points(
            (bx - brow_w, brow_y + side * brow_tilt * 0),
            (bx, brow_y - brow_tilt),
            (bx + brow_w, brow_y + side * brow_tilt * 0),
            n=10,
        )
        c.stroke_dot_path(pts, width=size * 0.028 * s, color=EYE)

    nose_x = cx + head_shift + turn * size * 0.05 * s
    nose_y = cy + size * 0.02 * s
    c.fill_ellipse(nose_x, nose_y, size * 0.025 * s, size * 0.025 * s, NOSE)

    mouth_y = cy + size * 0.17 * s
    mouth_w = size * (0.12 + 0.05 * expr) * s
    p0 = (cx + head_shift - mouth_w, mouth_y)
    p2 = (cx + head_shift + mouth_w, mouth_y)
    p1 = (cx + head_shift, mouth_y + expr * size * 0.22 * s)
    pts = quadratic_bezier_points(p0, p1, p2, n=36)
    c.stroke_dot_path(pts, width=size * 0.075 * s, color=MOUTH)

    if supersample == 1:
        return c

    out = Canvas(size, size)
    for y in range(size):
        for x in range(size):
            r = g = b = a = 0.0
            for sy in range(s):
                for sx in range(s):
                    px = c.px[(y * s + sy) * hi + (x * s + sx)]
                    r += px[0]
                    g += px[1]
                    b += px[2]
                    a += px[3]
            n = s * s
            out.px[y * size + x] = [r / n, g / n, b / n, a / n]
    return out


def read_grayscale_vector(canvas):
    """Flatten onto a white background and take luminance, for LTSA input."""
    vec = []
    for r, g, b, a in canvas.px:
        af = a / 255.0
        rr = r * af + 255 * (1 - af)
        gg = g * af + 255 * (1 - af)
        bb = b * af + 255 * (1 - af)
        vec.append(0.299 * rr + 0.587 * gg + 0.114 * bb)
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapkee-cli", default="bin/tapkee")
    parser.add_argument("--img-dir", default="website/resources/public/img/synthfaces")
    parser.add_argument("--out-json", default="website/resources/public/data/synthfaces.json")
    parser.add_argument("--neighbors", type=int, default=12)
    args = parser.parse_args()

    if not os.path.exists(args.tapkee_cli):
        sys.exit(f"tapkee CLI not found at {args.tapkee_cli}")

    os.makedirs(args.img_dir, exist_ok=True)

    turns = [-1 + 2 * i / (GRID_N - 1) for i in range(GRID_N)]
    exprs = [-1 + 2 * i / (GRID_N - 1) for i in range(GRID_N)]

    fnames, vectors = [], []
    for ti, turn in enumerate(turns):
        for ei, expr in enumerate(exprs):
            fname = f"face_{ti:02d}_{ei:02d}.png"
            canvas = render_face(turn, expr)
            write_png(os.path.join(args.img_dir, fname), canvas)
            vectors.append(read_grayscale_vector(canvas))
            fnames.append(fname)

    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "input.dat")
        output_file = os.path.join(tmp, "output.dat")
        with open(input_file, "w") as f:
            for vec in vectors:
                f.write(",".join(str(v) for v in vec) + "\n")

        subprocess.run(
            [args.tapkee_cli, "-i", input_file, "-o", output_file,
             "-m", "ltsa", "-k", str(args.neighbors), "-d", ","],
            check=True,
        )

        with open(output_file) as f:
            embedding = [[float(x) for x in line.split(",")] for line in f if line.strip()]

    data = [{"cx": embedding[i][0], "cy": embedding[i][1], "fname": fnames[i]}
            for i in range(len(fnames))]

    with open(args.out_json, "w") as f:
        json.dump({"data": data}, f)

    print(f"Wrote {len(data)} faces to {args.img_dir} and {args.out_json}")


if __name__ == "__main__":
    main()
