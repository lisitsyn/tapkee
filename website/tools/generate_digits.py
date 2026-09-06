#!/usr/bin/env python3
"""Generates the synthetic handwritten-digit dataset used by the "digits"
graphical example on the website (a replacement for real MNIST images: not
a license problem like MIT-CBCL, but MNIST isn't bundled in this repo either,
and regenerating from it would mean downloading the dataset from a third-party
mirror at every build -- an external, network-dependent build step this repo
otherwise doesn't have. This generates original 28x28 digit-like glyphs
instead, exercising the exact same demo mechanics: real pixel data, a real
t-SNE embedding computed by the tapkee CLI, hover-to-preview thumbnails).

Each glyph is a jittered seven-segment digit (0-9), rasterized from scratch
with the Python standard library only (no Pillow/ImageMagick), the same
approach as generate_synthfaces.py. Computing the plot layout requires the
tapkee CLI binary (a plain native build, see website/Makefile's
sync-digits target), which is run to produce a t-SNE embedding of the
generated images.

Usage:
    python3 website/tools/generate_digits.py \
        --tapkee-cli bin/tapkee \
        --img-dir website/resources/public/img/digits \
        --out-json website/resources/public/data/digits.json
"""
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib

IMAGE_SIZE = 28
SUPERSAMPLE = 4
SAMPLES_PER_DIGIT = 200  # 10 digits x 200 = 2000, matching the original demo

BG = (0, 0, 0, 0)
INK = (255, 255, 255, 255)

# Seven-segment layout on a unit square (x, y in [0, 1], y grows downward).
SEG_POINTS = {
    "a": ((0.22, 0.08), (0.78, 0.08)),
    "b": ((0.78, 0.08), (0.78, 0.50)),
    "c": ((0.78, 0.50), (0.78, 0.92)),
    "d": ((0.22, 0.92), (0.78, 0.92)),
    "e": ((0.22, 0.50), (0.22, 0.92)),
    "f": ((0.22, 0.08), (0.22, 0.50)),
    "g": ((0.22, 0.50), (0.78, 0.50)),
}
DIGIT_SEGMENTS = {
    0: "abcdef",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abcdfg",
}


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

    def stroke_dots(self, points, width, color):
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
        scanlines.append(0)
        scanlines.extend(raw[y * stride:(y + 1) * stride])

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def line_points(p0, p1, n):
    return [(p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n)
            for i in range(n + 1)]


def render_digit(digit, rng, size=IMAGE_SIZE, supersample=SUPERSAMPLE):
    hi = size * supersample
    c = Canvas(hi, hi)
    s = supersample
    cx, cy = hi / 2, hi / 2

    angle = rng.uniform(-0.09, 0.09)
    scale = rng.uniform(0.90, 1.02) * hi * 0.82
    shift_x = rng.uniform(-0.02, 0.02) * hi
    shift_y = rng.uniform(-0.02, 0.02) * hi
    stroke_w = rng.uniform(0.12, 0.15) * size * s
    jitter = 0.015

    def to_canvas(px, py):
        x = (px - 0.5) + rng.uniform(-jitter, jitter)
        y = (py - 0.5) + rng.uniform(-jitter, jitter)
        ca, sa = math.cos(angle), math.sin(angle)
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        return (cx + shift_x + rx * scale, cy + shift_y + ry * scale)

    for seg in DIGIT_SEGMENTS[digit]:
        p0, p1 = SEG_POINTS[seg]
        a, b = to_canvas(*p0), to_canvas(*p1)
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(dist / (size * s * 0.03)))
        pts = line_points(a, b, n)
        ink = (INK[0], INK[1], INK[2], rng.randint(210, 255))
        c.stroke_dots(pts, stroke_w, ink)

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


def alpha_vector(canvas):
    """Flatten to the ink-intensity channel used as the pixel feature."""
    return [px[3] / 255.0 for px in canvas.px]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapkee-cli", default="bin/tapkee")
    parser.add_argument("--img-dir", default="website/resources/public/img/digits")
    parser.add_argument("--out-json", default="website/resources/public/data/digits.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.tapkee_cli):
        sys.exit(f"tapkee CLI not found at {args.tapkee_cli}")

    os.makedirs(args.img_dir, exist_ok=True)
    rng = random.Random(args.seed)

    fnames, vectors = [], []
    idx = 0
    for digit in range(10):
        for _ in range(SAMPLES_PER_DIGIT):
            canvas = render_digit(digit, rng)
            fname = f"digit_{idx:04d}.png"
            write_png(os.path.join(args.img_dir, fname), canvas)
            vectors.append(alpha_vector(canvas))
            fnames.append(fname)
            idx += 1

    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "input.dat")
        output_file = os.path.join(tmp, "output.dat")
        with open(input_file, "w") as f:
            for vec in vectors:
                f.write(",".join(str(v) for v in vec) + "\n")

        subprocess.run(
            [args.tapkee_cli, "-i", input_file, "-o", output_file,
             "-m", "t-sne", "-d", ","],
            check=True,
        )

        with open(output_file) as f:
            embedding = [[float(x) for x in line.split(",")] for line in f if line.strip()]

    data = [{"cx": embedding[i][0], "cy": embedding[i][1], "fname": fnames[i]}
            for i in range(len(fnames))]

    with open(args.out_json, "w") as f:
        json.dump({"data": data}, f)

    print(f"Wrote {len(data)} digits to {args.img_dir} and {args.out_json}")


if __name__ == "__main__":
    main()
