"""Score the tensors capture_sr_tensors.sh pulled off the device (that script was removed from
examples/deployment on 2026-10-06; recover it from git history to capture new tensors).

The screen recording could not answer "is the model sharper than the anchor".
Both panes go through an H.264 encode at a fixed bitrate and a VLC upscale, and
a stalled branch shows as a frozen pane rather than an error, so what the panes
compare is the display path as much as the model. On 2026-09-08 the same
comparison scored 0.813 off the recording and 1.379 off the tensors -- the two
disagree by a factor of 1.7 and only one of them is measuring the model.

So this reads the raw RGBA the branches hand to the encoder and reports:

  선명도    mean Sobel gradient magnitude, and the ratio against the anchor.
            This is the number the field complaint is about.
  고주파    fraction of spectral energy above half-Nyquist. Cross-checks the
            Sobel figure -- blocking artefacts raise Sobel without raising this,
            which is how a compressed frame fakes sharpness.
  오버슈트  fraction of pixels ringing past what the frame's own low
            frequencies support. Rising gradient with rising overshoot is a
            halo bought, not detail recovered. cv2 bilinear is the calibration
            row: it cannot overshoot, so it must land near zero.

Measured 2026-09-08 at x2 -- anchor 1.0000, cv2 bilinear 1.0005, cv2 bicubic
1.2807, v2 model 1.3399, overshoot 0.01 / 0.01 / 0.35 / 1.11 %. The anchor
sitting on top of the bilinear row is the check that qtivtransform is still the
scaler it was; a model that gains sharpness without gaining overshoot is
recovering detail.

The captures are separate gst-launch runs, so the frames do not line up. Every
figure is a median over frames and the comparison is only valid on a static
scene; nothing here pairs frame i of one file with frame i of another.

    python3 examples/evaluation/score_sr_tensors.py <디렉터리>

The directory is what capture_sr_tensors.sh wrote: lr/anchor/model .rgba plus
meta.txt. Pass --cv2 to add cv2 bilinear and bicubic upscales of the captured
LR as reference rows; qtivtransform measured within 0.55 DN of cv2 bilinear, so
an anchor that lands far from that row means the scaler changed.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def read_meta(path):
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("scale="):
                for tok in line.split():
                    k, _, v = tok.partition("=")
                    meta[k] = v
            elif "=" in line and " " not in line:
                k, _, v = line.partition("=")
                meta[k] = v
    return meta


def frames(path, w, h, limit):
    """Yield up to `limit` frames spread across the file, as HxWx3 uint8 RGB."""
    stride = w * h * 4
    total = os.path.getsize(path) // stride
    if total == 0:
        return
    # A file the capture script already thinned holds only the frames it chose,
    # exposure settling included; skipping a tenth of that would throw away a
    # real sample. Only a full capture needs the head dropped.
    lo = total // 10 if total > 64 else 0
    idx = np.unique(np.linspace(lo, total - 1, min(limit, total - lo)).astype(int))
    with open(path, "rb") as fh:
        for i in idx:
            fh.seek(int(i) * stride)
            buf = np.frombuffer(fh.read(stride), dtype=np.uint8)
            if buf.size < stride:
                continue
            yield buf.reshape(h, w, 4)[:, :, :3]


def sobel_mean(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def hf_ratio(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.abs(np.fft.fft2(g)) ** 2)
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h // 2) / (h / 2)) ** 2 + ((xx - w // 2) / (w / 2)) ** 2)
    return float(f[r > 0.5].sum() / f.sum())


def overshoot(rgb, scale):
    """Fraction of pixels that ring: outside the envelope of a smooth upscale.

    Measured against the frame's own content, not against another file. The
    three captures are separate gst-launch runs and see different frames, so
    anything that pairs frame i of one file with frame i of another measures
    the scene changing rather than the model -- an earlier version of this did
    exactly that and scored cv2 bilinear at 56% overshoot, which is impossible.

    Instead: drop the frame to the SR input resolution and bring it back with
    the smoothest interpolation available. That reconstructs what this image's
    own low frequencies support. Pixels outside the local range of that
    reconstruction are halo the sharpening added, which is the thing overshoot
    is supposed to count. Bilinear scores near zero by construction, and that
    is the row to sanity-check against.
    """
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = a.shape
    small = cv2.resize(a, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(smooth, k)
    lo = cv2.erode(smooth, k)
    # 2 DN of slack keeps sensor noise out of the count.
    return float(np.mean((a > hi + 2.0) | (a < lo - 2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="capture_sr_tensors.sh 가 쓴 디렉터리")
    ap.add_argument("--frames", type=int, default=12, help="파일당 표본 프레임 수")
    ap.add_argument("--cv2", action="store_true", help="cv2 보간 기준선 추가")
    args = ap.parse_args()

    meta_path = os.path.join(args.dir, "meta.txt")
    if not os.path.isfile(meta_path):
        sys.exit(f"meta.txt 없음: {meta_path}")
    meta = read_meta(meta_path)
    scale = int(meta["scale"])
    iw, ih = (int(v) for v in meta["in"].split("x"))
    ow, oh = (int(v) for v in meta["out"].split("x"))

    paths = {n: os.path.join(args.dir, f"{n}.rgba") for n in ("lr", "anchor", "model")}
    for n, p in paths.items():
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            sys.exit(f"{n}.rgba 가 비었다 -- 캡처가 실패했다. meta.txt 의 frames 를 볼 것.")

    anchor = list(frames(paths["anchor"], ow, oh, args.frames))
    model = list(frames(paths["model"], ow, oh, args.frames))
    if not anchor or not model:
        sys.exit("표본 프레임을 얻지 못했다.")
    rows = [("앵커 (qtivtransform)", anchor)]
    if args.cv2:
        lr = list(frames(paths["lr"], iw, ih, args.frames))
        for label, interp in (("cv2 bilinear", cv2.INTER_LINEAR),
                              ("cv2 bicubic", cv2.INTER_CUBIC)):
            rows.append((label, [cv2.resize(f, (ow, oh), interpolation=interp) for f in lr]))
    rows.append(("모델", model))

    base = float(np.median([sobel_mean(f) for f in rows[0][1]]))
    print(f"{meta.get('model', '?')}")
    print(f"x{meta.get('scale', '?')}  {iw}x{ih} -> {ow}x{oh}  "
          f"module={meta.get('sr_module', '?')}\n")
    print(f"{'':<22}{'선명도':>9}{'/앵커':>9}{'고주파':>10}{'오버슈트':>10}")
    for label, fs in rows:
        s = float(np.median([sobel_mean(f) for f in fs]))
        h = float(np.median([hf_ratio(f) for f in fs]))
        o = float(np.median([overshoot(f, scale) for f in fs]))
        print(f"{label:<22}{s:9.3f}{s / base:9.4f}{h:10.5f}{o * 100:9.2f}%")

    print(f"\n프레임 표본 {len(anchor)}장/파일. 캡처가 서로 다른 런이므로 "
          f"중앙값 비교만 유효하다 (정지 장면 전제).")
    print("판정: /앵커 가 1 을 넘어야 모델이 단순 확대보다 낫다. "
          "선명도가 오르는데 고주파가 따라 오르지 않으면 블로킹, "
          "오버슈트만 오르면 링잉이다.")


if __name__ == "__main__":
    main()
