"""Build raw input tensors for on-device DLC benchmarking with snpe-net-run.

snpe-net-run reads its inputs as headerless raw tensors, so every deployment
input size needs its own directory of .raw files and an input_list.txt naming
them. This writes one directory per (domain, input size) pair:

    bench/IR_640x512/0.raw ... 7.raw
    bench/IR_640x512/input_list.txt      # lines are IR_640x512/0.raw, relative
                                         # to the bench root the device runs in

Layout and dtype match what the DLCs were converted for: float32 NHWC in [0,1],
the same range the QAT graph was trained and exported with, so these files are
also valid for dumping outputs and comparing them against the PyTorch model --
not just for timing, where the values are irrelevant.

Frames come from the validation HR sets, centre-cropped to the target aspect
ratio and bicubic-downscaled to the input size. The 1280x720 HR is 16:9 while
the IR 1024-profile inputs are 5:4, which is why the crop step exists.

Point it at a release folder and the sizes come from the exported graphs
themselves, so a new version with different input sizes needs no edit here:

    python3 examples/deployment/make_bench_raw.py --from-results results/release_v3
    scp -r bench root@<device>:/data/
    ssh root@<device> 'cd /data && ./bench_dlc.sh v3'

With no --from-results it uses the built-in table below: every size SR v3 and
BicubicPPDenoise[v1] deploy (--from-results reads only SR runs, x<s>_<eo|ir>[_1024]).

Related: make_snpe_raw_inputs.py (removed 2026-10-06, see git history) wrote the same tensor format for ONE size
with device-absolute paths, for AIMET calibration. This one is the multi-size
benchmark front end; the encoding convention (float32 NHWC RGB in [0,1]) is
identical and must stay that way in both.
"""
import argparse
import os
import re

import cv2
import numpy as np

# Every input size the deployed DLCs use. EO ships the 720 profile only, plus the
# 1280x720 denoise input; IR additionally runs the 1280x1024 profile, whose x3
# entry is 342 rows because 1024/3 is not an integer (the surplus is trimmed after
# the model).
SIZES = {
    "EO": [(640, 360), (428, 240), (320, 180), (1280, 720)],
    "IR": [(640, 360), (640, 512), (428, 240), (428, 342), (320, 180), (320, 256)],
}

DEFAULT_SOURCES = {
    "EO": "/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/HR",
    "IR": "/mnt/data_server/etc/jshong/SuperResolution/Finetunning_Dataset/val/HR",
}


def make_input(img, w, h):
    """Centre-crop to the target aspect, then bicubic down to (w, h)."""
    ih, iw = img.shape[:2]
    want = w / h
    if iw / ih > want:  # too wide: trim the sides
        cw = int(round(ih * want))
        x0 = (iw - cw) // 2
        img = img[:, x0:x0 + cw]
    else:               # too tall: trim top and bottom
        ch = int(round(iw / want))
        y0 = (ih - ch) // 2
        img = img[y0:y0 + ch, :]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def write_set(domain, w, h, srcs, out_root):
    name = f"{domain}_{w}x{h}"
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for i, src in enumerate(srcs):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable source image: {src}")
        rgb = cv2.cvtColor(make_input(img, w, h), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0)  # HWC, [0,1] -- NHWC with N=1
        tensor.tofile(os.path.join(out_dir, f"{i}.raw"))
        lines.append(f"{name}/{i}.raw")
    with open(os.path.join(out_dir, "input_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return name, len(lines), w * h * 3 * 4


def sizes_from_results(root):
    """{domain: [(w, h)]} read from every exported qat_model.onnx under a
    results root, so the benchmark inputs always match the graphs that shipped
    -- no size table to keep in sync when a new version changes them."""
    import onnx

    found = {}
    for run in sorted(os.listdir(root)):
        m = re.match(r"x\d_(eo|ir)(_1024)?$", run)
        path = os.path.join(root, run, "qat", "qat_model.onnx")
        if not m or not os.path.isfile(path):
            continue
        dims = [d.dim_value for d in
                onnx.load(path).graph.input[0].type.tensor_type.shape.dim]  # NCHW
        found.setdefault(m.group(1).upper(), []).append((dims[3], dims[2]))
    return {d: sorted(set(v)) for d, v in found.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench", help="output root (copied to the device)")
    ap.add_argument("--frames", type=int, default=8, help="raw files per input size")
    ap.add_argument("--domains", nargs="*", default=["EO", "IR"])
    ap.add_argument("--from-results", default=None, metavar="ROOT",
                    help="read the input sizes from ROOT/*/qat/qat_model.onnx "
                         "instead of the built-in table")
    ap.add_argument("--eo_src", default=DEFAULT_SOURCES["EO"], help="EO source images (default %(default)s)")
    ap.add_argument("--ir_src", default=DEFAULT_SOURCES["IR"], help="IR source images (default %(default)s)")
    args = ap.parse_args()
    sources = {"EO": args.eo_src, "IR": args.ir_src}

    sizes = sizes_from_results(args.from_results) if args.from_results else SIZES
    if not sizes:
        raise RuntimeError(f"no exported runs under {args.from_results}")
    if args.from_results:
        print(f"sizes from {args.from_results}: "
              + "  ".join(f"{d}[{', '.join(f'{w}x{h}' for w, h in v)}]"
                          for d, v in sorted(sizes.items())))

    os.makedirs(args.out, exist_ok=True)
    manifest = ["# domain_WxH  frames  bytes/frame  (float32 NHWC, [0,1])"]
    for domain in args.domains:
        if domain not in sizes:
            print(f"  {domain}: 해당 도메인의 export 없음 -- 건너뜀")
            continue
        src_dir = sources[domain]
        srcs = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        if len(srcs) < args.frames:
            raise RuntimeError(f"{domain}: only {len(srcs)} source images")
        srcs = srcs[:args.frames]
        for w, h in sizes[domain]:
            name, n, nbytes = write_set(domain, w, h, srcs, args.out)
            print(f"  {name:<14} {n} frames x {nbytes:,} bytes")
            manifest.append(f"{name}  {n}  {nbytes}")
    with open(os.path.join(args.out, "manifest.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest) + "\n")
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"{args.out}: {total / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
