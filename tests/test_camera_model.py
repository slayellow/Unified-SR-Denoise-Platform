"""CameraModel (MOD-CAM): camera-matched (LR, target) pairs for the EO denoiser.

TC ids follow docs/modules/20261008-MOD-CameraModel.md section 4.
"""

import copy
import os

import cv2
import numpy as np
import pytest
import yaml

from src.data.datasets import DenoiseDataset, SRDataset, log_active_degradations
from src.data.degradations import (
    apply_camera_model,
    bgr_to_ycbcr601,
    camera_chroma_noise,
    camera_luma_noise,
    camera_sharpen,
    shaped_noise,
    sparse_impulses,
    validate_camera_model_cfg,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOT = {"density": [20, 60], "amplitude": [8, 25], "footprint": [[0, 0.35, 0], [0.35, 1, 0.35], [0, 0.35, 0]]}
IMPULSES = {"density": [200, 600], "amplitude": [30, 55], "bright_share": 0.31}
DATA_CFG = os.path.join(REPO, "configs", "data", "bicubicpp", "denoise.yaml")


@pytest.fixture()
def cam():
    with open(DATA_CFG, encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f)["camera_model"])
    cfg["target"] = "camera"  # the shipped config trains on `source`; these tests start from the module default
    return cfg


def scene(h=256, w=256, seed=0):
    """Smooth gradient plus a few flat shapes, uint8 BGR."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([40 + 150 * xx / w, 60 + 120 * yy / h, 90 + 60 * (xx + yy) / (h + w)], axis=-1)
    img[h // 4 : h // 2, w // 4 : w // 2] = (200, 190, 180)
    img[h // 2 : 3 * h // 4, w // 2 : 3 * w // 4] = (30, 40, 50)
    return np.clip(img + rng.normal(0, 1, img.shape), 0, 255).astype(np.uint8)


def kernel_response(kernel, fy, fx):
    k = np.asarray(kernel, dtype=np.float64)
    r = k.shape[0] // 2
    return sum(
        k[i + r, j + r] * np.cos(2 * np.pi * (fy * i + fx * j)) for i in range(-r, r + 1) for j in range(-r, r + 1)
    )


def sinusoid_gain(sh, base, f=0.375):
    x = np.arange(256)
    y = (base + 1.0 * np.sin(2 * np.pi * f * x))[None].repeat(64, 0).astype(np.float32)
    out = camera_sharpen(y, sh)[16:-16, 16:-16].mean(0)
    xs = x[16:-16]
    basis = np.stack([np.sin(2 * np.pi * f * xs), np.cos(2 * np.pi * f * xs), np.ones_like(xs, dtype=float)], 1)
    a = np.linalg.lstsq(basis, out, rcond=None)[0]
    return float(np.hypot(a[0], a[1]))


# ---------- MREQ-CAM-01: pair generation ----------
def test_TC_CAM_01_1_PairShapeAndType(cam):
    lr, target = apply_camera_model(scene(), cam, rng=np.random.default_rng(0))
    assert lr.shape == target.shape == (256, 256, 3)
    assert lr.dtype == target.dtype == np.uint8
    assert not np.array_equal(lr, target)


def test_TC_CAM_01_2_ZeroNoiseGivesIdenticalPair(cam):
    cam["noise_scale"] = [0.0, 0.0]
    lr, target = apply_camera_model(scene(), cam, rng=np.random.default_rng(0))
    assert np.array_equal(lr, target)


def test_TC_CAM_01_3_OddSizePatch(cam):
    lr, target = apply_camera_model(scene(255, 257), cam, rng=np.random.default_rng(0))
    assert lr.shape == target.shape == (255, 257, 3)


# ---------- MREQ-CAM-02: sharpening ----------
def test_TC_CAM_02_1_SmallSignalGainMatchesKernel(cam):
    sh = cam["sharpen"]
    b = np.interp(160.0, sh["brightness_y"], sh["brightness_gain"])
    expected = 1 + b * (sh["gain_pos"] + sh["gain_neg"]) / 2 * kernel_response(sh["kernel"], 0.0, 0.375)
    assert sinusoid_gain(sh, 160.0) == pytest.approx(expected, rel=0.10)


def test_TC_CAM_02_2_DarkAreasAreSharpenedLess(cam):
    sh = cam["sharpen"]
    assert sinusoid_gain(sh, 45.0) < sinusoid_gain(sh, 160.0)


def test_TC_CAM_02_3_SoftEdgeOvershootInMeasuredRange(cam):
    # A 120 DN step blurred like a printed gray patch edge; the real camera gave 4.8 % on such edges.
    e = np.where(np.arange(256) < 128, 40.0, 160.0)[None].repeat(64, 0).astype(np.float32)
    e = cv2.GaussianBlur(e, (0, 0), 1.5)
    out = camera_sharpen(e, cam["sharpen"])[32]
    overshoot = (out.max() - 160.0) / 120.0
    assert 0.02 <= overshoot <= 0.10


# ---------- MREQ-CAM-03: noise ----------
def test_TC_CAM_03_1_LumaNoiseFollowsNLF(cam):
    mode = cam["noise_modes"][0]
    y = np.full((256, 256), 120.0, dtype=np.float32)
    n = camera_luma_noise(y, mode, 1.3, np.random.default_rng(1))
    expected = np.interp(120.0, [p[0] for p in mode["luma_nlf"]], [p[1] for p in mode["luma_nlf"]]) * 1.3
    assert float(n.std()) == pytest.approx(expected, rel=0.10)


def test_TC_CAM_03_2_ChromaNoiseOnHalfResPlanes(cam):
    mode = cam["noise_modes"][0]
    n_cb, n_cr = camera_chroma_noise(128, 128, mode, 0.8, np.random.default_rng(2))
    assert n_cb.shape == n_cr.shape == (128, 128)
    assert float(n_cb.std()) == pytest.approx(mode["chroma_sd"][0] * 0.8, rel=0.15)
    assert float(n_cr.std()) == pytest.approx(mode["chroma_sd"][1] * 0.8, rel=0.15)


def test_TC_CAM_03_3_ShapedNoiseHasUnitVarianceAndShape(cam):
    mode = cam["noise_modes"][0]
    n = shaped_noise(256, 256, mode["nps_bands"], mode["luma_nps"], np.random.default_rng(3))
    assert float(n.std()) == pytest.approx(1.0, rel=1e-3)
    lag1 = float((n[:, 1:] * n[:, :-1]).mean() / n.var())
    assert lag1 > 0.3  # the measured pre-sharpening spectrum is low-pass


def test_TC_CAM_03_5_TinyPlaneGivesFiniteNoise(cam):
    mode = cam["noise_modes"][0]
    assert np.all(shaped_noise(1, 1, mode["nps_bands"], mode["luma_nps"], np.random.default_rng(0)) == 0)
    lr, target = apply_camera_model(np.full((2, 2, 3), 120, np.uint8), cam, rng=np.random.default_rng(0))
    assert int(np.abs(lr.astype(int) - target.astype(int)).max()) < 40


def test_TC_CAM_03_4_PairNoiseLevelOnFlatGray(cam):
    cam["noise_scale"] = [1.0, 1.0]
    cam["noise_modes"][0]["prob"], cam["noise_modes"][1]["prob"] = 1.0, 0.0
    cam["sharpen"]["brightness_gain"] = [0.0] * len(cam["sharpen"]["brightness_gain"])
    lr, target = apply_camera_model(np.full((256, 256, 3), 121, np.uint8), cam, rng=np.random.default_rng(4))
    d = bgr_to_ycbcr601(lr.astype(np.float32)) - bgr_to_ycbcr601(target.astype(np.float32))
    expected = np.interp(
        120.0, [p[0] for p in cam["noise_modes"][0]["luma_nlf"]], [p[1] for p in cam["noise_modes"][0]["luma_nlf"]]
    )
    assert float(d[..., 0].std()) == pytest.approx(expected, rel=0.10)
    # chroma noise is added on the half-resolution planes and bilinearly upsampled, which keeps ~88 % of its sd
    for c, sd in ((1, cam["noise_modes"][0]["chroma_sd"][0]), (2, cam["noise_modes"][0]["chroma_sd"][1])):
        assert 0.75 * sd < float(d[..., c].std()) < 1.0 * sd


# ---------- MREQ-CAM-04: config rejection ----------
def _drop(cfg, path):
    node = cfg
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]


@pytest.mark.parametrize(
    "mutate, key",
    [
        (lambda c: c["sharpen"].__setitem__("kernel", np.zeros((8, 8)).tolist()), "kernel"),
        (lambda c: c["noise_modes"][0].__setitem__("luma_nlf", [[120, 3.0], [45, 1.7]]), "luma_nlf"),
        (lambda c: c["noise_modes"][0].__setitem__("prob", 0.5), "prob"),
        (lambda c: c.__setitem__("optics_sigma", [0.9, 0.55]), "optics_sigma"),
        (lambda c: _drop(c, ["sharpen"]), "sharpen"),
        (lambda c: c["sharpen"].__setitem__("brightness_y", [45, 45, 120, 160, 208]), "brightness_y"),
        (lambda c: c["noise_modes"][1].__setitem__("luma_nps", [0.0] * 10), "luma_nps"),
        (lambda c: c.__setitem__("gain_jitter", 1.5), "gain_jitter"),
        (lambda c: c.__setitem__("noise_scale", [0.0, 1.4]), "noise_scale"),
        (
            lambda c: [c["noise_modes"][0].__setitem__("prob", 1.2), c["noise_modes"][1].__setitem__("prob", -0.2)],
            "prob",
        ),
        (lambda c: c.__setitem__("target", "raw"), "target"),
        (lambda c: c.__setitem__("hot_pixels", dict(HOT, footprint=[[1.0, 0.35]])), "footprint"),
        (lambda c: c["noise_modes"][1].__setitem__("impulses", dict(IMPULSES, bright_share=1.5)), "bright_share"),
        (lambda c: c.__setitem__("hot_pixels", dict(HOT, density=[60, 20])), "hot_pixels.density"),
    ],
)
def test_TC_CAM_04_1_InvalidConfigRejected(cam, mutate, key):
    mutate(cam)
    with pytest.raises(ValueError, match=key):
        validate_camera_model_cfg(cam)


def test_TC_CAM_04_2_ShippedConfigIsValid(cam):
    validate_camera_model_cfg(cam)


# ---------- MREQ-CAM-05: determinism ----------
def test_TC_CAM_05_1_SameSeedSameOutput(cam):
    a = apply_camera_model(scene(), cam, rng=np.random.default_rng(7))
    b = apply_camera_model(scene(), cam, rng=np.random.default_rng(7))
    c = apply_camera_model(scene(), cam, rng=np.random.default_rng(8))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_TC_CAM_05_2_ValSetRegeneratesBitIdentically(tmp_path):
    from examples.dataset_preparation.make_val_camera import main as make_val

    src = tmp_path / "hr"
    src.mkdir()
    for i in range(2):
        cv2.imwrite(str(src / f"{i:04d}.png"), scene(64, 96, seed=i))
    for out in ("a", "b"):
        make_val(["--data_config", DATA_CFG, "--hr_root", str(src), "--out_dir", str(tmp_path / out)])
    for sub in ("HR", "LR"):
        for name in ("0000.png", "0001.png"):
            img_a = cv2.imread(str(tmp_path / "a" / sub / name))
            img_b = cv2.imread(str(tmp_path / "b" / sub / name))
            assert img_a is not None and img_b is not None, f"{sub}/{name} was not written"
            assert np.array_equal(img_a, img_b)


def test_TC_CAM_04_3_ValSetRejectsDisabledCameraModel(tmp_path):
    from examples.dataset_preparation.make_val_camera import main as make_val

    with open(DATA_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["camera_model"]["enabled"] = False
    p = tmp_path / "off.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "hr").mkdir()
    with pytest.raises(ValueError, match="camera_model"):
        make_val(["--data_config", str(p), "--hr_root", str(tmp_path / "hr"), "--out_dir", str(tmp_path / "out")])


# ---------- MREQ-CAM-06: dataset integration ----------
def _dataset(tmp_path, cam_cfg, clean_prob=0.0):
    root = tmp_path / "train"
    root.mkdir(exist_ok=True)
    cv2.imwrite(str(root / "img.png"), scene())
    cfg = {
        "clean_prob": clean_prob,
        "camera_model": cam_cfg,
        "degradation": {
            "stage1": {
                "blur": {"enabled": False},
                "resize": {"enabled": False},
                "jpeg": {"enabled": False},
                "sinc": {"enabled": False},
            },
            "stage2": {
                "blur": {"enabled": False},
                "target_resize": {"enabled": False},
                "ir_noise": {"enabled": False},
                "final_jpeg": {"enabled": False},
            },
        },
    }
    return DenoiseDataset(str(root), patch_size=256, config=cfg)


def _as_tensor(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0


def test_TC_CAM_06_1_DatasetUsesCameraTarget(tmp_path, cam):
    cam["prob"] = 1.0
    sample = _dataset(tmp_path, cam)[0]
    assert not np.allclose(sample["hr"].numpy(), _as_tensor(scene()))
    assert not np.allclose(sample["lr"].numpy(), sample["hr"].numpy())


def test_TC_CAM_06_2_DatasetLegacyPathUnchanged(tmp_path, cam):
    cam["enabled"] = False
    sample = _dataset(tmp_path, cam)[0]
    assert np.allclose(sample["hr"].numpy(), _as_tensor(scene()))


def test_TC_CAM_06_3_CleanPairUsesCameraTarget(tmp_path, cam):
    cam["prob"] = 1.0
    sample = _dataset(tmp_path, cam, clean_prob=1.0)[0]
    assert np.array_equal(sample["lr"].numpy(), sample["hr"].numpy())
    assert not np.allclose(sample["hr"].numpy(), _as_tensor(scene()))


def test_TC_CAM_06_4_StartupLogReportsCameraModel(capsys, cam):
    log_active_degradations({"camera_model": cam}, tag="t")
    out = capsys.readouterr().out
    assert "camera_model on" in out and "target=camera" in out and "hot_pixels=off" in out
    cam["hot_pixels"] = HOT
    cam["noise_modes"][1]["impulses"] = IMPULSES
    log_active_degradations({"camera_model": cam}, tag="t")
    out = capsys.readouterr().out
    assert "hot_pixels=[20, 60]" in out and "impulses" in out
    log_active_degradations({}, tag="t")
    assert "camera_model off" in capsys.readouterr().out


def test_TC_CAM_06_5_SRDatasetRejectsCameraModel(tmp_path, cam):
    root = tmp_path / "sr"
    root.mkdir()
    cv2.imwrite(str(root / "img.png"), scene())
    with pytest.raises(ValueError, match="camera_model"):
        SRDataset(str(root), scale_factor=2, patch_size=64, config={"camera_model": cam})


def test_TC_CAM_04_4_MalformedBlockStopsDatasetInit(tmp_path, cam):
    del cam["noise_modes"]
    with pytest.raises(ValueError, match="noise_modes"):
        _dataset(tmp_path, cam)


def test_TC_CAM_06_7_ProbSplitsCameraAndReplay(tmp_path, cam):
    import random

    cam["prob"] = 0.5
    ds = _dataset(tmp_path, cam)
    original = _as_tensor(scene())
    random.seed(0)
    np.random.seed(0)
    n = 200
    camera = sum(not np.allclose(ds[0]["hr"].numpy(), original) for _ in range(n))
    assert 0.38 * n < camera < 0.62 * n


def test_TC_CAM_06_8_DatasetSourceTargetIsHR(tmp_path, cam):
    cam["prob"] = 1.0
    cam["target"] = "source"
    sample = _dataset(tmp_path, cam)[0]
    assert np.allclose(sample["hr"].numpy(), _as_tensor(scene()))
    assert not np.allclose(sample["lr"].numpy(), sample["hr"].numpy())


# ---------- MREQ-CAM-07: target mode ----------
@pytest.mark.parametrize("mode", ["optics", "source"])
def test_TC_CAM_07_1_TargetModeChangesOnlyTarget(cam, mode):
    img = scene()
    lr_cam, target_cam = apply_camera_model(img, cam, rng=np.random.default_rng(5))
    lr, target = apply_camera_model(img, dict(cam, target=mode), rng=np.random.default_rng(5))
    assert np.array_equal(lr, lr_cam)
    sigma = np.random.default_rng(5).uniform(*cam["optics_sigma"])
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    expected = img if mode == "source" else np.clip(np.round(blurred), 0, 255).astype(np.uint8)
    assert np.array_equal(target, expected)
    assert not np.array_equal(target, target_cam)


def test_TC_CAM_07_2_ZeroNoiseSourceTargetKeepsCameraInput(cam):
    cam["noise_scale"] = [0.0, 0.0]
    img = scene()
    _, target_cam = apply_camera_model(img, cam, rng=np.random.default_rng(0))
    lr, target = apply_camera_model(img, dict(cam, target="source"), rng=np.random.default_rng(0))
    assert np.array_equal(lr, target_cam)
    assert np.array_equal(target, img)


# ---------- MREQ-CAM-08: sparse defects ----------
def _noise_free_modes(cam):
    for m in cam["noise_modes"]:
        m["luma_nlf"] = [[45.0, 0.0], [208.0, 0.0]]
        m["chroma_sd"] = [0.0, 0.0]


def test_TC_CAM_08_1_HotPixelLooksLikeMeasuredDefect(cam):
    _noise_free_modes(cam)
    cam["gain_jitter"] = 0.0
    cam["hot_pixels"] = dict(HOT, density=[60, 60], amplitude=[13, 13])
    flat = np.full((512, 512, 3), 160, np.uint8)
    lr, target = apply_camera_model(flat, cam, rng=np.random.default_rng(3))
    d = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    ys, xs = np.nonzero((d == cv2.dilate(d, np.ones((7, 7)))) & (d > 20))
    keep = (ys >= 3) & (ys < 509) & (xs >= 3) & (xs < 509)
    patch = np.mean([d[y - 3 : y + 4, x - 3 : x + 4] for y, x in zip(ys[keep], xs[keep])], axis=0)
    assert 8 <= keep.sum() <= 25  # 60 per Mpx on 0.26 Mpx, minus border and overlaps
    assert 52.5 * 0.75 < patch[3, 3] < 57.0 * 1.25  # measured a9 centre 52.5 / 57.0 from 12-13 DN at a0
    assert patch[3, 1] < -5 and patch[3, 5] < -5  # measured dark ring -0.22 x centre at +-2 px
    assert np.array_equal(target, apply_camera_model(flat, dict(cam, hot_pixels=None), rng=np.random.default_rng(3))[1])


def test_TC_CAM_08_2_ImpulseCountAmplitudeAndSign():
    rng = np.random.default_rng(0)
    out = sparse_impulses(1000, 1000, [400, 400], [30, 55], rng, bright_share=0.31)
    vals = out[out != 0]
    assert abs(len(vals) - 400) < 5 * np.sqrt(400)
    assert np.all((np.abs(vals) >= 30) & (np.abs(vals) <= 110))  # two impulses may land on one pixel
    assert np.mean(vals > 0) == pytest.approx(0.31, abs=0.07)
    assert not sparse_impulses(64, 64, [0, 0], [30, 55], rng).any()


def test_TC_CAM_08_3_ImpulsesOnlyInTheirMode(cam):
    _noise_free_modes(cam)
    cam["noise_modes"][1]["impulses"] = dict(IMPULSES, density=[5000, 5000])
    flat = np.full((256, 256, 3), 120, np.uint8)
    for i, m in enumerate(cam["noise_modes"]):
        m["prob"] = float(i == 1)
    lr, target = apply_camera_model(flat, cam, rng=np.random.default_rng(1))
    assert np.count_nonzero(np.abs(lr.astype(int) - target.astype(int)).max(axis=2) > 15) > 100
    for i, m in enumerate(cam["noise_modes"]):
        m["prob"] = float(i == 0)
    lr, target = apply_camera_model(flat, cam, rng=np.random.default_rng(1))
    assert np.abs(lr.astype(int) - target.astype(int)).max() <= 2
