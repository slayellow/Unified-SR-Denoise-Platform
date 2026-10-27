"""gen_convert_all.py: the script that turns export folders into DLCs inside qairt-env (or on another QAIRT machine).

A wrong line here fails only at conversion time, after the export is done:
a W16 run converted without --restrict_quantization_steps reports success and ships
clipped weights, and tensor names guessed instead of read move the failure to the board.
"""

import hashlib
import json
import os
import subprocess
import sys

import onnx
import pytest
from onnx import TensorProto, helper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "examples", "deployment"))
import gen_convert_all as g  # noqa: E402


def _run(root, run, bw=16, w=428, h=240, in_name="t.1", out_name="34", encodings=True):
    """A minimal export folder: qat_model.onnx with the given I/O names and NCHW size, plus encodings."""
    qat = root / run / "qat"
    qat.mkdir(parents=True)
    x = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 3, h, w])
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 3, h, w])
    graph = helper.make_graph([helper.make_node("Identity", [in_name], [out_name])], "g", [x], [y])
    onnx.save(helper.make_model(graph), str(qat / "qat_model.onnx"))
    if encodings:
        (qat / "qat_model.encodings").write_text(json.dumps({"quantizer_args": {"param_bitwidth": bw}}))


@pytest.mark.parametrize("bw,line", [(16, "convert r t.1 37 R.dlc 16"), (8, "convert r t.1 37 R.dlc")])
def test_convert_line_carries_w16_bitwidth(bw, line, tmp_path):
    (tmp_path / "r" / "qat").mkdir(parents=True)
    (tmp_path / "r" / "qat" / "qat_model.encodings").write_text(json.dumps({"quantizer_args": {"param_bitwidth": bw}}))
    lines = g.script_lines([("r", "t.1", "37", "R.dlc")], str(tmp_path))
    assert line in lines and f"{line} 16" not in lines
    # the W16 flag itself lives in convert(); bw 16 must select it, bw 8 must not
    assert '    if [ "$bw" -ge 16 ]; then' in lines


def test_missing_bitwidth_raises(tmp_path):
    """Defaulting to 8 would drop the W16 quantizer flag and ship a silently clipped DLC."""
    (tmp_path / "r" / "qat").mkdir(parents=True)
    (tmp_path / "r" / "qat" / "qat_model.encodings").write_text("{}")
    with pytest.raises(ValueError, match="param_bitwidth"):
        g.script_lines([("r", "t.1", "37", "R.dlc")], str(tmp_path))


def test_denoise_release_script_unchanged():
    """script_lines() for BicubicPPDenoise[v1]; results/denoise_release_v1/convert_all.sh is this
    text with a hand-written second line (the generator's RUN_RE only matches x<s>_eo/ir runs)."""
    root = os.path.join(REPO, "results", "denoise_release_v1")
    if not os.path.isfile(os.path.join(root, "dn_bpp_half4_w16a16", "qat", "qat_model.encodings")):
        pytest.skip("denoise release export not on this machine")
    text = (
        "\n".join(g.script_lines([("dn_bpp_half4_w16a16", "t.259", "55", "BicubicPPDenoise_EO_1280x720_v1.dlc")], root))
        + "\n"
    )
    assert (
        hashlib.sha256(text.encode()).hexdigest() == "a6801598134da599b8c6f3a5da620dbf8e3957f241481b92906447b820db1259"
    )


def test_collect_takes_names_and_size_from_each_runs_onnx(tmp_path):
    _run(tmp_path, "x3_ir", w=428, h=240)
    _run(tmp_path, "x3_ir_1024", w=428, h=342)
    _run(tmp_path, "x2_eo", w=640, h=360, out_name="37")
    entries, warnings = g.collect(str(tmp_path))
    assert entries == [
        ("x2_eo", "t.1", "37", "BicubicPP_EO_640x360_x2_v3.dlc"),
        ("x3_ir", "t.1", "34", "BicubicPP_IR_428x240_x3_v3.dlc"),
        ("x3_ir_1024", "t.1", "34", "BicubicPP_IR_428x342_x3_v3.dlc"),
    ]
    assert warnings == []


def test_collect_skips_half_exported_runs_and_filters(tmp_path):
    """No run is listed before both artifacts exist, and no names are borrowed from another run."""
    _run(tmp_path, "x3_ir")
    _run(tmp_path, "x3_eo", encodings=False)
    _run(tmp_path, "x4_ir", w=320, h=180, out_name="47")
    entries, warnings = g.collect(str(tmp_path), scales=[3], domains=("ir",))
    assert [e[0] for e in entries] == ["x3_ir"]
    entries, warnings = g.collect(str(tmp_path))
    assert [e[0] for e in entries] == ["x3_ir", "x4_ir"]
    assert any("x3_eo" in w for w in warnings)


def test_main_requires_root_and_keeps_existing_script(tmp_path):
    script = os.path.join(REPO, "examples", "deployment", "gen_convert_all.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert r.returncode == 2 and "root" in r.stderr

    _run(tmp_path, "x3_ir")
    (tmp_path / "convert_all.sh").write_text("old\n")
    r = subprocess.run([sys.executable, script, str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and (tmp_path / "convert_all.sh").read_text() == "old\n"


def test_failed_generation_leaves_no_script(tmp_path):
    """An empty convert_all.sh left by a crash would make every rerun refuse to overwrite it."""
    _run(tmp_path, "x3_ir")
    (tmp_path / "x3_ir" / "qat" / "qat_model.encodings").write_text("{}")
    script = os.path.join(REPO, "examples", "deployment", "gen_convert_all.py")
    r = subprocess.run([sys.executable, script, str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "param_bitwidth" in r.stderr
    assert not (tmp_path / "convert_all.sh").exists()


def test_no_runs_prints_why(tmp_path):
    _run(tmp_path, "x3_ir", encodings=False)
    script = os.path.join(REPO, "examples", "deployment", "gen_convert_all.py")
    r = subprocess.run([sys.executable, script, str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "x3_ir" in r.stdout
