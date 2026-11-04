"""Demo Gradio: carga de checkpoints, entradas arbitrarias, interfaz y descarga de pesos.

Checkpoints con pesos ALEATORIOS (pretrained=False) en tmp_path: se prueba que la
demo no se cae con casos nuevos, no la calidad de las predicciones.
"""
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")
gr = pytest.importorskip("gradio")

from camanchaca.demo import app as demo_app  # noqa: E402
from camanchaca.labeling.koschmieder import BAND_LABELS_ES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXP = "configs/experiment/smoke.yaml"
SEEDS = {"resnet50": [0, 1], "deit_s": [0]}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """runs/smoke/<modelo>/seed<k>/best.pt con pesos aleatorios."""
    import torch

    from camanchaca.config import load_experiment, run_config
    from camanchaca.models.factory import build_model
    from camanchaca.training.checkpoint import save_checkpoint

    root = tmp_path_factory.mktemp("runs")
    exp = load_experiment(EXP)
    for mcfg in exp["models"]:
        for seed in SEEDS.get(mcfg["name"], []):
            torch.manual_seed(seed)
            cfg = run_config(exp, mcfg, seed)
            model = build_model(cfg, pretrained=False)
            save_checkpoint(root / exp["name"] / mcfg["name"] / f"seed{seed}" / "best.pt",
                            model, cfg, {"mu": 2.0, "sigma": 0.5}, step=1)
    # carpetas que NO deben descubrirse
    (root / exp["name"] / "sweep" / "resnet50" / "seed0").mkdir(parents=True)
    return root


@pytest.fixture(scope="module")
def predictors(runs):
    return demo_app.load_predictors(exp=EXP, runs_root=runs, device="cpu", verbose=False)


def _check_outputs(out, n_bands=4):
    v_text, bands, speed, unc, latency, overlay, warns = out
    assert "visibilidad estimada (mor)" in v_text.lower() and " m" in v_text
    assert set(bands) == set(BAND_LABELS_ES) and len(bands) == n_bands
    assert sum(bands.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in bands.values())
    assert "ILUSTRATIVO" in speed and "km/h" in speed
    assert isinstance(unc, str) and unc
    assert "ms" in latency
    assert isinstance(warns, str)
    return overlay


# ------------------------------------------------------------- descubrimiento
def test_discover_and_load(runs, predictors):
    found = demo_app.discover_checkpoints("smoke", runs_root=runs)
    assert [p.parent.parent.name for p in found] == ["resnet50", "resnet50", "deit_s"]
    assert list(predictors) == ["resnet50", "deit_s"]
    assert [p.seed for p in predictors["resnet50"]] == [0, 1]
    explicit = demo_app.load_predictors([found[2], found[0].parent], device="cpu")
    assert {k: len(v) for k, v in explicit.items()} == {"resnet50": 1, "deit_s": 1}
    assert demo_app.load_predictors(exp="nada", runs_root=runs, verbose=False) == {}
    assert "semillas 0, 1" in demo_app.describe_predictors(predictors)


def test_corrupt_checkpoint_is_skipped(tmp_path, capsys):
    bad = tmp_path / "smoke" / "resnet50" / "seed0" / "best.pt"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"no es un checkpoint")
    assert demo_app.load_predictors(exp="smoke", runs_root=tmp_path) == {}
    assert "no se pudo cargar" in capsys.readouterr().err


# ------------------------------------------------------------------- entradas
def _inputs():
    rng = np.random.default_rng(0)
    return {
        "rgb_array": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        "rgba": Image.fromarray(rng.integers(0, 256, (120, 160, 4), dtype=np.uint8), "RGBA"),
        "grayscale": Image.fromarray(rng.integers(0, 256, (100, 100), dtype=np.uint8), "L"),
        "tiny_8x8": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        "large": Image.fromarray(rng.integers(0, 256, (2600, 3400, 3), dtype=np.uint8)),
        "float_2d": rng.random((50, 60)),
        "uint16": Image.fromarray(rng.integers(0, 65535, (64, 64), dtype=np.uint16)),
    }


@pytest.mark.parametrize("kind", list(_inputs()))
def test_analyze_any_input(predictors, kind):
    image = _inputs()[kind]
    overlay = _check_outputs(demo_app.analyze(image, "resnet50", "gradcam",
                                              predictors=predictors))
    assert overlay.dtype == np.uint8 and overlay.ndim == 3 and overlay.shape[2] == 3


@pytest.mark.parametrize("model_name,kind", [("deit_s", "rollout"), ("deit_s", "gradcam"),
                                             ("resnet50", "rollout"), ("resnet50", "none")])
def test_analyze_maps(predictors, model_name, kind):
    img = _inputs()["rgb_array"]
    out = demo_app.analyze(img, model_name, kind, predictors=predictors)
    overlay = _check_outputs(out)
    if kind == "none":
        assert overlay is None
    else:
        assert overlay.shape == (demo_app.OVERLAY_SIZE, demo_app.OVERLAY_SIZE, 3)
    if model_name == "resnet50" and kind == "rollout":
        assert "solo existe para DeiT" in out[6]


def _plain(predictor, img):
    """Lo que ve el modelo, al tamaño del mapa (para comparar con la superposición)."""
    x = predictor.transform(Image.fromarray(img)).unsqueeze(0)
    base = demo_app._denormalize(x[0], predictor)
    return x, np.asarray(Image.fromarray(base).resize((demo_app.OVERLAY_SIZE,) * 2,
                                                      Image.BILINEAR)).astype(float)


@pytest.mark.parametrize("model_name,kind", [("deit_s", "gradcam"), ("deit_s", "rollout"),
                                             ("resnet50", "gradcam")])
def test_explanation_map_uses_interpret_api(predictors, model_name, kind):
    """El mapa viene de camanchaca.interpret y colorea la imagen (salvo Grad-CAM vacío)."""
    img = _inputs()["rgb_array"]
    for pred in predictors[model_name]:
        x, plain = _plain(pred, img)
        overlay, note = demo_app.explanation_map(pred, x, kind)
        assert overlay.shape == (demo_app.OVERLAY_SIZE, demo_app.OVERLAY_SIZE, 3)
        diff = float(np.abs(overlay.astype(float) - plain).mean())
        if "VACÍO" in note:      # ReLU(Σ w·A) = 0: legítimo con pesos aleatorios
            assert kind == "gradcam" and diff < 1.0
        else:
            assert diff > 5.0, f"{model_name}/{kind}: superposición sin color ({note})"
            assert ("rollout" if kind == "rollout" else "Grad-CAM") in note


def test_explanation_map_empty_and_failing(predictors, monkeypatch):
    import torch

    from camanchaca.interpret import gradcam

    pred = predictors["resnet50"][0]
    x, plain = _plain(pred, _inputs()["rgb_array"])

    class Zero:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, x, sign=-1.0):
            return torch.zeros(x.shape[0], *x.shape[-2:]), torch.zeros(x.shape[0])

    monkeypatch.setattr(gradcam, "for_model", lambda model, cfg=None: Zero())
    overlay, note = demo_app.explanation_map(pred, x, "gradcam")
    assert "VACÍO" in note and float(np.abs(overlay - plain).mean()) < 1.0

    def boom(model, cfg=None):
        raise RuntimeError("capa inexistente")

    monkeypatch.setattr(gradcam, "for_model", boom)
    overlay, note = demo_app.explanation_map(pred, x, "gradcam")
    assert overlay is None and "capa inexistente" in note
    out = demo_app.analyze(_inputs()["rgb_array"], "resnet50", "gradcam", predictors=predictors)
    assert out[5] is None and "no se pudo calcular el mapa" in out[6]


def test_uncertainty_from_seed_spread(predictors):
    img = _inputs()["rgb_array"]
    unc_two = demo_app.analyze(img, "resnet50", "none", predictors=predictors)[3]
    unc_one = demo_app.analyze(img, "deit_s", "none", predictors=predictors)[3]
    assert unc_two.startswith("×") and "2 semillas" in unc_two
    assert "solo 1 semilla" in unc_one


def test_none_and_bad_inputs_raise_gr_error(predictors):
    with pytest.raises(gr.Error):
        demo_app.analyze(None, "resnet50", "gradcam", predictors=predictors)
    with pytest.raises(gr.Error):
        demo_app.analyze(np.zeros((4, 4, 7)), "resnet50", "none", predictors=predictors)
    with pytest.raises(gr.Error):
        demo_app.analyze(_inputs()["rgb_array"], "no_existe", "none", predictors=predictors)
    with pytest.raises(gr.Error):  # sin checkpoints: error claro, no excepción cruda
        demo_app.analyze(_inputs()["rgb_array"], None, "none", predictors={})


def test_model_label_is_accepted(predictors):
    out = demo_app.analyze(_inputs()["rgb_array"], "DeiT-S (1 semilla)", "none",
                           predictors=predictors)
    assert "DeiT-S" in out[4]


def test_exif_rotation_and_alpha():
    img = Image.new("RGB", (40, 20), (10, 200, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotar 90°
    rotated, info = demo_app.to_pil_rgb(_roundtrip_jpeg(img, exif))
    assert rotated.size == (20, 40) and info["orig_size"] == (20, 40)
    rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    out, info = demo_app.to_pil_rgb(rgba)
    assert info["alpha"] and out.mode == "RGB" and out.getpixel((0, 0)) == (128, 128, 128)


def _roundtrip_jpeg(img, exif):
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


def test_ood_warnings():
    def warns(img):
        pil, info = demo_app.to_pil_rgb(img)
        return " ".join(demo_app.ood_warnings(pil, info)).lower()

    rng = np.random.default_rng(1)
    assert "noche" in warns(rng.integers(0, 25, (64, 64, 3), dtype=np.uint8))
    assert "escala de grises" in warns(Image.fromarray(
        rng.integers(0, 256, (64, 64), dtype=np.uint8), "L"))
    assert "diminuta" in warns(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    assert "proporción" in warns(rng.integers(0, 256, (64, 400, 3), dtype=np.uint8))
    color = rng.integers(60, 256, (128, 128, 3), dtype=np.uint8)
    assert warns(color) == ""


def test_safe_speed_text_is_illustrative():
    txt = demo_app.format_speed(100.0, factor=1.5)
    assert "ILUSTRATIVO" in txt and "pesimista" in txt and "μ = 0,4" in txt


# ------------------------------------------------------------- simular niebla
def test_simulate_fog(predictors):
    img = _inputs()["rgb_array"]
    fogged, md = demo_app.simulate_fog(img, 60, "resnet50", 0.5, predictors=predictors)
    assert fogged.shape == img.shape and fogged.dtype == np.uint8
    assert "APROXIMADA" in md and "V fijada" in md and "V̂" in md
    # el cielo (sobre el horizonte) queda en el color del aire: menos contraste
    assert fogged[:100].std() < img[:100].std()
    fogged2, md2 = demo_app.simulate_fog(img, 60, None, 0.5, predictors={})
    assert fogged2.shape == img.shape and "Sin predicción" in md2
    with pytest.raises(gr.Error):
        demo_app.simulate_fog(None, 60, "resnet50", 0.5, predictors=predictors)


# ------------------------------------------------------------------ interfaz
def test_examples_synthetic_fallback(tmp_path):
    empty = tmp_path / "examples"
    empty.mkdir()
    (empty / "README.md").write_text("x")
    ex = demo_app.example_images(empty, out_dir=tmp_path / "gen")
    assert len(ex) == 2 and all(Path(p).is_file() for p in ex)
    Image.new("RGB", (32, 32)).save(empty / "propia.jpg")
    assert demo_app.example_images(empty, out_dir=tmp_path / "gen") == [str(empty / "propia.jpg")]


def test_build_demo(predictors, tmp_path):
    ex = demo_app.synthetic_examples(tmp_path)
    demo = demo_app.build_demo(predictors, examples=ex,
                               sim_examples=[demo_app.synthetic_clear_example(tmp_path)])
    assert isinstance(demo, gr.Blocks)
    names = {d.get("api_name") for d in demo.config["dependencies"]}
    assert {"analyze", "simulate_fog"} <= names
    assert isinstance(demo_app.build_demo({}, examples=[], sim_examples=[]), gr.Blocks)


def test_main_starts_without_checkpoints(tmp_path, monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kw: calls.update(kw))
    monkeypatch.setattr(demo_app, "example_images", lambda *a, **k: [])
    monkeypatch.setattr(demo_app, "synthetic_clear_example",
                        lambda *a, **k: str(tmp_path / "x.jpg"))
    rc = demo_app.main(["--exp", "main", "--runs-root", str(tmp_path), "--port", "7999",
                        "--device", "cpu"])
    assert rc == 0 and calls["server_port"] == 7999 and calls["share"] is False
    assert "No hay checkpoints" in capsys.readouterr().out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_launch_and_query_with_client(predictors, tmp_path):
    gradio_client = pytest.importorskip("gradio_client")
    ex = demo_app.synthetic_examples(tmp_path)
    demo = demo_app.build_demo(predictors, examples=ex, sim_examples=[])
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                quiet=True, allowed_paths=[str(tmp_path)])
    try:
        client = gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)
        res = client.predict(gradio_client.handle_file(ex[0]), "deit_s", "rollout",
                             api_name="/analyze")
        assert len(res) == 7 and "MOR" in res[0]
        assert res[1]["label"] in BAND_LABELS_ES
        assert res[5] is not None  # ruta del mapa
        fog = client.predict(gradio_client.handle_file(ex[1]), 80, "resnet50", 0.5,
                             api_name="/simulate_fog")
        assert "V fijada" in fog[1]
    finally:
        demo.close()


def _wait_port(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@pytest.mark.slow
@pytest.mark.parametrize("with_ckpts", [True, False], ids=["con_checkpoints", "sin_checkpoints"])
def test_app_script_subprocess(runs, tmp_path, with_ckpts):
    """`python app/app.py --runs-root ... --exp smoke` real: arranca, responde y se detiene."""
    gradio_client = pytest.importorskip("gradio_client")
    port = _free_port()
    root = runs if with_ckpts else tmp_path / "vacio"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), HF_HUB_OFFLINE="1",
               GRADIO_ANALYTICS_ENABLED="False", CAMANCHACA_RUNS=str(tmp_path / "runs_no"))
    log = tmp_path / "app.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "app" / "app.py"), "--runs-root", str(root), "--exp",
             "smoke", "--port", str(port), "--server-name", "127.0.0.1", "--device", "cpu"],
            stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=tmp_path)
        try:
            assert _wait_port(port, proc), log.read_text()[-2000:]
            client = gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)
            img = tmp_path / "foto.png"
            Image.fromarray(_inputs()["rgb_array"]).save(img)
            if with_ckpts:
                for model, kind in [("resnet50", "gradcam"), ("deit_s", "gradcam"),
                                    ("deit_s", "rollout")]:
                    res = client.predict(gradio_client.handle_file(str(img)), model, kind,
                                         api_name="/analyze")
                    assert "Visibilidad estimada (MOR)" in res[0]
                    assert {c["label"] for c in res[1]["confidences"]} == set(BAND_LABELS_ES)
                    ov = np.asarray(Image.open(res[5]))
                    assert ov.shape == (demo_app.OVERLAY_SIZE, demo_app.OVERLAY_SIZE, 3)
            else:
                with pytest.raises(Exception, match="No hay checkpoints"):
                    client.predict(gradio_client.handle_file(str(img)), None, "gradcam",
                                   api_name="/analyze")
                fog = client.predict(gradio_client.handle_file(str(img)), 80, None, 0.5,
                                     api_name="/simulate_fog")
                assert Path(fog[0]).is_file() and "Sin predicción" in fog[1]
            assert proc.poll() is None, "la app se cayó"
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
    text = log.read_text()
    assert ("Modelos cargados" if with_ckpts else "No hay checkpoints cargados") in text
    assert "Traceback" not in text


# ------------------------------------------------------- scripts/fetch_weights.py
def _load_fetch():
    spec = importlib.util.spec_from_file_location("fetch_weights",
                                                  REPO / "scripts" / "fetch_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_weights_url_drive_and_sha(tmp_path):
    fw = _load_fetch()
    src = tmp_path / "release"
    src.mkdir()
    blob = src / "resnet50_seed0.pt"
    blob.write_bytes(b"pesos de prueba")
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({
        "resnet50/seed0": {"url": blob.as_uri(), "sha256": sha},
        "deit_s/seed1": {"url": "https://ejemplo.invalid/deit_s_seed1.pt", "sha256": sha},
    }))
    runs = tmp_path / "runs"
    # deit_s/seed1 se toma de la carpeta "Drive" (sin red) con el nombre <modelo>_seed<k>.pt
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "deit_s_seed1.pt").write_bytes(blob.read_bytes())
    rc = fw.main(["--manifest", str(manifest), "--exp", "main", "--runs-root", str(runs),
                  "--drive-dir", str(drive)])
    assert rc == 0
    assert (runs / "main" / "resnet50" / "seed0" / "best.pt").read_bytes() == b"pesos de prueba"
    assert (runs / "main" / "deit_s" / "seed1" / "best.pt").is_file()
    # sha256 incorrecto -> error y no deja archivo
    manifest.write_text(json.dumps({"resnet50/seed2": {"url": blob.as_uri(), "sha256": "0" * 64}}))
    assert fw.main(["--manifest", str(manifest), "--exp", "main", "--runs-root", str(runs)]) != 0
    assert not (runs / "main" / "resnet50" / "seed2" / "best.pt").exists()
    # sin manifiesto: copia lo que haya en --drive-dir (sin verificar)
    (drive / "resnet50").mkdir()
    (drive / "resnet50" / "seed3").mkdir()
    (drive / "resnet50" / "seed3" / "best.pt").write_bytes(b"x")
    assert fw.main(["--manifest", str(tmp_path / "no_existe.json"), "--exp", "main",
                    "--runs-root", str(runs), "--drive-dir", str(drive)]) == 0
    assert (runs / "main" / "resnet50" / "seed3" / "best.pt").read_bytes() == b"x"
    assert fw.main(["--manifest", str(tmp_path / "no_existe.json"), "--exp", "main",
                    "--runs-root", str(runs)]) == 1


def test_fetch_weights_sha256_verification(tmp_path, capsys):
    """sha256 por bloques = hashlib; un destino con sha distinto se reemplaza; sin sha avisa."""
    fw = _load_fetch()
    big = tmp_path / "big.bin"
    big.write_bytes(np.random.default_rng(0).bytes(fw.CHUNK * 2 + 123))
    assert fw.sha256_of(big) == hashlib.sha256(big.read_bytes()).hexdigest()
    good = fw.sha256_of(big)
    root = tmp_path / "runs" / "main"
    dest = root / "deit_s" / "seed0" / "best.pt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupto")
    assert fw.fetch_one("deit_s/seed0", {"url": str(big), "sha256": good.upper()}, root)
    assert dest.read_bytes() == big.read_bytes()
    assert "no coincide" in capsys.readouterr().out
    assert fw.fetch_one("deit_s/seed0", {"url": "https://x.invalid/a.pt", "sha256": good}, root)
    assert "ya existe" in capsys.readouterr().out          # sha OK -> no se descarga de nuevo
    assert not fw.fetch_one("deit_s/seed1", {"url": str(big), "sha256": "f" * 64}, root)
    assert not list(root.rglob("*.part")) and not (root / "deit_s" / "seed1" / "best.pt").exists()
    assert fw.fetch_one("resnet50/seed0", {"url": str(big)}, root)
    assert "no verificado" in capsys.readouterr().out
    with pytest.raises(ValueError):
        fw.parse_key("deit_s/0")


def test_scripts_cli_help():
    for script in ["scripts/fetch_weights.py", "scripts/verify_env.py", "app/app.py"]:
        r = subprocess.run([sys.executable, str(REPO / script), "--help"], capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (script, r.stderr[-500:])


def test_verify_env_runs():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "verify_env.py")],
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert "torch" in r.stdout and "Repositorio" in r.stdout, r.stderr[-1000:]
    assert r.returncode in (0, 1)  # 1 = falta algo requerido (se informa, no se cae)


def test_fetch_weights_make_manifest(tmp_path, runs):
    fw = _load_fetch()
    out = tmp_path / "weights.json"
    release = tmp_path / "assets"
    rc = fw.main(["--make-manifest", "--exp", "smoke", "--runs-root", str(runs),
                  "--out", str(out), "--release-dir", str(release),
                  "--url-template", "https://github.com/o/r/releases/download/v1/{file}"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert set(data) == {"resnet50/seed0", "resnet50/seed1", "deit_s/seed0"}
    entry = data["deit_s/seed0"]
    assert entry["url"].endswith("/deit_s_seed0.pt") and len(entry["sha256"]) == 64
    assert (release / "deit_s_seed0.pt").is_file()
