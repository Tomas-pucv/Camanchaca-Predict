"""Notebooks delgados: nbformat v4 válido, sin salidas, sin def/class, sintaxis IPython
correcta, markdown en español, configuración común de Colab y cada línea
`!camanchaca ...` (y `python scripts/*.py`, `app/app.py`) aceptada por el parser REAL:
los marcadores `{VAR}` se reemplazan por los valores que el notebook asigna a VAR (todas
las ramas de un `x if c else y`) o por un valor ficticio, y se llama a
`camanchaca.cli.build_parser().parse_args(argv)`, que valida subcomando, banderas,
valores obligatorios y `choices`.

La ejecución completa en modo SMOKE (nbclient) es opcional:
    CAMANCHACA_RUN_NOTEBOOKS=1 pytest tests/test_notebooks.py -m slow
"""
import argparse
import ast
import contextlib
import importlib.util
import io
import itertools
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
EXPECTED = ["00_setup_colab", "01_prepare_data", "02_train", "03_evaluate",
            "04_interpretability", "05_cars_vs_fog", "06_demo"]
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))
CMD_RE = re.compile(r"^\s*!\s*camanchaca\s+(.*)$")
SCRIPT_RE = re.compile(r"^\s*!\s*(?:python|\{sys\.executable\})\s+((?:scripts|app)/\S+\.py)(.*)$")
MD_CMD_RE = re.compile(r"`!?((?:camanchaca|python)\s[^`]+)`")
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def _read(path):
    return nbformat.read(path, as_version=4)


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def _cli_commands():
    """{comando: (parser, {subcomando: parser})} desde camanchaca.cli.build_parser()."""
    from camanchaca.cli import build_parser

    def subparsers(p):
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction):
                return a.choices
        return {}

    return {name: (sp, subparsers(sp)) for name, sp in subparsers(build_parser()).items()}


def _options(parser):
    return {o for a in parser._actions for o in a.option_strings}


def _camanchaca_calls():
    """[(notebook, línea, tokens)] de cada `!camanchaca ...` en las celdas de código."""
    calls = []
    for path in NOTEBOOKS:
        for cell in _code_cells(_read(path)):
            for line in cell.source.splitlines():
                m = CMD_RE.match(line)
                if m:
                    calls.append((path.name, line.strip(), shlex.split(m.group(1), comments=True)))
    return calls


def _string_options(node):
    """Valores de texto posibles de una expresión (constante, f-string, `a if c else b`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):   # f"--labels {LABELS}" -> "--labels X"
        return ["".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)]
    if isinstance(node, ast.IfExp):
        a, b = _string_options(node.body), _string_options(node.orelse)
        return a + b if a is not None and b is not None else None
    return None


def _notebook_strings(nb):
    """{VAR: [valores de texto posibles]} de las asignaciones simples del notebook."""
    from IPython.core.inputtransformer2 import TransformerManager

    tm = TransformerManager()
    out = {}
    for cell in _code_cells(nb):
        for node in ast.walk(ast.parse(tm.transform_cell(cell.source))):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                opts = _string_options(node.value)
                if opts is not None:
                    out.setdefault(node.targets[0].id, []).extend(opts)
    return out


def _expand(line, strings):
    """Todas las variantes de `line` con cada {expr} sustituido (valores reales o ficticios)."""
    exprs = list(dict.fromkeys(PLACEHOLDER_RE.findall(line)))
    choices = []
    for e in exprs:
        e = e.strip()
        dummy = "/tmp/dummy/" + re.sub(r"\W+", "_", e).strip("_")
        choices.append(list(dict.fromkeys(strings.get(e, [dummy]))))
    variants = []
    for combo in itertools.product(*choices):
        text = line
        for e, val in zip(exprs, combo):
            text = text.replace("{" + e + "}", val)
        variants.append(text)
    return variants


def _parse_error(parse_args, argv):
    """None si `parse_args(argv)` acepta los argumentos; si no, el mensaje de argparse."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            parse_args(argv)
    except SystemExit as e:
        return err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else f"exit {e}"
    return None


def _cli_errors(line, strings):
    """Errores de `camanchaca <args>` (sin el `!camanchaca`) en cada variante de la línea."""
    from camanchaca.cli import build_parser

    errors = []
    for text in _expand(line, strings):
        argv = shlex.split(text, comments=True)
        msg = _parse_error(build_parser().parse_args, argv)
        if msg:
            errors.append(f"camanchaca {text!s} -> {msg}")
    return errors


def _script_parser(script):
    """parse_args del script (scripts/*.py) o de la demo (app/app.py)."""
    if script == "app/app.py":
        from camanchaca.demo.app import parse_args
        return parse_args
    path = REPO / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_args


def _script_errors(script, rest, strings):
    errors = []
    for text in _expand(rest, strings):
        msg = _parse_error(_script_parser(script), shlex.split(text, comments=True))
        if msg:
            errors.append(f"{script} {text.strip()} -> {msg}")
    return errors


def test_expected_notebooks_present():
    assert [p.stem for p in NOTEBOOKS] == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_valid_v4_without_outputs(path):
    nb = _read(path)
    nbformat.validate(nb)
    assert nb.nbformat == 4
    assert nb.metadata.get("kernelspec", {}).get("name") == "python3"
    for cell in _code_cells(nb):
        assert cell.get("outputs") == [], f"{path.name}: celda con salidas"
        assert cell.get("execution_count") is None, f"{path.name}: execution_count"
    assert nb.cells[0].cell_type == "markdown" and nb.cells[0].source.startswith("# ")
    first = nb.cells[0].source.lower()
    assert "propósito" in first and "tiempo esperado" in first and "salidas" in first


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_code_cells_are_thin_and_parse(path):
    """Sin def/class (la lógica vive en el paquete) y sintaxis válida tras las magias de IPython."""
    from IPython.core.inputtransformer2 import TransformerManager

    tm = TransformerManager()
    nb = _read(path)
    setup = [c for c in _code_cells(nb) if "IN_COLAB = " in c.source]
    assert len(setup) == 1, "falta (o se repite) la celda de configuración común"
    assert "SMOKE = not IN_COLAB" in setup[0].source
    for cell in _code_cells(nb):
        tree = ast.parse(tm.transform_cell(cell.source))
        bad = [type(n).__name__ for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert not bad, f"{path.name}: {bad} en una celda de código"


def test_setup_cell_identical_everywhere():
    sources = {p.name: next(c.source for c in _code_cells(_read(p)) if "IN_COLAB = " in c.source)
               for p in NOTEBOOKS}
    assert len(set(sources.values())) == 1, "la configuración común difiere entre notebooks"
    src = next(iter(sources.values()))
    for needle in ('CAMANCHACA_DATA"] = "/content/data"', "CAMANCHACA_RUNS", 'pip install -q -e ".[demo,data]"',
                   "camanchaca unpack", "REF = ", "drive.mount"):
        assert needle in src
    # nunca reinstalar torch en Colab (rompe CUDA)
    pip_lines = [ln.split("#")[0] for ln in src.splitlines() if "pip install" in ln]
    assert pip_lines and not any("torch" in ln for ln in pip_lines)


def test_every_camanchaca_line_parses_with_real_cli():
    """Cada `!camanchaca ...` (todas las variantes de sus variables) pasa build_parser().parse_args."""
    calls = 0
    errors = []
    for path in NOTEBOOKS:
        nb = _read(path)
        strings = _notebook_strings(nb)
        for cell in _code_cells(nb):
            for line in cell.source.splitlines():
                m = CMD_RE.match(line)
                if m:
                    calls += 1
                    errors += [f"{path.name}: {e}" for e in _cli_errors(m.group(1), strings)]
    assert calls >= 20
    assert not errors, "líneas que la CLI real rechaza:\n" + "\n".join(errors)


def test_commands_in_markdown_parse():
    """`camanchaca ...` y `python app/app.py ...` citados en el markdown también son válidos."""
    errors = []
    for path in NOTEBOOKS:
        for cell in (c for c in _read(path).cells if c.cell_type == "markdown"):
            for cmd in MD_CMD_RE.findall(cell.source):
                if cmd.startswith("camanchaca "):
                    errors += [f"{path.name}: {e}" for e in _cli_errors(cmd[len("camanchaca "):], {})]
                else:
                    m = re.match(r"python\s+((?:scripts|app)/\S+\.py)(.*)$", cmd)
                    if m:
                        errors += [f"{path.name}: {e}" for e in _script_errors(*m.groups(), {})]
    assert not errors, errors


def test_script_lines_parse():
    """`!{sys.executable} scripts/X.py ...` / `app/app.py ...`: existen y aceptan sus banderas."""
    seen, errors = set(), []
    for path in NOTEBOOKS:
        nb = _read(path)
        strings = _notebook_strings(nb)
        for cell in _code_cells(nb):
            for line in cell.source.splitlines():
                m = SCRIPT_RE.match(line)
                if m:
                    script, rest = m.groups()
                    assert (REPO / script).is_file(), f"{path.name}: {script}"
                    seen.add(script)
                    errors += [f"{path.name}: {e}" for e in _script_errors(script, rest, strings)]
    assert {"scripts/verify_env.py", "scripts/fetch_weights.py", "app/app.py"} <= seen
    assert not errors, errors


def test_checker_rejects_bad_lines():
    """El verificador detecta banderas, subcomandos y choices inexistentes."""
    strings = {"FLAG": ["--final", "--no-existe"]}
    assert len(_cli_errors("interpret --exp x.yaml {FLAG}", strings)) == 1
    assert _cli_errors("prep nada --raw x", {})
    assert _cli_errors("cars --images a --gt b --weights imagenet", {})
    assert _cli_errors("predict --sets val --final", {})          # falta --exp (obligatoria)
    assert not _cli_errors("cars --images a --gt b --weights none", {})
    assert _script_errors("app/app.py", " --exp main --share --bandera-falsa", {})
    assert not _script_errors("scripts/fetch_weights.py", " --exp main --drive-dir /x", {})


def test_flags_inside_string_variables_exist():
    """Banderas armadas en variables (p. ej. LAT_ARGS = "--device cpu ...") también existen."""
    from IPython.core.inputtransformer2 import TransformerManager

    cli = _cli_commands()
    known = set()
    for parser, subs in cli.values():
        known |= _options(parser)
        for sp in subs.values():
            known |= _options(sp)
    tm = TransformerManager()
    bad = []
    for path in NOTEBOOKS:
        for cell in _code_cells(_read(path)):
            assigns = [n.value for n in ast.walk(ast.parse(tm.transform_cell(cell.source)))
                       if isinstance(n, ast.Assign)]
            for node in (c for a in assigns for c in ast.walk(a)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    flags = re.findall(r"(?:^|\s)(--[a-z][\w-]*)", node.value)
                    bad += [f"{path.name}: {f}" for f in flags if f not in known]
    assert not bad, bad


def test_markdown_is_spanish():
    english = re.compile(r"\b(Purpose|Expected time|Outputs|Setup|Run this|Next steps|Note:)\b")
    for path in NOTEBOOKS:
        text = "\n".join(c.source for c in _read(path).cells if c.cell_type == "markdown")
        assert not english.findall(text), f"{path.name}: {english.findall(text)}"
        words = re.findall(r"\b(de|la|los|las|el|que|con|para|en)\b", text.lower())
        assert len(words) >= 20, f"{path.name}: markdown no parece estar en español"


def test_prepare_data_ends_with_git_commit():
    """01 termina con los comandos para versionar manifiestos y splits."""
    nb = _read(NB_DIR / "01_prepare_data.ipynb")
    last = nb.cells[-1].source
    for needle in ("git add data/manifests/*.csv data/splits/v1/*", "git commit", "git push"):
        assert needle in last, needle
    src = "\n".join(c.source for c in _code_cells(nb))
    assert "manifests_splits.zip" in src and "data/manifests" in src and "data/splits" in src


def test_demo_notebook_launches_with_share():
    nb = _read(NB_DIR / "06_demo.ipynb")
    src = "\n".join(c.source for c in _code_cells(nb))
    assert "launch_demo(" in src and "share=IN_COLAB" in src
    assert re.search(r"app/app\.py .*--share", src)
    from camanchaca.demo import app as demo_app
    assert callable(demo_app.launch_demo)   # la API que usa el notebook existe


def test_contract_commands_are_used():
    used = {(t[0], t[1] if t[0] == "prep" else None) for _, _, t in _camanchaca_calls()}
    for cmd in ["smoke", "unpack", "pack", "split", "sweep", "train-all", "predict", "baselines",
                "evaluate", "latency", "probes", "report", "interpret", "counterfactual", "cars"]:
        assert (cmd, None) in used, cmd
    for sub in ["reside", "foggy-cityscapes", "cs-fog", "cars", "hazerd", "ohaze", "folder"]:
        assert ("prep", sub) in used, sub


def test_final_flag_guarded():
    """`--final` solo dentro de bloques condicionados (bloqueo de test)."""
    for path in NOTEBOOKS:
        for cell in _code_cells(_read(path)):
            for line in cell.source.splitlines():
                if CMD_RE.match(line) and "--final" in line:
                    assert line.startswith((" ", "\t")), f"{path.name}: --final sin condición"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CAMANCHACA_RUN_NOTEBOOKS") != "1",
                    reason="ejecución completa opcional (CAMANCHACA_RUN_NOTEBOOKS=1)")
@pytest.mark.parametrize("name", EXPECTED)
def test_execute_in_smoke_mode(name, tmp_path_factory):
    """Ejecuta el notebook fuera de Colab (SMOKE=True) con un `camanchaca` en el PATH."""
    nbclient = pytest.importorskip("nbclient")
    shim = tmp_path_factory.getbasetemp() / "shim"
    shim.mkdir(exist_ok=True)
    exe = shim / "camanchaca"
    if not exe.exists() and shutil.which("camanchaca") is None:
        exe.write_text(f"#!/bin/sh\nPYTHONPATH={REPO / 'src'} exec {sys.executable} -m camanchaca \"$@\"\n")
        exe.chmod(0o755)
    work = Path(os.environ.get("CAMANCHACA_SMOKE_DIR") or tmp_path_factory.getbasetemp() / "nb_smoke")
    env_backup = dict(os.environ)
    os.environ.update(PATH=f"{shim}:{os.environ['PATH']}", CAMANCHACA_SMOKE_DIR=str(work),
                      HF_HUB_OFFLINE="1", MPLBACKEND="Agg")
    try:
        nb = _read(NB_DIR / f"{name}.ipynb")
        nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}}).execute()
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
