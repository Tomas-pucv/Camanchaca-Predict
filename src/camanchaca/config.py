"""Configuración por YAML: protocolo compartido + modelo + datos + semilla.

Un experimento (`configs/experiment/*.yaml`) referencia:
- `protocol`: UN archivo (`configs/protocol/base.yaml`) con TODO lo que debe ser
  idéntico entre arquitecturas (entrada, aumentación, cabeza, pérdida,
  optimizador, pasos, batch, AMP, criterio de selección);
- `models`: la lista de backbones (`configs/model/*.yaml`); solo esto cambia;
- `data`: carpeta de splits versionados y nombres de los CSV;
- `seeds`, `sweep`: semillas y barrido de LR (misma grilla para todos).

`run_config()` produce la configuración resuelta de UNA corrida (modelo x
semilla), que se guarda junto al checkpoint. `assert_same_protocol()` verifica
que dos corridas solo difieren en el backbone (y el LR elegido en el barrido).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .paths import resolve_repo_path


def load_yaml(path) -> dict:
    with open(resolve_repo_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(cfg: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_by_dotted(cfg: dict, key: str, default=None):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def apply_overrides(cfg: dict, overrides=()) -> dict:
    """Aplica overrides 'a.b.c=valor' (valor parseado como YAML)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override inválido (use clave=valor): {ov}")
        k, v = ov.split("=", 1)
        set_by_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def load_experiment(path, overrides=()) -> dict:
    """Carga un experimento resolviendo protocolo y modelos referenciados."""
    exp = load_yaml(path)
    exp["protocol"] = load_yaml(exp["protocol"]) if isinstance(exp.get("protocol"), str) \
        else exp.get("protocol", {})
    exp["protocol"] = deep_merge(exp["protocol"], exp.pop("protocol_overrides", {}) or {})
    if isinstance(exp.get("sweep"), str):
        exp["sweep"] = load_yaml(exp["sweep"])
    models = []
    for m in exp.get("models", []):
        mc = load_yaml(m) if isinstance(m, str) else copy.deepcopy(m)
        mc = mc.get("model", mc)
        mc = deep_merge(mc, exp.get("model_overrides", {}))
        models.append(mc)
    exp["models"] = models
    exp = apply_overrides(exp, overrides)
    exp.setdefault("seeds", [0])
    exp.setdefault("name", Path(str(path)).stem)
    return exp


def get_model_cfg(exp: dict, name: str) -> dict:
    for m in exp["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"modelo '{name}' no está en el experimento ({[m['name'] for m in exp['models']]})")


def run_config(exp: dict, model_cfg: dict, seed: int, lr_backbone: float | None = None,
               protocol_overrides: dict | None = None) -> dict:
    """Configuración resuelta de una corrida (modelo x semilla)."""
    protocol = deep_merge(exp["protocol"], protocol_overrides or {})
    if lr_backbone is not None:
        protocol.setdefault("optim", {})["lr_backbone"] = float(lr_backbone)
    return {
        "experiment": exp["name"],
        "protocol": protocol,
        "model": copy.deepcopy(model_cfg),
        "data": copy.deepcopy(exp.get("data", {})),
        "seed": int(seed),
    }


# Claves que pueden diferir entre arquitecturas: el backbone y el LR que eligió
# el barrido (misma grilla y mismo presupuesto para todos).
_TUNED_KEYS = ("lr_backbone", "lr_head")
# Claves que solo afectan la E/S (cada cuánto se escribe last.pt en Drive), no
# el aprendizaje: no cuentan para la igualdad de protocolo ni para el hash.
_IO_ONLY_KEYS = (("schedule", "save_last_every"),)


def _strip_io_only(protocol: dict) -> dict:
    p = copy.deepcopy(protocol)
    for section, key in _IO_ONLY_KEYS:
        if isinstance(p.get(section), dict):
            p[section].pop(key, None)
    return p


def _protocol_for_compare(cfg: dict) -> dict:
    p = _strip_io_only(cfg["protocol"])
    for k in _TUNED_KEYS:
        p.get("optim", {}).pop(k, None)
    return {"protocol": p, "data": cfg.get("data", {})}


def assert_same_protocol(cfg_a: dict, cfg_b: dict) -> None:
    """Falla si dos corridas difieren en algo más que backbone/semilla/LR."""
    a, b = _protocol_for_compare(cfg_a), _protocol_for_compare(cfg_b)
    if a != b:
        diff = _diff(a, b)
        raise AssertionError(f"protocolos distintos entre arquitecturas: {diff}")


def _diff(a, b, prefix=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{prefix}{k}.")
    elif a != b:
        out.append(f"{prefix[:-1]}: {a!r} != {b!r}")
    return out


def config_hash(cfg: dict) -> str:
    if isinstance(cfg.get("protocol"), dict):
        cfg = {**cfg, "protocol": _strip_io_only(cfg["protocol"])}
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
