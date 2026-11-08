"""CLI única del proyecto: `camanchaca <comando> ...` (o `python -m camanchaca`).

Cada subpaquete registra sus comandos en su módulo `commands.py` con una
función `register(subparsers)`; las importaciones pesadas (torch, timm) se
hacen dentro de cada comando para que la CLI arranque rápido.

Comandos (ver `camanchaca -h`):
  datos      prep / split / pack / unpack      (camanchaca.data.commands)
  training   train / sweep / train-all         (camanchaca.training.commands)
  evaluación predict / evaluate / latency / report (camanchaca.evaluation.commands)
  baselines  baselines / probes                (camanchaca.baselines.commands)
  interpret  interpret / counterfactual        (camanchaca.interpret.commands)
  autos      cars                              (camanchaca.detection.commands)
  smoke      prueba de humo de punta a punta   (camanchaca.smoke)
"""
from __future__ import annotations

import argparse
import importlib
import sys

COMMAND_MODULES = (
    "camanchaca.data.commands",
    "camanchaca.training.commands",
    "camanchaca.evaluation.commands",
    "camanchaca.baselines.commands",
    "camanchaca.interpret.commands",
    "camanchaca.detection.commands",
    "camanchaca.smoke",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camanchaca",
        description="Camanchaca-Predict: visibilidad bajo niebla (ResNet-50 vs DeiT-S)")
    sub = parser.add_subparsers(dest="command", metavar="<comando>")
    sub.required = True
    for modname in COMMAND_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:  # subpaquete aún no disponible
            if e.name == modname:
                continue
            raise
        mod.register(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return int(rc or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
