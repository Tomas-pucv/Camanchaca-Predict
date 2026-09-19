"""Ciclo de entrenamiento genérico (CLI). Los notebooks traen su versión didáctica."""
from .loop import evaluate, train_one_epoch

__all__ = ["train_one_epoch", "evaluate"]
