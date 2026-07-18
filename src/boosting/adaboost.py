"""Backward-compatible import path for the binding AdaBoost implementation."""

from src.trees.adaboost import AdaBoostClassifier, DecisionStump

__all__ = ["AdaBoostClassifier", "DecisionStump"]
