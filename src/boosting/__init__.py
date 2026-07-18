"""Boosting algorithms, including the optional gradient-boosting extension."""

from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.trees.adaboost import AdaBoostClassifier, DecisionStump

__all__ = ["AdaBoostClassifier", "DecisionStump", "GradientBoostingClassifier"]
