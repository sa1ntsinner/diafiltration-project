"""Matplotlib house style and reusable figure builders."""

from .panels import (estimator_figure, histogram_figure, horizon_figure,
                     metric_table_figure, pareto_figure, process_schematic,
                     switching_figure, tariff_figure, trajectory_figure)
from .style import CTRL_COLORS, PALETTE, figure, save, use_style

__all__ = ["PALETTE", "CTRL_COLORS", "use_style", "figure", "save",
           "trajectory_figure", "horizon_figure", "metric_table_figure",
           "pareto_figure", "histogram_figure", "tariff_figure",
           "switching_figure", "estimator_figure", "process_schematic"]
