"""Recalculate descriptive metrics and charts from the committed daily CSV.

This checks the supplied return series; it does not reconstruct Factor 70
signals or fills from raw market prices.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/daily_top3_returns_2025_2026.csv"
STRATEGY_COLUMN = "daily_top3"
CHART_DIR = ROOT / "docs/figures"


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No daily returns in {SOURCE}")
    previous = None
    for row in rows:
        day = date.fromisoformat(row["date"])
        if previous is not None and day != previous + timedelta(days=1):
            raise ValueError(f"Missing or duplicate daily date at {day}")
        float(row[STRATEGY_COLUMN])
        previous = day
    return rows


def equity_and_drawdown(returns: list[float]) -> tuple[list[float], list[float]]:
    equity = 1.0
    peak = 1.0
    equities = []
    drawdowns = []
    for daily_return in returns:
        equity *= 1.0 + daily_return
        peak = max(peak, equity)
        equities.append(equity)
        drawdowns.append(equity / peak - 1.0)
    return equities, drawdowns


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    returns = [float(row[STRATEGY_COLUMN]) for row in rows]
    equities, drawdowns = equity_and_drawdown(returns)
    count = len(returns)
    total = equities[-1] - 1.0
    annualized = equities[-1] ** (365.0 / count) - 1.0
    average = statistics.mean(returns)
    daily_std = statistics.stdev(returns) if count > 1 else 0.0
    downside = math.sqrt(sum(min(0.0, value) ** 2 for value in returns) / count)
    max_drawdown = min(drawdowns)
    largest = max(returns)
    return {
        "observations": count,
        "total_return": total,
        "max_drawdown": max_drawdown,
        "sharpe": average / daily_std * math.sqrt(365) if daily_std else 0.0,
        "sortino": average / downside * math.sqrt(365) if downside else 0.0,
        "calmar": annualized / abs(max_drawdown) if max_drawdown else 0.0,
        "annualized_return": annualized,
        "largest_day_return": largest,
        "without_largest_day": equities[-1] / (1.0 + largest) - 1.0,
    }


def svg_chart(
    dates: list[date], values: list[float], title: str, axis_label: str,
    ticks: list[float], color: str, output: Path,
) -> None:
    width, height = 960, 360
    left, right, top, bottom = 82, 26, 52, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    lower, upper = min(ticks), max(ticks)

    def x(index: int) -> float:
        return left + index * plot_width / (len(values) - 1)

    def y(value: float) -> float:
        return top + (upper - value) * plot_height / (upper - lower)

    grid = []
    for tick in ticks:
        yy = y(tick)
        grid.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        grid.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" fill="#475569" font-size="13">{tick:.0%}</text>')
    for label in ("2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2026-09-30"):
        target = date.fromisoformat(label)
        if dates[0] <= target <= dates[-1]:
            index = (target - dates[0]).days
            xx = x(index)
            grid.append(f'<text x="{xx:.1f}" y="{height-22}" text-anchor="middle" fill="#475569" font-size="12">{label[:7]}</text>')
    points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
    markup = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f'<title>{title}</title><desc>Derived from committed daily net returns, {dates[0]} through {dates[-1]}.</desc>'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="{left}" y="29" fill="#0f172a" font-family="Arial,sans-serif" font-size="19" font-weight="bold">{title}</text>'
        f'<text x="{left}" y="{height-5}" fill="#64748b" font-family="Arial,sans-serif" font-size="11">{axis_label}</text>'
        f'<g font-family="Arial,sans-serif">{"".join(grid)}</g>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" points="{points}"/>'
        '</svg>\n'
    )
    output.write_text(markup, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    dates = [date.fromisoformat(row["date"]) for row in rows]
    returns = [float(row[STRATEGY_COLUMN]) for row in rows]
    equities, drawdowns = equity_and_drawdown(returns)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    svg_chart(dates, [value - 1.0 for value in equities], "Factor 70: cumulative net return", "Growth of 1.0, less 1", [-0.75, -0.5, 0, 0.5, 1, 1.5], "#2563eb", CHART_DIR / "factor70_equity.svg")
    svg_chart(dates, drawdowns, "Factor 70: drawdown", "From running peak", [-0.75, -0.6, -0.45, -0.3, -0.15, 0], "#dc2626", CHART_DIR / "factor70_drawdown.svg")
    groups = {
        "2025": [row for row in rows if row["date"].startswith("2025")],
        "2026 Jan-Sep": [row for row in rows if row["date"].startswith("2026")],
        "Full period": rows,
    }
    print(json.dumps({label: metrics(group) for label, group in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
