#!/usr/bin/env python
"""
Batch runner — EOLES multi-node model.

For each climate year in CLIMATE_YEARS, solves two scenarios:
  - FR alone
  - FR interconnected (FR + ES + DE + IT + UK, configurable via INTERCONNECTED_AREAS)

Results are written under OUTPUT_BASE/{year}_{scenario_name}/.
A run_log.csv is written at OUTPUT_BASE/run_log.csv upon completion.

Usage (on the server):
    python run_batch.py
    nohup python run_batch.py > batch.log 2>&1 &
"""

import os
import traceback
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch use

from utils_io import get_config
from utils_results import save_results, save_interconnection_stats, save_duals
from utils_plots import save_residual_load
from modelEoles_multiN_v3_7 import ModelEOLES

# ── Parameters ────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/config_multi_nodes.json"
OUTPUT_BASE = "outputs/batch_2606_res"

CLIMATE_YEARS = list(range(2040,2061)) # 2040 to 2060 inclusive

INTERCONNECTED_AREAS = ["FR", "ES", "DE", "IT", "UK", "NL", "BE", "CH", "PT", "IE"]

SCENARIOS = {
    "FR_alone":          ["FR"],
    "FR_interconnected": INTERCONNECTED_AREAS,
}

INCLUDE_RESERVE = True

# Set to True to also save hourly dispatch CSVs for every country (large files).
EXPORT_HOURLY = False
# ──────────────────────────────────────────────────────────────────────────────


def run_one(year, scenario_name, restricted_area, base_config, output_dir):
    """
    Build, solve and extract results for one (year, scenario) pair.
    Returns (status_str, system_cost_or_None).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_name = f"{year}_{scenario_name}"

    m = ModelEOLES(
        name=run_name,
        config=base_config,
        output_path=output_dir,
        include_reserve=INCLUDE_RESERVE,
        restricted_area=restricted_area,
        years_of_interest=[year],
    )
    m.build_model()
    status, term = m.solve(solver_name="gurobi")

    ok = (status == "ok" and term == "optimal") or \
         (status == "warning" and term == "other")
    suboptimal = (status == "ok" and term == "suboptimal")

    if not ok and not suboptimal:
        print(f"    [SKIP] solve failed: status={status}, term={term}")
        return "failed", None

    if suboptimal:
        print(f"    [WARN] suboptimal solution accepted: status={status}, term={term}")

    m.extract_optimisation_results_linopy()
    save_results(m, output_dir, export_hourly=EXPORT_HOURLY)
    save_duals(m, output_dir)
    save_residual_load(m, output_dir)
    save_interconnection_stats(m, output_dir)
    run_status = "suboptimal" if suboptimal else "ok"
    return run_status, float(m.system_social_cost)


def main():
    base_config = get_config(CONFIG_PATH)
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)

    log_rows = []
    total = len(CLIMATE_YEARS) * len(SCENARIOS)
    done = 0

    for year in CLIMATE_YEARS:
        for scenario_name, restricted_area in SCENARIOS.items():
            done += 1
            run_name = f"{year}_{scenario_name}"
            output_dir = os.path.join(OUTPUT_BASE, run_name)
            print(f"\n[{done}/{total}] {run_name}")

            try:
                run_status, cost = run_one(
                    year, scenario_name, restricted_area, base_config, output_dir
                )
                log_rows.append({
                    "run":              run_name,
                    "year":             year,
                    "scenario":         scenario_name,
                    "status":           run_status,
                    "system_cost_M€":   cost,
                })
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                log_rows.append({
                    "run":            run_name,
                    "year":           year,
                    "scenario":       scenario_name,
                    "status":         "error",
                    "system_cost_M€": None,
                    "error_msg":      str(e),
                })

    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(OUTPUT_BASE, "run_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\n{'='*60}")
    print(f"All {total} runs processed.")
    n_ok         = (log_df["status"] == "ok").sum()
    n_suboptimal = (log_df["status"] == "suboptimal").sum()
    n_fail       = (log_df["status"].isin(["failed", "error"])).sum()
    print(f"  OK: {n_ok}  |  suboptimal: {n_suboptimal}  |  failed/skipped: {n_fail}")
    print(f"Run log -> {log_path}")


if __name__ == "__main__":
    main()
