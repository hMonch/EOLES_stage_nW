"""
Minimal end-to-end example: build, solve and extract results for a small EOLES run.

Run directly:
    python example.py

This solves France + Germany for a single climate year (see config/config_multi_nodes.json)
and writes the full result set to outputs/example_run/ — the exact same files/layout as
run_batch.py's save_results()/save_duals()/save_residual_load()/save_interconnection_stats(),
so anything written for notebook_batch_comparison.ipynb can also be read back from a single
example.py run. Solving more countries or more years increases run time (roughly linearly
with the number of country-years modelled) — a FR-alone, single-year, no-reserve solve takes
about a minute with Gurobi on a laptop.

For a full batch across many climate years and country sets, see run_batch.py instead.
"""

from pathlib import Path

from utils_io import get_config
from utils_results import save_results, save_interconnection_stats, save_duals
from utils_plots import save_residual_load
from modelEoles_multiN_v3_7 import ModelEOLES

OUTPUT_PATH = "outputs/example_run"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

config = get_config("config/config_multi_nodes.json")

example = ModelEOLES(
    name="example_fr_de",
    config=config,
    output_path=OUTPUT_PATH,
    include_reserve=False,           # set True to also model FCR/FRR reserves (slower to solve)
    restricted_area=["FR", "DE"],    # subset of countries to model; remove this argument
                                      # (or pass restricted_area=None) to run the full country
                                      # set defined in inputs/area_indexed/links.csv
    # detailed_countries=["FR"],     # default when omitted: only France gets the full
                                      # operational detail (DSM, FCR/FRR, hydro spillage/
                                      # min-volume/min-outflow). Pass e.g. ["FR", "DE"] to
                                      # extend that level of detail to Germany too.
)

example.build_model()
status, termination = example.solve(solver_name="gurobi")
print(f"Solve status: {status} / {termination}")

example.extract_optimisation_results_linopy()
print(f"Total system cost: {example.system_social_cost:.1f} M€/yr")

# Full result set — same files as run_batch.py, so example_run/ can be read back with
# utils_batch.py exactly like any {year}_{scenario} batch folder.
save_results(example, OUTPUT_PATH, export_hourly=True)
save_duals(example, OUTPUT_PATH)
save_residual_load(example, OUTPUT_PATH)
save_interconnection_stats(example, OUTPUT_PATH)

print(f"Results written to {OUTPUT_PATH}/")
