"""
Load and aggregate result CSVs written by run_batch.py (via utils_results.save_results and
friends) across many {year}_{scenario} run folders.

Distinct from utils_results.py: everything here reads already-saved CSVs from disk, for
comparing years/scenarios after the fact (see notebook_batch_comparison.ipynb). utils_results.py
extracts results from a solved ModelEOLES instance still in memory, right after solve().
"""

from pathlib import Path

import numpy as np
import pandas as pd


def run_dir(batch_dir, year, scenario):
    """Path to one run's output folder, matching run_batch.py's naming convention."""
    return Path(batch_dir) / f"{year}_{scenario}"


def load_csv(path, index_col=0):
    """Read a CSV, returning None if it doesn't exist or fails to parse."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, index_col=index_col)
    except Exception:
        return None


def find_csv(directory, pattern):
    """Glob-safe file finder (handles special characters in filenames, e.g. the euro sign
    in 'investment_costs_annualized_M€yr.csv', which can trip up a hardcoded path)."""
    hits = list(Path(directory).glob(pattern))
    return hits[0] if hits else None


def get_area_series(path, area):
    """Read a tech-by-area (or area-by-tech) CSV and return the row/column for `area`.

    Handles both CSV orientations transparently (installed_power_GW.csv is area x tech,
    generation_per_tech_TWh.csv is tech x area), and falls back to the first column for
    single-area runs (e.g. FR_alone's summary.csv, whose only column is 'FR').
    """
    df = load_csv(path)
    if df is None:
        return None
    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    if area in df.index:
        return df.loc[area]
    elif area in df.columns:
        return df[area]
    return df.iloc[:, 0]


def load_across_years(batch_dir, file_name, scenario, years, area):
    """DataFrame indexed by year (columns = whatever the CSV's other axis is: tech, usually)
    for one (scenario, area) pair across years. Missing years/files are silently skipped."""
    rows = {}
    for year in years:
        path = run_dir(batch_dir, year, scenario) / f"{file_name}.csv"
        s = get_area_series(path, area)
        if s is not None:
            rows[year] = s
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def compute_stats(df):
    """min/mean/max per column, e.g. per technology across climate years."""
    return pd.DataFrame({"min": df.min(), "mean": df.mean(), "max": df.max()})


def filter_techs(stats_df, min_val=0.01, exclude=frozenset()):
    """Keep only rows whose max exceeds min_val and that are not in `exclude`, sorted
    ascending by mean (convenient for horizontal bar charts, largest bar on top)."""
    mask = (stats_df["max"] >= min_val) & (~stats_df.index.isin(exclude))
    return stats_df[mask].sort_values("mean", ascending=True)


def aggregate_df(df, groups):
    """Sum individual tech columns into named groups, e.g. {'Batteries': ['battery_1h', ...]}.
    Groups with no matching column become an all-zero column (not dropped), so downstream
    code can rely on every requested group being present."""
    result = {}
    for group, techs in groups.items():
        avail = [t for t in techs if t in df.columns]
        result[group] = df[avail].sum(axis=1) if avail else pd.Series(0.0, index=df.index)
    return pd.DataFrame(result)


def load_om_across_years(batch_dir, scenario, years, area=None):
    """Load OM_costs_M€yr.csv across years — per-country (area x tech, like the investment
    cost CSVs) since the OM_cost per-area fix. Pass area to select one country's breakdown;
    otherwise sums across all areas present in the file (system-wide)."""
    rows = {}
    for year in years:
        path = run_dir(batch_dir, year, scenario) / "OM_costs_M€yr.csv"
        df = load_csv(path)
        if df is None:
            continue
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        rows[year] = df.loc[area] if (area is not None and area in df.index) else df.sum(axis=0)
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def load_area_investment(batch_dir, year, scenario, area):
    """Return (invest_power_all_areas, invest_power_area, invest_energy_area) for one run.

    invest_power_all_areas (area x tech) is kept alongside invest_power_area for callers
    that need the full table (e.g. to compute an area's share of system-wide investment).
    """
    d = run_dir(batch_dir, year, scenario)
    ip_path = find_csv(d, "investment_costs_annualized_M*yr.csv")
    ie_path = find_csv(d, "investment_costs_energy_annualized_M*yr.csv")
    if ip_path is None or ie_path is None:
        return None, None, None
    ip_all = load_csv(ip_path).apply(pd.to_numeric, errors="coerce").fillna(0)
    ie_all = load_csv(ie_path).apply(pd.to_numeric, errors="coerce").fillna(0)
    key = area if area in ip_all.index else ip_all.index[0]
    ie_area = ie_all.loc[key] if key in ie_all.index else pd.Series(dtype=float)
    return ip_all, ip_all.loc[key], ie_area


def load_area_om(batch_dir, year, scenario, area, ip_all=None, ip_area=None):
    """Read `area`'s own O&M cost directly from OM_costs_M€yr.csv (area x tech, like the
    investment cost CSVs, since the OM_cost per-area fix). ip_all/ip_area are no longer
    needed (O&M used to be system-wide and was allocated proportionally to investment share)
    but are kept as accepted (ignored) parameters for call-site compatibility."""
    d = run_dir(batch_dir, year, scenario)
    om_path = find_csv(d, "OM_costs_M*yr.csv")
    if om_path is None:
        return pd.Series(dtype=float)
    om_all = load_csv(om_path).apply(pd.to_numeric, errors="coerce").fillna(0)
    key = area if area in om_all.index else om_all.index[0]
    return om_all.loc[key]


def load_area_cost_by_vector(batch_dir, scenario, years, area, tech_vector):
    """Per-area annualised cost (CAPEX + storage CAPEX + allocated O&M) broken down by
    energy vector, across years. `tech_vector` maps tech name -> vector (e.g. {'nuclear':
    'elec', 'methanization': 'CH4', ...}). Unit: M€/yr."""
    rows = {}
    for year in years:
        ip_all, ip_area, ie_area = load_area_investment(batch_dir, year, scenario, area)
        if ip_area is None:
            continue
        om_area = load_area_om(batch_dir, year, scenario, area, ip_all, ip_area)
        result = {"elec": 0.0, "CH4": 0.0, "H2": 0.0}
        for tech, vec in tech_vector.items():
            c = ip_area.get(tech, 0) + ie_area.get(tech, 0) + (om_area.get(tech, 0) if tech in om_area.index else 0)
            result[vec] = result.get(vec, 0.0) + float(c)
        rows[year] = result
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def build_area_cost_by_group(batch_dir, scenario, years, area, groups):
    """Per-area annualised cost (CAPEX + storage CAPEX + allocated O&M), summed into named
    display groups (e.g. {'Batteries': ['battery_1h', ...]}), across years. Unit: M€/yr."""
    rows = {}
    for year in years:
        ip_all, ip_area, ie_area = load_area_investment(batch_dir, year, scenario, area)
        if ip_area is None:
            continue
        om_area = load_area_om(batch_dir, year, scenario, area, ip_all, ip_area)
        row = {}
        for group, techs in groups.items():
            total = 0.0
            for t in techs:
                total += float(ip_area.get(t, 0)) + float(ie_area.get(t, 0))
                total += float(om_area.get(t, 0)) if t in om_area.index else 0.0
            row[group] = total
        rows[year] = row
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def compute_cost_per_production(batch_dir, scenario, years, area, tech_vector):
    """LCOE per vector (elec/CH4/H2) for `area`, across years.
    LCOE = (CAPEX annuity + storage CAPEX + allocated O&M) / production.  Unit: €/MWh (= M€/TWh).
    Falls back to total vector supply if no production is attributed to `area` for that
    vector's own techs (e.g. an area with no domestic CH4 production but that still has a
    CH4 balance via imports)."""
    rows = {}
    for year in years:
        d = run_dir(batch_dir, year, scenario)
        if not d.exists():
            continue
        ip_all, ip_area, ie_area = load_area_investment(batch_dir, year, scenario, area)
        if ip_area is None:
            continue
        om_area = load_area_om(batch_dir, year, scenario, area, ip_all, ip_area)
        cost = ip_area.add(ie_area.reindex(ip_area.index, fill_value=0), fill_value=0)
        cost = cost.add(om_area.reindex(cost.index, fill_value=0), fill_value=0)

        def _series(fname):
            s = get_area_series(d / fname, area)
            return s.apply(pd.to_numeric, errors="coerce").fillna(0) if s is not None else pd.Series(dtype=float)

        prod_src = {
            "elec": _series("generation_per_tech_TWh.csv"),
            "CH4": _series("balance_CH4_supply_TWh.csv"),
            "H2": _series("balance_H2_supply_TWh.csv"),
        }
        row = {}
        for vec in ["elec", "CH4", "H2"]:
            techs_v = [t for t, v in tech_vector.items() if v == vec]
            cost_v = sum(float(cost.get(t, 0)) for t in techs_v)
            ps = prod_src[vec]
            prod_v = sum(float(ps.get(t, 0)) for t in techs_v if t in ps.index)
            if prod_v < 1e-3:
                prod_v = float(ps.sum())
            row[f"cost_{vec}_ME"] = cost_v
            row[f"prod_{vec}_TWh"] = prod_v
            row[f"lcoe_{vec}"] = cost_v / prod_v if prod_v > 1e-3 else float("nan")
        rows[year] = row
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def find_typical_year(value_series, years=None):
    """Year whose value is closest to the mean across years (e.g. to pick a representative
    year for a detailed single-year plot such as an energy-flow diagram)."""
    s = value_series.dropna()
    if years is not None:
        s = s.reindex(years).dropna()
    if s.empty:
        return None
    return int((s - s.mean()).abs().idxmin())


def load_lost_load(batch_dir, scenario, years, countries, row_name="lost_load [% of demand]"):
    """Lost load per country/year from summary.csv, pivoted (index=year, columns=country).
    Value is a raw fraction (not %) — the theoretical cap is 2e-5 of demand (0.002% of demand)
    on average over the years."""
    rows = []
    for year in years:
        df = load_csv(run_dir(batch_dir, year, scenario) / "summary.csv")
        if df is None or row_name not in df.index:
            continue
        for country in countries:
            if country not in df.columns:
                continue
            rows.append({"year": year, "country": country,
                        "lost_load": pd.to_numeric(df.loc[row_name, country], errors="coerce")})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="year", columns="country", values="lost_load")


def load_curtailment(batch_dir, scenario, years, countries):
    """Curtailment [%] and [TWh] per country/year from summary.csv, each pivoted
    (index=year, columns=country). Returns (pct_df, twh_df)."""
    rows_pct, rows_twh = [], []
    for year in years:
        df = load_csv(run_dir(batch_dir, year, scenario) / "summary.csv")
        if df is None:
            continue
        for country in countries:
            if country not in df.columns:
                continue
            col = df[country].apply(pd.to_numeric, errors="coerce")
            rows_pct.append({"year": year, "country": country, "curtailment_%": col.get("gene_curtailed [%]", np.nan)})
            rows_twh.append({"year": year, "country": country, "curtailment_TWh": col.get("gene_curtailed [TWh]", np.nan)})
    pct = pd.DataFrame(rows_pct).pivot(index="year", columns="country", values="curtailment_%") if rows_pct else pd.DataFrame()
    twh = pd.DataFrame(rows_twh).pivot(index="year", columns="country", values="curtailment_TWh") if rows_twh else pd.DataFrame()
    return pct, twh


def load_trade_balance(batch_dir, scenario, years, countries):
    """Net imports (imports − exports) per vector, per country and year: electricity, CH4
    (+ biogas import), H2. Long-format DataFrame, one row per (year, country)."""
    rows = []
    for year in years:
        d = run_dir(batch_dir, year, scenario)
        df = load_csv(d / "summary.csv")
        if df is None:
            continue
        h2_imp = load_csv(d / "balance_H2_trade_import_TWh.csv")
        h2_exp = load_csv(d / "balance_H2_trade_export_TWh.csv")
        for country in countries:
            if country not in df.columns:
                continue
            col = df[country].apply(pd.to_numeric, errors="coerce")
            elec_net = col.get("net_imports [TWh/yr]", np.nan)
            ch4_net = col.get("CH4_net_imports [TWh/yr]", np.nan)
            biogas_imp = col.get("biogas_import [TWh/yr]", 0.0)
            h2_net = np.nan
            if h2_imp is not None and h2_exp is not None and country in h2_imp.index and country in h2_exp.index:
                h2_net = h2_imp.loc[country].sum() - h2_exp.loc[country].sum()
            rows.append({
                "year": year, "country": country,
                "elec_net_TWh": elec_net, "CH4_net_TWh": ch4_net,
                "biogas_import_TWh": biogas_imp,
                "CH4+biogas_net_TWh": (ch4_net + biogas_imp) if pd.notna(ch4_net) else np.nan,
                "H2_net_TWh": h2_net,
            })
    return pd.DataFrame(rows)


def load_residual_load_stats(batch_dir, scenario, years, area):
    """Load residual_load_stats.csv (written by run_batch.save_residual_load) across years
    for one area. Index = year."""
    rows = {}
    for year in years:
        df = load_csv(run_dir(batch_dir, year, scenario) / "residual_load_stats.csv")
        if df is None or area not in df.index:
            continue
        rows[year] = df.loc[area]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df.index = df.index.astype(int)
    return df.apply(pd.to_numeric, errors="coerce")


def load_capacity_duals(batch_dir, scenario, years, group):
    """Load dual_max_capacity_{group}.csv ('prod'/'conv'/'str') across years into one long
    DataFrame with columns [tech, area, dual, year]."""
    frames = []
    for year in years:
        path = run_dir(batch_dir, year, scenario) / "duals" / f"dual_max_capacity_{group}.csv"
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)  # long format (tech, area, dual) - no index column
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_annual_dual(batch_dir, scenario, years, name):
    """Load dual_{name}.csv (e.g. 'annual_methanization', 'methanation_CO2') across years into
    one long DataFrame. Columns depend on whether the constraint has an 'area' dimension."""
    frames = []
    for year in years:
        path = run_dir(batch_dir, year, scenario) / "duals" / f"dual_{name}.csv"
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)  # long format - no index column
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_interconnection_stats(batch_dir, scenario, years):
    """Average interconnection_stats.csv (written by save_interconnection_stats) across years.
    Returns a DataFrame with one row per (direction, partner), columns cap_gw, mean_util_pct,
    mean_flow_gw, pct_hours_at_full, pct_hours_idle, energy_twh averaged across the years found."""
    numeric_cols = ["cap_gw", "mean_util_pct", "mean_flow_gw", "pct_hours_at_full", "pct_hours_idle", "energy_twh"]
    frames = []
    for year in years:
        path = run_dir(batch_dir, year, scenario) / "interconnection_stats.csv"
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)  # direction/partner are plain columns, not an index
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df[numeric_cols] = all_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return all_df.groupby(["direction", "partner"], as_index=False)[numeric_cols].mean()
