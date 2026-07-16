"""
Builds the three hourly demand curves consumed by the EOLES model — electricity, methane and
hydrogen, for France (broken down by region) and its European neighbours (by country) — from
the scenario Excel file, and writes them to inputs/time_varying_inputs/.

Run directly to regenerate all three CSVs:
    python create_demand/create_demand_complete.py

Inputs needed by the pipeline (see DemandBuilder class attributes for exact paths):
1) The baseline load curve. FR elec/gas baselines run 2010-2026, EU elec baseline 2016-2026.
The baselines are the demand patterns without the additional effect of thermosensitive demand
(additional demand linked to temperature).
2) Temperature chronicles over the target period: MPI model, SSP2-4.5 scenario, 2040-2060 for
   both the French regions and the European countries.
3) Per-region/country thermosensitivity coefficients and thresholds.
4) Daily hourly profiles broken down by season / day-of-week / region (or country for rest of EU).
"""

import pandas as pd
import numpy as np

from pipeline_build_demand import run_pipeline
from pipeline_build_demand_EU import run_pipeline_EU


def _drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    """Removes February 29th of a dataframe"""
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def compute_thermosensitive_demand(temp_path, thermo_coeffs_path,
                                   alpha_hdd, alpha_cdd,
                                   heating_rate_corr, cooling_rate_corr = 1,
                                   vector="elec", EU: bool = False,
                                   years_limit: list = None):
    """
    Computes the thermosensitive demand, in order to scale appropriately the baseline for the
    regions/countries.

    Parameters:
    temp_path: path for temperatures
    thermo_coeffs_path: path for thermosensitive coeffs (heating and cooling rates and thresholds)
    alpha_hdd, alpha_cdd: parameters of smoothing for the effective temperature
    heating_rate_corr: correction of the heating rate between present and targeted period (2050 fa)
    cooling_rate_corr: similar
    vector: "elec" by default, anything else will be considered as gas or CH4
    EU: True if working with european data
    years_limit: interval of the years considered for the calculation of the thermosensitive demand
    """

    temp = _drop_feb29(pd.read_csv(temp_path, index_col=0, parse_dates=True))
    thermo_coeffs = pd.read_csv(thermo_coeffs_path, index_col=0)

    if years_limit:
        temp = temp[(temp.index >= f"{years_limit[0]}") & (temp.index < f"{years_limit[1]+1}")]

    heating_demand_tot = 0
    cooling_demand_tot = 0
    if EU:
        thermo_demand_EU = {}
        temp = temp[temp.columns.intersection(thermo_coeffs.index)]

    nb_years = len(list(temp.index.year.unique()))

    T_h = temp.copy()
    T_c = temp.copy()
    for r in temp.columns:

        T_h[r] = temp[r].ewm(alpha=alpha_hdd, adjust=False).mean()
        T_c[r] = temp[r].ewm(alpha=alpha_cdd, adjust=False).mean()

        T_hdd = thermo_coeffs["heating_threshold"].loc[r]
        heating_rate = thermo_coeffs["heating_rate"].loc[r]
        if vector == "elec":
            cooling_rate = thermo_coeffs["cooling_rate"].loc[r]
            T_cdd = thermo_coeffs["cooling_threshold"].loc[r]
        else:
            cooling_rate = 0
            T_cdd = 0

        if EU:
            heating_demand = np.maximum(0, T_hdd - T_h[r]) * heating_rate * heating_rate_corr[r]
        else:
            heating_demand = np.maximum(0, T_hdd - T_h[r]) * heating_rate * heating_rate_corr
        cooling_demand = np.maximum(0, T_c[r] - T_cdd) * cooling_rate * cooling_rate_corr

        if EU:
            thermo_demand_EU[r] = (heating_demand.sum() + cooling_demand.sum()) * 24 / nb_years / 1e6

        heating_demand_tot += heating_demand.sum()
        cooling_demand_tot += cooling_demand.sum()

    print(heating_demand_tot / 1e6 * 24 / 21, cooling_demand_tot / 1e6 * 24 / 21)

    if EU:
        return thermo_demand_EU

    if vector != "elec":
        return heating_demand_tot * 24 / nb_years / 1e6

    yearly_thermo = (heating_demand_tot + cooling_demand_tot) * 24 / nb_years
    return yearly_thermo / 1e6


class DemandBuilder:
    """
    Reads the scenario Excel file once (constructor) and builds the three hourly demand curves
    used by the EOLES model — electricity, methane, hydrogen — for France and its European
    neighbours.

    Usage:
        DemandBuilder("Scenario_data_EUR_plus.xlsx").run()

    Or step by step (build_gas/build_h2 each reuse the previous step's output as scaffold):
        db = DemandBuilder()
        db.build_elec()
        db.build_gas()
        db.build_h2()
    """

    # Baselines
    BASELINE_PATH = "create_demand/baselines/baseline_no_month_elec_regions.csv"
    BASELINE_PATH_GAS = "create_demand/baselines/baseline_no_month_gas_regions.csv"
    BASELINE_PATH_EU = "create_demand/baselines/baseline_noT_EU.csv"
    # Temperatures
    TEMP_PATH = "create_demand/temperatures/temp_FR_daily_pecd.csv"
    TEMP_PATH_EU = "create_demand/temperatures/temp_EU_daily_pecd.csv"
    # Thermosensitive coefficients and thresholds
    THERMO_COEFFS_PATH = "create_demand/thresholds_and_rates/elec_thermo_parameters_no_month.csv"
    THERMO_COEFFS_PATH_GAS = "create_demand/thresholds_and_rates/gas_thermo_parameters_no_month.csv"
    THERMO_COEFFS_PATH_EU = "create_demand/thresholds_and_rates/elec_thermo_parameters_EU.csv"
    # Hourly profiles
    PROFILES_DIR = "create_demand/hourly_profiles"
    PROFILES_DIR_EU = "create_demand/hourly_profiles/EU"

    # Target period for the reconstructed demand (climate years)
    YEARS_LIMIT = [2040, 2060]

    def __init__(self, xlsx_path="Scenario_data_EUR_plus.xlsx", output_dir="inputs/time_varying_inputs"):
        self.output_dir = output_dir

        xl = pd.ExcelFile(xlsx_path)
        demand_EU = pd.read_excel(xl, "demande_EU", index_col=0)
        demand_FR = pd.read_excel(xl, "create_demand_param", index_col=0, header=None)

        self.conv_heating_factor = float(demand_FR.loc["coef_elec"].iloc[0])
        self.conv_cooling_factor = float(demand_FR.loc["coef_clim_elec"].iloc[0])
        self.conv_heating_factor_gas = float(demand_FR.loc["coef_CH4"].iloc[0])
        self.demand_elec_tot = float(demand_FR.loc["demand_elec_tot"].iloc[0])
        self.demand_gas_tot = float(demand_FR.loc["demand_CH4_tot"].iloc[0])
        self.demand_H2_tot = float(demand_FR.loc["demand_H2_tot"].iloc[0])

        self.demand_elec_tot_EU = demand_EU.loc["elec"]
        self.demand_CH4_tot_EU = demand_EU.loc["CH4"]
        self.demand_H2_tot_EU = demand_EU.loc["H2"]
        self.conv_heating_factor_EU = demand_EU.loc["heating_factor"]

        # Populated by build_elec() / build_gas(), reused by the next step and by run()'s checks.
        self._hourly_demand_elec_EU = None
        self._hourly_demand_gas_EU = None
        self._hourly_demand_H2_EU = None

    def build_elec(self):
        """Builds and saves demand_elec.csv (France by region + EU by country, in GW)."""
        thermo_demand = compute_thermosensitive_demand(
            self.TEMP_PATH, self.THERMO_COEFFS_PATH,
            alpha_hdd=0.4, alpha_cdd=0.6, 
            heating_rate_corr=self.conv_heating_factor, 
            cooling_rate_corr=self.conv_cooling_factor )
        thermo_demand_EU = compute_thermosensitive_demand(
            self.TEMP_PATH_EU, self.THERMO_COEFFS_PATH_EU,
            alpha_hdd=0.7, alpha_cdd=1, heating_rate_corr=self.conv_heating_factor_EU,
            cooling_rate_corr=1.5, EU=True, years_limit=self.YEARS_LIMIT)

        print("Thermo demand FR:", thermo_demand, "TWh/year")
        print("Thermo demand EU:", thermo_demand_EU)

        # Non-thermosensitive part of the baseline = total demand - thermosensitive demand.
        base_demand = {"elec": self.demand_elec_tot - thermo_demand}
        base_demand_EU = {c: self.demand_elec_tot_EU[c] - thermo_demand_EU[c] for c in thermo_demand_EU}
        print("Base demand EU:", base_demand_EU)

        hourly_demand, _ = run_pipeline(
            self.BASELINE_PATH, self.TEMP_PATH, self.THERMO_COEFFS_PATH, self.PROFILES_DIR,
            base_demand=base_demand, conv_heating_2050=self.conv_heating_factor,
            conv_cooling_2050=1, alpha_hdd=0.4, alpha_cdd=0.6)

        hourly_demand_EU, _ = run_pipeline_EU(
            self.BASELINE_PATH_EU, self.TEMP_PATH_EU, self.THERMO_COEFFS_PATH_EU, self.PROFILES_DIR_EU,
            base_demand_EU, conv_heating_2050=self.conv_heating_factor_EU, conv_cooling_2050=1.5)

        hourly_demand_EU["FR"] = hourly_demand.sum(axis=1)
        hourly_demand_EU = hourly_demand_EU / 1e3  # MW -> GW

        hourly_demand_EU.to_csv(f"{self.output_dir}/demand_elec.csv")
        print("Done for the electricity demand in Europe")

        self._hourly_demand_elec_EU = hourly_demand_EU
        return hourly_demand_EU

    def build_gas(self):
        """Builds and saves demand_CH4.csv. Requires build_elec() to have run first (reuses
        its EU frame as scaffold for the country columns)."""
        if self._hourly_demand_elec_EU is None:
            raise RuntimeError("build_gas() requires build_elec() to run first.")

        thermo_demand_CH4 = compute_thermosensitive_demand(
            self.TEMP_PATH, self.THERMO_COEFFS_PATH_GAS,
            alpha_hdd=0.6, alpha_cdd=1, heating_rate_corr=self.conv_heating_factor_gas, vector="CH4")
        print("Thermo demand CH4 FR:", thermo_demand_CH4, "TWh/year")

        base_demand_gas = {"gas": self.demand_gas_tot - thermo_demand_CH4}

        hourly_demand_gas, _ = run_pipeline(
            self.BASELINE_PATH_GAS, self.TEMP_PATH, self.THERMO_COEFFS_PATH_GAS, self.PROFILES_DIR,
            base_demand=base_demand_gas, conv_heating_2050=self.conv_heating_factor_gas,
            alpha_hdd=0.4, vector="gas")

        hourly_demand_gas_EU = self._hourly_demand_elec_EU.copy()
        for c in self.demand_CH4_tot_EU.index:
            hourly_demand_gas_EU[c] = float(self.demand_CH4_tot_EU[c]) * 1e3 / 8760  # GWh/yr -> GW, flat profile
        hourly_demand_gas_EU["FR"] = hourly_demand_gas.sum(axis=1) / 1e3  # MW -> GW

        hourly_demand_gas_EU.to_csv(f"{self.output_dir}/demand_CH4.csv")
        print("Done for the CH4 demand in Europe")

        self._hourly_demand_gas_EU = hourly_demand_gas_EU
        return hourly_demand_gas_EU

    def build_h2(self):
        """Builds and saves demand_H2.csv (flat profile, no thermosensitivity modelled).
        Requires build_gas() to have run first (reuses its EU frame as scaffold)."""
        if self._hourly_demand_gas_EU is None:
            raise RuntimeError("build_h2() requires build_gas() to run first.")

        hourly_demand_H2_EU = self._hourly_demand_gas_EU.copy()
        for c in self.demand_H2_tot_EU.index:
            hourly_demand_H2_EU[c] = float(self.demand_H2_tot_EU[c]) * 1e3 / 8760
        hourly_demand_H2_EU["FR"] = self.demand_H2_tot * 1e3 / 8760

        hourly_demand_H2_EU.to_csv(f"{self.output_dir}/demand_H2.csv")
        print("Done for the H2 demand in Europe")

        self._hourly_demand_H2_EU = hourly_demand_H2_EU
        return hourly_demand_H2_EU

    def run(self):
        """Builds all three demand curves in order and prints a yearly sanity check
        (average yearly total per vector, in TWh)."""
        self.build_elec()
        self.build_gas()
        self.build_h2()

        print("========================")
        print("Check of the outputs")
        print("========================")
        print(self._hourly_demand_elec_EU.groupby(self._hourly_demand_elec_EU.index.year).sum().mean() / 1e3)
        print(self._hourly_demand_gas_EU.groupby(self._hourly_demand_gas_EU.index.year).sum().mean() / 1e3)
        print(self._hourly_demand_H2_EU.groupby(self._hourly_demand_H2_EU.index.year).sum().mean() / 1e3)


if __name__ == "__main__":
    DemandBuilder().run()
