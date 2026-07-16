import pandas as pd
import os
import numpy as np

"""
Ce fichier contient les fonctions principales qui permettent d'adapter a un scenario de temperature donnee,
la courbe de demande electrique finale associee.
Il prend en consideration le fait que les courbes fournies sont nationales ou regionales
"""
# These are the factor for France to reconstruct demand,
# supposing that heating thermosensitivity is reduced and cooling enhanced
heating_factor_19_50 = 0.56
cooling_factor_19_50 = 1
heating_factor_19_50_gas = 0.165

# Parametres du modele pour le smoothing de la temperature determines par validation process
alpha_hdd = 0.4
alpha_cdd = 0.6

_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# Demande totale non thermosensible, qui va nous permettre de scale la baseline
base_demand_default = {"elec": 440, "gas": 112}



def _drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def load_csv(baseline_path, temp_path, thermo_coeffs_path):
    baseline = _drop_feb29(pd.read_csv(baseline_path, index_col=0, parse_dates=True))
    temp     = _drop_feb29(pd.read_csv(temp_path,     index_col=0, parse_dates=True))
    coeffs   = pd.read_csv(thermo_coeffs_path, index_col=0)

    common_cols = baseline.columns.intersection(temp.columns)
    baseline = baseline[common_cols]
    temp     = temp[common_cols]
    print("-----------------------------")
    print("Zones considérées:")
    print(common_cols)
    print("-----------------------------")


    years = list(temp.index.year.unique())

    return baseline, temp, coeffs, years


def get_french_public_holidays(years: list[int]) -> pd.DatetimeIndex:
    """
    Genere les jours feries francais pour une liste d'annees.
    Paques calcule via l'algorithme de Gauss.
    """
    from datetime import date, timedelta

    def easter(year):
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day   = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    holidays = []
    for year in years:
        e = easter(year)
        holidays += [
            date(year, 1,  1), # 1er janvier
            e + timedelta(1), #Lundi de Paques
            date(year, 5,  1), # 1er mai
            date(year, 5,  8), # 8 mai
            e + timedelta(39), #Ascension
            e + timedelta(50), #Pentecoste
            date(year, 7, 14), # 14 juillet
            date(year, 8, 15), # Assomption
            date(year, 11, 1), # Fete des morts
            date(year, 11,11), # Armistice
            date(year, 12,25), # Noel
        ]
    return pd.DatetimeIndex(holidays)


def _get_day_type(d: pd.Timestamp, holiday_set: set, bridge_set: set, year_map: dict) -> str:
    """Retourne le type de jour pour la selection du profil horaire."""
    if d in holiday_set:
        return "dim_jf"
    # Day-of-week comes from the source year to stay consistent with the demand profile
    source_dow = d.replace(year=year_map[d.year]).dayofweek
    if source_dow == 6:
        return "dim_jf"
    if source_dow == 5:
        return "sam_pont"
    if d in bridge_set:
        return "sam_pont"
    if source_dow == 0:
        return "lun"
    if source_dow in (1, 2, 3):
        return "mar_jeu"
    return "ven"  # source_dow == 4


def set_baseline_to_target_year(baseline: pd.DataFrame, years: list):

    # We collect the 4 most recent years for baseline
    source_years = sorted(baseline.index.year.unique())[-4:]
    baseline_last4 = baseline[baseline.index.year.isin(source_years)]

    # Holidays for each source year (used as reference for the cycle)
    source_holidays_index = get_french_public_holidays(list(source_years))

    chunks = []
    all_holidays = []
    all_bridges = []
    year_map = {}

    for i, target_year in enumerate(years):
        # We attribute a source year for each future year
        source_year = source_years[i % len(source_years)]
        year_map[target_year] = source_year
        year_data = baseline_last4[baseline_last4.index.year == source_year].copy()

        year_data.index = year_data.index.map(lambda dt, y=target_year: dt.replace(year=y))
        chunks.append(year_data)

        # Shift holidays from source year to target year (same cycle as baseline)
        source_year_holidays = source_holidays_index[source_holidays_index.year == source_year]
        target_holidays = pd.DatetimeIndex([h.replace(year=target_year) for h in source_year_holidays])
        all_holidays.extend(target_holidays)

        # Bridge detection using SOURCE year's day-of-week to stay consistent with the demand mapping
        holiday_set = set(target_holidays)
        for d in pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D"):
            if d.month == 2 and d.day == 29:
                continue
            source_dow = d.replace(year=source_year).dayofweek
            if source_dow == 0 and (d - pd.Timedelta(days=3)) in holiday_set:
                all_bridges.append(d)
            elif source_dow == 4 and (d + pd.Timedelta(days=3)) in holiday_set:
                all_bridges.append(d)

    holidays_index = pd.DatetimeIndex(sorted(all_holidays))
    bridges_index  = pd.DatetimeIndex(sorted(all_bridges))

    return pd.concat(chunks).sort_index(), holidays_index, bridges_index, year_map


def build_daily_demand(baseline: pd.DataFrame,
                       base_demand: dict,
                       temp: pd.DataFrame,
                       coeffs: pd.DataFrame,
                       conv_factor_h = heating_factor_19_50,
                       conv_factor_c = cooling_factor_19_50,
                       conv_factor_gas = heating_factor_19_50_gas,
                       alpha_hdd: float = alpha_hdd,
                       alpha_cdd: float = alpha_cdd,
                       vector: str = "elec"):
    # We first build demand using daily baseline and daily temperature and regional coeffs
    # We start by computing daily hdd and cdd using effective temperature.

    #Dataframe for effective temperature
    Th = temp.copy()
    Tc = temp.copy()

    #dataframes for hdd and cdd and heating/cooling energy demand
    hdd = pd.DataFrame()
    cdd = pd.DataFrame()

    #df for daily demand
    daily_demand = baseline.copy()

    # Scale baseline so that the annual energy across regions matches base_demand[vector] (TWh)
    # daily_demand is in GW (daily average power), so annual energy = mean * 365 * 24 / 1000 TWh
    annual_total_twh = daily_demand.sum(axis=1).mean() * 365 * 24 / 1e6
    scaling_factor = base_demand[vector] / annual_total_twh
    daily_demand = daily_demand * scaling_factor

    for r in temp.columns:
        # We compute the temperature using exponential moving window
        Th[r] = temp[r].ewm(alpha= alpha_hdd, adjust=False).mean()

        T_hdd = coeffs["heating_threshold"].loc[r]

        if vector != "gas":
            heat_rate = coeffs["heating_rate"].loc[r] * conv_factor_h
        else :
            heat_rate = coeffs["heating_rate"].loc[r] * conv_factor_gas

        hdd[r] = np.maximum(0, T_hdd - Th[r])*heat_rate
        daily_demand[r] += hdd[r]

        if vector != "gas":
            Tc[r] = temp[r].ewm(alpha = alpha_cdd, adjust=False).mean()
            T_cdd = coeffs["cooling_threshold"].loc[r]
            cool_rate = coeffs["cooling_rate"].loc[r] * conv_factor_c
            cdd[r] = np.maximum(0, Tc[r] - T_cdd)*cool_rate
            daily_demand[r] += cdd[r]

    yearly_heating = hdd.groupby(hdd.index.year).sum()
    if vector == "gas":
        yearly_thermo = yearly_heating.copy()
        yearly_thermo.columns = pd.MultiIndex.from_product([yearly_thermo.columns, ["heating"]])
    else:
        yearly_thermo = pd.concat(
            {"heating": yearly_heating,
             "cooling": cdd.groupby(cdd.index.year).sum()},
            axis=1
        ).swaplevel(axis=1).sort_index(axis=1)
    yearly_thermo.index.name = "year"

    return daily_demand, yearly_thermo


def build_hourly_demand(daily_demand: pd.DataFrame,
                        profiles_dir: str,
                        holidays: pd.DatetimeIndex,
                        bridges: pd.DatetimeIndex,
                        year_map: dict,
                        vector: str = "elec") -> pd.DataFrame:

    holiday_set = set(holidays.normalize())
    bridge_set  = set(bridges.normalize())

    result = {}

    for r in daily_demand.columns:
        if vector != "gas":
            path = os.path.join(profiles_dir, f"{r}_profiles.csv")
            profiles = pd.read_csv(path, index_col=[0, 1])
            profiles.columns = profiles.columns.astype(int)  # "0".."23" -> 0..23

        chunks = []
        for day, daily_val in daily_demand[r].items():
            timestamps = pd.date_range(day, periods=24, freq="h")
            if vector == "gas":
                hourly_vals = np.full(24, daily_val)
            else:
                season   = _SEASON_MAP[day.month]
                day_type = _get_day_type(day, holiday_set, bridge_set, year_map)
                hourly_vals = daily_val * 24 * profiles.loc[(day_type, season)].values
            chunks.append(pd.Series(hourly_vals, index=timestamps))

        result[r] = pd.concat(chunks)

    return pd.DataFrame(result)


def run_pipeline(baseline_path: str,
                 temp_path: str,
                 thermo_coeffs_path: str,
                 profiles_dir: str,
                 base_demand: dict = base_demand_default,
                 conv_heating_2050: float = heating_factor_19_50,
                 conv_cooling_2050: float = cooling_factor_19_50,
                 conv_heating_gas_2050: float = heating_factor_19_50_gas,
                 alpha_hdd: float = alpha_hdd,
                 alpha_cdd: float = alpha_cdd,
                 vector: str = "elec"
                 ):

    # We collect the years from the temperature file
    baseline, temp, coeffs, years = load_csv(baseline_path, temp_path, thermo_coeffs_path)
    print(f"Years considered : {min(years)} to {max(years)}")
    print("------------------------------------------------")

    baseline, holidays, bridges, year_map = set_baseline_to_target_year(baseline, years)
    print("Baseline constructed for the years considered")
    print("------------------------------------------------")

    daily_demand, yearly_thermo = build_daily_demand(baseline, base_demand = base_demand, temp=temp, coeffs=coeffs, 
                                                     conv_factor_h = conv_heating_2050, conv_factor_c = conv_cooling_2050, conv_factor_gas = conv_heating_gas_2050, 
                                                     alpha_hdd = alpha_hdd, alpha_cdd = alpha_cdd, 
                                                     vector = vector
                                                     )

    hourly_demand = build_hourly_demand(daily_demand, profiles_dir, holidays, bridges, year_map, vector=vector)
    print("Hourly demand constructed")

    return hourly_demand, yearly_thermo




######################################################################
######################################################################
##########################TEST########################################
######################################################################
######################################################################


#run_pipeline("final_predictions/daily_baselines/baseline_noT_elec_regions.csv", "base_data/temp_FR_daily_pecd.csv")
