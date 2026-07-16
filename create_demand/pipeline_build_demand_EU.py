import pandas as pd
import os
import numpy as np

"""
Ce fichier contient les fonctions principales qui permettent d'adapter a un scenario de temperature donnee,
la courbe de demande electrique finale associee pour les pays europeens voisins de la France.
"""
# These are the factor for each country to reconstruct demand. 
# These factors are approximations and should be broken down by countries
heating_factor_19_50 = 1.4
cooling_factor_19_50 = 1.1

# Parametres du modele pour le smoothing de la temperature determines par validation process
alpha_hdd = 0.7
alpha_cdd = 1

_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# Demande annuelle non thermosensible par pays (TWh)
base_demand_default = {"BE": 115, "CH": 70, "IT": 380, "ES": 340, "DE": 660, "UK": 540, "NL":100, "IE":100, "PT":100}


def _drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def load_csv(baseline_path, temp_path, thermo_coeffs_path, first_year, last_year):
    baseline = _drop_feb29(pd.read_csv(baseline_path, index_col=0, parse_dates=True))
    temp     = _drop_feb29(pd.read_csv(temp_path,     index_col=0, parse_dates=True))
    coeffs   = pd.read_csv(thermo_coeffs_path, index_col=0)

    common_cols = baseline.columns.intersection(temp.columns)
    baseline = baseline[common_cols]
    temp     = temp[common_cols]
    temp     = temp[(temp.index >= f"{first_year}") & (temp.index < f"{last_year+1}")]
    print("-----------------------------")
    print("Zones considerees:")
    print(common_cols)
    print("-----------------------------")

    years = list(temp.index.year.unique())

    return baseline, temp, coeffs, years


def get_european_public_holidays(years: list[int], countries=["UK", "BE", "DE", "CH", "IT", "ES", "NL", "IE", "PT"]):

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

    def first_monday(year, month):
        d = date(year, month, 1)
        return d + timedelta((7 - d.weekday()) % 7)

    def last_monday(year, month):
        last = date(year, month + 1, 1) - timedelta(1) if month < 12 else date(year + 1, 1, 1) - timedelta(1)
        return last - timedelta((last.weekday()) % 7)

    holidays = {c: [] for c in countries}

    for year in years:
        e = easter(year)
        good_friday   = e - timedelta(2)
        easter_monday = e + timedelta(1)
        ascension     = e + timedelta(39)
        whit_monday   = e + timedelta(50)
        corpus_christi = e + timedelta(60)

        if "UK" in countries:
            holidays["UK"] += [
                date(year, 1,  1),
                good_friday,
                easter_monday,
                first_monday(year, 5),
                last_monday(year, 5),
                last_monday(year, 8),
                date(year, 12, 25),
                date(year, 12, 26),
            ]

        if "DE" in countries:
            holidays["DE"] += [
                date(year, 1,  1),
                good_friday,
                easter_monday,
                date(year, 5,  1),
                ascension,
                whit_monday,
                date(year, 10, 3),
                date(year, 12, 25),
                date(year, 12, 26),
            ]

        if "BE" in countries:
            holidays["BE"] += [
                date(year, 1,  1),
                easter_monday,
                date(year, 5,  1),
                ascension,
                whit_monday,
                date(year, 7, 21),
                date(year, 8, 15),
                date(year, 11, 1),
                date(year, 11, 11),
                date(year, 12, 25),
            ]

        if "IT" in countries:
            holidays["IT"] += [
                date(year, 1,  1),
                date(year, 1,  6),
                easter_monday,
                date(year, 4, 25),
                date(year, 5,  1),
                date(year, 6,  2),
                date(year, 8, 15),
                date(year, 11, 1),
                date(year, 12, 8),
                date(year, 12, 25),
                date(year, 12, 26),
            ]

        if "ES" in countries:
            holidays["ES"] += [
                date(year, 1,  1),
                date(year, 1,  6),
                good_friday,
                date(year, 5,  1),
                date(year, 8, 15),
                date(year, 10, 12),
                date(year, 11, 1),
                date(year, 12, 6),
                date(year, 12, 8),
                date(year, 12, 25),
            ]

        if "CH" in countries:
            holidays["CH"] += [
                date(year, 1,  1),
                good_friday,
                easter_monday,
                ascension,
                whit_monday,
                date(year, 8,  1),
                date(year, 12, 25),
                date(year, 12, 26),
            ]
        if "NL" in countries:
            holidays["NL"] += [
                date(year, 1,  1),           # Nieuwjaarsdag
                good_friday,                 # Goede Vrijdag
                easter_monday,               # Tweede Paasdag
                ascension,                   # Hemelvaartsdag
                whit_monday,                 # Tweede Pinksterdag
                date(year, 12, 25),          # Eerste Kerstdag
                date(year, 12, 26),          # Tweede Kerstdag
            ]

            # King's / Queen's Day : 30 avril avant 2014, 27 avril depuis
            if year < 2014:
                holidays["NL"].append(date(year, 4, 30))
            else:
                holidays["NL"].append(date(year, 4, 27))

            # Bevrijdingsdag : férié uniquement les années multiples de 5
            if year % 5 == 0:
                holidays["NL"].append(date(year, 5, 5))

        if "PT" in countries:
            holidays["PT"] += [
                date(year, 1,  1),           # Ano Novo
                good_friday,                 # Sexta-Feira Santa
                e,                           # Domingo de Páscoa
                date(year, 4, 25),           # Dia da Liberdade
                date(year, 5,  1),           # Dia do Trabalhador
                corpus_christi,              # Corpo de Deus
                date(year, 6, 10),           # Dia de Portugal
                date(year, 8, 15),           # Assunção
                date(year, 10, 5),           # Implantação da República
                date(year, 11, 1),           # Todos os Santos
                date(year, 12, 1),           # Restauração da Independência
                date(year, 12, 8),           # Imaculada Conceição
                date(year, 12, 25),          # Natal
            ]

        if "IE" in countries:
            # St Patrick's Day : 17 mars, observé le lundi suivant si week-end
            st_patrick = date(year, 3, 17)
            if st_patrick.weekday() == 5:        # samedi → lundi 19
                st_patrick = date(year, 3, 19)
            elif st_patrick.weekday() == 6:      # dimanche → lundi 18
                st_patrick = date(year, 3, 18)

            holidays["IE"] += [
                date(year, 1,  1),               # New Year's Day
                st_patrick,                      # St Patrick's Day
                easter_monday,                   # Easter Monday
                first_monday(year, 5),           # May Day Bank Holiday
                first_monday(year, 6),           # June Bank Holiday
                first_monday(year, 8),           # August Bank Holiday
                last_monday(year, 10),           # October Bank Holiday
                date(year, 12, 25),              # Christmas Day
                date(year, 12, 26),              # St Stephen's Day
            ]

            # St Brigid's Day : férié depuis 2023
            if year >= 2023:
                feb1 = date(year, 2, 1)
                if feb1.weekday() == 4:          # 1er février est un vendredi
                    st_brigid = feb1
                else:
                    st_brigid = first_monday(year, 2)
                holidays["IE"].append(st_brigid)

            # Bank Holiday exceptionnel COVID — 18 mars 2022
            if year == 2022:
                holidays["IE"].append(date(2022, 3, 18))

            # Reports si Noël tombe le week-end
            xmas = date(year, 12, 25)
            if xmas.weekday() == 5:              # 25 = samedi → 27, 28 observés
                holidays["IE"] += [date(year, 12, 27), date(year, 12, 28)]
            elif xmas.weekday() == 6:            # 25 = dimanche → 27 observé
                holidays["IE"].append(date(year, 12, 27))

            # Report Jour de l'An si dimanche
            if date(year, 1, 1).weekday() == 6:
                holidays["IE"].append(date(year, 1, 2))

        
    return {c: pd.DatetimeIndex(holidays[c]) for c in countries}


def _get_day_type(d: pd.Timestamp, holiday_set: set, bridge_set: set, year_map: dict) -> str:
    """Retourne le type de jour pour la selection du profil horaire."""
    if d in holiday_set:
        return "dim_jf"
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
    return "ven"


def set_baseline_to_target_year(baseline: pd.DataFrame, years: list):

    source_years = sorted(baseline.index.year.unique())[-4:]
    baseline_last4 = baseline[baseline.index.year.isin(source_years)]
    countries = list(baseline.columns)

    # Holidays per country for source years
    source_holidays = get_european_public_holidays(list(source_years), countries=countries)

    chunks = []
    all_holidays = {c: [] for c in countries}
    all_bridges  = {c: [] for c in countries}
    year_map = {}

    for i, target_year in enumerate(years):
        source_year = source_years[i % len(source_years)]
        year_map[target_year] = source_year
        year_data = baseline_last4[baseline_last4.index.year == source_year].copy()

        year_data.index = year_data.index.map(lambda dt, y=target_year: dt.replace(year=y))
        chunks.append(year_data)

        for c in countries:
            # Shift source-year holidays to target year
            src_hols = source_holidays[c]
            src_year_hols = src_hols[src_hols.year == source_year]
            target_hols = pd.DatetimeIndex([h.replace(year=target_year) for h in src_year_hols])
            all_holidays[c].extend(target_hols)

            # Bridge detection using SOURCE year's day-of-week
            holiday_set = set(target_hols)
            for d in pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D"):
                if d.month == 2 and d.day == 29:
                    continue
                source_dow = d.replace(year=source_year).dayofweek
                if source_dow == 0 and (d - pd.Timedelta(days=3)) in holiday_set:
                    all_bridges[c].append(d)
                elif source_dow == 4 and (d + pd.Timedelta(days=3)) in holiday_set:
                    all_bridges[c].append(d)

    holidays_dict = {c: pd.DatetimeIndex(sorted(all_holidays[c])) for c in countries}
    bridges_dict  = {c: pd.DatetimeIndex(sorted(all_bridges[c]))  for c in countries}

    return pd.concat(chunks).sort_index(), holidays_dict, bridges_dict, year_map


def build_daily_demand(baseline: pd.DataFrame,
                       base_demand: dict,
                       temp: pd.DataFrame,
                       coeffs: pd.DataFrame,
                       conv_factor_h: float = heating_factor_19_50,
                       conv_factor_c: float = cooling_factor_19_50,
                       alpha_hdd: float = alpha_hdd,
                       alpha_cdd: float = alpha_cdd,
                       vector: str = "elec"):

    Th = temp.copy()
    Tc = temp.copy()
    hdd = pd.DataFrame()
    cdd = pd.DataFrame()
    daily_demand = baseline.copy()

    # Scale each country individually to its base_demand target (TWh/year)
    for r in daily_demand.columns:
        if r in base_demand:
            annual_twh = daily_demand[r].mean() * 365 * 24 / 1e6
            daily_demand[r] *= base_demand[r] / annual_twh

    for r in temp.columns:
        Th[r] = temp[r].ewm(alpha=alpha_hdd, adjust=False).mean()

        T_hdd = coeffs["heating_threshold"].loc[r]
        heat_rate = coeffs["heating_rate"].loc[r] * conv_factor_h[r]

        hdd[r] = np.maximum(0, T_hdd - Th[r]) * heat_rate
        daily_demand[r] += hdd[r]

        if vector != "gas":
            Tc[r] = temp[r].ewm(alpha=alpha_cdd, adjust=False).mean()
            T_cdd = coeffs["cooling_threshold"].loc[r]
            cool_rate = coeffs["cooling_rate"].loc[r] * conv_factor_c
            cdd[r] = np.maximum(0, Tc[r] - T_cdd) * cool_rate
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
                        holidays: dict,
                        bridges: dict,
                        year_map: dict,
                        vector: str = "elec") -> pd.DataFrame:

    result = {}

    for r in daily_demand.columns:
        holiday_set = set(holidays[r].normalize()) if r in holidays else set()
        bridge_set  = set(bridges[r].normalize())  if r in bridges  else set()

        if vector != "gas":
            path = os.path.join(profiles_dir, f"{r}_profiles.csv")
            profiles = pd.read_csv(path, index_col=[0, 1])
            profiles.columns = profiles.columns.astype(int)

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


def run_pipeline_EU(baseline_path: str,
                 temp_path: str,
                 thermo_coeffs_path: str,
                 profiles_dir: str,
                 base_demand: dict = base_demand_default,
                 conv_heating_2050: float = heating_factor_19_50,
                 conv_cooling_2050: float = cooling_factor_19_50,
                 alpha_hdd: float = alpha_hdd,
                 alpha_cdd: float = alpha_cdd,
                 vector: str = "elec",
                 years_limit: list = [2040,2060]
                 ):

    baseline, temp, coeffs, years = load_csv(baseline_path, temp_path, thermo_coeffs_path, years_limit[0], years_limit[1])
    print(f"Years considered : {min(years)} to {max(years)}")
    print("------------------------------------------------")

    baseline, holidays, bridges, year_map = set_baseline_to_target_year(baseline, years)
    print("Baseline constructed for the years considered")
    print("------------------------------------------------")

    daily_demand, yearly_thermo = build_daily_demand(
        baseline, base_demand, temp, coeffs,
        conv_factor_h=conv_heating_2050, conv_factor_c=conv_cooling_2050,
        alpha_hdd=alpha_hdd, alpha_cdd=alpha_cdd, vector=vector
    )

    hourly_demand = build_hourly_demand(daily_demand, profiles_dir, holidays, bridges, year_map, vector=vector)
    print("Hourly demand constructed")

    return hourly_demand, yearly_thermo




######################################################################
######################################################################
##########################TEST########################################
######################################################################
######################################################################
