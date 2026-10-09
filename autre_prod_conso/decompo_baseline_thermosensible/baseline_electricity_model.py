"""
=============================================================================
MODÈLE DE DÉCOMPOSITION DE LA CONSOMMATION ÉLECTRIQUE
Baseline + Effets température (spline) + Effets calendaires
=============================================================================

Structure :
    0. Imports & configuration
    1. Chargement des données
    2. Génération des variables calendaires (JF, ponts, vacances)
    3. Construction des variables explicatives (spline T, dummies)
    4. Estimation OLS par région
    5. Diagnostic & validation
    6. Extraction de la baseline
    7. Export des résultats

Modélisation de l'effet température :
    On remplace les HDD/CDD (qui présupposent des seuils fixes) par une
    Natural Cubic Spline (NCS) sur T. Le modèle apprend lui-même la forme
    de la relation conso~T, y compris le point de retournement et les
    asymétries chauffage/climatisation.

    f(T) = Σ βₖ · Nₖ(T)   où Nₖ sont les bases de la NCS

    Choix des nœuds : 5 nœuds aux quantiles 5/27.5/50/72.5/95 de T
    (recommandation de Harrell, 2001 — robuste, évite le surapprentissage).

Données attendues :
    - conso.csv      : colonnes = régions, lignes = jours (index = date)
    - temperature.csv : même format
    Les deux fichiers doivent couvrir 2010-2024 (COVID 2020-2021 exclus).
=============================================================================
"""

# =============================================================================
# 0. IMPORTS & CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# --- Paramètres globaux ---
RUPTURE_YEAR  = 2022        # rupture de niveau (post-sobriété)
# Grille de recherche des seuils de température (°C)
T1_GRID   = np.arange(10.0, 18.5, 0.5)   # seuil chauffage électricité T_c1
T2_GRID   = np.arange(16.0, 24.5, 0.5)   # seuil climatisation électricité T_c2
T_GAS_GRID = np.arange(8.0, 20.0, 0.05)  # seuil unique gaz (rupture chauffage/confort)
EXCLUDE_YEARS = [2020, 2021] # COVID
EXCLUDE_PERIOD = ["2020-03", "2021-05"] #COVID

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(conso_path: str, temp_path: str, training_period = None, exclude_full_covid_y = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les fichiers de consommation et de température.

    Format attendu :
        - Index = dates (parsées automatiquement)
        - Colonnes = noms des 13 régions métropolitaines

    Returns
    -------
    conso : pd.DataFrame  shape (n_days, 13)
    temp  : pd.DataFrame  shape (n_days, 13)
    """
    conso = pd.read_csv(conso_path, index_col=0, parse_dates=True)
    temp  = pd.read_csv(temp_path,  index_col=0, parse_dates=True)

    if training_period is not None:
        conso = conso[conso.index.year.isin(training_period)]
        temp = temp[temp.index.year.isin(training_period)]

    common_idx = conso.index.intersection(temp.index)
    conso = conso.loc[common_idx]
    temp  = temp.loc[common_idx]

    if exclude_full_covid_y:
        mask  = ~conso.index.year.isin(EXCLUDE_YEARS)
    else:
        start = pd.Timestamp(EXCLUDE_PERIOD[0])                # 2020-03-01
        end   = pd.Timestamp(EXCLUDE_PERIOD[1]) + pd.offsets.MonthEnd(0)  # 2021-05-31
        mask  = (conso.index < start) | (conso.index > end)

    conso = conso[mask]
    temp  = temp[mask]

    print(f"[load_data] Période retenue : {conso.index.min().date()} → {conso.index.max().date()}")
    print(f"[load_data] Régions : {list(conso.columns)}")
    print(f"[load_data] N jours : {len(conso)}")

    return conso, temp


# =============================================================================
# 2. GÉNÉRATION DES VARIABLES CALENDAIRES
# =============================================================================

def get_french_public_holidays(years: list[int]) -> pd.DatetimeIndex:
    """
    Génère les jours fériés français pour une liste d'années.
    Pâques calculé via l'algorithme de Gauss.
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
            e + timedelta(1), #Lundi de Pâques
            date(year, 5,  1), # 1er mai
            date(year, 5,  8), # 8 mai
            e + timedelta(39), #Ascension
            e + timedelta(50), #Pentecôte
            date(year, 7, 14), # 14 juillet
            date(year, 8, 15), # Assomption
            date(year, 11, 1), # Fête des morts
            date(year, 11,11), # Armistice
            date(year, 12,25), # Noël
        ]
    return pd.DatetimeIndex(holidays)


def get_european_public_holidays(years: list[int], countries = ["UK", "BE", "DE", "CH", "IT", "ES", "NL", "IE", "PT"]):

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
        good_friday  = e - timedelta(2)
        easter_monday = e + timedelta(1)
        ascension    = e + timedelta(39)
        whit_monday  = e + timedelta(50)

        if "UK" in countries:
            holidays["UK"] += [
                date(year, 1,  1),           # New Year's Day
                good_friday,                 # Good Friday
                easter_monday,               # Easter Monday
                first_monday(year, 5),       # Early May Bank Holiday
                last_monday(year, 5),        # Spring Bank Holiday
                last_monday(year, 8),        # Summer Bank Holiday
                date(year, 12, 25),          # Christmas Day
                date(year, 12, 26),          # Boxing Day
            ]

        if "DE" in countries:
            holidays["DE"] += [
                date(year, 1,  1),           # Neujahr
                good_friday,                 # Karfreitag
                easter_monday,               # Ostermontag
                date(year, 5,  1),           # Tag der Arbeit
                ascension,                   # Christi Himmelfahrt
                whit_monday,                 # Pfingstmontag
                date(year, 10, 3),           # Tag der Deutschen Einheit
                date(year, 12, 25),          # 1. Weihnachtstag
                date(year, 12, 26),          # 2. Weihnachtstag
            ]

        if "BE" in countries:
            holidays["BE"] += [
                date(year, 1,  1),           # Nouvel An
                easter_monday,               # Lundi de Pâques
                date(year, 5,  1),           # Fête du Travail
                ascension,                   # Ascension
                whit_monday,                 # Lundi de Pentecôte
                date(year, 7, 21),           # Fête nationale
                date(year, 8, 15),           # Assomption
                date(year, 11, 1),           # Toussaint
                date(year, 11, 11),          # Armistice
                date(year, 12, 25),          # Noël
            ]

        if "IT" in countries:
            holidays["IT"] += [
                date(year, 1,  1),           # Capodanno
                date(year, 1,  6),           # Epifania
                easter_monday,               # Pasquetta
                date(year, 4, 25),           # Festa della Liberazione
                date(year, 5,  1),           # Festa del Lavoro
                date(year, 6,  2),           # Festa della Repubblica
                date(year, 8, 15),           # Ferragosto
                date(year, 11, 1),           # Ognissanti
                date(year, 12, 8),           # Immacolata Concezione
                date(year, 12, 25),          # Natale
                date(year, 12, 26),          # Santo Stefano
            ]

        if "ES" in countries:
            holidays["ES"] += [
                date(year, 1,  1),           # Año Nuevo
                date(year, 1,  6),           # Epifanía
                good_friday,                 # Viernes Santo
                date(year, 5,  1),           # Día del Trabajo
                date(year, 8, 15),           # Asunción
                date(year, 10, 12),          # Fiesta Nacional
                date(year, 11, 1),           # Todos los Santos
                date(year, 12, 6),           # Día de la Constitución
                date(year, 12, 8),           # Inmaculada Concepción
                date(year, 12, 25),          # Navidad
            ]

        if "CH" in countries:
            holidays["CH"] += [
                date(year, 1,  1),           # Nouvel An
                good_friday,                 # Vendredi saint (quasi-national, sauf Valais/Tessin)
                easter_monday,               # Lundi de Pâques (quasi-national, sauf Valais)
                ascension,                   # Ascension (tous cantons)
                whit_monday,                 # Lundi de Pentecôte (quasi-national, sauf Valais)
                date(year, 8,  1),           # Fête nationale
                date(year, 12, 25),          # Noël (tous cantons)
                date(year, 12, 26),          # St Étienne (quasi-national, sauf Valais/Genève)
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

            if year < 2014:
                holidays["NL"].append(date(year, 4, 30)) # King's day
            else:
                holidays["NL"].append(date(year, 4, 27))

            if year % 5 == 0:
                holidays["NL"].append(date(year, 5, 5))   
            
        
        if "PT" in countries:
            corpus_christi = e + timedelta(60)        # Fête-Dieu
            holidays["PT"] += [
                date(year, 1,  1),           # Ano Novo
                good_friday,                 # Sexta-Feira Santa
                e,                           # Domingo de Páscoa (déjà dimanche, optionnel)
                date(year, 4, 25),           # Dia da Liberdade (révolution des œillets)
                date(year, 5,  1),           # Dia do Trabalhador
                corpus_christi,              # Corpo de Deus
                date(year, 6, 10),           # Dia de Portugal
                date(year, 8, 15),           # Assunção de Nossa Senhora
                date(year, 10, 5),           # Implantação da República
                date(year, 11, 1),           # Dia de Todos os Santos
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
                st_patrick,                      # St Patrick's Day (observé)
                easter_monday,                   # Easter Monday
                first_monday(year, 5),           # May Day Bank Holiday
                first_monday(year, 6),           # June Bank Holiday
                first_monday(year, 8),           # August Bank Holiday
                last_monday(year, 10),           # October Bank Holiday
                date(year, 12, 25),              # Christmas Day
                date(year, 12, 26),              # St Stephen's Day
            ]

            # St Brigid's Day : férié depuis 2023
            # = 1er février si vendredi, sinon premier lundi de février
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


def load_vacation_periods_from_csv(csv_path: str) -> dict:
    """
    Lit un CSV de vacances scolaires et reconstruit vacation_periods.

    Format CSV attendu :
        date, vacances_zone_a, vacances_zone_b, vacances_zone_c, nom_vacances
        2010-01-01, False, False, False,
        2010-02-13, True,  False, False, Hiver

    Algorithme : détecte les blocs consécutifs de True par zone et enregistre
    (date_début, date_fin) sous l'année du premier jour de chaque bloc.

    Returns
    -------
    dict { année: { "A": [("YYYY-MM-DD","YYYY-MM-DD"), ...], "B": [...], "C": [...] } }

    Exemple
    -------
    >>> vp = load_vacation_periods_from_csv("data.csv")
    >>> vp[2022]["A"]
    [('2022-02-05', '2022-02-20'), ('2022-04-09', '2022-04-24'), ...]
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    zone_map = {
        "A": "vacances_zone_a",
        "B": "vacances_zone_b",
        "C": "vacances_zone_c",
    }
    vacation_periods: dict = {}

    for zone, col in zone_map.items():
        # Normalise en booléen ("True"/"False" strings, 0/1, ou bool natif)
        active = df[col].astype(str).str.lower() == "true"

        # Identifiant de groupe : s'incrémente à chaque changement de valeur
        group_id = (active != active.shift()).cumsum()

        # Extrait les périodes de True consécutifs
        for gid, grp in df[active].groupby(group_id[active]):
            start = grp["date"].min()
            end   = grp["date"].max()
            year  = start.year
            vacation_periods.setdefault(year, {"A": [], "B": [], "C": []})
            vacation_periods[year][zone].append(
                (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            )

    return dict(sorted(vacation_periods.items()))


def get_school_holidays(years: list[int]) -> tuple[pd.DataFrame, dict]:
    """
    Génère les périodes de vacances scolaires par zone (A, B, C).

    Returns
    -------
    df           : pd.DataFrame [date, zone_A, zone_B, zone_C]
    REGION_ZONE  : dict région → zone scolaire
    """
    REGION_ZONE = {
        "Auvergne-Rhône-Alpes":       "A",
        "Bourgogne-Franche-Comté":     "A",
        "Nouvelle-Aquitaine":          "A",
        "Centre-Val de Loire":         "B",
        "Île-de-France":               "C",
        "Normandie":                   "B",
        "Hauts-de-France":             "B",
        "Bretagne":                    "B",
        "Pays de la Loire":            "B",
        "Grand Est":                   "B",
        "Occitanie":                   "C",
        "Provence-Alpes-Côte d'Azur": "B",
        "Corse":                       "B",
    }

    vacation_periods = load_vacation_periods_from_csv("data_vacances/data.csv")

    all_dates = pd.date_range("2010-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame(index=all_dates, data={"zone_A": 0, "zone_B": 0, "zone_C": 0})

    for year, zones in vacation_periods.items():
        for zone, periods in zones.items():
            col = f"zone_{zone}"
            for start, end in periods:
                mask = (df.index >= start) & (df.index <= end)
                df.loc[mask, col] = 1

    df = df.reset_index().rename(columns={"index": "date"})
    return df, REGION_ZONE


def build_calendar_features(dates: pd.DatetimeIndex, month_effects: bool = True) -> pd.DataFrame:
    """
    Construit toutes les variables calendaires.

    Variables produites :
        dow_0…dow_5     : lundi–samedi (dimanche = référence, absent)
        month_1…month_11: jan–nov (décembre = référence, absent)
        is_holiday      : jour férié
        is_eve          : veille de jour férié
        is_bridge       : pont (lundi/vendredi encadrant un JF)
        zone_A/B/C      : vacances scolaires par zone
        post_rupture    : 1 si année ≥ RUPTURE_YEAR
    """
    years         = list(range(dates.year.min(), dates.year.max() + 1))
    public_hols   = get_french_public_holidays(years)
    holiday_set   = set(public_hols)

    df = pd.DataFrame(index=dates)

    for i in range(6):
        df[f"dow_{i}"] = (dates.dayofweek == i).astype(int)

    if month_effects:
        for m in range(1, 12):
            df[f"month_{m}"] = (dates.month == m).astype(int)

    df["is_holiday"] = dates.isin(public_hols).astype(int)

    df["is_eve"] = pd.DatetimeIndex(
        [d + pd.Timedelta(days=1) for d in dates]
    ).isin(holiday_set).astype(int)

    def is_bridge(d):
        dow = d.dayofweek
        if dow == 4:
            return int((d - pd.Timedelta(days=1)) in holiday_set)
        elif dow == 0:
            return int((d + pd.Timedelta(days=1)) in holiday_set)
        return 0

    df["is_bridge"] = [is_bridge(d) for d in dates]

    vac_df, _ = get_school_holidays(years)
    vac_df = vac_df.set_index("date")
    vac_df.index = pd.DatetimeIndex(vac_df.index)
    df = df.join(vac_df[["zone_A", "zone_B", "zone_C"]], how="left")
    df[["zone_A", "zone_B", "zone_C"]] = df[["zone_A", "zone_B", "zone_C"]].fillna(0)

    df["post_rupture"] = (dates.year >= RUPTURE_YEAR).astype(int)

    return df

def build_calendar_features_v2(dates: pd.DatetimeIndex, month_effect: bool = True) -> pd.DataFrame:
    """
    Version enrichie de build_calendar_features.

    Changements vs v1
    ─────────────────
    Jours de semaine
      • dow_0..dow_4      lundi–vendredi (jours fériés exclus → traités comme dimanche)
      • is_sat_or_bridge  samedi OU pont (lundi/vendredi encadrant un JF)
      • Référence         dimanche + jours fériés (activité minimale identique)
      → Supprimées : dow_5, is_holiday, is_bridge

    Vacances scolaires (par zone A/B/C)
      • vac_noel_zone_X   Noël (décembre + 1–7 janvier)
      • vac_ete_zone_X    grandes vacances (juillet + août)
      • vac_other_zone_X  autres vacances (hiver fév, printemps avr, toussaint)
      → Supprimée : zone_A/B/C (remplacée par les 3 types)

    Période industrielle estivale
      • is_early_august   1–20 août : creux maximal, fermetures d'usines
                          Additif à month_8 → capte l'écart début/fin août
                          sans créer de colinéarité (β_month_8 = niveau moyen août,
                          β_early_august = dépression supplémentaire 1–20)

    Inchangé
      • month_1..month_11  dummies mois (décembre = référence)
      • is_eve             veille de JF
      • post_rupture       1 si année ≥ RUPTURE_YEAR
    """
    years       = list(range(dates.year.min(), dates.year.max() + 1))
    public_hols = get_french_public_holidays(years)
    holiday_set = set(public_hols)
    is_holiday  = dates.isin(public_hols)

    df = pd.DataFrame(index=dates)

    # ── Jours de semaine ─────────────────────────────────────────────────────
    # Lundi–vendredi hors JF (les JF sont fusionnés avec le dimanche = référence)
    for i in range(5):
        df[f"dow_{i}"] = ((dates.dayofweek == i) & ~is_holiday).astype(int)

    # Samedi OU pont (hors JF)
    def _is_bridge(d):
        dow = d.dayofweek
        if dow == 4:
            return (d - pd.Timedelta(days=1)) in holiday_set
        if dow == 0:
            return (d + pd.Timedelta(days=1)) in holiday_set
        return False

    is_sat    = (dates.dayofweek == 5) & ~is_holiday
    is_bridge = pd.Series([_is_bridge(d) for d in dates], index=dates, dtype=bool) & ~is_holiday
    df["is_sat_or_bridge"] = (is_sat | is_bridge).astype(int)

    # ── Veille de JF ─────────────────────────────────────────────────────────
    df["is_eve"] = pd.DatetimeIndex(
        [d + pd.Timedelta(days=1) for d in dates]
    ).isin(holiday_set).astype(int)

    # ── Dummies mois ─────────────────────────────────────────────────────────
    if month_effect:
        for m in range(1, 12):
            df[f"month_{m}"] = (dates.month == m).astype(int)

    # ── Vacances scolaires décomposées ───────────────────────────────────────
    vac_df, _ = get_school_holidays(years)
    vac_df = vac_df.set_index("date")
    vac_df.index = pd.DatetimeIndex(vac_df.index)
    df = df.join(vac_df[["zone_A", "zone_B", "zone_C"]], how="left")
    df[["zone_A", "zone_B", "zone_C"]] = df[["zone_A", "zone_B", "zone_C"]].fillna(0)

    is_xmas   = (dates.month == 12) | ((dates.month == 1) & (dates.day <= 7))
    is_summer = dates.month.isin([7, 8])

    for zone in ["A", "B", "C"]:
        base = df[f"zone_{zone}"].astype(bool)
        df[f"vac_noel_zone_{zone}"]  = (base & is_xmas).astype(int)
        df[f"vac_ete_zone_{zone}"]   = (base & is_summer).astype(int)
        df[f"vac_other_zone_{zone}"] = (base & ~is_xmas & ~is_summer).astype(int)

    df = df.drop(columns=["zone_A", "zone_B", "zone_C"])

    # ── Début août — creux industriel ────────────────────────────────────────
    # Additionnel à month_8 : capte le surcroît de dépression des 20 premiers
    # jours (fermetures usines) sans colinéarité exacte avec le dummy mensuel.
    df["is_early_august"] = ((dates.month == 8) & (dates.day <= 20)).astype(int)

    # ── Rupture structurelle ─────────────────────────────────────────────────
    df["post_rupture"] = (dates.year >= RUPTURE_YEAR).astype(int)

    return df

def build_calendar_features_EU(dates: pd.DatetimeIndex, countries: list[str] = ["UK", "BE", "DE", "CH", "IT", "ES", "NL", "IE", "PT"], month_effect: bool = True) -> dict[str, pd.DataFrame]:
    """
    Construit les features calendaires pour chaque pays européen voisin.

    Structure identique à build_calendar_features_v2 :
      • dow_0..dow_4       lundi–vendredi hors JF (dimanche + JF = référence)
      • is_sat_or_bridge   samedi OU pont
      • is_eve             veille de JF
      • month_1..month_11  dummies mois (décembre = référence)
      • vac_noel           décembre + 1–4 janvier
      • vac_early_august   1–20 août (creux industriel estival)

    Returns
    -------
    dict[country_code, pd.DataFrame]
    """
    years       = list(range(dates.year.min(), dates.year.max() + 1))
    public_hols = get_european_public_holidays(years, countries)

    is_xmas         = (dates.month == 12) & (dates.day >= 21)| ((dates.month == 1) & (dates.day <= 4))
    is_early_august = (dates.month == 8) & (dates.day <= 20)

    result = {}

    for country, hols in public_hols.items():
        holiday_set = set(hols)
        is_holiday  = dates.isin(hols)

        df = pd.DataFrame(index=dates)

        # ── Jours de semaine ─────────────────────────────────────────────────
        for i in range(5):
            df[f"dow_{i}"] = ((dates.dayofweek == i) & ~is_holiday).astype(int)

        def _is_bridge(d):
            dow = d.dayofweek
            if dow == 0:
                return (d - pd.Timedelta(days=3)) in holiday_set
            if dow == 4:
                return (d + pd.Timedelta(days=3)) in holiday_set
            return False

        is_sat    = (dates.dayofweek == 5) & ~is_holiday
        is_bridge = pd.Series([_is_bridge(d) for d in dates], index=dates, dtype=bool) & ~is_holiday
        df["is_sat_or_bridge"] = (is_sat | is_bridge).astype(int)

        # ── Veille de JF ─────────────────────────────────────────────────────
        df["is_eve"] = pd.DatetimeIndex(
            [d + pd.Timedelta(days=1) for d in dates]
        ).isin(holiday_set).astype(int)

        # ── Dummies mois ─────────────────────────────────────────────────────
        if month_effect:
            for m in range(1, 12):
                df[f"month_{m}"] = (dates.month == m).astype(int)

        # ── Vacances simplifiées ──────────────────────────────────────────────
        df["vac_noel"]         = is_xmas.astype(int)
        df["vac_early_august"] = is_early_august.astype(int)

        df["post_rupture"] = (dates.year >= RUPTURE_YEAR).astype(int)

        result[country] = df

    return result

# =============================================================================
# 3. MODÈLE TEMPÉRATURE LINÉAIRE PAR MORCEAUX + MATRICE DE FEATURES
# =============================================================================

def build_feature_matrix(temp_region: pd.Series,
                          calendar_df: pd.DataFrame,
                          region: str,
                          region_zone_map: dict,
                          T_c1: float,
                          T_c2: float,
                          vector: str = "elec",
                          month_effect: bool = True) -> pd.DataFrame:
    """
    Construit la matrice X complète pour une région.

    Effet température élec — linéaire par morceaux (forme en U) :
        HDD = max(T_c1 - T, 0)   : degrés-jours chauffage
        CDD = max(T - T_c2, 0)   : degrés-jours climatisation
        Zone neutre [T_c1, T_c2] : plat (HDD = CDD = 0)
    
    Effet température gaz:
        HDD = max(T_c1 − T, 0)  — seule variable température (pas de CDD)

    Autres variables :
        - Dummies jour de semaine (lun-sam, dim = réf.)
        - Dummies mois (jan-nov, déc = réf.)
        - is_holiday, is_eve, is_bridge
        - is_vacances (zone de la région)
        - post_rupture
        - Constante
    """

    zone    = region_zone_map.get(region, "B") #par défaut on attribue la zone B si ça marche pas
    vac_col = f"zone_{zone}"

    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_holiday", "is_eve", "is_bridge", "post_rupture"]

    if vector == "elec":
        hdd_cdd = pd.DataFrame({
            "HDD": np.maximum(T_c1 - temp_region, 0),
            "CDD": np.maximum(temp_region - T_c2, 0),
        }, index=temp_region.index)

        X = pd.concat([
            hdd_cdd,
            calendar_df[dow_cols + month_cols + other_cols],
            calendar_df[[vac_col]].rename(columns={vac_col: "is_vacances"}),
        ], axis=1)
    
    else: #vector is gas
        hdd = pd.DataFrame({
            "HDD": np.maximum(T_c1 - temp_region, 0),
        }, index=temp_region.index)

        X = pd.concat([
            hdd,
            calendar_df[dow_cols + month_cols + other_cols],
            calendar_df[[vac_col]].rename(columns={vac_col: "is_vacances"}),
        ], axis=1)


    common = temp_region.dropna().index.intersection(X.dropna().index)
    return sm.add_constant(X.loc[common], has_constant="add")


def find_optimal_breakpoints(conso_region: pd.Series,
                              temp_region: pd.Series,
                              calendar_df: pd.DataFrame,
                              region: str,
                              region_zone_map: dict,
                              t1_grid: np.ndarray = T1_GRID,
                              t2_grid: np.ndarray = T2_GRID,
                              t_grid_gas: np.ndarray = T_GAS_GRID,
                              vector: str = "elec",
                              month_effect: bool = True) -> tuple[float, float]:
    """
    Trouve (T_c1, T_c2) minimisant le BIC par recherche de grille.

    Pour chaque paire (t1, t2) avec t1 < t2 :
        - Calcule HDD/CDD
        - Résout OLS via lstsq (rapide, pas de statsmodels)
        - Calcule BIC = n*log(RSS/n) + k*log(n)
    Retourne la paire optimale.
    """
    zone    = region_zone_map.get(region, "B")
    vac_col = f"zone_{zone}"
    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_holiday", "is_eve", "is_bridge", "post_rupture"]
    cal_part = calendar_df[dow_cols + month_cols + other_cols + [vac_col]].rename(
        columns={vac_col: "is_vacances"}
    )
    # Constante + calendrier (fixe pour toute la grille)
    cal_const = np.hstack([
        np.ones((len(cal_part), 1)),
        cal_part.values,
    ])

    common = (conso_region.dropna().index
              .intersection(temp_region.dropna().index)
              .intersection(cal_part.dropna().index))

    y   = conso_region.loc[common].values.astype(float)
    T   = temp_region.loc[common].values.astype(float)
    cal = cal_const[cal_part.index.get_indexer(common)]
    n   = len(y)


    if vector == "elec":
        best_bic = np.inf
        best_t1, best_t2 = t1_grid[0], t2_grid[-1]

        for t1 in t1_grid:
            for t2 in t2_grid:
                if t1 >= t2:
                    continue
                hdd = np.maximum(t1 - T, 0).reshape(-1, 1)
                cdd = np.maximum(T - t2, 0).reshape(-1, 1)
                X_np = np.hstack([hdd, cdd, cal])
                try:
                    beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
                    if len(res_arr) == 0:
                        resid = y - X_np @ beta
                        rss   = float(np.dot(resid, resid))
                    else:
                        rss = float(res_arr[0])
                    k   = X_np.shape[1]
                    bic = n * np.log(rss / n) + k * np.log(n)
                    if bic < best_bic:
                        best_bic  = bic
                        best_t1, best_t2 = t1, t2
                except np.linalg.LinAlgError:
                    continue

        return best_t1, best_t2
    
    else: #vector is gas
        """
        Trouve le seuil T_c1 unique minimisant le BIC — modèle gaz.

        Modèle gaz :  f(T) = β_h * max(T_c1 − T, 0)
            T < T_c1  : pente négative (part thermosensible chauffage)
            T ≥ T_c1  : plat (zone de confort, pas de chauffage gaz)
            
        Pas de terme CDD (le gaz ne sert pas à la climatisation).
        Recherche 1-D sur t_grid (grille fine par défaut : 8–20°C, pas 0.05°C).
        """
        best_bic = np.inf
        best_t1  = t_grid_gas[len(t_grid_gas) // 2]

        for t1 in t_grid_gas:
            hdd  = np.maximum(t1 - T, 0).reshape(-1, 1)
            X_np = np.hstack([hdd, cal])
            try:
                beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
                if len(res_arr) == 0:
                    resid = y - X_np @ beta
                    rss   = float(np.dot(resid, resid))
                else:
                    rss = float(res_arr[0])
                k   = X_np.shape[1]
                bic = n * np.log(rss / n) + k * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_t1  = t1
            except np.linalg.LinAlgError:
                continue

        return best_t1

def find_optimal_breakpoints_EU(conso_region: pd.Series,
                              temp_region_hdd: pd.Series,
                              temp_region_cdd: pd.Series,
                              calendar_df: pd.DataFrame,
                              region: str,
                              t1_grid: np.ndarray = T1_GRID,
                              t2_grid: np.ndarray = T2_GRID,
                              month_effect: bool = True) -> tuple[float, float]:

    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_sat_or_bridge", "is_eve", "post_rupture", "vac_early_august", "vac_noel"]

    cal_part = calendar_df[dow_cols + month_cols + other_cols]
    cal_const = np.hstack([np.ones((len(cal_part), 1)), cal_part.values])

    common = (conso_region.dropna().index
              .intersection(temp_region_hdd.dropna().index)
              .intersection(cal_part.dropna().index))

    y   = conso_region.loc[common].values.astype(float)
    Th   = temp_region_hdd.loc[common].values.astype(float)
    Tc   = temp_region_cdd.loc[common].values.astype(float)
    cal = cal_const[cal_part.index.get_indexer(common)]
    n   = len(y)

    best_bic = np.inf
    best_t1, best_t2 = t1_grid[0], t2_grid[-1]

    for t1 in t1_grid:
        for t2 in t2_grid:
            if t1 >= t2:
                continue
            hdd = np.maximum(t1 - Th, 0).reshape(-1, 1)
            cdd = np.maximum(Tc - t2, 0).reshape(-1, 1)
            X_np = np.hstack([hdd, cdd, cal])
            try:
                beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
                if len(res_arr) == 0:
                    resid = y - X_np @ beta
                    rss   = float(np.dot(resid, resid))
                else:
                    rss = float(res_arr[0])
                k   = X_np.shape[1]
                bic = n * np.log(rss / n) + k * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_t1, best_t2 = t1, t2
            except np.linalg.LinAlgError:
                continue

    return best_t1, best_t2


def build_feature_matrix_EU(temp_region: pd.Series,
                              calendar_df: pd.DataFrame,
                              T_c1: float,
                              T_c2: float,
                              month_effect: bool = True) -> pd.DataFrame:
    """
    Construit la matrice X pour un pays européen (calendrier EU).

    Pas de region_zone_map : vacances simplifiées (vac_noel, vac_early_august).
    HDD/CDD sont initialisés ici puis écrasés dans fit_all_countries pour
    appliquer l'asymétrie thermique (T_eff pour HDD, T_obs pour CDD).
    """
    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_sat_or_bridge", "is_eve", "post_rupture", "vac_noel", "vac_early_august"]

    hdd_cdd = pd.DataFrame({
        "HDD": np.maximum(T_c1 - temp_region, 0),
        "CDD": np.maximum(temp_region - T_c2, 0),
    }, index=temp_region.index)

    X = pd.concat([hdd_cdd, calendar_df[dow_cols + month_cols + other_cols]], axis=1)
    common = temp_region.dropna().index.intersection(X.dropna().index)
    return sm.add_constant(X.loc[common], has_constant="add")


# =============================================================================
# 4. ESTIMATION OLS PAR RÉGION
# =============================================================================

def fit_region_model(conso_region: pd.Series, X: pd.DataFrame):
    """OLS avec erreurs robustes HAC (Newey-West, maxlags=10).
    HAC corrige à la fois l'hétéroscédasticité ET l'autocorrélation des résidus,
    contrairement à HC3 qui ne corrige que l'hétéroscédasticité.
    Les coefficients OLS restent non biaisés malgré l'autocorrélation.
    """
    common = conso_region.dropna().index.intersection(X.index)
    return sm.OLS(conso_region.loc[common], X.loc[common]).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}
    )


def fit_all_regions(conso: pd.DataFrame,
                    temp: pd.DataFrame,
                    calendar_df: pd.DataFrame,
                    region_zone_map: dict,
                    vector:str = "elec",
                    month_effect: bool = True) -> dict:
    """
    Estime un modèle OLS par région.

    Returns
    -------
    models : dict { région : {"results", "make_basis", "knot_values"} }
    """
    models = {}
    print("\n=== ESTIMATION OLS PAR RÉGION ===\n")
    print(f"{'Région':<35} {'R²':>6} {'R²adj':>6} {'RMSE':>8} {'Tc1':>5} {'Tc2':>5} {'N':>5}")
    print("-" * 75)

    for region in conso.columns:
        if region not in temp.columns:
            print(f"  [WARN] {region} absent de temp — ignorée")
            continue

        print(f"  {region:<33} recherche seuils...", end="\r")

        if vector == "elec":
            T_c1, T_c2 = find_optimal_breakpoints(
                conso[region], temp[region], calendar_df, region, region_zone_map, month_effect=month_effect
            )
            X       = build_feature_matrix(temp[region], calendar_df, region, region_zone_map, T_c1, T_c2, month_effect=month_effect)

        else: # vector is gas
            """Différence clé vs elec :
            - Recherche de seuil unique T_c1 (grille fine T_GAS_GRID)
            - Seul HDD dans la matrice X (pas de CDD)
            - T_c2 = None dans le model_dict"""

            T_c1    = find_optimal_breakpoints(
                conso[region], temp[region], calendar_df, region, region_zone_map, vector="gas", month_effect=month_effect
            )
            X       = build_feature_matrix(temp[region], calendar_df, region, region_zone_map, T_c1, T_c2=None, vector="gas", month_effect=month_effect)

        results = fit_region_model(conso[region], X)
        rmse    = np.sqrt(results.mse_resid)

        if vector == "elec":
            models[region] = {"results": results, "T_c1": T_c1, "T_c2": T_c2}
            print(f"  {region:<33} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
                f"{rmse:>8.1f} {T_c1:>5.1f} {T_c2:>5.1f} {int(results.nobs):>5}")
        
        else: #vector is gas
            models[region] = {"results": results, "T_c1": T_c1, "T_c2": None}
            print(f"  {region:<33} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
                   f"{rmse:>8.1f} {T_c1:>6.2f} {int(results.nobs):>5}")

    return models



def fit_all_countries(conso: pd.DataFrame,
                      temp: pd.DataFrame,
                      calendar_df: dict,
                      exclude_years: list = None,
                      use_effective_temp: bool = True,
                      alpha_eff: float = 0.7,
                      alpha_cdd: float = 1,
                      month_effect: bool = True) -> dict:
    """
    Estime un modèle OLS par pays européen.

    Parameters
    ----------
    conso              : DataFrame (index DatetimeIndex, colonnes = pays)
    temp               : DataFrame températures (idem)
    calendar_df        : dict[country, pd.DataFrame] issu de build_calendar_features_EU
    exclude_years      : années à exclure (ex: [2020, 2021])
    use_effective_temp : True → HDD sur T_eff = EWMA(T, alpha), False → T brute
    alpha_eff,_cdd     : paramètre EWMA pour l'inertie thermique
    month_effect       : True → inclut les dummies mois, False → les supprime

    Returns
    -------
    models : dict { pays : {"results", "T_c1", "T_c2", "alpha_eff", "alpha_cdd"} }
    """
    if exclude_years is None:
        exclude_years = EXCLUDE_YEARS

    label_eff = f"T_eff(alpha={alpha_eff})" if use_effective_temp else "T_obs"
    label_cdd = f"T_cdd(alpha={alpha_cdd})" if use_effective_temp else "T_obs"
    print("\n=== ESTIMATION OLS PAR PAYS ===\n")
    print(f"  Température HDD  : {label_eff}")
    print(f"  Température CDD  : {label_cdd}")
    print(f"  Années exclues   : {exclude_years}\n")
    print(f"{'Pays':<10} {'R²':>6} {'R²adj':>6} {'RMSE':>8} {'Tc1':>5} {'Tc2':>5} {'N':>5}")
    print("-" * 55)

    models = {}

    for country in conso.columns:
        if country not in temp.columns:
            print(f"  [WARN] {country} absent de temp — ignoré")
            continue
        if country not in calendar_df:
            print(f"  [WARN] {country} absent de calendar_df — ignoré")
            continue

        cal       = calendar_df[country]
        all_dates = temp[country].index.union(conso[country].index)
        keep_mask = ~all_dates.year.isin(exclude_years)

        T_hdd = compute_effective_temperature(temp[country], alpha=alpha_eff) \
                if use_effective_temp else temp[country]
        T_cdd = compute_effective_temperature(temp[country], alpha=alpha_cdd)

        train_idx   = T_hdd.dropna().index[T_hdd.dropna().index.isin(all_dates[keep_mask])]
        conso_train = conso[country].reindex(train_idx)

        print(f"  {country:<8} recherche seuils...", end="\r")

        T_c1, T_c2 = find_optimal_breakpoints_EU(
            conso_train, T_hdd.reindex(train_idx), T_cdd.reindex(train_idx), cal.reindex(train_idx), country,
            month_effect=month_effect
        )

        X = build_feature_matrix_EU(
            T_hdd.reindex(train_idx), cal.reindex(train_idx), T_c1, T_c2,
            month_effect=month_effect
        )
        # Asymétrie thermique : T_eff pour chauffage, T_obs pour climatisation
        X["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X.index), 0)
        X["CDD"] = np.maximum(T_cdd.reindex(X.index) - T_c2, 0)

        results = fit_region_model(conso[country], X)
        rmse    = np.sqrt(results.mse_resid)

        models[country] = {
            "results":   results,
            "T_c1":      T_c1,
            "T_c2":      T_c2,
            "alpha_eff": alpha_eff if use_effective_temp else None,
            "alpha_cdd": alpha_cdd if use_effective_temp else None,
        }

        print(f"  {country:<8} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
              f"{rmse:>8.1f} {T_c1:>5.1f} {T_c2:>5.1f} {int(results.nobs):>5}")

    return models

    



# =============================================================================
# 5. DIAGNOSTIC & VALIDATION
# =============================================================================

def plot_temperature_response(models: dict,
                               temp: pd.DataFrame,
                               save: bool = True,
                               lan: str = "FR"):
    """
    Trace la réponse estimée f(T) = beta_h*HDD + beta_c*CDD pour chaque région.
    Les lignes pointillées indiquent les seuils T_c1 et T_c2 estimés.
    """
    n_regions = len(models)
    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    is_gas = all(m.get("T_c2") is None for m in models.values())
    fig.suptitle(
        "Estimated response f(T) — " + ("gas (HDD only)" if is_gas else "electricity (PWL)"),
        fontsize=14, fontweight="bold"
    )

    for idx, (region, m) in enumerate(models.items()):
        ax      = axes[idx]
        results = m["results"]
        T_c1    = m["T_c1"]
        T_c2    = m.get("T_c2")
        beta_h  = results.params.get("HDD", 0.0)
        beta_c  = results.params.get("CDD", 0.0) if T_c2 is not None else 0.0

        t_grid = np.linspace(temp[region].min() - 1, temp[region].max() + 1, 300)
        if T_c2 is not None:
            f_T = beta_h * np.maximum(T_c1 - t_grid, 0) + beta_c * np.maximum(t_grid - T_c2, 0)
        else:
            f_T = beta_h * np.maximum(T_c1 - t_grid, 0)

        ax.plot(t_grid, f_T, color="steelblue", lw=2.5, label="f(T) estimated")
        ax.axvline(T_c1, color="red", lw=1.2, ls="--", label=f"Tc1={T_c1:.2f}°C")
        if T_c2 is not None:
            ax.axvline(T_c2, color="orange", lw=1.2, ls="--", label=f"Tc2={T_c2:.1f}°C")
        ax.axhline(0, color="black", lw=0.8, ls="--")

        ax2 = ax.twinx()
        ax2.hist(temp[region].dropna(), bins=40, alpha=0.15, color="gray")
        ax2.set_ylabel("Frequency", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

        ax.set_title(region, fontsize=9, fontweight="bold")
        ax.set_xlabel("Effective temperature (°C)")
        ax.set_ylabel("Impact on daily average demand (MW)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "temperature_response_pwl.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[diag] temperature_response_pwl.png sauvegardé")
    else:
        plt.show()


def plot_diagnostics(results, region: str, save: bool = True):
    """4 graphiques de diagnostic OLS standard."""
    residuals = results.resid
    fitted    = results.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"OLS Diagnostics — {region}", fontsize=13, fontweight="bold")

    axes[0,0].scatter(fitted, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0,0].axhline(0, color="red", lw=1.5)
    axes[0,0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

    stats.probplot(residuals, plot=axes[0,1])
    axes[0,1].set_title("QQ-Plot of residuals")

    plot_acf(residuals, lags=40, ax=axes[1,0], alpha=0.05)
    axes[1,0].set_title("Autocorrelation of residuals")

    axes[1,1].plot(residuals.index, residuals, lw=0.8, color="steelblue", alpha=0.7)
    axes[1,1].axhline(0, color="red", lw=1)
    axes[1,1].set_title("Residuals over time")
    axes[1,1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"diag_{region.replace(' ','_')}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_residuals_seasonality(results, region: str, save: bool = True):
    """
    Analyse de la structure saisonnière des résidus OLS.

    3 graphiques complémentaires à plot_diagnostics :
        (1) Boxplots par mois     — révèle un biais saisonnier résiduel
                                    Si centré sur 0 : l'effet mois est bien capturé.
                                    Si décalé sur jan/déc : manque irradiation ou
                                    variable chauffage supplémentaire.
        (2) Boxplots par année    — révèle une dérive structurelle ou un effet
                                    non capturé par post_rupture.
        (3) Heatmap mois × année  — combine les deux : localise précisément
                                    les écarts (biais = anomalie isolée vs bande
                                    entière = saisonnalité systématique).
    """
    resid = results.resid.copy()
    resid.index = pd.DatetimeIndex(resid.index)

    df = pd.DataFrame({
        "resid": resid,
        "month": resid.index.month,
        "year":  resid.index.year,
    })

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"OLS Residual Seasonality — {region}",
                 fontsize=13, fontweight="bold")

    # (1) Par mois
    ax = axes[0]
    data_m = [df[df["month"] == m]["resid"].values for m in range(1, 13)]
    bp = ax.boxplot(data_m, labels=month_names, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.axhline(0, color="red", lw=1.2, ls="--")
    ax.set_title("(1) Residuals by month")
    ax.set_ylabel("Residual (MWh/day)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # (2) Par année
    ax = axes[1]
    years_sorted = sorted(df["year"].unique())
    data_y = [df[df["year"] == y]["resid"].values for y in years_sorted]
    bp2 = ax.boxplot(data_y, labels=years_sorted, showfliers=False, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.axhline(0, color="red", lw=1.2, ls="--")
    ax.set_title("(2) Residuals by year")
    ax.set_ylabel("Residual (MWh/day)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # (3) Heatmap mois × année
    ax = axes[2]
    pivot = df.groupby(["year", "month"])["resid"].mean().unstack("month")
    pivot.columns = [month_names[c - 1] for c in pivot.columns]
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("(3) Mean residual (month × year)")
    fig.colorbar(im, ax=ax, label="Mean residual (MWh/day)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > vmax * 0.6 else "black")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"residuals_seasonality_{region.replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[diag] residuals_seasonality_{region.replace(' ', '_')}.png sauvegardé")
    else:
        plt.show()


def plot_daily_residuals_vs_temperature(
    models: dict,
    temp: pd.DataFrame,
) -> None:
    """
    For each region: scatter plot of daily OLS residuals vs temperature,
    MAE by temperature bin, and mean residual by (month × T-bin).

    Purpose: check whether the PWL temperature model (HDD/CDD) has left any
    residual temperature signal. A flat scatter around 0 means f(T) is well
    specified. A visible trend means there is remaining non-linearity that
    the model did not capture (e.g., wrong breakpoint, missing interaction).

    Three panels per region:
        (1) Scatter resid vs T + 1°C-bin mean   — remaining temperature structure
        (2) MAE by temperature bin              — where are errors largest?
        (3) Heatmap mean residual (month × T)   — month-temperature interaction
    """
    from scipy.stats import pearsonr

    n_regions = len(models)
    ncols = 3
    nrows = n_regions

    fig, all_axes = plt.subplots(n_regions, 3, figsize=(16, 4 * n_regions))
    if n_regions == 1:
        all_axes = all_axes[np.newaxis, :]

    fig.suptitle("Daily OLS residuals vs temperature", fontsize=14, fontweight="bold")

    for row_idx, (region, m) in enumerate(models.items()):
        resid = m["results"].resid.copy()
        resid.index = pd.DatetimeIndex(resid.index)

        T = temp[region].reindex(resid.index) if region in temp.columns else None
        if T is None or T.isna().all():
            for ax in all_axes[row_idx]:
                ax.text(0.5, 0.5, f"{region}\n(no temperature)", ha="center",
                        va="center", transform=ax.transAxes)
            continue

        df = pd.DataFrame({"resid": resid, "T": T, "month": resid.index.month}).dropna()

        t_bins = np.arange(np.floor(df["T"].min()), np.ceil(df["T"].max()) + 1, 1.0)
        df["T_bin"] = pd.cut(df["T"], bins=t_bins)
        bin_centers  = df.groupby("T_bin")["T"].mean()
        mean_resid   = df.groupby("T_bin")["resid"].mean()
        mae_resid    = df.groupby("T_bin")["resid"].apply(lambda x: x.abs().mean())

        # (1) Scatter + mean per °C
        ax = all_axes[row_idx, 0]
        ax.scatter(df["T"], df["resid"], s=4, alpha=0.25, color="steelblue")
        ax.axhline(0, color="black", lw=0.8)
        ax.plot(bin_centers.values, mean_resid.values, color="tomato", lw=2,
                label="mean per °C")
        r2 = pearsonr(df["T"], df["resid"])[0] ** 2
        ax.set_title(f"{region}  |  R²(resid~T)={r2:.4f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Residual (MWh/day)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (2) MAE by temperature bin
        ax = all_axes[row_idx, 1]
        ax.bar(bin_centers.values, mae_resid.values, width=0.8,
               color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("MAE (MWh/day)")
        ax.set_title(f"{region} — MAE by temperature bin", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # (3) Mean residual by (month × T-bin) heatmap
        ax = all_axes[row_idx, 2]
        pivot = df.groupby(["month", "T_bin"])["resid"].mean().unstack("T_bin")
        col_mids = [b.mid for b in pivot.columns]
        vmax = np.nanmax(np.abs(pivot.values)) if pivot.size > 0 else 1.0
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[min(col_mids), max(col_mids),
                                len(pivot.index) - 0.5, -0.5])
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([month_labels[m - 1] for m in pivot.index])
        ax.set_xlabel("Temperature (°C)")
        ax.set_title(f"{region} — mean residual (month × T)", fontsize=9)
        fig.colorbar(im, ax=ax, label="Mean residual (MWh/day)", shrink=0.8)

    fig.tight_layout()
    plt.show()


def ljung_box_test(results, region: str, lags: int = 10):
    """Test de Ljung-Box sur les résidus."""
    lb  = acorr_ljungbox(results.resid, lags=lags, return_df=True)
    sig = lb[lb["lb_pvalue"] < 0.05]
    tag = "[WARN]" if len(sig) > 0 else "[OK]  "
    msg = f"autocorrélation aux lags {list(sig.index)}" if len(sig) > 0 \
          else f"pas d'autocorrélation (lags 1-{lags})"
    print(f"  {tag} {region}: {msg}")


def summarize_model_stats(models: dict) -> pd.DataFrame:
    """Tableau récapitulatif R², RMSE, AIC, BIC, nœuds par région."""
    rows = []
    for region, m in models.items():
        res = m["results"]
        rows.append({
            "region":    region,
            "R2":        round(res.rsquared, 4),
            "R2_adj":    round(res.rsquared_adj, 4),
            "RMSE":      round(np.sqrt(res.mse_resid), 2),
            "nRMSE_%":   round(np.sqrt(res.mse_resid) / np.mean(res.model.endog) * 100, 2),
            "N_obs":     int(res.nobs),
            "AIC":       round(res.aic, 1),
            "BIC":       round(res.bic, 1),
            "T_c1_°C":   m["T_c1"],
            "T_c2_°C":   m.get("T_c2"),   # None pour les modèles gaz
        })
    return pd.DataFrame(rows).set_index("region")


# =============================================================================
# 6. EXTRACTION DE LA BASELINE
# =============================================================================

def compute_normal_temperature(temp_region: pd.Series) -> pd.Series:
    """
    Température "normale" = moyenne par jour de l'année (day of year)
    sur la période disponible.

    Note : si tu disposes des normales climatologiques 1981-2010 de
    Météo-France pour chaque région, utilise-les à la place — elles sont
    plus représentatives du "climat de référence".
    """
    doy_mean = temp_region.groupby(temp_region.index.dayofyear).mean()
    return pd.Series(
        [doy_mean.get(d.dayofyear, temp_region.mean()) for d in temp_region.index],
        index=temp_region.index,
        name="T_normal"
    )


def extract_baseline(model_dict: dict, temp_region: pd.Series) -> pd.Series:
    """
    Baseline = consommation prédite à température normale.

    Formule :
        baseline(t) = ŷ(t) − f(T_obs(t)) + f(T_normale(t))

    Avec f(T) = beta_h * max(T_c1 - T, 0) + beta_c * max(T - T_c2, 0).
    Tous les autres effets (calendaire, tendance) sont conservés.

    Parameters
    ----------
    model_dict  : {"results", "T_c1", "T_c2"}
    temp_region : série de températures observées
    """
    results = model_dict["results"]
    T_c1    = model_dict["T_c1"]
    T_c2    = model_dict.get("T_c2")          # None pour les modèles gaz
    beta_h  = results.params.get("HDD", 0.0)
    beta_c  = results.params.get("CDD", 0.0)  # 0 si absent (gaz)

    t_normal = compute_normal_temperature(temp_region)
    common   = results.fittedvalues.index.intersection(temp_region.dropna().index)

    y_pred  = results.fittedvalues.loc[common]
    t_obs   = temp_region.loc[common]
    t_norm  = t_normal.loc[common]

    f_obs  = beta_h * np.maximum(T_c1 - t_obs,  0)
    f_norm = beta_h * np.maximum(T_c1 - t_norm, 0)
    if T_c2 is not None:
        f_obs  += beta_c * np.maximum(t_obs  - T_c2, 0)
        f_norm += beta_c * np.maximum(t_norm - T_c2, 0)

    return pd.Series(y_pred.values - f_obs.values + f_norm.values, index=common, name="baseline")


def extract_all_baselines(models: dict, temp: pd.DataFrame) -> pd.DataFrame:
    """Extrait la baseline pour toutes les régions."""
    return pd.DataFrame({
        region: extract_baseline(m, temp[region])
        for region, m in models.items()
        if region in temp.columns
    })


def predict_demand(model_dict: dict,
                   region: str,
                   region_zone_map: dict,
                   temp_new: pd.Series, is_France: bool = True) -> pd.DataFrame:
    """
    Reconstruit le profil de consommation attendu pour de nouvelles données.

    À partir du modèle ajusté d'une région, construit automatiquement toutes
    les variables calendaires (jours fériés, vacances, ponts, etc.) et prédit
    la consommation jour par jour.

    Décomposition retournée :
        y_pred  : demande totale prédite (calendaire + effet T° + rupture structurelle)
        y_base  : composante structurelle pure — pas d'effet température (HDD=CDD=0) . Représente la conso "normale" par type de jour,
                  indépendamment des conditions météo.
        y_temp  : effet température = y_pred - y_base

    Parameters
    ----------
    model_dict      : dict retourné par fit_all_regions, contient "results", "T_c1", "T_c2"
    region          : nom de la région (ex. "Île-de-France")
    region_zone_map : dict région → zone scolaire (retourné par get_school_holidays)
    temp_new        : pd.Series de températures journalières — index DatetimeIndex

    Returns
    -------
    pd.DataFrame  index = dates, colonnes = [y_pred, y_base, y_temp]

    Exemple
    -------
    >>> _, region_zone_map = get_school_holidays(list(range(2010, 2025)))
    >>> df = predict_demand(models["Île-de-France"], "Île-de-France",
    ...                     region_zone_map, temp["Île-de-France"].loc["2022"])
    >>> df.plot()
    """
    results     = model_dict["results"]
    T_c1        = model_dict["T_c1"]
    T_c2        = model_dict.get("T_c2")          # None pour les modèles gaz
    calendar_v  = model_dict.get("calendar_v", 1) # 1 (défaut) ou 2
    expected    = results.model.exog_names         # colonnes exactes du modèle entraîné

    dates        = pd.DatetimeIndex(temp_new.index)

    if is_France:
        if calendar_v == 2: #if v2
            alpha_eff = model_dict.get("alpha_eff")
            alpha_cdd = model_dict.get("alpha_cdd")
            T_hdd = compute_effective_temperature(temp_new, alpha=alpha_eff) if alpha_eff is not None else temp_new
            T_cdd = compute_effective_temperature(temp_new, alpha=alpha_cdd) if alpha_cdd is not None else temp_new
            calendar_new = build_calendar_features_v2(dates)

            if T_c2 is None: #gas
                X_new = build_feature_matrix_v2(T_hdd, calendar_new, region, region_zone_map, T_c1, T_c2, vector="gas")
                X_new["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X_new.index), 0)
                HDD = X_new["HDD"]
            else:
                X_new = build_feature_matrix_v2(T_hdd, calendar_new, region, region_zone_map, T_c1, T_c2)
                X_new["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X_new.index), 0)
                X_new["CDD"] = np.maximum(T_cdd.reindex(X_new.index) - T_c2, 0)
                HDD = X_new["HDD"]
                CDD = X_new["CDD"]

        elif T_c2 is None: # if gas
            calendar_new = build_calendar_features(dates)
            X_new = build_feature_matrix(temp_new, calendar_new, region, region_zone_map, T_c1, T_c2, vector = "gas")
            HDD = 0
        else:
            calendar_new = build_calendar_features(dates)
            X_new = build_feature_matrix(temp_new, calendar_new, region, region_zone_map, T_c1, T_c2)
            HDD = 0
            CDD = 0

    else: # european countries
        alpha_eff = model_dict.get("alpha_eff")
        alpha_cdd = model_dict.get("alpha_cdd")
        T_hdd = compute_effective_temperature(temp_new, alpha=alpha_eff) if alpha_eff is not None else temp_new
        T_cdd = compute_effective_temperature(temp_new, alpha=alpha_cdd) if alpha_cdd is not None else temp_new

        cal_dict     = build_calendar_features_EU(dates, [region])
        calendar_new = cal_dict[region]
        X_new        = build_feature_matrix_EU(T_hdd, calendar_new, T_c1, T_c2)
        X_new["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X_new.index), 0)
        if T_c2 is not None:
            X_new["CDD"] = np.maximum(T_cdd.reindex(X_new.index) - T_c2, 0)
        HDD = X_new["HDD"] 
        CDD = X_new["CDD"]

    # Aligner les colonnes sur celles du modèle entraîné (ordre + présence exacte)
    X_new = X_new.reindex(columns=expected, fill_value=0.0)

    # Prédiction complète (calendaire + température)
    y_pred = pd.Series(results.predict(X_new), index=X_new.index)

    # Composante structurelle : annule l'effet température
    X_base = X_new.copy()
    X_base["HDD"] = 0.0
    if "CDD" in X_base.columns:
        X_base["CDD"] = 0.0

    y_base = pd.Series(results.predict(X_base), index=X_new.index)

    return pd.DataFrame({
        "y_pred": y_pred,
        "y_base": y_base,
        "y_temp": y_pred - y_base,
        "HDD": HDD,
        **({"CDD": CDD} if T_c2 is not None else {}),
    })


def plot_decomposition(conso: pd.DataFrame,
                       baselines: pd.DataFrame,
                       region: str,
                       save: bool = True):
    """
    Décomposition pour une région :
        Panel 1 : Consommation observée vs baseline
        Panel 2 : Effet température = conso - baseline
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Decomposition — {region}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(conso[region], lw=0.8, color="steelblue", alpha=0.6, label="Observed consumption")
    ax.plot(baselines[region], lw=1.5, color="red", label="Baseline (normal T)")
    ax.set_ylabel("MWh/day")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    effet = conso[region].loc[baselines[region].index] - baselines[region]
    ax.fill_between(effet.index, effet, 0, where=(effet > 0),
                    color="orangered", alpha=0.5, label="Positive deviation")
    ax.fill_between(effet.index, effet, 0, where=(effet < 0),
                    color="steelblue", alpha=0.5, label="Negative deviation")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Deviation from normal-T baseline (MWh/day)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"decomp_{region.replace(' ','_')}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =============================================================================
# 7. EXPORT DES RÉSULTATS
# =============================================================================

def export_results(models: dict, baselines: pd.DataFrame):
    """
    Exporte :
        baselines.csv     : baseline journalière par région
        summary_stats.csv : R², RMSE, AIC, BIC, positions des nœuds
        residuals.csv     : résidus par région
        spline_coefs.csv  : coefficients des termes spline par région
    """
    baselines.to_csv(OUTPUT_DIR / "baselines.csv")
    print(f"\n[export] baselines.csv")

    summarize_model_stats(models).to_csv(OUTPUT_DIR / "summary_stats.csv")
    print(f"[export] summary_stats.csv")

    pd.DataFrame({r: m["results"].resid for r, m in models.items()}).to_csv(
        OUTPUT_DIR / "residuals.csv"
    )
    print(f"[export] residuals.csv")

    pwl_thresholds = {
        region: {"T_c1": m["T_c1"], "T_c2": m["T_c2"],
                 "beta_HDD": m["results"].params.get("HDD"),
                 "beta_CDD": m["results"].params.get("CDD")}
        for region, m in models.items()
    }
    pd.DataFrame(pwl_thresholds).T.to_csv(OUTPUT_DIR / "pwl_thresholds.csv")
    print("[export] pwl_thresholds.csv")

    print(f"\nTous les fichiers dans : {OUTPUT_DIR.resolve()}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(conso_path: str, temp_path: str, vector = "elec", month_effect: bool = True):
    """Pipeline complet."""

    conso, temp = load_data(conso_path, temp_path)

    print("\n[calendar] Génération des variables calendaires...")
    calendar_df = build_calendar_features(conso.index)
    _, region_zone_map = get_school_holidays(list(range(2010, 2025)))
    print(f"[calendar] {len(calendar_df.columns)} variables générées")

    if vector == "gas":
        models = fit_all_regions(conso, temp, calendar_df, region_zone_map, vector=vector, month_effect=month_effect)
    else:
        models = fit_all_regions(conso, temp, calendar_df, region_zone_map, month_effect=month_effect)

    print("\n[diag] Réponses f(T) par région...")
    plot_temperature_response(models, temp, save=True)

    print("\n=== TESTS DE LJUNG-BOX ===\n")
    for region, m in models.items():
        ljung_box_test(m["results"], region)

    print("\n[diag] Graphiques de diagnostic OLS...")
    for region, m in models.items():
        plot_diagnostics(m["results"], region, save=True)

    print("\n[baseline] Extraction des baselines...")
    baselines = extract_all_baselines(models, temp)

    print("[baseline] Graphiques de décomposition...")
    for region in baselines.columns:
        plot_decomposition(conso, baselines, region, save=True)

    export_results(models, baselines)

    if vector == "gas":
        print("\n=== PIPELINE GAZ TERMINÉE ===")
    else:
        print("\n=== PIPELINE TERMINÉE ===")

    return models, baselines, calendar_df

def run_pipeline_v2(conso_path: str, temp_path: str, inertia_effect: bool = True, training_period: list = None, exclude_full_covid_y: bool = True, vector = "elec", alpha = 0.7, alpha_cdd = 1, month_effect: bool = True):
    """Pipeline complet."""

    conso, temp = load_data(conso_path, temp_path, training_period, exclude_full_covid_y)

    print("\n[calendar] Génération des variables calendaires...")
    calendar_df = build_calendar_features_v2(conso.index) ## ICI v2 car différents features
    _, region_zone_map = get_school_holidays(list(range(2010, 2025)))
    print(f"[calendar] {len(calendar_df.columns)} variables générées")

    if vector == "gas":
        models = fit_all_regions_v2(conso, temp, calendar_df, region_zone_map, use_effective_temp=inertia_effect, alpha_eff=alpha, vector="gas", month_effect=month_effect)
    else:
        models = fit_all_regions_v2(conso, temp, calendar_df, region_zone_map, use_effective_temp=inertia_effect, alpha_eff=alpha, alpha_cdd=alpha_cdd, month_effect=month_effect)

    print("\n[diag] Réponses f(T) par région...")
    plot_temperature_response(models, temp, save=True)

    print("\n=== TESTS DE LJUNG-BOX ===\n")
    for region, m in models.items():
        ljung_box_test(m["results"], region)

    print("\n[diag] Graphiques de diagnostic OLS...")
    for region, m in models.items():
        plot_diagnostics(m["results"], region, save=True)

    print("\n[baseline] Extraction des baselines...")
    baselines = extract_all_baselines(models, temp)

    print("[baseline] Graphiques de décomposition...")
    for region in baselines.columns:
        plot_decomposition(conso, baselines, region, save=True)

    export_results(models, baselines)

    print("\n=== PIPELINE TERMINÉ ===")
    return models, baselines, calendar_df


def run_pipeline_EU(conso_path: str,
                    temp_path: str,
                    countries: list[str] = ["UK", "BE", "DE", "CH", "IT", "ES", "NL", "IE", "PT"],
                    inertia_effect: bool = True,
                    training_period: list = None,
                    exclude_full_covid_y: bool = True,
                    month_effect: bool = True):
    """Pipeline complet pour les pays européens voisins.

    Parameters
    ----------
    conso_path           : chemin CSV consommation (colonnes = pays)
    temp_path            : chemin CSV températures  (colonnes = pays)
    countries            : liste des pays à modéliser
    inertia_effect       : True → HDD sur T_eff EWMA, False → T brute
    training_period      : [date_début, date_fin] ou None (toute la série)
    exclude_full_covid_y : True → exclut les années COVID (EXCLUDE_YEARS)
    month_effect         : True → inclut les dummies mois, False → les supprime
    """
    conso, temp = load_data(conso_path, temp_path, training_period, exclude_full_covid_y)

    print("\n[calendar] Génération des variables calendaires EU...")
    all_dates   = temp.index.union(conso.index)
    calendar_df = build_calendar_features_EU(all_dates, countries)
    print(f"[calendar] {len(next(iter(calendar_df.values())).columns)} variables par pays")

    exclude_years = EXCLUDE_YEARS if exclude_full_covid_y else []
    models = fit_all_countries(
        conso, temp, calendar_df,
        exclude_years=exclude_years,
        use_effective_temp=inertia_effect,
        month_effect=month_effect
    )

    print("\n[diag] Réponses f(T) par pays...")
    plot_temperature_response(models, temp, save=True)

    print("\n=== TESTS DE LJUNG-BOX ===\n")
    for country, m in models.items():
        ljung_box_test(m["results"], country)

    print("\n[diag] Graphiques de diagnostic OLS...")
    for country, m in models.items():
        plot_diagnostics(m["results"], country, save=True)

    print("\n[baseline] Extraction des baselines...")
    baselines = extract_all_baselines(models, temp)

    print("[baseline] Graphiques de décomposition...")
    for country in baselines.columns:
        plot_decomposition(conso, baselines, country, save=True)

    export_results(models, baselines)

    print("\n=== PIPELINE EU TERMINÉ ===")
    return models, baselines, calendar_df


# =============================================================================
# UTILITAIRES SUPPLÉMENTAIRES
# =============================================================================

def national_aggregate(baselines: pd.DataFrame) -> pd.Series:
    """Agrège les baselines régionales en total national."""
    return baselines.sum(axis=1).rename("baseline_nationale")


def plot_bic_landscape(conso_region: pd.Series,
                        temp_region: pd.Series,
                        calendar_df: pd.DataFrame,
                        region: str,
                        region_zone_map: dict,
                        t1_grid: np.ndarray = T1_GRID,
                        t2_grid: np.ndarray = T2_GRID):
    """
    Carte BIC sur la grille (T_c1, T_c2) pour une région.
    Utile pour vérifier l'unicité et la robustesse du minimum.

    Exemple :
        plot_bic_landscape(conso["Île-de-France"], temp["Île-de-France"],
                           calendar_df, "Île-de-France", region_zone_map)
    """
    bic_grid = np.full((len(t1_grid), len(t2_grid)), np.nan)

    zone    = region_zone_map.get(region, "B")
    vac_col = f"zone_{zone}"
    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")]
    other_cols = ["is_holiday", "is_eve", "is_bridge", "post_rupture"]
    cal_part = calendar_df[dow_cols + month_cols + other_cols + [vac_col]].rename(
        columns={vac_col: "is_vacances"}
    )
    cal_const = np.hstack([np.ones((len(cal_part), 1)), cal_part.values])
    common = (conso_region.dropna().index
              .intersection(temp_region.dropna().index)
              .intersection(cal_part.dropna().index))
    y   = conso_region.loc[common].values.astype(float)
    T   = temp_region.loc[common].values.astype(float)
    cal = cal_const[cal_part.index.get_indexer(common)]
    n   = len(y)

    for i, t1 in enumerate(t1_grid):
        for j, t2 in enumerate(t2_grid):
            if t1 >= t2:
                continue
            X_np = np.hstack([
                np.maximum(t1 - T, 0).reshape(-1, 1),
                np.maximum(T - t2, 0).reshape(-1, 1),
                cal,
            ])
            beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
            rss = float(res_arr[0]) if len(res_arr) else float(np.dot(y - X_np @ beta, y - X_np @ beta))
            bic_grid[i, j] = n * np.log(rss / n) + X_np.shape[1] * np.log(n)

    plt.figure(figsize=(8, 6))
    plt.contourf(t2_grid, t1_grid, bic_grid, levels=30, cmap="viridis_r")
    plt.colorbar(label="BIC")
    plt.xlabel("T_c2 — seuil climatisation (°C)")
    plt.ylabel("T_c1 — seuil chauffage (°C)")
    plt.title(f"Paysage BIC — {region}")
    imin, jmin = np.unravel_index(np.nanargmin(bic_grid), bic_grid.shape)
    plt.scatter(t2_grid[jmin], t1_grid[imin], color="red", s=80, zorder=5,
                label=f"min : Tc1={t1_grid[imin]:.1f}, Tc2={t2_grid[jmin]:.1f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 8. TEMPÉRATURE EFFECTIVE — INERTIE THERMIQUE DES BÂTIMENTS
# =============================================================================

def compute_effective_temperature(temp: pd.Series, alpha: float = 0.7) -> pd.Series:
    """
    Température effective par moyenne mobile exponentielle (EWMA).

    Modélise l'inertie thermique des bâtiments : la demande de chauffage
    dépend non seulement de la température du jour, mais aussi des jours
    précédents (masse thermique du bâti).

        T_eff(t) = alpha * T(t) + (1 - alpha) * T_eff(t-1)

    alpha proche de 1 → réponse quasi-instantanée (pas d'inertie)
    alpha proche de 0 → forte inertie (mémoire longue des vagues de froid)

    Note : l'inertie est asymétrique — elle affecte principalement le chauffage
    (hiver), pas la climatisation (AC réagit à la chaleur instantanée), car les gens ont tendance à ouvrir les fenêtres en été.
    Utiliser T_eff pour HDD, T brut pour CDD.

    Référence : Staffell & Pfenninger (2023), Nature — thermal demand inertia.

    Parameters
    ----------
    temp  : températures journalières observées
    alpha : poids de T(t) courant — 0.7 recommandé pour données journalières
    """
    return temp.ewm(alpha=alpha, adjust=False).mean().rename(f"T_eff(a={alpha})")


# =============================================================================
# 9. FEATURES MÉTÉO ADDITIONNELLES (VENT, PRÉCIPITATIONS, IRRADIATION)
# =============================================================================

def build_meteo_features(region_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les variables météorologiques additionnelles pour une région.

    Colonnes attendues dans region_meteo (toutes optionnelles) :
        wind_speed   : vitesse du vent (m/s)
                       → augmente la demande via effet éolien (wind chill)
                       → transformée en log(1 + v) pour effet sous-linéaire
        precip       : précipitations (mm/j)
                       → augmente légèrement la demande (humidité, froid ressenti)
                       → transformée en sqrt pour atténuer les extrêmes
        irradiation  : irradiation GHI (W/m² ou kWh/m²/j)
                       → réduit la demande de chauffage (gains solaires passifs)
                       → linéaire, signe attendu négatif en hiver

    Toutes les variables sont centrées-réduites (z-score) pour comparabilité.

    Returns
    -------
    pd.DataFrame  index = dates, colonnes transformées et normalisées
    """
    features = pd.DataFrame(index=region_meteo.index)

    if "wind_speed" in region_meteo.columns:
        ws = np.log1p(region_meteo["wind_speed"].clip(lower=0))
        features["wind_log"] = (ws - ws.mean()) / ws.std()

    if "precip" in region_meteo.columns:
        pr = np.sqrt(region_meteo["precip"].clip(lower=0))
        features["precip_sqrt"] = (pr - pr.mean()) / pr.std()

    if "irradiation" in region_meteo.columns:
        ir = region_meteo["irradiation"].clip(lower=0)
        features["irradiation"] = (ir - ir.mean()) / ir.std()

    return features


def build_feature_matrix_extended(temp_region: pd.Series,
                                   calendar_df: pd.DataFrame,
                                   region: str,
                                   region_zone_map: dict,
                                   T_c1: float,
                                   T_c2: float,
                                   extra_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Version étendue de build_feature_matrix avec features additionnelles.

    Appelle build_feature_matrix puis concatène extra_features si fourni.
    Les fonctions existantes ne sont pas modifiées.

    Parameters
    ----------
    extra_features : pd.DataFrame (ex: output de build_meteo_features)
                     doit avoir le même index DatetimeIndex que temp_region
    """
    X_base = build_feature_matrix(temp_region, calendar_df, region, region_zone_map, T_c1, T_c2)

    if extra_features is None or extra_features.empty:
        return X_base

    extra  = extra_features.reindex(X_base.index).dropna()
    common = X_base.index.intersection(extra.index)
    return pd.concat([X_base.loc[common], extra.loc[common]], axis=1)


def fit_all_regions_extended(conso: pd.DataFrame,
                              temp: pd.DataFrame,
                              calendar_df: pd.DataFrame,
                              region_zone_map: dict,
                              meteo_dict: dict = None,
                              use_effective_temp: bool = True,
                              alpha_eff: float = 0.7) -> dict:
    """
    Version étendue de fit_all_regions avec inertie thermique et météo.

    Améliorations par rapport à fit_all_regions :
        - HDD calculé sur T_eff = EWMA(T, alpha) au lieu de T instantanée
        - CDD calculé sur T brute (climatisation = réponse instantanée)
        - Features météo additionnelles (vent, précip, irradiation) si fournies

    Parameters
    ----------
    meteo_dict       : { région : pd.DataFrame } — output de build_meteo_features
                       par région. Si None, pas de features météo additionnelles.
    use_effective_temp : True → HDD sur T_eff, False → comportement identique à
                         fit_all_regions
    alpha_eff        : paramètre EWMA (0 < α ≤ 1). 0.7 = inertie modérée.

    Returns
    -------
    models : dict { région : {"results", "T_c1", "T_c2", "alpha_eff"} }
    """
    models = {}
    label_eff = f"T_eff(alpha={alpha_eff})" if use_effective_temp else "T_obs"
    print("\n=== ESTIMATION OLS ÉTENDUE PAR RÉGION ===\n")
    print(f"  Température HDD : {label_eff}")
    print(f"  Météo additionnelle : {'oui (' + str(list(next(iter(meteo_dict.values())).columns)) + ')' if meteo_dict else 'non'}\n")
    print(f"{'Région':<35} {'R²':>6} {'R²adj':>6} {'RMSE':>8} {'Tc1':>5} {'Tc2':>5} {'N':>5}")
    print("-" * 75)

    for region in conso.columns:
        if region not in temp.columns:
            print(f"  [WARN] {region} absent de temp — ignorée")
            continue

        # Température pour HDD : effective ou brute
        T_hdd = compute_effective_temperature(temp[region], alpha=alpha_eff) \
                if use_effective_temp else temp[region]

        # Pour CDD on garde la température instantanée
        T_cdd = temp[region]

        # Features météo additionnelles pour cette région
        extra = meteo_dict.get(region) if meteo_dict else None

        # Grid search sur T_hdd (inertie incluse si use_effective_temp)
        print(f"  {region:<33} recherche seuils...", end="\r")
        T_c1, T_c2 = find_optimal_breakpoints(
            conso[region], T_hdd, calendar_df, region, region_zone_map
        )

        # HDD sur T_eff, CDD sur T_obs → température asymétrique
        hdd_cdd_asym = pd.DataFrame({
            "HDD": np.maximum(T_c1 - T_hdd, 0),
            "CDD": np.maximum(T_cdd - T_c2, 0),
        }, index=T_hdd.index)

        # Construire X : calendaire + HDD_eff/CDD_obs + météo
        zone    = region_zone_map.get(region, "B")
        vac_col = f"zone_{zone}"
        dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
        month_cols = [c for c in calendar_df.columns if c.startswith("month_")]
        other_cols = ["is_holiday", "is_eve", "is_bridge", "post_rupture"]

        X = pd.concat([
            hdd_cdd_asym,
            calendar_df[dow_cols + month_cols + other_cols],
            calendar_df[[vac_col]].rename(columns={vac_col: "is_vacances"}),
        ], axis=1)

        if extra is not None:
            X = pd.concat([X, extra.reindex(X.index)], axis=1)

        common = conso[region].dropna().index.intersection(X.dropna().index)
        X = sm.add_constant(X.loc[common], has_constant="add")

        results = fit_region_model(conso[region], X)
        rmse    = np.sqrt(results.mse_resid)

        models[region] = {
            "results":   results,
            "T_c1":      T_c1,
            "T_c2":      T_c2,
            "alpha_eff": alpha_eff if use_effective_temp else None,
        }

        print(f"  {region:<33} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
              f"{rmse:>8.1f} {T_c1:>5.1f} {T_c2:>5.1f} {int(results.nobs):>5}")

    return models


# =============================================================================
# 9b. MODÈLE V2 — MATRICE DE FEATURES ET ESTIMATION (calendrier v2)
# =============================================================================

def build_feature_matrix_v2(temp_region: pd.Series,
                              calendar_df: pd.DataFrame,
                              region: str,
                              region_zone_map: dict,
                              T_c1: float,
                              T_c2: float,
                              vector: str = "elec",
                              month_effect: bool = True) -> pd.DataFrame:
    """
    Construit la matrice X pour une région à partir d'un calendrier v2
    (issu de build_calendar_features_v2).

    Colonnes calendrier utilisées
    ─────────────────────────────
    • dow_0..dow_4          lundi–vendredi hors JF
    • is_sat_or_bridge      samedi + ponts (hors JF)
    • is_eve                veille de JF
    • month_1..month_11     (omis si month_effect=False)
    • is_early_august       1–20 août
    • post_rupture
    • vac_noel_zone_X       vacances Noël  (déc + jan 1–7)  de la zone de la région
    • vac_ete_zone_X        grandes vacances (jul–août)
    • vac_other_zone_X      vacances restantes (fév, avr, toussaint)
    """

    zone = region_zone_map.get(region, "B")

    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_sat_or_bridge", "is_eve", "post_rupture", "is_early_august"]
    vac_cols   = [f"vac_noel_zone_{zone}", f"vac_ete_zone_{zone}", f"vac_other_zone_{zone}"]

    if vector == "elec":
        hdd_cdd = pd.DataFrame({
            "HDD": np.maximum(T_c1 - temp_region, 0),
            "CDD": np.maximum(temp_region - T_c2, 0),
        }, index=temp_region.index)

        X = pd.concat([
            hdd_cdd,
            calendar_df[dow_cols + month_cols + other_cols + vac_cols],
        ], axis=1)
    
    else: #vector is gas
        hdd = pd.DataFrame({
            "HDD": np.maximum(T_c1 - temp_region, 0),
        }, index=temp_region.index)

        X = pd.concat([
            hdd,
            calendar_df[dow_cols + month_cols + other_cols + vac_cols],
        ], axis=1)


    common = temp_region.dropna().index.intersection(X.dropna().index)
    return sm.add_constant(X.loc[common], has_constant="add")


def find_optimal_breakpoints_v2(conso_region: pd.Series,
                                  temp_region_hdd: pd.Series,
                                  temp_region_cdd: pd.Series,
                                  calendar_df: pd.DataFrame,
                                  region: str,
                                  region_zone_map: dict,
                                  t1_grid: np.ndarray = T1_GRID,
                                  t2_grid: np.ndarray = T2_GRID,
                                  t_grid_gas: np.ndarray = T_GAS_GRID,
                                  vector: str = "elec",
                                  month_effect: bool = True) -> tuple[float, float]:
    """
    Identique à find_optimal_breakpoints mais pour un calendrier v2.
    """
    zone = region_zone_map.get(region, "B")

    dow_cols   = [c for c in calendar_df.columns if c.startswith("dow_")]
    month_cols = [c for c in calendar_df.columns if c.startswith("month_")] if month_effect else []
    other_cols = ["is_sat_or_bridge", "is_eve", "post_rupture", "is_early_august"]
    vac_cols   = [f"vac_noel_zone_{zone}", f"vac_ete_zone_{zone}", f"vac_other_zone_{zone}"]

    cal_part = calendar_df[dow_cols + month_cols + other_cols + vac_cols]
    cal_const = np.hstack([np.ones((len(cal_part), 1)), cal_part.values])

    common = (conso_region.dropna().index
              .intersection(temp_region_hdd.dropna().index)
              .intersection(cal_part.dropna().index))

    y   = conso_region.loc[common].values.astype(float)
    Th   = temp_region_hdd.loc[common].values.astype(float)
    Tc   = temp_region_cdd.loc[common].values.astype(float)
    cal = cal_const[cal_part.index.get_indexer(common)]
    n   = len(y)


    if vector == "elec":
        best_bic = np.inf
        best_t1, best_t2 = t1_grid[0], t2_grid[-1]

        for t1 in t1_grid:
            for t2 in t2_grid:
                if t1 >= t2:
                    continue
                hdd = np.maximum(t1 - Th, 0).reshape(-1, 1)
                cdd = np.maximum(Tc - t2, 0).reshape(-1, 1)
                X_np = np.hstack([hdd, cdd, cal])
                try:
                    beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
                    if len(res_arr) == 0:
                        resid = y - X_np @ beta
                        rss   = float(np.dot(resid, resid))
                    else:
                        rss = float(res_arr[0])
                    k   = X_np.shape[1]
                    bic = n * np.log(rss / n) + k * np.log(n)
                    if bic < best_bic:
                        best_bic = bic
                        best_t1, best_t2 = t1, t2
                except np.linalg.LinAlgError:
                    continue

        return best_t1, best_t2
    
    else: #vector is gas
        best_bic = np.inf
        best_t1  = t_grid_gas[len(t_grid_gas) // 2]

        for t1 in t_grid_gas:
            hdd  = np.maximum(t1 - Th, 0).reshape(-1, 1)
            X_np = np.hstack([hdd, cal])
            try:
                beta, res_arr, _, _ = np.linalg.lstsq(X_np, y, rcond=None)
                if len(res_arr) == 0:
                    resid = y - X_np @ beta
                    rss   = float(np.dot(resid, resid))
                else:
                    rss = float(res_arr[0])
                k   = X_np.shape[1]
                bic = n * np.log(rss / n) + k * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_t1  = t1
            except np.linalg.LinAlgError:
                continue

        return best_t1


def fit_all_regions_v2(conso: pd.DataFrame,
                        temp: pd.DataFrame,
                        region_zone_map: dict,
                        exclude_years: list = None,
                        use_effective_temp: bool = True,
                        alpha_eff: float = 0.7,
                        alpha_cdd: float = 1,
                        vector = "elec",
                        month_effect: bool = True) -> dict:
    """
    Estime un modèle OLS par région avec le calendrier enrichi v2.

    Le calendrier v2 (build_calendar_features_v2) est construit automatiquement
    sur la plage de dates commune temp/conso.

    Parameters
    ----------
    conso              : DataFrame consommation (index DatetimeIndex, colonnes = régions)
    temp               : DataFrame températures  (idem)
    region_zone_map    : dict région → zone scolaire ("A"/"B"/"C")
    exclude_years      : années à exclure de l'entraînement (ex: [2020, 2021])
    use_effective_temp : True → HDD sur T_eff = EWMA(T, alpha), False → T brute
    alpha_eff, _cdd    : paramètre EWMA pour l'inertie thermique

    Returns
    -------
    models : dict { région : {"results", "T_c1", "T_c2", "alpha_eff", "alpha_cdd"
                               "calendar_v": 2} }
    """
    if exclude_years is None:
        exclude_years = EXCLUDE_YEARS

    # Calendrier v2 sur toute la plage de dates
    all_dates = temp.index.union(conso.index)
    calendar_df = build_calendar_features_v2(all_dates)

    # Masque d'exclusion (COVID, etc.)
    keep_mask = ~all_dates.year.isin(exclude_years)

    models = {}
    label_eff = f"T_eff(alpha={alpha_eff})" if use_effective_temp else "T_obs"
    label_cdd = f"T_eff(alpha={alpha_cdd})" if alpha_cdd < 1 else "T_obs"
    print("\n=== ESTIMATION OLS V2 PAR RÉGION ===\n")
    print(f"  Calendrier       : v2 (sat+pont, vac décomposées, début-août)")
    print(f"  Température HDD  : {label_eff}")
    print(f"  Température CDD  : {label_cdd}")
    print(f"  Années exclues   : {exclude_years}\n")
    print(f"{'Région':<35} {'R²':>6} {'R²adj':>6} {'RMSE':>8} {'Tc1':>5} {'Tc2':>5} {'N':>5}")
    print("-" * 75)

    for region in conso.columns:
        if region not in temp.columns:
            print(f"  [WARN] {region} absent de temp — ignorée")
            continue

        T_hdd = compute_effective_temperature(temp[region], alpha=alpha_eff) \
                if use_effective_temp else temp[region]
        if vector == "elec":
            T_cdd = compute_effective_temperature(temp[region], alpha=alpha_cdd)

        # Restreindre aux années d'entraînement
        train_idx = T_hdd.dropna().index[T_hdd.dropna().index.isin(all_dates[keep_mask])]
        conso_train = conso[region].reindex(train_idx)

        print(f"  {region:<33} recherche seuils...", end="\r")

        if vector == "elec":
            T_c1, T_c2 = find_optimal_breakpoints_v2(
                conso_train, T_hdd.reindex(train_idx), T_cdd.reindex(train_idx), calendar_df.reindex(train_idx),
                region, region_zone_map, month_effect=month_effect
            )

            X = build_feature_matrix_v2(
                T_hdd.reindex(train_idx), calendar_df.reindex(train_idx),
                region, region_zone_map, T_c1, T_c2, month_effect=month_effect
            )
            # HDD asymétrique (T_eff pour chauffage, T_obs pour clim)
            X["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X.index), 0)
            X["CDD"] = np.maximum(T_cdd.reindex(X.index) - T_c2, 0)

        else : #vector is gas
            T_c1 = find_optimal_breakpoints_v2(
                conso_train, T_hdd.reindex(train_idx), temp[region], calendar_df.reindex(train_idx),
                region, region_zone_map, vector=vector, month_effect=month_effect
            )
            X = build_feature_matrix_v2(
                T_hdd.reindex(train_idx), calendar_df.reindex(train_idx),
                region, region_zone_map, T_c1, T_c2=None, vector=vector, month_effect=month_effect
            )
            X["HDD"] = np.maximum(T_c1 - T_hdd.reindex(X.index), 0)


        results = fit_region_model(conso[region], X)
        rmse    = np.sqrt(results.mse_resid)

        if vector == "elec":
            models[region] = {
                "results":     results,
                "T_c1":        T_c1,
                "T_c2":        T_c2,
                "alpha_eff":   alpha_eff if use_effective_temp else None,
                "alpha_cdd": alpha_cdd if use_effective_temp else None,
                "calendar_v":  2,
            }

            print(f"  {region:<33} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
                f"{rmse:>8.1f} {T_c1:>5.1f} {T_c2:>5.1f} {int(results.nobs):>5}")
        
        else :
            models[region] = {
                "results":     results,
                "T_c1":        T_c1,
                "T_c2":        None,
                "alpha_eff":   alpha_eff if use_effective_temp else None,
                "alpha_cdd":   alpha_cdd if use_effective_temp else None,
                "calendar_v":  2,
            }

            print(f"  {region:<33} {results.rsquared:>6.3f} {results.rsquared_adj:>6.3f} "
                f"{rmse:>8.1f} {T_c1:>6.2f} {int(results.nobs):>5}")

    return models


# =============================================================================
# 10. ÉVALUATION — RMSE ET MÉTRIQUES OUT-OF-SAMPLE
# =============================================================================

def evaluate_predictions(y_true: pd.Series,
                          y_pred: pd.Series,
                          label: str = "") -> pd.Series:
    """
    Calcule RMSE, MAE et MAPE entre prédiction et réalisation.

    Parameters
    ----------
    y_true : consommation réelle     ex: conso["Île-de-France"].loc["2018"]
    y_pred : consommation prédite    ex: predict_demand(...)["y_pred"]
    label  : étiquette pour l'affichage

    Returns
    -------
    pd.Series  index = ["RMSE", "MAE", "MAPE_%", "N_jours"]

    Exemple
    -------
    >>> df_pred = predict_demand(models["Île-de-France"], "Île-de-France",
    ...                          region_zone_map, temp["Île-de-France"].loc["2018"])
    >>> evaluate_predictions(conso["Île-de-France"].loc["2018"], df_pred["y_pred"],
    ...                       label="IDF 2018")
    """
    common = y_true.dropna().index.intersection(y_pred.dropna().index)
    y_t = y_true.loc[common]
    y_p = y_pred.loc[common]

    rmse  = float(np.sqrt(((y_t - y_p) ** 2).mean()))
    mae   = float((y_t - y_p).abs().mean())
    mape  = float(((y_t - y_p).abs() / y_t.abs()).mean() * 100)
    nrmse = rmse / float(y_t.mean()) * 100

    metrics = pd.Series({
        "RMSE":    round(rmse, 1),
        "nRMSE_%": round(nrmse, 2),
        "MAE":     round(mae, 1),
        "MAPE_%":  round(mape, 2),
        "N_jours": len(common),
    }, name=label or "metrics")

    if label:
        print(f"[eval] {label}")
        print(f"  RMSE  : {rmse:>10,.0f} MWh/j")
        print(f"  nRMSE : {nrmse:>10.2f} %")
        print(f"  MAE   : {mae:>10,.0f} MWh/j")
        print(f"  MAPE  : {mape:>10.2f} %")
        print(f"  N     : {len(common):>10} jours")

    return metrics


# =============================================================================
# 11. VISUALISATION — PRÉDICTION VS RÉEL PAR RÉGION
# =============================================================================

def plot_predictions_vs_actual(models: dict,
                                conso: pd.DataFrame,
                                temp: pd.DataFrame,
                                region_zone_map: dict,
                                year: int = None,
                                ncols: int = 3,
                                figsize_per_panel: tuple = (7, 3),
                                save: str = None,
                                regs: list = None,
                                vector: str = "elec",
                                is_France: bool = True) -> None:
    """
    Trace, pour chaque région, la consommation réelle, la prédiction reconstruite
    et la baseline (sans effet température).

    Parameters
    ----------
    models          : dict renvoyé par fit_all_regions ou fit_all_regions_gas
    conso           : DataFrame consommation (index=dates, colonnes=régions)
    temp            : DataFrame températures  (idem)
    region_zone_map : dict région → zone vacances scolaires
    year            : si fourni, filtre sur cette année uniquement (ex: 2018)
                      si None, trace toute la période disponible
    ncols           : nombre de colonnes dans la grille de sous-graphes
    figsize_per_panel : largeur × hauteur (en pouces) de chaque panneau
    save            : chemin fichier si on veut sauvegarder (ex: "pred_vs_reel.png")

    Exemple
    -------
    >>> plot_predictions_vs_actual(models, conso, temp, region_zone_map, year=2018)
    >>> plot_predictions_vs_actual(models, conso, temp, region_zone_map, save="pred.png")
    """
    import math

    if regs is not None:
        regions = regs
    else:
        regions = [r for r in models if r in conso.columns and r in temp.columns]
    n       = len(regions)
    nrows   = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=False,
    )
    axes_flat = axes.flatten() if n > 1 else [axes]

    title_suffix = f" — {year}" if year is not None else " — toute la période"

    for idx, region in enumerate(regions):
        ax = axes_flat[idx]

        # Filtrage éventuel sur l'année
        temp_region = temp[region].dropna()
        if year is not None:
            temp_region = temp_region.loc[str(year)]

        try:
            df_pred = predict_demand(models[region], region, region_zone_map, temp_region, is_France)
        except Exception as e:
            ax.set_title(f"{region}\n[erreur: {e}]", fontsize=8)
            ax.axis("off")
            continue

        y_true = conso[region].reindex(df_pred.index)

        # Traces
        ax.plot(y_true.index,        y_true.values,           color="steelblue",
                lw=0.8, alpha=0.85,  label="True")
        ax.plot(df_pred.index,       df_pred["y_pred"].values, color="tomato",
                lw=0.8, alpha=0.85,  label="Predicted")
        ax.plot(df_pred.index,       df_pred["y_base"].values, color="seagreen",
                lw=0.8, alpha=0.75,  linestyle="--", label="Baseline")

        # Mise en forme
        ax.set_title(region, fontsize=9, fontweight="bold")
        ax.set_ylabel("MW", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # Légende uniquement sur le premier panneau
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Masquer les panneaux vides
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"True consumption vs prediction reconstructed by region{title_suffix}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save}")

    plt.show()


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":

    CONSO_PATH = "data/conso.csv"
    TEMP_PATH  = "data/temperature.csv"

    models, baselines, calendar_df = run_pipeline(CONSO_PATH, TEMP_PATH)

    # Agrégat national
    baseline_nat = national_aggregate(baselines)
    print(f"\nBaseline nationale — moyenne : {baseline_nat.mean():.0f} MWh/jour")

    # Résumé
    print("\n--- Résumé par région ---")
    print(summarize_model_stats(models)[["R2", "R2_adj", "RMSE", "AIC", "BIC"]])
