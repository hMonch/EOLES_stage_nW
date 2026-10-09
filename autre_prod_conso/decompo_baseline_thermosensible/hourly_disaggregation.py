"""
=============================================================================
DÉSAGRÉGATION HORAIRE DE LA CONSOMMATION ÉLECTRIQUE — ÉTAPE 1
Profils normalisés par région × type_jour × saison
=============================================================================

Contexte
────────
Ce module vient *en aval* du modèle quotidien (baseline_electricity_model.py).
Le modèle quotidien produit, pour chaque région et chaque jour, une estimation
de la consommation journalière en MWh. Ce module transforme cette série
journalière en série horaire (24 valeurs par jour, en MWh/h) via des profils
normalisés appris sur l'historique.

Principe (top-down)
───────────────────
Pour chaque région r, type de jour d, saison s :

    p̄_{r,d,s}(h) = moyenne empirique, sur les jours (d,s) de l'historique,
                    du ratio  conso_horaire(h) / conso_journalière

Par construction la somme sur h vaut 1, donc :

    conso_horaire(r, jour, h) = conso_quotidienne(r, jour) × p̄_{r,d(jour),s(jour)}(h)

→ la somme des 24 valeurs horaires d'un jour est exactement égale à la valeur
  journalière fournie en entrée. Cohérence multi-échelle garantie.

Choix de granularité (étape 1)
──────────────────────────────
  • type_jour  : 5 classes
      lun, mar-jeu, ven, sam+pont, dim+JF
  • saison     : 4 saisons météo
      DJF (déc-jan-fév), MAM, JJA, SON
  • région     : 12 régions métropolitaines (hors Corse, absente du CSV conso)

→ 5 × 4 = 20 profils de 24h par région
→ ~13 ans × 365 j / 20 ≈ 240 jours par cellule (hors COVID) → largement suffisant.

Données attendues
─────────────────
  • conso.csv   : conso horaire RTE, index datetime tz-aware (+01:00/+02:00),
                  colonnes = 12 régions, valeurs en MWh/h (2013-01-01 → 2025-12-31)
  • temp.csv    : températures horaires, index datetime naïf UTC ou local,
                  colonnes = 12 régions, valeurs en °C (2010-01-01 → 2025-12-31)
                  → Non utilisé à l'étape 1 (pas de dépendance T° dans la forme).
                    Conservé pour l'étape 2 future.

Sorties
───────
  • profiles : dict { region: DataFrame (20 × 24) indexé par MultiIndex
                     (type_jour, saison), colonnes = heure 0..23 }
  • fonction disaggregate_daily(daily_series, region, profiles)
    → renvoie la série horaire reconstruite

Évaluation
──────────
On reconstruit la conso horaire à partir de la conso **journalière réelle**
(agrégée depuis l'horaire observé). Cela isole l'erreur de désagrégation pure,
indépendamment de l'erreur du modèle quotidien.
=============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats as scipy_stats

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. PARAMÈTRES GLOBAUX
# =============================================================================

EXCLUDE_YEARS = [2020, 2021]   # COVID — cohérent avec le modèle quotidien

# Mapping mois → saison météo (DJF / MAM / JJA / SON)
MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
     3: "MAM", 4: "MAM", 5: "MAM",
     6: "JJA", 7: "JJA", 8: "JJA",
     9: "SON", 10: "SON", 11: "SON",
}

DAY_TYPES    = ["lun", "mar_jeu", "ven", "sam_pont", "dim_jf"]
SEASONS      = ["DJF", "MAM", "JJA", "SON"]
HOURS        = list(range(24))

OUTPUT_DIR = Path("outputs_hourly")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. JOURS FÉRIÉS & PONTS (repris du modèle quotidien pour cohérence)
# =============================================================================

def _easter(year: int) -> date:
    """Pâques via algorithme de Gauss."""
    a = year % 19
    b, c = year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_french_public_holidays(years: list[int]) -> set:
    """Retourne un set de pd.Timestamp (normalisés minuit, tz-naïf) des JF FR."""
    hols = []
    for year in years:
        e = _easter(year)
        hols += [
            date(year, 1, 1),
            e + timedelta(1),          # lundi de Pâques
            date(year, 5, 1),
            date(year, 5, 8),
            e + timedelta(39),         # Ascension
            e + timedelta(50),         # lundi de Pentecôte
            date(year, 7, 14),
            date(year, 8, 15),
            date(year, 11, 1),
            date(year, 11, 11),
            date(year, 12, 25),
        ]
    return set(pd.Timestamp(d) for d in hols)


def build_day_labels(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Pour un index de dates (jour), renvoie un DataFrame avec :
        - type_jour ∈ {lun, mar_jeu, ven, sam_pont, dim_jf}
        - saison   ∈ {DJF, MAM, JJA, SON}
        - is_holiday, is_bridge (flags utiles pour diagnostic)

    Règles type_jour :
        dimanche OU JF                                 → dim_jf
        samedi OU (lun/ven encadrant un JF = pont)     → sam_pont
        vendredi hors JF et hors pont                  → ven
        lundi hors JF et hors pont                     → lun
        mar/mer/jeu hors JF                            → mar_jeu

    Les JF "écrasent" le dow : un mardi férié est classé dim_jf.
    """
    # Normalisation : on veut des dates tz-naïves pour comparer aux JF
    idx = pd.DatetimeIndex(daily_index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    years  = list(range(idx.year.min(), idx.year.max() + 1))
    hols   = get_french_public_holidays(years)
    is_hol = idx.isin(hols)

    # Pont : lundi dont le vendredi précédent est JF, ou vendredi dont le lundi
    # suivant est JF (définition symétrique pour capter les deux cas courants).
    # On étend aussi au mardi après un lundi férié et jeudi avant un vendredi férié
    # — pratique courante en France pour "faire le pont".
    def _is_bridge(d: pd.Timestamp, is_hol_today: bool) -> bool:
        if is_hol_today:
            return False
        dow = d.dayofweek
        if dow == 4:  # vendredi : pont si jeudi précédent est JF (ex. Ascension)
            return (d - pd.Timedelta(days=1)) in hols
        if dow == 0:  # lundi : pont si mardi suivant est JF
            return (d + pd.Timedelta(days=1)) in hols
        return False

    is_bridge = np.array([_is_bridge(d, h) for d, h in zip(idx, is_hol)])
    dow       = idx.dayofweek.values

    type_jour = np.empty(len(idx), dtype=object)
    # Ordre important : on traite d'abord les cas "écrasants"
    mask_dim_jf   = (dow == 6) | is_hol
    mask_sam_pont = ((dow == 5) | is_bridge) & ~mask_dim_jf
    mask_ven      = (dow == 4) & ~mask_dim_jf & ~mask_sam_pont
    mask_lun      = (dow == 0) & ~mask_dim_jf & ~mask_sam_pont
    mask_mar_jeu  = np.isin(dow, [1, 2, 3]) & ~mask_dim_jf & ~mask_sam_pont

    type_jour[mask_dim_jf]   = "dim_jf"
    type_jour[mask_sam_pont] = "sam_pont"
    type_jour[mask_ven]      = "ven"
    type_jour[mask_lun]      = "lun"
    type_jour[mask_mar_jeu]  = "mar_jeu"

    saison = np.array([MONTH_TO_SEASON[m] for m in idx.month])

    out = pd.DataFrame({
        "type_jour":  type_jour,
        "saison":     saison,
        "is_holiday": is_hol,
        "is_bridge":  is_bridge,
    }, index=idx)
    return out


# =============================================================================
# 3. CHARGEMENT DES DONNÉES HORAIRES
# =============================================================================

def _deduplicate_index(df: pd.DataFrame, path: str = "") -> pd.DataFrame:
    """
    Supprime les doublons d'index en gardant la première occurrence.

    Cause courante : changement d'heure d'automne en heure locale naïve
    (la 2h→3h du dernier dimanche d'octobre apparaît deux fois quand on
    convertit un tz-aware en naïf, car 02:00 UTC+2 et 02:00 UTC+1 tombent
    sur le même timestamp naïf).

    Avertit s'il y a des doublons pour que l'utilisateur soit informé.
    """
    dup_mask = df.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        name = Path(path).name if path else "données"
        print(f"[load] {name} : {n_dup} timestamp(s) dupliqué(s) retirés "
              f"(probablement changements d'heure)")
        df = df[~dup_mask]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _read_hourly_csv(path: str) -> pd.DataFrame:
    """
    Lit un CSV horaire et garantit un index DatetimeIndex en heure locale
    naïve (Europe/Paris, sans fuseau).

    Gère trois cas :
        1. Index tz-aware avec offset unique ('+01:00')        → convert + drop tz
        2. Index tz-aware avec offsets mixtes ('+01:00'/'+02:00', cas RTE)
           → pd.read_csv laisse un Index d'objets strings. On force
             pd.to_datetime(utc=True) puis convert Europe/Paris + drop tz.
        3. Index tz-naïf                                       → laissé tel quel

    Le résultat est toujours un DatetimeIndex naïf représentant l'heure
    locale civile française — cohérent entre les fichiers conso et temp.
    """
    df = pd.read_csv(path, index_col=0)

    idx = df.index
    # Cas 3 : déjà un DatetimeIndex naïf parsé par pandas
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is None:
        df = _deduplicate_index(df, path)
        return df

    # Cas 1 : déjà un DatetimeIndex tz-aware (offset unique)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df.index = idx.tz_convert("Europe/Paris").tz_localize(None)
        df = _deduplicate_index(df, path)
        return df

    # Cas 2 : Index d'objets (strings) — offsets mixtes ou parsing échoué.
    # utc=True force la conversion de toutes les valeurs en UTC d'abord
    # (gère +01:00 et +02:00 simultanément sans lever d'erreur),
    # puis on convertit en Europe/Paris et on drop le fuseau.
    parsed = pd.to_datetime(df.index, utc=True, errors="raise")
    df.index = parsed.tz_convert("Europe/Paris").tz_localize(None)
    df = _deduplicate_index(df, path)
    return df


def load_hourly_data(conso_path: str,
                     temp_path: str = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Charge conso horaire (obligatoire) et température horaire (optionnelle).

    La conso RTE est en heure locale Europe/Paris avec fuseau explicite
    (+01:00 / +02:00). On la convertit en Europe/Paris puis on retire le
    fuseau pour manipuler des timestamps "heure locale nue" — plus simple
    pour grouper par heure-du-jour et par date calendaire.

    Returns
    -------
    conso : DataFrame index=datetime (heure locale naïve), colonnes=régions
    temp  : idem, ou None si non fourni
    """
    conso = _read_hourly_csv(conso_path)
    print(f"[load] Conso : {conso.index.min()} → {conso.index.max()}  |  "
          f"{len(conso):,} lignes  |  {conso.shape[1]} régions")

    temp = None
    if temp_path is not None:
        temp = _read_hourly_csv(temp_path)
        print(f"[load] Temp  : {temp.index.min()} → {temp.index.max()}  |  "
              f"{len(temp):,} lignes  |  {temp.shape[1]} régions")

    return conso, temp


# =============================================================================
# 4. CALCUL DES PROFILS NORMALISÉS
# =============================================================================

def _hourly_to_daily(conso_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège l'horaire en journalier (somme des 24h).

    Un jour n'est gardé que s'il contient exactement 24 observations valides,
    pour éviter les biais sur les jours de changement d'heure (23h ou 25h)
    et sur les jours partiellement manquants.
    """
    g = conso_hourly.groupby(conso_hourly.index.normalize())
    daily = g.sum(min_count=24)              # NaN si <24 obs
    counts = g.size()
    daily = daily[counts == 24]
    return daily


def _build_hour_index(conso_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 'date' (jour calendaire) et 'hour' (0..23) à l'index horaire.
    Renvoie un DataFrame long pratique pour le groupby.
    """
    idx = conso_hourly.index
    aux = pd.DataFrame({
        "date": idx.normalize(),
        "hour": idx.hour,
    }, index=idx)
    return aux


def compute_profiles(conso_hourly: pd.DataFrame,
                     exclude_years: list[int] = EXCLUDE_YEARS,
                     ) -> dict[str, pd.DataFrame]:
    """
    Calcule les profils normalisés par région × type_jour × saison.

    Méthode
    ───────
    1. Agréger l'horaire en journalier.
    2. Pour chaque heure h de chaque jour, calculer ratio(h) = conso(h) / conso_jour.
    3. Étiqueter chaque jour par (type_jour, saison).
    4. Moyenner les ratios par (région, type_jour, saison, heure).
       → Par construction la somme sur h vaut ≈1 (à l'arrondi près).
    5. Renormaliser proprement pour garantir sum_h = 1 exactement.

    Returns
    -------
    profiles : dict { region: DataFrame
                       index = MultiIndex (type_jour, saison),
                       columns = 0..23,
                       valeurs = part de la conso journalière sur chaque heure }
    """
    # 1. Aggrégation journalière + filtrage années
    daily = _hourly_to_daily(conso_hourly)
    mask_years = ~daily.index.year.isin(exclude_years)
    daily = daily[mask_years]

    # 2. On ne garde les heures que pour les jours complets retenus
    kept_dates = set(daily.index)
    hourly = conso_hourly[conso_hourly.index.normalize().isin(kept_dates)].copy()

    # 3. Étiquettes sur le journalier, puis jointure horaire
    labels = build_day_labels(daily.index)[["type_jour", "saison"]]

    # Construction d'un DF long : [datetime, date, hour, region, conso]
    hourly_long = hourly.reset_index().melt(
        id_vars=hourly.index.name or "index",
        var_name="region",
        value_name="conso_h",
    )
    dt_col = hourly.index.name or "index"
    hourly_long["date"] = hourly_long[dt_col].dt.normalize()
    hourly_long["hour"] = hourly_long[dt_col].dt.hour
    hourly_long = hourly_long.drop(columns=[dt_col])

    # Jointure avec la conso journalière (par date, par région)
    daily_long = daily.stack().rename("conso_d").reset_index()
    daily_long.columns = ["date", "region", "conso_d"]

    merged = hourly_long.merge(daily_long, on=["date", "region"], how="inner")

    # Étiquettes
    lab_long = labels.reset_index().rename(columns={"index": "date"})
    # reset_index renomme 'index' ou conserve le nom → on force 'date'
    lab_long.columns = ["date", "type_jour", "saison"]
    merged = merged.merge(lab_long, on="date", how="inner")

    # 4. Ratio horaire
    # Garde-fou : si conso_d = 0 ou négative (anomalie), on exclut
    merged = merged[merged["conso_d"] > 0]
    merged["ratio"] = merged["conso_h"] / merged["conso_d"]

    # 5. Moyenne par (region, type_jour, saison, hour)
    grp = merged.groupby(
        ["region", "type_jour", "saison", "hour"]
    )["ratio"].mean().reset_index()

    # 6. Pivot en format profil : index (type_jour, saison), colonnes heure
    profiles: dict[str, pd.DataFrame] = {}
    for region, sub in grp.groupby("region"):
        pivot = sub.pivot_table(
            index=["type_jour", "saison"],
            columns="hour",
            values="ratio",
            aggfunc="mean",
        )
        # Réindexer sur toutes les combinaisons attendues
        full_idx = pd.MultiIndex.from_product(
            [DAY_TYPES, SEASONS], names=["type_jour", "saison"]
        )
        pivot = pivot.reindex(full_idx)
        pivot = pivot.reindex(columns=HOURS)

        # Renormalisation : sum_h = 1 exactement
        # (évite les écarts dus à la moyenne non-exacte ratio)
        pivot = pivot.div(pivot.sum(axis=1), axis=0)

        profiles[region] = pivot

    # Diagnostic sur cellules manquantes éventuelles
    for region, p in profiles.items():
        nan_cells = p.isna().any(axis=1).sum()
        if nan_cells > 0:
            print(f"[compute_profiles] WARN : {region} a {nan_cells} cellule(s) "
                  f"(type_jour, saison) sans donnée — seront remplies par fallback")
            # Fallback : moyenne sur la saison tous type_jour confondus,
            #           sinon moyenne générale de la région
            season_avg = p.groupby(level="saison").mean()
            global_avg = p.mean(axis=0)
            global_avg = global_avg / global_avg.sum()
            for (tj, sa) in p.index:
                if p.loc[(tj, sa)].isna().any():
                    if not season_avg.loc[sa].isna().any():
                        row = season_avg.loc[sa]
                        p.loc[(tj, sa)] = (row / row.sum()).values
                    else:
                        p.loc[(tj, sa)] = global_avg.values
            profiles[region] = p

    return profiles


# =============================================================================
# 5. DÉSAGRÉGATION
# =============================================================================

def disaggregate_daily(daily_series: pd.Series,
                        region: str,
                        profiles: dict[str, pd.DataFrame],
                        ) -> pd.Series:
    """
    Transforme une série journalière en série horaire via les profils.

    Parameters
    ----------
    daily_series : pd.Series
        Index = dates (DatetimeIndex, un point par jour).
        Valeurs = conso journalière (MWh/jour).
    region : str
        Doit être une clé de `profiles`.
    profiles : dict
        Sortie de compute_profiles().

    Returns
    -------
    pd.Series  index = timestamps horaires, valeurs = MWh/h.
                Somme sur chaque jour = valeur journalière d'entrée.
    """
    if region not in profiles:
        raise KeyError(f"Région '{region}' absente des profils "
                       f"(disponibles : {list(profiles.keys())})")

    prof = profiles[region]
    labels = build_day_labels(daily_series.index)

    # Index horaire construit à partir des jours réellement présents
    # (gère correctement les trous éventuels dans daily_series).
    days = pd.DatetimeIndex(daily_series.index).normalize()
    hourly_index = pd.DatetimeIndex(
        np.repeat(days.values, 24) + np.tile(
            np.arange(24) * np.timedelta64(1, "h"), len(days)
        )
    )

    # Vectorisation : matrice (n_days, 24) = profil × conso_j
    profil_matrix = np.zeros((len(daily_series), 24))
    for i, day in enumerate(daily_series.index):
        tj = labels.loc[day, "type_jour"]
        sa = labels.loc[day, "saison"]
        try:
            p = prof.loc[(tj, sa)].values
        except KeyError:
            # Fallback défensif (ne devrait pas arriver après renormalisation)
            p = prof.mean(axis=0).values
            p = p / p.sum()
        profil_matrix[i] = p * daily_series.iloc[i]

    out = pd.Series(
        profil_matrix.flatten(),
        index=hourly_index,
        name=f"{region}_hourly",
    )
    return out


def disaggregate_all_regions(daily_df: pd.DataFrame,
                              profiles: dict[str, pd.DataFrame],
                              ) -> pd.DataFrame:
    """Applique disaggregate_daily à toutes les colonnes d'un DF journalier."""
    out = {}
    for region in daily_df.columns:
        if region not in profiles:
            print(f"  [WARN] {region} ignorée (pas de profil)")
            continue
        out[region] = disaggregate_daily(daily_df[region], region, profiles)
    return pd.DataFrame(out)


# =============================================================================
# 6. ÉVALUATION
# =============================================================================

def evaluate_disaggregation(conso_hourly: pd.DataFrame,
                             profiles: dict[str, pd.DataFrame],
                             years: list[int] | None = None,
                             ) -> pd.DataFrame:
    """
    Évalue la désagrégation sur des années données, en partant de la conso
    JOURNALIÈRE RÉELLE (= agrégation horaire→journalier des données observées).

    → Cela mesure l'erreur de désagrégation PURE, sans la contaminer par
      l'erreur du modèle quotidien.

    Métriques retournées par région :
        RMSE  (MWh/h)
        MAE   (MWh/h)
        MAPE  (%)
        R²    (variance expliquée)
    """
    daily = _hourly_to_daily(conso_hourly)

    if years is not None:
        mask = daily.index.year.isin(years)
        daily = daily[mask]
        target = conso_hourly[conso_hourly.index.normalize().isin(set(daily.index))]
    else:
        target = conso_hourly.copy()
        # Aligner target sur daily (pour le cas exclude years)
        target = target[target.index.normalize().isin(set(daily.index))]

    pred_hourly = disaggregate_all_regions(daily, profiles)
    pred_hourly = pred_hourly.reindex(target.index)

    rows = []
    for region in target.columns:
        if region not in pred_hourly.columns:
            continue
        y  = target[region].dropna()
        yp = pred_hourly[region].reindex(y.index).dropna()
        common = y.index.intersection(yp.index)
        y, yp = y.loc[common], yp.loc[common]
        if len(y) == 0:
            continue
        err   = y - yp
        rmse  = float(np.sqrt((err ** 2).mean()))
        nrmse = rmse / float(y.mean()) * 100
        mae   = float(err.abs().mean())
        mape  = float((err.abs() / y.abs()).mean() * 100)
        ss_res = float((err ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            "region":   region,
            "RMSE":     round(rmse, 1),
            "nRMSE_%":  round(nrmse, 2),
            "MAE":      round(mae, 1),
            "MAPE_%":   round(mape, 2),
            "R2":       round(r2, 4),
            "N":        len(y),
        })

    metrics = pd.DataFrame(rows).set_index("region")
    return metrics


def evaluate_by_hour(conso_hourly: pd.DataFrame,
                      profiles: dict[str, pd.DataFrame],
                      region: str,
                      years: list[int] | None = None,
                      ) -> pd.DataFrame:
    """
    Décomposition de l'erreur par heure de la journée et par saison.
    Utile pour repérer si certaines heures sont systématiquement mal prédites
    (p. ex. pic du soir en hiver) — ce qui justifierait l'étape 2.
    """
    daily = _hourly_to_daily(conso_hourly[[region]])
    if years is not None:
        daily = daily[daily.index.year.isin(years)]

    target = conso_hourly[[region]][
        conso_hourly.index.normalize().isin(set(daily.index))
    ]
    pred = disaggregate_daily(daily[region], region, profiles)
    pred = pred.reindex(target.index)

    df = pd.DataFrame({
        "y":  target[region],
        "yp": pred,
    }).dropna()
    df["err"]    = df["y"] - df["yp"]
    df["hour"]   = df.index.hour
    df["month"]  = df.index.month
    df["saison"] = df["month"].map(MONTH_TO_SEASON)

    agg = df.groupby(["saison", "hour"]).apply(
        lambda g: pd.Series({
            "RMSE":    float(np.sqrt((g["err"] ** 2).mean())),
            "nRMSE_%": float(np.sqrt((g["err"] ** 2).mean()) / g["y"].mean() * 100),
            "MAE":     float(g["err"].abs().mean()),
            "MAPE_%":  float((g["err"].abs() / g["y"].abs()).mean() * 100),
            "N":       len(g),
        })
    ).reset_index()
    return agg


# =============================================================================
# 7. VISUALISATION
# =============================================================================

def plot_profiles(profiles: dict[str, pd.DataFrame],
                   region: str,
                   save: str | None = None) -> None:
    """
    Trace les 20 profils (5 types × 4 saisons) pour une région.
    Une figure 4×1 (une ligne par saison), chaque subplot empile les 5 types.
    """
    prof = profiles[region]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    colors = {
        "lun":      "#1f77b4",
        "mar_jeu":  "#2ca02c",
        "ven":      "#ff7f0e",
        "sam_pont": "#d62728",
        "dim_jf":   "#9467bd",
    }
    labels_en = {
        "lun":      "Mon",
        "mar_jeu":  "Tue–Thu",
        "ven":      "Fri",
        "sam_pont": "Sat+bridge",
        "dim_jf":   "Sun+hol",
    }
    for ax, saison in zip(axes, SEASONS):
        for tj in DAY_TYPES:
            try:
                row = prof.loc[(tj, saison)]
            except KeyError:
                continue
            ax.plot(row.index, row.values, label=labels_en[tj], color=colors[tj], lw=1.5)
        ax.set_title(f"{saison}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of the day")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of daily consumption")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Profiles normalized — {region}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()


def plot_week_comparison(conso_hourly: pd.DataFrame,
                          profiles: dict[str, pd.DataFrame],
                          region: str,
                          start_date: str,
                          n_days: int = 7,
                          save: str | None = None) -> None:
    """Compare réel vs reconstruit sur une semaine donnée."""
    daily = _hourly_to_daily(conso_hourly[[region]])
    pred  = disaggregate_daily(daily[region], region, profiles)

    start = pd.Timestamp(start_date)
    end   = start + pd.Timedelta(days=n_days)
    mask  = (conso_hourly.index >= start) & (conso_hourly.index < end)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(conso_hourly.index[mask], conso_hourly.loc[mask, region],
            label="Actual", color="steelblue", lw=1.2)
    ax.plot(pred.index[(pred.index >= start) & (pred.index < end)],
            pred[(pred.index >= start) & (pred.index < end)],
            label="Reconstructed", color="tomato", lw=1.2, linestyle="--")
    ax.set_title(f"{region} — {start_date} to {end.date()}", fontweight="bold")
    ax.set_ylabel("MW")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()


def plot_error_heatmap(conso_hourly: pd.DataFrame,
                        profiles: dict[str, pd.DataFrame],
                        region: str,
                        years: list[int] | None = None,
                        save: str | None = None) -> None:
    """
    Carte thermique MAPE par (saison × heure) — diagnostic clé pour décider
    si l'étape 2 est nécessaire.
    """
    agg = evaluate_by_hour(conso_hourly, profiles, region, years=years)
    pivot = agg.pivot(index="saison", columns="hour", values="MAPE_%")
    pivot = pivot.reindex(SEASONS)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(SEASONS))); ax.set_yticklabels(SEASONS)
    ax.set_xlabel("Hour")
    ax.set_title(f"MAPE (%) by (season, hour) — {region}", fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 5 else "white")
    fig.colorbar(im, ax=ax, label="MAPE %")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()


# =============================================================================
# 7bis. DIAGNOSTICS DES RÉSIDUS
# =============================================================================
#
# Un résidu = y_réel - y_prédit. Si le modèle a tout capté, les résidus
# doivent ressembler à du bruit : pas de structure, pas de tendance, centrés
# sur zéro, variance à peu près constante.
#
# Toute structure visible dans les résidus = signal non capturé = piste
# d'amélioration. Pas de structure = on est proche de la limite irréductible.
# =============================================================================


def compute_residuals(conso_hourly: pd.DataFrame,
                       profiles: dict[str, pd.DataFrame],
                       region: str,
                       years: list[int] | None = None,
                       temp_hourly: pd.DataFrame | None = None,
                       ) -> pd.DataFrame:
    """
    Calcule les résidus horaires pour une région, en partant de la conso
    journalière réelle (= désagrégation pure, sans erreur du modèle quotidien).

    Returns
    -------
    DataFrame index = datetime horaire, colonnes :
        y        : conso réelle
        y_pred   : conso reconstruite
        resid    : résidu brut (MWh/h)
        resid_pct: résidu relatif (%)  100 × resid / y
        hour, dow, month, saison, type_jour
        T        : température (si temp_hourly fourni) — utile pour le test clé
    """
    daily = _hourly_to_daily(conso_hourly[[region]])
    if years is not None:
        daily = daily[daily.index.year.isin(years)]

    target = conso_hourly[[region]][
        conso_hourly.index.normalize().isin(set(daily.index))
    ]
    pred = disaggregate_daily(daily[region], region, profiles).reindex(target.index)

    df = pd.DataFrame({
        "y":      target[region],
        "y_pred": pred,
    }).dropna()
    df["resid"]     = df["y"] - df["y_pred"]
    df["resid_pct"] = 100.0 * df["resid"] / df["y"]

    df["hour"]      = df.index.hour
    df["dow"]       = df.index.dayofweek
    df["month"]     = df.index.month
    df["saison"]    = df["month"].map(MONTH_TO_SEASON)

    labels = build_day_labels(df.index.normalize().unique())
    df["type_jour"] = df.index.normalize().map(labels["type_jour"])

    if temp_hourly is not None and region in temp_hourly.columns:
        df["T"] = temp_hourly[region].reindex(df.index)

    return df


def plot_residuals_diagnostics(conso_hourly: pd.DataFrame,
                                profiles: dict[str, pd.DataFrame],
                                region: str,
                                years: list[int] | None = None,
                                temp_hourly: pd.DataFrame | None = None,
                                save: str | None = None) -> pd.DataFrame:
    """
    Les 4 graphiques classiques de diagnostic des résidus.

    (1) Résidus vs temps         : patterns saisonniers / temporels résiduels
    (2) Résidus vs prédit        : hétéroscédasticité (variance change avec niveau)
    (3) Résidus vs température   : SIGNAL-CLÉ — si nuage plat, la T° n'ajoute
                                    rien et l'étape 2 est inutile. Si tendance
                                    visible, la T° aiderait.
    (4) Histogramme + QQ-plot    : normalité, symétrie, queues épaisses

    Returns
    -------
    Le DataFrame des résidus (pratique pour analyses complémentaires).
    """
    res = compute_residuals(conso_hourly, profiles, region,
                             years=years, temp_hourly=temp_hourly)

    # Sous-échantillonnage pour la lisibilité si trop de points
    n = len(res)
    sample = res.sample(min(n, 8000), random_state=0) if n > 8000 else res

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = f"Residual Diagnostics — {region}"
    if years:
        title += f" ({', '.join(map(str, years))})"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # (1) Residuals vs time
    ax = axes[0, 0]
    ax.scatter(res.index, res["resid"], s=1, alpha=0.3, color="steelblue")
    roll = res["resid"].rolling(168, min_periods=24).mean()
    ax.plot(roll.index, roll.values, color="tomato", lw=1.2,
             label="7-day rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(1) Residuals vs time")
    ax.set_ylabel("Residual (MW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (2) Residuals vs fitted
    ax = axes[0, 1]
    ax.scatter(sample["y_pred"], sample["resid"], s=2, alpha=0.3, color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    bins = pd.cut(sample["y_pred"], bins=20)
    band = sample.groupby(bins)["resid"].agg(["mean", "std"])
    centers = [b.mid for b in band.index]
    ax.plot(centers, band["mean"] + 2 * band["std"], color="tomato",
             lw=1, linestyle="--", label="±2σ per bin")
    ax.plot(centers, band["mean"] - 2 * band["std"], color="tomato",
             lw=1, linestyle="--")
    ax.set_title("(2) Residuals vs fitted — heteroscedasticity?")
    ax.set_xlabel("Fitted (MW)")
    ax.set_ylabel("Residual (MW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) Residuals vs temperature (key signal)
    ax = axes[1, 0]
    if "T" in res.columns and res["T"].notna().any():
        ax.scatter(sample["T"], sample["resid"], s=2, alpha=0.3, color="steelblue")
        ax.axhline(0, color="black", lw=0.8)
        tb = pd.cut(res["T"], bins=np.arange(
            np.floor(res["T"].min()), np.ceil(res["T"].max()) + 1, 1.0))
        trend = res.groupby(tb)["resid"].mean()
        tc = [b.mid for b in trend.index]
        ax.plot(tc, trend.values, color="tomato", lw=2,
                 label="mean per °C")
        ax.set_title("(3) Residuals vs temperature — key signal")
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("Residual (MW)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Temperature not provided",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(3) Residuals vs temperature")
    ax.grid(alpha=0.3)

    # (4) Histogram + normal fit
    ax = axes[1, 1]
    ax.hist(res["resid"], bins=80, color="steelblue", alpha=0.7,
             edgecolor="white")
    mu, sigma = res["resid"].mean(), res["resid"].std()
    x = np.linspace(res["resid"].min(), res["resid"].max(), 200)
    bin_w = (res["resid"].max() - res["resid"].min()) / 80
    ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma) * len(res) * bin_w,
             color="tomato", lw=1.5, label=f"N(μ={mu:.1f}, σ={sigma:.1f})")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("(4) Residual distribution")
    ax.set_xlabel("Residual (MW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()

    # Quelques chiffres récap
    nrmse_glob = float(np.sqrt((res['resid'] ** 2).mean()) / res['y'].mean() * 100)
    print(f"\nResidual summary — {region}")
    print(f"  mean        : {res['resid'].mean():>8.2f} MW  "
          f"(should be close to 0)")
    print(f"  std dev     : {res['resid'].std():>8.2f} MW")
    print(f"  nRMSE       : {nrmse_glob:>8.2f} %  "
          f"(RMSE / mean load)")
    print(f"  skewness    : {res['resid'].skew():>8.3f}  "
          f"(symmetry: 0 = perfect, >0 = right tail)")
    print(f"  kurtosis    : {res['resid'].kurt():>8.3f}  "
          f"(tails: 0 = normal, >0 = heavy tails)")
    print(f"  global MAPE : {res['resid_pct'].abs().mean():>8.2f} %")

    return res


def plot_residuals_by_typology(residuals_df: pd.DataFrame,
                                 save: str | None = None) -> None:
    """
    Décompose les résidus par (saison × type_jour) — boxplots.

    Ce qu'on cherche : des boîtes toutes centrées sur 0 avec une dispersion
    comparable. Si certaines cases sont clairement décalées (moyenne ≠ 0),
    c'est un biais systématique sur ce type de jour → affiner la typologie.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, saison in zip(axes, SEASONS):
        sub = residuals_df[residuals_df["saison"] == saison]
        data = [sub[sub["type_jour"] == tj]["resid"].values for tj in DAY_TYPES]
        ax.boxplot(data, labels=DAY_TYPES, showfliers=False)
        ax.axhline(0, color="tomato", lw=1)
        ax.set_title(saison, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Residual (MW)")
    fig.suptitle("Residuals by (season × day-type) — systematic bias?",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()


def plot_residuals_by_hour_season(residuals_df: pd.DataFrame,
                                    save: str | None = None) -> None:
    """
    Résidu moyen par (heure × saison) — corollaire de la heatmap MAPE
    mais avec le SIGNE (pas la valeur absolue).

    Permet de voir où le modèle sur-estime (résidu < 0) ou sous-estime
    (résidu > 0) systématiquement. Par exemple, un résidu > 0 le soir
    en DJF = "la vraie conso est au-dessus du profil" = le pic du soir
    réel est plus marqué que le profil moyen.
    """
    agg = residuals_df.groupby(["saison", "hour"])["resid"].mean().reset_index()
    pivot = agg.pivot(index="saison", columns="hour", values="resid")
    pivot = pivot.reindex(SEASONS)

    fig, ax = plt.subplots(figsize=(12, 3))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(SEASONS))); ax.set_yticklabels(SEASONS)
    ax.set_xlabel("Hour")
    ax.set_title("Mean signed residual by (season, hour) "
                  "— red = under-estimate, blue = over-estimate",
                  fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if abs(v) > vmax * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Mean residual (MW)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()


def plot_residuals_vs_temperature(residuals_df: pd.DataFrame,
                                   region: str,
                                   save: str | None = None) -> None:
    """
    Quantifies how much hourly reconstruction errors correlate with temperature.

    Requires `residuals_df` to have a column 'T' (built by compute_residuals
    when temp_hourly is provided).

    Four panels:
        (1) Scatter residual vs T, with 1°C-bin mean — reveals any remaining
            temperature-driven bias in the disaggregation step.
        (2) Mean absolute residual by temperature bin — shows where errors are
            largest (cold peaks, heat waves, mild weather).
        (3) Mean residual by (T-bin × season) heatmap — checks whether the
            temperature signal interacts with seasonality.
        (4) R² of a linear fit resid ~ T, by season and hour — quantifies the
            fraction of reconstruction error variance explained by temperature.
    """
    if "T" not in residuals_df.columns or residuals_df["T"].isna().all():
        print(f"[{region}] No temperature column in residuals_df — pass temp_hourly to compute_residuals.")
        return

    res = residuals_df.dropna(subset=["T", "resid"])

    t_bins = np.arange(np.floor(res["T"].min()), np.ceil(res["T"].max()) + 1, 1.0)
    res = res.copy()
    res["T_bin"] = pd.cut(res["T"], bins=t_bins)
    bin_centers = res.groupby("T_bin")["T"].mean()

    mean_resid = res.groupby("T_bin")["resid"].mean()
    mae_resid  = res.groupby("T_bin")["resid"].apply(lambda x: x.abs().mean())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Residuals vs Temperature — {region}", fontsize=13, fontweight="bold")

    # (1) Scatter + mean per °C
    ax = axes[0, 0]
    sample = res.sample(min(len(res), 6000), random_state=0)
    ax.scatter(sample["T"], sample["resid"], s=2, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(bin_centers.values, mean_resid.values, color="tomato", lw=2, label="mean per °C")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Residual (MW)")
    ax.set_title("(1) Residual vs temperature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (2) MAE by temperature bin
    ax = axes[0, 1]
    ax.bar(bin_centers.values, mae_resid.values, width=0.8,
           color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("(2) Mean absolute error by temperature bin")
    ax.grid(axis="y", alpha=0.3)

    # (3) Heatmap mean residual by (T-bin × season)
    ax = axes[1, 0]
    pivot_s = (
        res.groupby(["saison", "T_bin"])["resid"].mean()
        .unstack("T_bin")
    )
    pivot_s = pivot_s.reindex(SEASONS)
    col_mids = [b.mid for b in pivot_s.columns]
    vmax = np.nanmax(np.abs(pivot_s.values))
    im = ax.imshow(pivot_s.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[min(col_mids), max(col_mids),
                            len(SEASONS) - 0.5, -0.5])
    ax.set_yticks(range(len(SEASONS)))
    ax.set_yticklabels(SEASONS)
    ax.set_xlabel("Temperature (°C)")
    ax.set_title("(3) Mean residual by (season × temperature)")
    fig.colorbar(im, ax=ax, label="Mean residual (MW)")
    ax.grid(False)

    # (4) R² of resid ~ T by (season × hour)
    ax = axes[1, 1]
    from scipy.stats import pearsonr

    r2_grid = np.full((len(SEASONS), 24), np.nan)
    for si, saison in enumerate(SEASONS):
        sub_s = res[res["saison"] == saison]
        for h in range(24):
            sub_h = sub_s[sub_s["hour"] == h]
            if len(sub_h) >= 10 and sub_h["T"].std() > 0:
                r, _ = pearsonr(sub_h["T"], sub_h["resid"])
                r2_grid[si, h] = r ** 2

    im2 = ax.imshow(r2_grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(range(0, 24, 3))
    ax.set_yticks(range(len(SEASONS)))
    ax.set_yticklabels(SEASONS)
    ax.set_xlabel("Hour of day")
    ax.set_title("(4) R²(resid ~ T) by season × hour")
    fig.colorbar(im2, ax=ax, label="R²")
    for i in range(len(SEASONS)):
        for j in range(24):
            v = r2_grid[i, j]
            if np.isfinite(v) and v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  → {save}")
    plt.show()

    # Summary stat: overall R² of resid ~ T
    r_all, _ = pearsonr(res["T"], res["resid"])
    print(f"\nTemperature–residual correlation — {region}")
    print(f"  R²(resid ~ T) overall : {r_all**2:.4f}  "
          f"({'strong' if r_all**2 > 0.1 else 'weak'} temperature signal in disaggregation errors)")
    print(f"  Mean residual          : {res['resid'].mean():+.2f} MW")
    print(f"  MAE                    : {res['resid'].abs().mean():.2f} MW")


# =============================================================================
# 8. PIPELINE COMPLET
# =============================================================================

def run_pipeline(conso_path: str,
                  temp_path: str | None = None,
                  eval_years: list[int] = (2019, 2023, 2024),
                  exclude_years: list[int] | None = None,
                  ) -> dict:
    """
    Pipeline principal : charge les données, calcule les profils, évalue.

    Parameters
    ----------
    eval_years : années sur lesquelles évaluer les profils.
    exclude_years : années exclues du calcul des profils (défaut : EXCLUDE_YEARS,
        soit [2020, 2021] pour la période COVID). Pour une évaluation
        out-of-sample, passer exclude_years=EXCLUDE_YEARS + list(eval_years)
        afin que les années d'évaluation ne soient pas vues à l'entraînement.

    Returns
    -------
    dict avec clés :
        'conso'    : DataFrame horaire original
        'profiles' : dict { region: DataFrame profil }
        'metrics'  : DataFrame métriques par région (moyenne sur eval_years)
        'per_year' : dict { year: DataFrame métriques }
    """
    if exclude_years is None:
        exclude_years = list(EXCLUDE_YEARS)

    print("=" * 75)
    print("PIPELINE DÉSAGRÉGATION HORAIRE — ÉTAPE 1")
    print("=" * 75)

    conso, temp = load_hourly_data(conso_path, temp_path)

    print("\n[1/3] Calcul des profils normalisés…")
    print(f"  Années exclues de l'entraînement : {sorted(exclude_years)}")
    profiles = compute_profiles(conso, exclude_years=exclude_years)
    print(f"  ✓ {len(profiles)} régions, {len(DAY_TYPES) * len(SEASONS)} profils/région")

    print("\n[2/3] Évaluation par année…")
    per_year = {}
    for yr in eval_years:
        m = evaluate_disaggregation(conso, profiles, years=[yr])
        per_year[yr] = m
        print(f"\n--- {yr} ---")
        print(m.to_string())

    print("\n[3/3] Moyenne multi-années…")
    all_metrics = pd.concat(per_year.values(), keys=per_year.keys(), names=["year"])
    avg = all_metrics.groupby("region")[["RMSE", "MAE", "MAPE_%", "R2"]].mean()
    avg = avg.round({"RMSE": 1, "MAE": 1, "MAPE_%": 2, "R2": 4})
    print(avg.to_string())

    # Agrégat national (moyenne pondérée par taille de région)
    nat_mape = (all_metrics["MAPE_%"] * all_metrics["N"]).groupby(
        all_metrics.index.get_level_values("year")
    ).sum() / all_metrics["N"].groupby(
        all_metrics.index.get_level_values("year")
    ).sum()
    print("\nMAPE pondéré national par année :")
    for y, v in nat_mape.items():
        print(f"  {y} : {v:.2f} %")

    return {
        "conso":    conso,
        "temp":     temp,
        "profiles": profiles,
        "metrics":  avg,
        "per_year": per_year,
    }


# =============================================================================
# 9. EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":

    CONSO_PATH = "data/conso_horaire.csv"
    TEMP_PATH  = "data/temp_horaire.csv"       # non utilisé étape 1

    result = run_pipeline(CONSO_PATH, TEMP_PATH,
                           eval_years=[2019, 2023, 2024])

    conso    = result["conso"]
    profiles = result["profiles"]

    # --- Visualisations sur une région exemple ---
    region_demo = "Île-de-France"

    plot_profiles(profiles, region_demo,
                   save=OUTPUT_DIR / f"profiles_{region_demo}.png")

    # Semaine type hiver
    plot_week_comparison(conso, profiles, region_demo,
                          start_date="2019-01-14", n_days=7,
                          save=OUTPUT_DIR / f"week_winter_{region_demo}.png")

    # Semaine type été
    plot_week_comparison(conso, profiles, region_demo,
                          start_date="2019-07-15", n_days=7,
                          save=OUTPUT_DIR / f"week_summer_{region_demo}.png")

    # Heatmap erreur
    plot_error_heatmap(conso, profiles, region_demo, years=[2019, 2023, 2024],
                        save=OUTPUT_DIR / f"heatmap_err_{region_demo}.png")
    

    # --- Diagnostic des résidus ---
    temp = result.get("temp")
    res = plot_residuals_diagnostics(
        conso, profiles, region_demo,
        years=[2019, 2023, 2024],
        temp_hourly=temp,
        save=OUTPUT_DIR / f"residuals_{region_demo}.png",
    )
    plot_residuals_by_typology(
        res, save=OUTPUT_DIR / f"residuals_typology_{region_demo}.png"
    )
    plot_residuals_by_hour_season(
        res, save=OUTPUT_DIR / f"residuals_signed_{region_demo}.png"
    )

    # --- Utilisation typique : désagréger une série journalière issue du modèle quotidien ---
    #
    # >>> from baseline_electricity_model import run_pipeline as run_daily
    # >>> daily_out = run_daily("data/conso.csv", "data/temp.csv")
    # >>> baseline_journaliere = daily_out[1]["Île-de-France"]   # MWh/jour
    # >>> baseline_horaire = disaggregate_daily(
    # ...     baseline_journaliere, "Île-de-France", profiles
    # ... )
    # >>> # baseline_horaire est en MWh/h, somme sur chaque jour = valeur journalière
