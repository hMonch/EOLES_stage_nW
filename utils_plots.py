"""
Plotting functions for ModelEOLES outputs (weekly dispatch, installed
capacity, generation mix, storage state, residual load, spot prices, LRMC,
price-setter analysis). All take already-extracted result objects (see
utils_results.py) - none of these touch the linopy model directly.

The bottom section (from TECH_COLORS onward) holds cross-year / cross-scenario
comparison charts, meant to be fed by utils_batch.py's loading functions - see
notebook_batch_comparison.ipynb.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.dates as mdates

from utils_results import compute_residual_demand, compute_lrmc_per_tech, extract_capacity_duals


def extract_time_index(hourly_balance, start_hour=0, nb_hours=None):
    """
    Extract time coordinates from xarray dataset as pd.DatetimeIndex.
    
    Supports two formats:
    - datetime64: directly extracts the datetime values
    - tuple (date, hour): combines date and hour components
    
    Parameters:
    -----------
    hourly_balance : xr.Dataset
        Dataset with 'hour' coordinates in datetime64 or tuple format
    start_hour : int
        Starting hour index
    nb_hours : int, optional
        Number of hours to extract. If None, extracts all remaining hours
        
    Returns:
    --------
    pd.DatetimeIndex
    """
    coords = hourly_balance.coords['hour'].values
    
    # Calculate end index
    end_hour = start_hour + nb_hours if nb_hours else len(coords)
    
    # Extract the time values
    time_vals = coords[start_hour:end_hour]
    
    # Convert to DatetimeIndex
    # If already datetime64, pd.to_datetime will handle it directly
    # If tuples (date, hour), this will fail but we assume datetime64 format now
    try:
        time_index = pd.to_datetime(time_vals)
    except (TypeError, ValueError):
        # Fallback for old format with tuples (date, hour)
        time_index = pd.to_datetime([pd.Timestamp(d) + pd.Timedelta(hours=int(h)) for d, h in time_vals])
    
    return time_index




'''[[[ Functions used to plot outputs_main_scenario ]]]'''



def plot_load_shift_week(model_instance, hourly_balance, hour, area=None, lang="EN"):
    """
    Plot demand flexibility over a week.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle (pour accéder aux sets)
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Si le dataset a une dimension
        'area', passer area= pour sélectionner un pays, sinon les données sont
        agrégées sur tous les pays.
    hour : int
        Heure de démarrage
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    lang : str
        Langue ("EN" ou "FR")
    """
    # --- résolution de la dimension area ---
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")

    area_label = f" – {area}" if area is not None else ""

    fig, ax = plt.subplots(figsize=(16, 10))

    # Extraction des données
    load_shift_up = hourly_balance["load_shift_up"].isel(hour=slice(hour, hour+7*24)).values
    load_shift_down = hourly_balance["load_shift_down"].isel(hour=slice(hour, hour+7*24)).values
    net_shift = load_shift_up - load_shift_down
    
    positive_net_shift = np.where(net_shift < 0, 0, net_shift)
    negative_net_shift = -np.where(net_shift > 0, 0, net_shift)
    
    elec_demand = hourly_balance["elec_demand"].isel(hour=slice(hour, hour+7*24)).values
    elec_demand_w_shift = hourly_balance["elec_demand_w_shift"].isel(hour=slice(hour, hour+7*24)).values
    
    # Somme des productions non pilotables
    nonoperable_prod = np.zeros(7*24)
    for tech in model_instance.solar.values:
        nonoperable_prod += hourly_balance[tech].isel(hour=slice(hour, hour+7*24)).values
    
    for tech in ["biomass_coge", "geothermal_coge", "waste", "ocgt_coge", "river", "marine", "onshore", "offshore_ground", "offshore_float"]:
        if tech in hourly_balance.data_vars:
            nonoperable_prod += hourly_balance[tech].isel(hour=slice(hour, hour+7*24)).values
    
    # Création de l'axe temps
    time = extract_time_index(hourly_balance, hour, 7*24)

    if lang == "EN":
        ax.stackplot(time, nonoperable_prod, lw=0, color="#d66b0d60", labels=["Non-operable production"])
        ax.stackplot(time, elec_demand, positive_net_shift, lw=0, colors=[(0,0,0,0), "orange"], labels=[None, "Positive net shift"])
        ax.stackplot(time, elec_demand_w_shift, negative_net_shift, lw=0, colors=[(0,0,0,0), "steelblue"], labels=[None, "Negative net shift"])

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demand before load shift")
        ax.plot(time, elec_demand_w_shift, lw=1.8, color="black", label="Demand after load shift")

        ax.legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.38, +1.06), columnspacing=0.5)
        ax.text(x=0.113, y=0.93, s=f"Electricity demand load shift over one week{area_label}", transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_ylabel("Electricity demand [GW]", fontsize = 13)

    if lang == "FR":
        ax.stackplot(time, nonoperable_prod, lw=0, color="#d66b0d60", labels=["Production non pilotable"])
        ax.stackplot(time, elec_demand, positive_net_shift, lw=0, colors=[(0,0,0,0), "orange"], labels=[None, "Décalage net positif"])
        ax.stackplot(time, elec_demand_w_shift, negative_net_shift, lw=0, colors=[(0,0,0,0), "steelblue"], labels=[None, "Décalage net négatif"])

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demande avant pilotage")
        ax.plot(time, elec_demand_w_shift, lw=1.8, color="black", label="Demande après pilotage")

        ax.legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.38, +1.06), columnspacing=0.5)
        ax.text(x=0.113, y=0.93, s=f"Pilotage de la demande sur une semaine{area_label}", transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_ylabel("Demande d'électricité [GW]", fontsize = 13)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.grid(True)
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig




def plot_elec_balance_week(model_instance, hourly_balance, installed_power, hour, area=None, include_str=True, net_import=True, lang="EN", n_hours=7*24):
    """
    Affiche l'équilibre du réseau électrique sur une semaine.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Peut avoir une dimension 'area' ;
        passer area= pour sélectionner un pays.
    installed_power : xr.DataArray
        Puissances installées par technologie. Peut avoir une dimension 'area'.
    hour : int
        Heure de démarrage (index entier)
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    include_str : bool
        Inclure les flux de stockage/exports (négatifs) dans le graphe. Défaut True.
    lang : str
        Langue ("EN" ou "FR")

    Example:
    --------
    fig = plot_elec_balance_week(model, model.hourly_balance, model.installed_power,
                                  hour=168, area="ES", lang="FR")
    """
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")
    if isinstance(installed_power, xr.DataArray) and "area" in installed_power.dims:
        installed_power = installed_power.sel(area=area) if area is not None else installed_power.sum("area")

    area_label = f" – {area}" if area is not None else ""
    _use_str_dummy = (
        "str_dummy" in hourly_balance.data_vars
        and float(hourly_balance["str_dummy"].sum("hour").values) > 10
    )
    n = n_hours
    sl = slice(hour, hour + n)

    def _w(var):
        if var in hourly_balance.data_vars:
            return hourly_balance[var].isel(hour=sl).values
        return np.zeros(n)

    def _ip(tech):
        if tech in installed_power.coords['tech'].values:
            return float(installed_power.sel(tech=tech).values)
        return 0.0

    _ed = _w("elec_demand_w_shift") if "elec_demand_w_shift" in hourly_balance.data_vars else _w("elec_demand")
    elec_demand = _ed if not np.any(np.isnan(_ed)) else _w("elec_demand")
    time = extract_time_index(hourly_balance, hour, n)

    # ---- positive stacks (production + imports) ----
    elec_prod = {}
    colors_prod = []

    def _add_prod(label, val, color):
        elec_prod[label] = val
        colors_prod.append(color)

    _imp = _w("imports")
    _exp = _w("exports")
    if net_import:
        _net     = _imp - _exp
        _net_pos = np.maximum(_net, 0.0)
        _net_neg = np.minimum(_net, 0.0)

    if lang == "EN":
        _add_prod("Cogeneration",      _w("biomass_coge") + _w("geothermal_coge") + _w("waste") + _w("ocgt_coge"), "#156956")
        _add_prod("Hydropower - Other",_w("river") + _w("marine"),                                                  "#2672b0")
        _add_prod("Wind - onshore",    _w("onshore"),                                                                "#74cb2e")
        _add_prod("Wind - offshore",   _w("offshore_ground") + _w("offshore_float"),                                 "#72cbb7")
        _add_prod("Hydropower - Dams", _w("lake"),                                                                   "#1a4f85")
        _add_prod("Solar",             sum(_w(t) for t in model_instance.solar.values),                              "#d66b0d")
        if _ip("nuclear") > 0: _add_prod("Nuclear",  _w("nuclear"), "#e4a701")
        if _ip("coal")    > 0: _add_prod("Coal",     _w("coal"),    "#a68832")
        _add_prod("PHS",           _w("phs"),                                                                        "#0e4269")
        _add_prod("Batteries",     sum(_w(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]),        "#80549f")
        _add_prod("CH4 turbines",  _w("ch4_ocgt") + _w("ch4_ccgt"),                                                 "#f20809")
        _add_prod("H2 turbines",   _w("h2_ccgt"),                                                                    "#f252c0")
        if _use_str_dummy:
            _add_prod("Missing storage", _w("str_dummy"), "#757575")
        if net_import:
            if _net_pos.max() > 1e-6: _add_prod("Net imports", _net_pos, "#00b4d8")
        else:
            if _imp.sum() > 1e-6: _add_prod("Imports", _imp, "#00b4d8")
    else:
        _add_prod("Cogénération",          _w("biomass_coge") + _w("geothermal_coge") + _w("waste") + _w("ocgt_coge"), "#156956")
        _add_prod("Hydraulique - Autres",  _w("river") + _w("marine"),                                                  "#2672b0")
        _add_prod("Eolien - Terrestre",    _w("onshore"),                                                                "#74cb2e")
        _add_prod("Eolien - En mer",       _w("offshore_ground") + _w("offshore_float"),                                 "#72cbb7")
        _add_prod("Hydraulique - Barrages",_w("lake"),                                                                   "#1a4f85")
        _add_prod("Photovoltaïque",        sum(_w(t) for t in model_instance.solar.values),                              "#d66b0d")
        if _ip("nuclear") > 0: _add_prod("Nucléaire", _w("nuclear"), "#e4a701")
        if _ip("coal")    > 0: _add_prod("Charbon",   _w("coal"),    "#a68832")
        _add_prod("STEP",          _w("phs"),                                                                            "#0e4269")
        _add_prod("Batteries",     sum(_w(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]),            "#80549f")
        _add_prod("Turbines CH4",  _w("ch4_ocgt") + _w("ch4_ccgt"),                                                     "#f20809")
        _add_prod("Turbines H2",   _w("h2_ccgt"),                                                                        "#f252c0")
        if _use_str_dummy:
            _add_prod("Stockage manquant", _w("str_dummy"), "#757575")
        if net_import:
            if _net_pos.max() > 1e-6: _add_prod("Imports nets", _net_pos, "#00b4d8")
        else:
            if _imp.sum() > 1e-6: _add_prod("Imports", _imp, "#00b4d8")

    fig, ax = plt.subplots(figsize=(16, 10))
    handles_prod = ax.stackplot(time, *elec_prod.values(), labels=elec_prod.keys(), colors=colors_prod)

    # ---- negative stacks (storage input + exports) ----
    elec_str = {}
    colors_str = []

    def _add_str(label, val, color):
        elec_str[label] = val
        colors_str.append(color)

    if include_str:
        if lang == "EN":
            _add_str("PHS",       -_w("phs_input"),                                                                    "#0e4269")
            _add_str("Batteries", -sum(_w(t + "_input") for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
            if "methanation_input"  in hourly_balance.data_vars: _add_str("Methanation", -_w("methanation_input"),  "#f252c0")
            if "electrolysis_input" in hourly_balance.data_vars: _add_str("Electrolysis",-_w("electrolysis_input"), "#9370DB")
            if _use_str_dummy: _add_str("Missing storage", -_w("str_dummy_input"), "#757575")
            if net_import:
                if _net_neg.min() < -1e-6: _add_str("Net exports", _net_neg, "#e07a5f")
            else:
                if _exp.sum() > 1e-6: _add_str("Exports", -_exp, "#e07a5f")
        else:
            _add_str("STEP",      -_w("phs_input"),                                                                    "#0e4269")
            _add_str("Batteries", -sum(_w(t + "_input") for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
            if "methanation_input"  in hourly_balance.data_vars: _add_str("Méthanation", -_w("methanation_input"),  "#f252c0")
            if "electrolysis_input" in hourly_balance.data_vars: _add_str("Electrolyse", -_w("electrolysis_input"), "#9370DB")
            if _use_str_dummy: _add_str("Stockage manquant", -_w("str_dummy_input"), "#757575")
            if net_import:
                if _net_neg.min() < -1e-6: _add_str("Exports nets", _net_neg, "#e07a5f")
            else:
                if _exp.sum() > 1e-6: _add_str("Exports", -_exp, "#e07a5f")
        handles_str = ax.stackplot(time, *elec_str.values(), labels=elec_str.keys(), colors=colors_str)
    else:
        handles_str = []

    # ---- demand line (drawn on top) ----
    if lang == "EN":
        handle_demand = ax.plot(time, elec_demand, lw=2, color="black", zorder=5, label="Electricity demand")
        ax.set_ylabel('Electricity Production and Usage [GW]', fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Electricity Balance Over One Week{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Production by source (+ imports) and usage (storage charging + exports)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_str  = ax.legend(handles=handles_str + handle_demand, loc='upper center', ncol=len(elec_str) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.21, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=len(elec_prod),
                             title="Production", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.5, +1.14), frameon=False, columnspacing=0.5)
    else:
        handle_demand = ax.plot(time, elec_demand, lw=2, color="black", zorder=5, label="Demande d'électricité")
        ax.set_ylabel("Production et utilisation de l'électricité [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Equilibre du réseau électrique sur une semaine{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Production par source (+ imports) et usage (stockage + exports) sur une semaine", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_str  = ax.legend(handles=handles_str + handle_demand, loc='upper center', ncol=len(elec_str) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.241, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=max(1, len(elec_prod) - 2),
                             title="Production", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.45, +1.165), frameon=False, columnspacing=0.5)

    ax.add_artist(leg_str)
    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig


def plot_CH4_balance(model_instance, hourly_balance, installed_power, hour, area=None,
                     n_hours=7*24, include_str=True, lang="EN"):
    """
    Affiche l'équilibre du réseau CH4 sur une période configurable.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Peut avoir une dimension 'area'.
    installed_power : xr.DataArray
        Puissances installées par technologie. Peut avoir une dimension 'area'.
    hour : int
        Heure de démarrage (index entier)
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    n_hours : int
        Nombre d'heures à afficher. Défaut 7*24 (une semaine). Utiliser 30*24 pour un mois.
    include_str : bool
        Inclure les flux de stockage (négatifs) dans le graphe. Défaut True.
    lang : str
        Langue ("EN" ou "FR")

    Example:
    --------
    fig = plot_CH4_balance(model, model.hourly_balance, model.installed_power,
                           hour=0, area="FR", n_hours=30*24, lang="FR")
    """
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")
    if isinstance(installed_power, xr.DataArray) and "area" in installed_power.dims:
        installed_power = installed_power.sel(area=area) if area is not None else installed_power.sum("area")

    area_label = f" – {area}" if area is not None else ""
    sl = slice(hour, hour + n_hours)
    n = n_hours

    def _w(var):
        if var in hourly_balance.data_vars:
            return hourly_balance[var].isel(hour=sl).values
        return np.zeros(n)

    time = extract_time_index(hourly_balance, hour, n)
    ch4_demand = _w("CH4_demand")

    if n_hours <= 7 * 24:
        period_label_en, period_label_fr = "One Week", "une semaine"
    elif n_hours <= 31 * 24:
        period_label_en, period_label_fr = "One Month", "un mois"
    else:
        period_label_en = f"{n_hours // 24} Days"
        period_label_fr = f"{n_hours // 24} jours"

    # ---- positive stacks (supply) ----
    ch4_prod = {}
    colors_prod = []

    def _add_prod(label, val, color):
        if val.sum() > 1e-6:
            ch4_prod[label] = val
            colors_prod.append(color)

    if lang == "EN":
        _add_prod("Natural gas",      _w("natural_gas"),     "#c0392b")
        _add_prod("Biogas import",    _w("biogas_import"),   "#27ae60")
        _add_prod("Methanization",    _w("methanization"),   "#2ecc71")
        _add_prod("Pyrogazification", _w("pyrogazification"), "#16a085")
        _add_prod("Methanation",      _w("methanation"),     "#8e44ad")
        _add_prod("CH4 storage",      _w("ch4_reservoir"),   "#e67e22")
    else:
        _add_prod("Gaz naturel",      _w("natural_gas"),     "#c0392b")
        _add_prod("Import biogaz",    _w("biogas_import"),   "#27ae60")
        _add_prod("Méthanisation",    _w("methanization"),   "#2ecc71")
        _add_prod("Pyrogazification", _w("pyrogazification"), "#16a085")
        _add_prod("Méthanation",      _w("methanation"),     "#8e44ad")
        _add_prod("Stockage CH4",     _w("ch4_reservoir"),   "#e67e22")

    fig, ax = plt.subplots(figsize=(16, 8))
    handles_prod = ax.stackplot(time, *ch4_prod.values(), labels=ch4_prod.keys(), colors=colors_prod) if ch4_prod else []

    # ---- negative stacks (usage: turbines + storage charging) ----
    ch4_use = {}
    colors_use = []

    def _add_use(label, val, color):
        if np.abs(val).sum() > 1e-6:
            ch4_use[label] = val
            colors_use.append(color)

    if include_str:
        if lang == "EN":
            _add_use("CH4 turbines (OCGT)",  -_w("ch4_ocgt_input"),      "#e74c3c")
            _add_use("CH4 turbines (CCGT)",  -_w("ch4_ccgt_input"),      "#c0392b")
            _add_use("Cogeneration (OCGT)",  -_w("ocgt_coge_input"),     "#922b21")
            _add_use("CH4 storage charge",   -_w("ch4_reservoir_input"), "#e67e22")
        else:
            _add_use("Turbines CH4 (OCGT)",  -_w("ch4_ocgt_input"),      "#e74c3c")
            _add_use("Turbines CH4 (CCGT)",  -_w("ch4_ccgt_input"),      "#c0392b")
            _add_use("Cogénération (OCGT)",  -_w("ocgt_coge_input"),     "#922b21")
            _add_use("Charge stockage CH4",  -_w("ch4_reservoir_input"), "#e67e22")
        handles_use = ax.stackplot(time, *ch4_use.values(), labels=ch4_use.keys(), colors=colors_use) if ch4_use else []
    else:
        handles_use = []

    if lang == "EN":
        handle_demand = ax.plot(time, ch4_demand, lw=2, color="black", zorder=5, label="CH4 demand")
        ax.set_ylabel("CH4 Production and Usage [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"CH4 Balance Over {period_label_en}{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Supply by source and usage (turbines + storage charging)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_use  = ax.legend(handles=handles_use + handle_demand, loc='upper center', ncol=len(ch4_use) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.21, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=max(1, len(ch4_prod)),
                             title="Supply", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.5, +1.14), frameon=False, columnspacing=0.5)
    else:
        handle_demand = ax.plot(time, ch4_demand, lw=2, color="black", zorder=5, label="Demande CH4")
        ax.set_ylabel("Production et utilisation CH4 [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Équilibre CH4 sur {period_label_fr}{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Offre par source et usage (turbines + charge stockage)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_use  = ax.legend(handles=handles_use + handle_demand, loc='upper center', ncol=len(ch4_use) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.241, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=max(1, len(ch4_prod)),
                             title="Offre", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.45, +1.165), frameon=False, columnspacing=0.5)

    ax.add_artist(leg_use)
    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    if n_hours <= 7 * 24:
        ax.xaxis.set_major_locator(mdates.DayLocator())
    elif n_hours <= 31 * 24:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig


def plot_H2_balance(model_instance, hourly_balance, installed_power, hour, area=None,
                    n_hours=7*24, include_str=True, lang="EN"):
    """
    Affiche l'équilibre du réseau H2 sur une période configurable.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Peut avoir une dimension 'area'.
    installed_power : xr.DataArray
        Puissances installées par technologie. Peut avoir une dimension 'area'.
    hour : int
        Heure de démarrage (index entier)
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    n_hours : int
        Nombre d'heures à afficher. Défaut 7*24 (une semaine). Utiliser 30*24 pour un mois.
    include_str : bool
        Inclure les flux de stockage (négatifs) dans le graphe. Défaut True.
    lang : str
        Langue ("EN" ou "FR")

    Example:
    --------
    fig = plot_H2_balance(model, model.hourly_balance, model.installed_power,
                          hour=0, area="FR", n_hours=30*24, lang="FR")
    """
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")
    if isinstance(installed_power, xr.DataArray) and "area" in installed_power.dims:
        installed_power = installed_power.sel(area=area) if area is not None else installed_power.sum("area")

    area_label = f" – {area}" if area is not None else ""
    sl = slice(hour, hour + n_hours)
    n = n_hours

    def _w(var):
        if var in hourly_balance.data_vars:
            return hourly_balance[var].isel(hour=sl).values
        return np.zeros(n)

    time = extract_time_index(hourly_balance, hour, n)
    h2_demand = _w("H2_demand")

    if n_hours <= 7 * 24:
        period_label_en, period_label_fr = "One Week", "une semaine"
    elif n_hours <= 31 * 24:
        period_label_en, period_label_fr = "One Month", "un mois"
    else:
        period_label_en = f"{n_hours // 24} Days"
        period_label_fr = f"{n_hours // 24} jours"

    # ---- positive stacks (supply) ----
    h2_prod = {}
    colors_prod = []

    def _add_prod(label, val, color):
        if val.sum() > 1e-6:
            h2_prod[label] = val
            colors_prod.append(color)

    if lang == "EN":
        _add_prod("Electrolysis", _w("electrolysis"),   "#9370DB")
        _add_prod("H2 storage",   _w("h2_saltcavern"),  "#3498db")
    else:
        _add_prod("Électrolyse",  _w("electrolysis"),   "#9370DB")
        _add_prod("Stockage H2",  _w("h2_saltcavern"),  "#3498db")

    fig, ax = plt.subplots(figsize=(16, 8))
    handles_prod = ax.stackplot(time, *h2_prod.values(), labels=h2_prod.keys(), colors=colors_prod) if h2_prod else []

    # ---- negative stacks (usage: turbines + storage charging) ----
    h2_use = {}
    colors_use = []

    def _add_use(label, val, color):
        if np.abs(val).sum() > 1e-6:
            h2_use[label] = val
            colors_use.append(color)

    if include_str:
        if lang == "EN":
            _add_use("H2 turbines (CCGT)",  -_w("h2_ccgt_input"),      "#f252c0")
            _add_use("H2 storage charge",   -_w("h2_saltcavern_input"), "#3498db")
        else:
            _add_use("Turbines H2 (CCGT)",  -_w("h2_ccgt_input"),      "#f252c0")
            _add_use("Charge stockage H2",  -_w("h2_saltcavern_input"), "#3498db")
        handles_use = ax.stackplot(time, *h2_use.values(), labels=h2_use.keys(), colors=colors_use) if h2_use else []
    else:
        handles_use = []

    if lang == "EN":
        handle_demand = ax.plot(time, h2_demand, lw=2, color="black", zorder=5, label="H2 demand")
        ax.set_ylabel("H2 Production and Usage [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"H2 Balance Over {period_label_en}{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Supply (electrolysis + discharge) and usage (turbines + storage charging)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_use  = ax.legend(handles=handles_use + handle_demand, loc='upper center', ncol=len(h2_use) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.21, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=max(1, len(h2_prod)),
                             title="Supply", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.5, +1.14), frameon=False, columnspacing=0.5)
    else:
        handle_demand = ax.plot(time, h2_demand, lw=2, color="black", zorder=5, label="Demande H2")
        ax.set_ylabel("Production et utilisation H2 [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Équilibre H2 sur {period_label_fr}{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Offre (électrolyse + décharge) et usage (turbines + charge stockage)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_use  = ax.legend(handles=handles_use + handle_demand, loc='upper center', ncol=len(h2_use) + 1,
                             title="Usage", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.241, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=max(1, len(h2_prod)),
                             title="Offre", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.45, +1.165), frameon=False, columnspacing=0.5)

    ax.add_artist(leg_use)
    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    if n_hours <= 7 * 24:
        ax.xaxis.set_major_locator(mdates.DayLocator())
    elif n_hours <= 31 * 24:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig




def plot_elec_residual_balance_week(model_instance, hourly_balance, installed_power, hour, area=None, net_import=True, lang="EN"):
    """
    Affiche l'équilibre résiduel du réseau électrique sur une semaine.

    La demande résiduelle = demande ajustée (avec pilotage de charge) - production non pilotable (ENR).
    Les imports apparaissent en positif (au-dessus de 0), les exports en négatif (en-dessous).
    La courbe de demande résiduelle peut être négative si l'ENR dépasse la demande locale.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Peut avoir une dimension 'area'.
    installed_power : xr.DataArray
        Puissances installées par technologie. Peut avoir une dimension 'area'.
    hour : int
        Heure de démarrage (index entier)
    area : str or None
        Identifiant de zone (ex. "FR", "ES"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    lang : str
        Langue ("EN" ou "FR")

    Example:
    --------
    fig = plot_elec_residual_balance_week(model, model.hourly_balance, model.installed_power,
                                           hour=168, area="DE", lang="EN")
    """
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")
    if isinstance(installed_power, xr.DataArray) and "area" in installed_power.dims:
        installed_power = installed_power.sel(area=area) if area is not None else installed_power.sum("area")

    area_label = f" – {area}" if area is not None else ""
    _use_str_dummy = (
        "str_dummy" in hourly_balance.data_vars
        and float(hourly_balance["str_dummy"].sum("hour").values) > 10
    )
    n = 7 * 24
    sl = slice(hour, hour + n)

    def _w(var):
        if var in hourly_balance.data_vars:
            return hourly_balance[var].isel(hour=sl).values
        return np.zeros(n)

    def _ip(tech):
        if tech in installed_power.coords['tech'].values:
            return float(installed_power.sel(tech=tech).values)
        return 0.0

    time = extract_time_index(hourly_balance, hour, n)

    # Demand with load-shift adjustment; fall back to raw demand if NaN (non-FR areas have no load shift)
    _ed = _w("elec_demand_w_shift") if "elec_demand_w_shift" in hourly_balance.data_vars else _w("elec_demand")
    elec_demand = _ed if not np.any(np.isnan(_ed)) else _w("elec_demand")

    # Non-dispatchable (VRE) production
    nonoperable_prod = sum(_w(t) for t in model_instance.solar.values)
    for tech in ["biomass_coge", "geothermal_coge", "waste", "ocgt_coge",
                 "river", "marine", "onshore", "offshore_ground", "offshore_float"]:
        nonoperable_prod = nonoperable_prod + _w(tech)

    # Residual demand = adjusted demand minus non-dispatchable (can be negative)
    residual_elec_demand = elec_demand - nonoperable_prod

    # ---- positive stacks (operable production + imports) ----
    operable_prod = {}
    colors_prod = []

    def _add_op(label, val, color):
        operable_prod[label] = val
        colors_prod.append(color)

    _imp = _w("imports")
    _exp = _w("exports")
    if net_import:
        _net     = _imp - _exp
        _net_pos = np.maximum(_net, 0.0)
        _net_neg = np.minimum(_net, 0.0)

    if lang == "EN":
        _add_op("Hydropower - Dams", _w("lake"),                              "#1a4f85")
        if _ip("nuclear") > 0: _add_op("Nuclear",  _w("nuclear"),             "#e4a701")
        if _ip("coal")    > 0: _add_op("Coal",     _w("coal"),                "#a68832")
        _add_op("PHS",           _w("phs"),                                    "#0e4269")
        _add_op("Batteries",     sum(_w(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
        _add_op("CH4 turbines",  _w("ch4_ocgt") + _w("ch4_ccgt"),             "#f20809")
        _add_op("H2 turbines",   _w("h2_ccgt"),                               "#f252c0")
        if _use_str_dummy:
            _add_op("Missing storage", _w("str_dummy"),                        "#757575")
        if net_import:
            if _net_pos.max() > 1e-6: _add_op("Net imports", _net_pos,        "#00b4d8")
        else:
            if _imp.sum() > 1e-6:    _add_op("Imports",     _imp,             "#00b4d8")
    else:
        _add_op("Hydraulique - Barrages", _w("lake"),                         "#1a4f85")
        if _ip("nuclear") > 0: _add_op("Nucléaire", _w("nuclear"),            "#e4a701")
        if _ip("coal")    > 0: _add_op("Charbon",   _w("coal"),               "#a68832")
        _add_op("STEP",          _w("phs"),                                    "#0e4269")
        _add_op("Batteries",     sum(_w(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
        _add_op("Turbines CH4",  _w("ch4_ocgt") + _w("ch4_ccgt"),             "#f20809")
        _add_op("Turbines H2",   _w("h2_ccgt"),                               "#f252c0")
        if _use_str_dummy:
            _add_op("Stockage manquant", _w("str_dummy"),                      "#757575")
        if net_import:
            if _net_pos.max() > 1e-6: _add_op("Imports nets", _net_pos,     "#00b4d8")
        else:
            if _imp.sum() > 1e-6:    _add_op("Imports",        _imp,          "#00b4d8")

    # ---- negative stacks (storage charging + exports) ----
    elec_str = {}
    colors_str = []

    def _add_str(label, val, color):
        elec_str[label] = val
        colors_str.append(color)

    if lang == "EN":
        _add_str("PHS",       -_w("phs_input"),                                                                     "#0e4269")
        _add_str("Batteries", -sum(_w(t + "_input") for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
        if "methanation_input"  in hourly_balance.data_vars: _add_str("Methanation", -_w("methanation_input"),  "#f252c0")
        if "electrolysis_input" in hourly_balance.data_vars: _add_str("Electrolysis",-_w("electrolysis_input"), "#9370DB")
        if "str_dummy_input"    in hourly_balance.data_vars: _add_str("Missing storage", -_w("str_dummy_input"), "#757575")
        if net_import:
            if _net_neg.min() < -1e-6: _add_str("Net exports", _net_neg, "#e07a5f")
        else:
            if _exp.sum() > 1e-6: _add_str("Exports", -_exp, "#e07a5f")
    else:
        _add_str("STEP",      -_w("phs_input"),                                                                     "#0e4269")
        _add_str("Batteries", -sum(_w(t + "_input") for t in ["battery_1h","battery_2h","battery_4h","battery_8h"]), "#80549f")
        if "methanation_input"  in hourly_balance.data_vars: _add_str("Méthanation", -_w("methanation_input"),  "#f252c0")
        if "electrolysis_input" in hourly_balance.data_vars: _add_str("Electrolyse", -_w("electrolysis_input"), "#9370DB")
        if "str_dummy_input"    in hourly_balance.data_vars: _add_str("Stockage manquant", -_w("str_dummy_input"), "#757575")
        if net_import:
            if _net_neg.min() < -1e-6: _add_str("Exports nets", _net_neg, "#e07a5f")
        else:
            if _exp.sum() > 1e-6: _add_str("Exports", -_exp, "#e07a5f")

    fig, ax = plt.subplots(figsize=(16, 10))
    handles_prod = ax.stackplot(time, *operable_prod.values(), labels=operable_prod.keys(), colors=colors_prod)
    handles_str  = ax.stackplot(time, *elec_str.values(),      labels=elec_str.keys(),      colors=colors_str)

    if lang == "EN":
        handle_demand = ax.plot(time, residual_elec_demand, lw=2.5, color="black", zorder=5,
                                label="Residual electricity demand (w/ load shift)")
        ax.set_ylabel('Electricity [GW]', fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Residual Electricity Balance Over One Week{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Dispatchable production + imports (above 0) | Storage charging + exports (below 0)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_str  = ax.legend(handles=handles_str + handle_demand, loc='upper center', ncol=len(elec_str) + 1,
                             title="Usage/Exports", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.24, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=len(operable_prod),
                             title="Operable production + Imports", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.28, +1.14), frameon=False, columnspacing=0.5)
    else:
        handle_demand = ax.plot(time, residual_elec_demand, lw=2.5, color="black", zorder=5,
                                label="Demande résiduelle (avec pilotage)")
        ax.set_ylabel("Electricité [GW]", fontsize=12, labelpad=10)
        ax.text(x=0.05, y=1.02, s=f"Equilibre résiduel du réseau électrique sur une semaine{area_label}", transform=fig.transFigure, ha='left', fontsize=14, weight='bold')
        ax.text(x=0.05, y=1.0,  s="Production pilotable + imports (> 0) | Stockage en entrée + exports (< 0)", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)
        leg_str  = ax.legend(handles=handles_str + handle_demand, loc='upper center', ncol=len(elec_str) + 1,
                             title="Usage/Exports", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.272, +1.08), frameon=False, columnspacing=0.5)
        leg_prod = ax.legend(handles=handles_prod, loc='upper center', ncol=len(operable_prod),
                             title="Production pilotable + Imports", title_fontsize="large", alignment="left",
                             bbox_to_anchor=(0.28, +1.14), frameon=False, columnspacing=0.5)

    ax.add_artist(leg_str)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig


def plot_residual_load_duration(hourly_balance, area=None, lang="FR",
                                show_peak=True, figsize=(14, 6),
                                save_path=None, show=True):
    """
    Courbe de charge (monotone) de la demande résiduelle.

    Trie la demande résiduelle de la plus haute à la plus basse valeur sur toute l'année
    simulée. Permet d'identifier :
      - le pic de besoin en capacité pilotable (sommet de la courbe)
      - les heures avec excès ENR (demande résiduelle < 0, écrêtement potentiel)

    Parameters
    ----------
    hourly_balance : xr.Dataset
    area           : str or None
    lang           : "FR" or "EN"
    show_peak      : bool — annoter le pic et la zone excès ENR
    figsize        : tuple

    Returns
    -------
    fig   : matplotlib.figure.Figure
    stats : dict — peak_gw, hours_negative, pct_negative, vre_coverage_pct, …

    Example
    -------
    fig, stats = plot_residual_load_duration(m.hourly_balance, area="FR", lang="FR")
    """
    residual, demand, vre = compute_residual_demand(hourly_balance, area=area)
    n = len(residual)
    sorted_res = np.sort(residual)[::-1]
    hours = np.arange(n)

    peak_gw  = float(sorted_res[0])
    min_gw   = float(sorted_res[-1])
    h_neg    = int((residual < 0).sum())
    h_pos    = n - h_neg
    area_lbl = f" — {area}" if area else ""

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hours, sorted_res, color="#e8834c", lw=1.8, zorder=3,
            label="Demande résiduelle triée" if lang == "FR" else "Sorted residual demand")
    ax.axhline(0, color="#333", lw=0.9, ls="--", zorder=2)
    ax.fill_between(hours, sorted_res, 0,
                    where=sorted_res >= 0, alpha=0.15, color="#e8834c", zorder=1,
                    label="Besoin pilotable" if lang == "FR" else "Dispatchable need")
    ax.fill_between(hours, sorted_res, 0,
                    where=sorted_res < 0, alpha=0.20, color="#4c9be8", zorder=1,
                    label=("Excès ENR (écrêtement potentiel)"
                           if lang == "FR" else "VRE surplus (potential curtailment)"))

    if show_peak and peak_gw > 0:
        ax.annotate(
            f"Pic : {peak_gw:.0f} GW" if lang == "FR" else f"Peak: {peak_gw:.0f} GW",
            xy=(0, peak_gw), xytext=(n * 0.06, peak_gw * 0.97),
            fontsize=10, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )
    if show_peak and h_neg > 0:
        ax.axvline(h_pos, color="#4c9be8", lw=1.2, ls=":", alpha=0.9)
        ax.text(h_pos + n * 0.01, min_gw * 0.5,
                f"{h_neg} h\nexcès ENR" if lang == "FR" else f"{h_neg} h\nVRE surplus",
                fontsize=9, color="#2980b9", va="center")

    title_info = (
        f"Pic : {peak_gw:.0f} GW   |   Heures excès ENR : {h_neg}/{n} ({h_neg / n * 100:.1f}%)"
        if lang == "FR" else
        f"Peak: {peak_gw:.0f} GW   |   VRE surplus hours: {h_neg}/{n} ({h_neg / n * 100:.1f}%)"
    )
    ax.set_xlabel("Heures (ordre décroissant)" if lang == "FR" else "Hours (decreasing order)", fontsize=11)
    ax.set_ylabel("Demande résiduelle [GW]" if lang == "FR" else "Residual demand [GW]", fontsize=11)
    ax.set_title(
        f"{'Courbe de charge — Demande résiduelle' if lang == 'FR' else 'Residual load duration curve'}"
        f"{area_lbl}\n{title_info}",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    import matplotlib.ticker as _mticker
    ax.yaxis.set_major_locator(_mticker.MultipleLocator(20))
    ax.xaxis.set_major_locator(_mticker.MultipleLocator(500))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()

    stats = {
        "peak_gw":          peak_gw,
        "peak_hour":        int(np.argmax(residual)),
        "min_gw":           min_gw,
        "hours_negative":   h_neg,
        "pct_negative":     h_neg / n * 100,
        "mean_demand_gw":   float(demand.mean()),
        "mean_vre_gw":      float(vre.mean()),
        "mean_residual_gw": float(residual.mean()),
        "vre_coverage_pct": float(vre.sum() / demand.sum() * 100),
    }
    w = max(len(k) for k in stats) + 2
    print(f"\nStatistiques — demande résiduelle{area_lbl}:")
    for k, v in stats.items():
        unit = "%" if ("pct" in k or "coverage" in k) else ("h" if k == "peak_hour" else "GW")
        fmt  = ".1f" if unit == "GW" else (".0f" if unit == "h" else ".2f")
        print(f"  {k:<{w}}: {v:{fmt}} {unit}")

    return fig, stats


def plot_storage_state_year(hourly_balance, hour, nb_years=1, area=None, select_tech=["ch4_reservoir", "h2_saltcavern", "lake", "phs", "battery", "str_dummy"], lang="EN"):
    """
    Affiche l'état de charge des technologies de stockage sur une année complète.

    Parameters:
    -----------
    hourly_balance : xr.Dataset
        Dataset contenant les données horaires. Si le dataset a une dimension
        'area', passer area= pour sélectionner un pays.
    hour : int
        Heure de démarrage
    nb_years : int
        Nombre d'années à afficher
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    select_tech : list
        Liste des technologies de stockage à inclure
    lang : str
        Langue ("EN" ou "FR")
    """
    # --- résolution de la dimension area ---
    if "area" in hourly_balance.dims:
        hourly_balance = hourly_balance.sel(area=area) if area is not None else hourly_balance.sum("area")

    area_label = f" – {area}" if area is not None else ""

    fig, ax = plt.subplots(figsize=(16, 10))

    elec_str = {}
    if lang == "EN":
        if "ch4_reservoir" in select_tech and "ch4_reservoir_state_charge" in hourly_balance.data_vars:
            elec_str["Methane"] = hourly_balance["ch4_reservoir_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "h2_saltcavern" in select_tech and "h2_saltcavern_state_charge" in hourly_balance.data_vars:
            elec_str["Hydrogen"] = hourly_balance["h2_saltcavern_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "lake" in select_tech and "lake_state_charge" in hourly_balance.data_vars:
            elec_str["Hydropower"] = hourly_balance["lake_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "phs" in select_tech and "phs_state_charge" in hourly_balance.data_vars:
            elec_str["PHS"] = hourly_balance["phs_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "battery" in select_tech:
            battery_state = np.zeros(8760*nb_years)
            if "battery_1h_state_charge" in hourly_balance.data_vars:
                battery_state += hourly_balance["battery_1h_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
            if "battery_4h_state_charge" in hourly_balance.data_vars:
                battery_state += hourly_balance["battery_4h_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
            if battery_state.sum() > 0:
                elec_str["Batteries"] = battery_state
        if "str_dummy" in select_tech and "str_dummy_state_charge" in hourly_balance.data_vars:
            elec_str["Missing storage"] = hourly_balance["str_dummy_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
            
    if lang == "FR":
        if "ch4_reservoir" in select_tech and "ch4_reservoir_state_charge" in hourly_balance.data_vars:
            elec_str["Méthane"] = hourly_balance["ch4_reservoir_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "h2_saltcavern" in select_tech and "h2_saltcavern_state_charge" in hourly_balance.data_vars:
            elec_str["Hydrogène"] = hourly_balance["h2_saltcavern_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "lake" in select_tech and "lake_state_charge" in hourly_balance.data_vars:
            elec_str["Hydraulique"] = hourly_balance["lake_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "phs" in select_tech and "phs_state_charge" in hourly_balance.data_vars:
            elec_str["STEP"] = hourly_balance["phs_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
        if "battery" in select_tech:
            battery_state = np.zeros(8760*nb_years)
            if "battery_1h_state_charge" in hourly_balance.data_vars:
                battery_state += hourly_balance["battery_1h_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
            if "battery_4h_state_charge" in hourly_balance.data_vars:
                battery_state += hourly_balance["battery_4h_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values
            if battery_state.sum() > 0:
                elec_str["Batteries"] = battery_state
        if "str_dummy" in select_tech and "str_dummy_state_charge" in hourly_balance.data_vars:
            elec_str["Stockage manquant"] = hourly_balance["str_dummy_state_charge"].isel(hour=slice(hour, hour+8760*nb_years)).values

    # Création de l'axe temps
    time = extract_time_index(hourly_balance, hour, 8760*nb_years)

    # Couleurs consistantes
    colors_str = []
    color_map = {
        "Methane": "#f20809", "Méthane": "#f20809",
        "Hydrogen": "#f252c0", "Hydrogène": "#f252c0",
        "Hydropower": "#2672b0", "Hydraulique": "#2672b0",
        "PHS": "#0e4269", "STEP": "#0e4269",
        "Batteries": "#80549f",
        "Missing storage": "#757575", "Stockage manquant": "#757575"
    }
    
    for tech in elec_str.keys():
        if tech in color_map:
            colors_str.append(color_map[tech])

    if elec_str:
        handles_str = ax.stackplot(time, *elec_str.values(), labels=elec_str.keys(), colors=colors_str)
    else:
        print("[warn] Aucune donnée de stockage trouvée")
        return fig

    if lang == "EN":
        ax.set_ylabel('State of charge [GWh]', fontsize=12, labelpad=10)
        ax.text(x=0.06, y=0.93, s=f"Storage State-of-Charge Over {nb_years} Year(s){area_label}", transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        leg_str = ax.legend(handles=handles_str, loc='upper center', ncol=len(elec_str) + 1,
                  bbox_to_anchor=(0.19, +1.06), frameon=False, columnspacing=0.5)
    if lang == "FR":
        ax.set_ylabel('Niveau de charge [GWh]', fontsize=12, labelpad=10)
        ax.text(x=0.06, y=0.93, s=f"Niveau de charge sur {nb_years} an(s){area_label}", transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        leg_str = ax.legend(handles=handles_str, loc='upper center', ncol=len(elec_str) + 1,
                  bbox_to_anchor=(0.22, +1.06), frameon=False, columnspacing=0.5)

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    if elec_str:
        total_sum = np.column_stack(list(elec_str.values())).sum(axis=1).min()
        ax.set_ylim([total_sum, None])
    
    ax.spines[['top','right','bottom']].set_visible(False)
    if nb_years < 4:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig




def plot_installed_power(model_instance, installed_power, name="Scenario", area=None, values=False, vector="elec", lang="EN", highlight_saturation=True):
    """
    Affiche les puissances installées par technologie.

    Supports single-scenario (backward compatible) and multi-scenario comparisons.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle pour accéder aux sets
    installed_power : xr.DataArray, list of xr.DataArray, or dict {str: xr.DataArray}
        Puissances installées indexées par technologie.
        - Single DataArray: backward compatible
        - list: each element is a scenario; use name=["A","B"] for labels
        - dict {"A": DA1, "B": DA2}: keys are scenario labels
    name : str or list of str
        Nom(s) du/des scénario(s). Ignored when installed_power is a dict.
    area : str or None
        Zone à sélectionner si installed_power a une dimension 'area'.
    values : bool
        Afficher les valeurs sur les barres (défaut: False)
    vector : str
        "elec", "CH4", or "H2"
    lang : str
        "EN" or "FR"

    Examples:
    ---------
    fig = plot_installed_power(model, model.installed_power, area="FR", values=True)
    fig = plot_installed_power(model, [s1.installed_power, s2.installed_power], name=["Base","High RE"], area="FR")
    fig = plot_installed_power(model, {"Base": s1.installed_power, "High RE": s2.installed_power}, area="FR")
    """
    # Normalize input to {scenario_name: DataArray}
    if isinstance(installed_power, dict):
        scenarios = installed_power
    elif isinstance(installed_power, list):
        names = name if isinstance(name, list) else [f"Scenario {i+1}" for i in range(len(installed_power))]
        scenarios = {n: da for n, da in zip(names, installed_power)}
    else:
        sc_label = name if isinstance(name, str) else "Scenario"
        scenarios = {sc_label: installed_power}

    area_label = f" – {area}" if area is not None else ""

    # Maximum capacity per tech for saturation detection
    def _max_cap_fn():
        mc = model_instance.maximum_capacity
        if isinstance(mc, xr.DataArray) and "area" in mc.dims:
            if area is not None and area in mc.coords["area"].values:
                mc = mc.sel(area=area)
            else:
                return lambda tech: np.inf
        def _mc(tech):
            if tech in mc.coords["tech"].values:
                v = float(mc.sel(tech=tech).values)
                return v if not np.isinf(v) else np.inf
            return np.inf
        return _mc
    _mc = _max_cap_fn() if highlight_saturation else (lambda tech: np.inf)

    def _build_dict(ip):
        if isinstance(ip, xr.DataArray) and "area" in ip.dims:
            ip = ip.sel(area=area) if area is not None else ip.sum("area")

        def _ip(tech):
            if tech in ip.coords['tech'].values:
                v = float(ip.sel(tech=tech).values)
                return v if v > 0 else 0.0
            return 0.0

        d = {}
        d_max = {}  # parallel dict: max capacity per label (for saturation)

        def _add(label, techs_list):
            v = sum(_ip(t) for t in techs_list)
            if v > 0:
                d[label] = v
                d_max[label] = sum(_mc(t) for t in techs_list)

        def _add1(label, tech):
            _add(label, [tech])

        if vector == "CH4":
            if lang == "EN":
                entries = [
                    ("Natural gas",      "natural_gas"),
                    ("Biogas import",    "biogas_import"),
                    ("Methanization",    "methanization"),
                    ("Pyrogasification", "pyrogazification"),
                    ("Methanation",      "methanation"),
                    ("CH4 OCGT",         "ch4_ocgt"),
                    ("CH4 CCGT",         "ch4_ccgt"),
                    ("CH4 reservoir",    "ch4_reservoir"),
                ]
            else:
                entries = [
                    ("Gaz naturel",       "natural_gas"),
                    ("Import biogaz",     "biogas_import"),
                    ("Méthanisation",     "methanization"),
                    ("Pyrogazéification", "pyrogazification"),
                    ("Méthanation",       "methanation"),
                    ("Turbine CH4 OCGT",  "ch4_ocgt"),
                    ("Turbine CH4 CCGT",  "ch4_ccgt"),
                    ("Réservoir CH4",     "ch4_reservoir"),
                ]
            for label, tech in entries:
                _add1(label, tech)

        elif vector == "H2":
            if lang == "EN":
                entries = [
                    ("Electrolysis",  "electrolysis"),
                    ("H2 CCGT",       "h2_ccgt"),
                    ("H2 saltcavern", "h2_saltcavern"),
                ]
            else:
                entries = [
                    ("Electrolyse", "electrolysis"),
                    ("Turbine H2",  "h2_ccgt"),
                    ("Caverne H2",  "h2_saltcavern"),
                ]
            for label, tech in entries:
                _add1(label, tech)

        else:  # elec
            if model_instance is not None:
                _solar_techs = list(model_instance.solar.values)
            else:
                _all_techs = ip.coords["tech"].values if isinstance(ip, xr.DataArray) else []
                _solar_techs = [t for t in _all_techs if "solar" in t.lower() or "pv" in t.lower()]
            if lang == "EN":
                _add("Cogeneration",      ["biomass_coge","geothermal_coge","waste","ocgt_coge"])
                _add("Hydropower - Other",["river","marine"])
                _add1("Hydropower - Dams","lake")
                _add1("Hydropower - PHS", "phs")
                _add1("Wind - Onshore",   "onshore")
                _add1("Nuclear power",    "nuclear")
                _add("Wind - Offshore",   ["offshore_ground","offshore_float"])
                _add("Solar",             _solar_techs)
                _add("Batteries",         ["battery_1h","battery_2h","battery_4h","battery_8h"])
                _add("CH4 turbines",      ["ch4_ocgt","ch4_ccgt"])
                _add1("H2 turbines",      "h2_ccgt")
            else:
                _add("Cogénération",       ["biomass_coge","geothermal_coge","waste","ocgt_coge"])
                _add("Hydraulique - Autres",["river","marine"])
                _add1("Hydraulique - Barrages","lake")
                _add1("Hydraulique - STEP","phs")
                _add1("Eolien - Terrestre","onshore")
                _add1("Nucléaire",         "nuclear")
                _add("Eolien - En mer",    ["offshore_ground","offshore_float"])
                _add("Photovoltaïque",     _solar_techs)
                _add("Batteries",          ["battery_1h","battery_2h","battery_4h","battery_8h"])
                _add("Turbines CH4",       ["ch4_ocgt","ch4_ccgt"])
                _add1("Turbines H2",       "h2_ccgt")
        return d, d_max

    _built = {sc_name: _build_dict(ip) for sc_name, ip in scenarios.items()}
    sc_data = {sc_name: v[0] for sc_name, v in _built.items()}
    sc_max  = {sc_name: v[1] for sc_name, v in _built.items()}
    # Saturated labels: installed >= 99% of max capacity (use first scenario)
    _first_max = next(iter(sc_max.values()))
    _first_d   = next(iter(sc_data.values()))
    saturated_labels = {
        l for l, v in _first_d.items()
        if _first_max.get(l, np.inf) < np.inf and v >= 0.99 * _first_max[l]
    } if highlight_saturation else set()
    n_sc = len(sc_data)

    # Union of all tech labels in deterministic order, sorted ascending by max value
    ordered_labels = []
    for d in sc_data.values():
        for k in d:
            if k not in ordered_labels:
                ordered_labels.append(k)
    ordered_labels.sort(key=lambda l: max(d.get(l, 0) for d in sc_data.values()))

    unit_label = {"elec": "GW", "CH4": "GW", "H2": "GW"}[vector]
    title_en = f"Installed Power – {vector} vector{area_label}"
    title_fr = f"Puissance installée – vecteur {vector}{area_label}"

    sat_color = "#d32f2f"  # red for saturated bars

    if n_sc == 1:
        sc_name_single = list(sc_data.keys())[0]
        d = sc_data[sc_name_single]
        fig, ax = plt.subplots(figsize=(16, 10))
        vals = [d.get(l, 0) for l in ordered_labels]
        bar_colors = [sat_color if l in saturated_labels else "C0" for l in ordered_labels]
        _cap_label = "Installed capacity [GW]" if lang == "EN" else "Puissance installée [GW]"
        bars = ax.barh(ordered_labels, vals, height=0.6, label=_cap_label, color=bar_colors)
        if values:
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
        if saturated_labels and highlight_saturation:
            from matplotlib.patches import Patch
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(color=sat_color, label="At max capacity" if lang == "EN" else "À capacité max"))
            ax.legend(handles=handles, loc='lower right', frameon=True, fontsize=9)
        else:
            ax.legend(loc='lower right', frameon=True, fontsize=9)
    else:
        bar_h = 0.7 / n_sc
        colors_list = [mpl.cm.tab10(i % 10) for i in range(n_sc)]
        y = np.arange(len(ordered_labels))
        fig, ax = plt.subplots(figsize=(16, max(6, len(ordered_labels) * 0.5 * n_sc + 2)))
        for i, (sc_name_i, d) in enumerate(sc_data.items()):
            vals = [d.get(l, 0) for l in ordered_labels]
            offset = (i - (n_sc - 1) / 2) * bar_h
            bar_colors = [sat_color if l in saturated_labels else colors_list[i] for l in ordered_labels]
            bars = ax.barh(y + offset, vals, height=bar_h, color=bar_colors, label=sc_name_i)
            if values:
                ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(ordered_labels)
        if saturated_labels and highlight_saturation:
            from matplotlib.patches import Patch
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(color=sat_color, label="At max capacity" if lang == "EN" else "À capacité max"))
            ax.legend(handles=handles, loc='lower right', frameon=True, fontsize=9)
        else:
            ax.legend(loc='lower right', frameon=True, fontsize=9)

    if lang == "EN":
        ax.text(x=0.132, y=0.93, s=title_en, transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_xlabel(f"Installed power [{unit_label}]", fontsize=12)
    else:
        ax.text(x=0.132, y=0.93, s=title_fr, transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_xlabel(f"Puissance installée [{unit_label}]", fontsize=12)

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig, ax







def plot_gene_per_tech(model_instance, generation_per_tech, name="Scenario", area=None, values=False, vector="elec", lang="EN"):
    """
    Affiche la production énergétique par technologie.

    Supports single-scenario (backward compatible) and multi-scenario comparisons.

    Parameters:
    -----------
    model_instance : ModelEOLES
        Instance du modèle pour accéder aux sets
    generation_per_tech : xr.DataArray, list of xr.DataArray, or dict {str: xr.DataArray}
        Production par technologie en TWh.
        - Single DataArray: backward compatible (pass model.generation_per_technology)
        - list: each element is a scenario; use name=["A","B"] for labels
        - dict {"A": DA1, "B": DA2}: keys are scenario labels
    name : str or list of str
        Nom(s) du/des scénario(s). Ignored when generation_per_tech is a dict.
    area : str or None
        Identifiant de zone. If None and 'area' is a dimension, data is summed.
    values : bool
        Afficher les valeurs sur les barres (défaut: False)
    vector : str
        "elec", "CH4", or "H2"
    lang : str
        "EN" or "FR"

    Examples:
    ---------
    fig = plot_gene_per_tech(model, model.generation_per_technology, area="FR", values=True)
    fig = plot_gene_per_tech(model, [s1.generation_per_technology, s2.generation_per_technology], name=["Base","High RE"], area="FR")
    fig = plot_gene_per_tech(model, {"Base": s1.generation_per_technology, "High RE": s2.generation_per_technology}, area="FR")
    """
    # Normalize input to {scenario_name: DataArray}
    if isinstance(generation_per_tech, dict):
        scenarios = generation_per_tech
    elif isinstance(generation_per_tech, list):
        names = name if isinstance(name, list) else [f"Scenario {i+1}" for i in range(len(generation_per_tech))]
        scenarios = {n: da for n, da in zip(names, generation_per_tech)}
    else:
        sc_label = name if isinstance(name, str) else "Scenario"
        scenarios = {sc_label: generation_per_tech}

    area_label = f" – {area}" if area is not None else ""

    def _build_dict(gpt):
        if isinstance(gpt, xr.DataArray) and "area" in gpt.dims:
            gpt = gpt.sel(area=area) if area is not None else gpt.sum("area")

        # Convert to pandas for reliable tech lookup — avoids xarray index issues
        if isinstance(gpt, xr.DataArray) and "tech" in gpt.coords:
            _gen_pd = pd.Series(gpt.values, index=gpt.coords["tech"].values)
        else:
            _gen_pd = pd.Series(dtype=float)

        def _g(tech):
            if tech in _gen_pd.index:
                v = float(_gen_pd.at[tech])
                return v if v > 0 else 0.0
            return 0.0

        d = {}
        if vector == "CH4":
            if lang == "EN":
                entries = [
                    ("Natural gas",       "natural_gas"),
                    ("Biogas import",     "biogas_import"),
                    ("Methanization",     "methanization"),
                    ("Pyrogasification",  "pyrogazification"),
                    ("Methanation",       "methanation"),
                ]
                unit_local = "TWh CH4"
            else:
                entries = [
                    ("Gaz naturel",       "natural_gas"),
                    ("Import biogaz",     "biogas_import"),
                    ("Méthanisation",     "methanization"),
                    ("Pyrogazéification", "pyrogazification"),
                    ("Méthanation",       "methanation"),
                ]
                unit_local = "TWh CH4"
            for label, tech in entries:
                v = _g(tech)
                if v > 0:
                    d[label] = v

        elif vector == "H2":
            if lang == "EN":
                entries = [("Electrolysis", "electrolysis")]
                unit_local = "TWh H2"
            else:
                entries = [("Electrolyse",  "electrolysis")]
                unit_local = "TWh H2"
            for label, tech in entries:
                v = _g(tech)
                if v > 0:
                    d[label] = v

        else:  # elec
            unit_local = "TWh"
            if lang == "EN":
                cogen = sum(_g(t) for t in ["biomass_coge","geothermal_coge","waste","ocgt_coge"])
                if cogen > 0: d["Cogeneration"] = cogen
                hydro_other = sum(_g(t) for t in ["river","marine"])
                if hydro_other > 0: d["Hydropower - Other"] = hydro_other
                v = _g("onshore"); (d.update({"Wind - onshore": v}) if v > 0 else None)
                v = _g("nuclear"); (d.update({"Nuclear power": v})  if v > 0 else None)
                wind_off = sum(_g(t) for t in ["offshore_ground","offshore_float"])
                if wind_off > 0: d["Wind - offshore"] = wind_off
                v = _g("lake"); (d.update({"Hydropower - Dams": v}) if v > 0 else None)
                solar = sum(_g(t) for t in model_instance.solar.values)
                if solar > 0: d["Solar"] = solar
                v = _g("phs"); (d.update({"PHS": v}) if v > 0 else None)
                batt = sum(_g(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"])
                if batt > 0: d["Batteries"] = batt
                ch4 = sum(_g(t) for t in ["ch4_ocgt","ch4_ccgt"])
                if ch4 > 0: d["CH4 turbines"] = ch4
                v = _g("h2_ccgt");      (d.update({"H2 turbines": v})   if v > 0 else None)
                v = _g("biogas_import");(d.update({"Biogas import": v}) if v > 0 else None)
            else:
                cogen = sum(_g(t) for t in ["biomass_coge","geothermal_coge","waste","ocgt_coge"])
                if cogen > 0: d["Cogénération"] = cogen
                hydro_other = sum(_g(t) for t in ["river","marine"])
                if hydro_other > 0: d["Hydraulique - Autres"] = hydro_other
                v = _g("nuclear"); (d.update({"Nucléaire": v})           if v > 0 else None)
                v = _g("onshore"); (d.update({"Eolien - Terrestre": v})  if v > 0 else None)
                wind_off = sum(_g(t) for t in ["offshore_ground","offshore_float"])
                if wind_off > 0: d["Eolien - En mer"] = wind_off
                v = _g("lake"); (d.update({"Hydraulique - Barrages": v}) if v > 0 else None)
                solar = sum(_g(t) for t in model_instance.solar.values)
                if solar > 0: d["Photovoltaïque"] = solar
                v = _g("phs"); (d.update({"STEP": v}) if v > 0 else None)
                batt = sum(_g(t) for t in ["battery_1h","battery_2h","battery_4h","battery_8h"])
                if batt > 0: d["Batteries"] = batt
                ch4 = sum(_g(t) for t in ["ch4_ocgt","ch4_ccgt"])
                if ch4 > 0: d["Turbines CH4"] = ch4
                v = _g("h2_ccgt");      (d.update({"Turbines H2": v})   if v > 0 else None)
                v = _g("biogas_import");(d.update({"Import biogaz": v}) if v > 0 else None)
        return d, unit_local

    sc_results = {sc_name: _build_dict(gpt) for sc_name, gpt in scenarios.items()}
    sc_data = {sc_name: res[0] for sc_name, res in sc_results.items()}
    unit = list(sc_results.values())[0][1]
    n_sc = len(sc_data)

    # Union of all tech labels in deterministic order, sorted ascending by max value
    ordered_labels = []
    for d in sc_data.values():
        for k in d:
            if k not in ordered_labels:
                ordered_labels.append(k)
    ordered_labels.sort(key=lambda l: max(d.get(l, 0) for d in sc_data.values()))

    title_en = f"Energy Generation – {vector} vector{area_label}"
    title_fr = f"Production énergétique – vecteur {vector}{area_label}"

    if n_sc == 1:
        sc_name_single = list(sc_data.keys())[0]
        d = sc_data[sc_name_single]
        fig, ax = plt.subplots(figsize=(16, 10))
        vals = [d.get(l, 0) for l in ordered_labels]
        bars = ax.barh(ordered_labels, vals, height=0.6, label=sc_name_single)
        if values:
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
        ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0, +1.06), frameon=False)
    else:
        bar_h = 0.7 / n_sc
        colors_list = [mpl.cm.tab10(i % 10) for i in range(n_sc)]
        y = np.arange(len(ordered_labels))
        fig, ax = plt.subplots(figsize=(16, max(6, len(ordered_labels) * 0.5 * n_sc + 2)))
        for i, (sc_name_i, d) in enumerate(sc_data.items()):
            vals = [d.get(l, 0) for l in ordered_labels]
            offset = (i - (n_sc - 1) / 2) * bar_h
            bars = ax.barh(y + offset, vals, height=bar_h, color=colors_list[i], label=sc_name_i)
            if values:
                ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(ordered_labels)
        ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0, +1.06), frameon=False)

    if lang == "EN":
        ax.text(x=0.132, y=0.93, s=title_en, transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_xlabel(f"Total energy generated [{unit}]", fontsize=12)
    else:
        ax.text(x=0.132, y=0.93, s=title_fr, transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
        ax.set_xlabel(f"Énergie totale générée [{unit}]", fontsize=12)

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig




def plot_selected_techs_generation(generation_per_tech, selected_techs, title="Selected Technologies Generation", area=None, values=False, lang="EN"):
    """
    Affiche la production énergétique pour une liste de technologies sélectionnées.

    Parameters:
    -----------
    generation_per_tech : xr.DataArray
        Production par technologie en TWh. Peut avoir une dimension 'area' ;
        sera automatiquement réduite selon area=.
    selected_techs : list
        Liste des noms de technologies à afficher
    title : str
        Titre du graphique
    area : str or None
        Identifiant de zone (ex. "FR"). Si None et que 'area' est présent,
        les données sont sommées sur toutes les zones.
    values : bool
        Afficher les valeurs sur les barres (défaut: False)
    lang : str
        Langue ("EN" ou "FR")
    """
    if isinstance(generation_per_tech, xr.DataArray) and "area" in generation_per_tech.dims:
        generation_per_tech = generation_per_tech.sel(area=area) if area is not None else generation_per_tech.sum("area")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Filtrer les technologies sélectionnées qui existent dans les données
    available_techs = []
    gen_values = []

    for tech in selected_techs:
        if tech in generation_per_tech.coords['tech'].values:
            available_techs.append(tech)
            gen_values.append(generation_per_tech.sel(tech=tech).values)
        else:
            print(f"[warn] Technologie '{tech}' non trouvée ou génération nulle")

    if not available_techs:
        print("[error] Aucune technologie sélectionnée trouvée!")
        return fig
    
    # Créer une Series pandas pour les barplots
    gen_series = pd.Series(gen_values, index=available_techs)
    bars = ax.barh(gen_series.index, gen_series.values, height=0.6, color="#2672b0")

    # Afficher les valeurs sur les barres si demandé
    if values:
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)

    # Mise en forme du graphique
    ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0, +1.06), frameon=False)
    ax.text(x=0.05, y=0.93, s=title, transform=fig.transFigure, ha='left', fontsize=16, weight='bold')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.grid(True)
    
    if lang == "EN":
        ax.set_xlabel("Total energy generated [TWh]", fontsize=12)
    elif lang == "FR":
        ax.set_xlabel("Énergie totale générée [TWh]", fontsize=12)
    
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    return fig




def compare_operable_mix(names, results, lang="EN"):

    fig, ax = plt.subplots(figsize=(16, 10))
    bottom = np.zeros(len(names))

    if lang=="EN":
        ax.bar(names, [result.at["lake", "Installed power [GW]"] for result in results], width=0.5, label="Hydropower - Dams", bottom=bottom, color="#2672b0")
        bottom += [result.at["lake", "Installed power [GW]"] for result in results]
        if any(nuc>0 for nuc in [result.at["nuclear", "Installed power [GW]"] for result in results]):
            ax.bar(names, [result.at["nuclear", "Installed power [GW]"] for result in results], width=0.5, label="Nuclear", bottom=bottom, color="#e4a701")
            bottom += [result.at["nuclear", "Installed power [GW]"] for result in results]
        if any(coal>0 for coal in [result.at["coal", "Installed power [GW]"] for result in results]):
            ax.bar(names, [result.at["coal", "Installed power [GW]"] for result in results], width=0.5, label="Coal", bottom=bottom, color="#a68832")
            bottom += [result.at["coal", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["phs", "Installed power [GW]"] for result in results], width=0.5, label="PHS", bottom=bottom, color="#0e4269")
        bottom += [result.at["phs", "Installed power [GW]"] for result in results]
        ax.bar(names, [(result.at["battery_1h", "Installed power [GW]"] + result.at["battery_4h", "Installed power [GW]"]) for result in results], width=0.5, label="Batteries", bottom=bottom, color="#80549f")
        bottom += [(result.at["battery_1h", "Installed power [GW]"] + result.at["battery_4h", "Installed power [GW]"]) for result in results]
        ax.bar(names, [result.at["ch4_ocgt", "Installed power [GW]"] for result in results], width=0.5, label="Open-cycle gas turbines", bottom=bottom, color="#f20809")
        bottom += [result.at["ch4_ocgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["ch4_ccgt", "Installed power [GW]"] for result in results], width=0.5, label="Combined-cycle gas turbines", bottom=bottom, color="#a90506")
        bottom += [result.at["ch4_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["h2_ccgt", "Installed power [GW]"] for result in results], width=0.5, label="Hydrogen gas turbines", bottom=bottom, color="#f252c0")
        bottom += [result.at["h2_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["str_dummy", "Installed power [GW]"] for result in results], width=0.5, label="Missing storage", bottom=bottom, color="#757575")
        bottom += [result.at["str_dummy", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["rsv_dummy", "Installed power [GW]"] for result in results], width=0.5, label="Missing reserves", bottom=bottom, color="#3d3d3d")
        bottom += [result.at["rsv_dummy", "Installed power [GW]"] for result in results]
        ax.bar(names, bottom/5, bottom=bottom, alpha=0) # margin

        ax.set_ylabel("Installed power [GW]", fontsize=12)
        ax.set_title("Operable part of the energy mix for different scenarios")

    if lang=="FR":
        ax.bar(names, [result.at["lake", "Installed power [GW]"] for result in results], width=0.5, label="Hydraulique - Barrages", bottom=bottom, color="#2672b0")
        bottom += [result.at["lake", "Installed power [GW]"] for result in results]
        if any(nuc>0 for nuc in [result.at["nuclear", "Installed power [GW]"] for result in results]):
            ax.bar(names, [result.at["nuclear", "Installed power [GW]"] for result in results], width=0.5, label="Nucléaire", bottom=bottom, color="#e4a701")
            bottom += [result.at["nuclear", "Installed power [GW]"] for result in results]
        if any(coal>0 for coal in [result.at["coal", "Installed power [GW]"] for result in results]):
            ax.bar(names, [result.at["coal", "Installed power [GW]"] for result in results], width=0.5, label="Charbon", bottom=bottom, color="#a68832")
            bottom += [result.at["coal", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["phs", "Installed power [GW]"] for result in results], width=0.5, label="STEP", bottom=bottom, color="#0e4269")
        bottom += [result.at["phs", "Installed power [GW]"] for result in results]
        ax.bar(names, [(result.at["battery_1h", "Installed power [GW]"] + result.at["battery_4h", "Installed power [GW]"]) for result in results], width=0.5, label="Batteries", bottom=bottom, color="#80549f")
        bottom += [(result.at["battery_1h", "Installed power [GW]"] + result.at["battery_4h", "Installed power [GW]"]) for result in results]
        ax.bar(names, [result.at["ch4_ocgt", "Installed power [GW]"] for result in results], width=0.5, label="Turbines CH4 cycle ouvert", bottom=bottom, color="#f20809")
        bottom += [result.at["ch4_ocgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["ch4_ccgt", "Installed power [GW]"] for result in results], width=0.5, label="Turbines CH4 cycle combiné", bottom=bottom, color="#a90506")
        bottom += [result.at["ch4_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["h2_ccgt", "Installed power [GW]"] for result in results], width=0.5, label="Turbines H2", bottom=bottom, color="#f252c0")
        bottom += [result.at["h2_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["str_dummy", "Installed power [GW]"] for result in results], width=0.5, label="Stockage manquant", bottom=bottom, color="#757575")
        bottom += [result.at["str_dummy", "Installed power [GW]"] for result in results]
        ax.bar(names, [result.at["rsv_dummy", "Installed power [GW]"] for result in results], width=0.5, label="Réserves manquantes", bottom=bottom, color="#3d3d3d")
        bottom += [result.at["rsv_dummy", "Installed power [GW]"] for result in results]
        ax.bar(names, bottom/5, bottom=bottom, alpha=0) # margin

        ax.set_ylabel("Puissance installée [GW]", fontsize=12)
        ax.set_title("Partie pilotable du mix énergétique pour différents scénarios")



    ax.legend(loc='upper left', ncol=2, frameon=False)

    plt.show()

    return fig



def plot_spot_price(spot_price, period, area="FR", vector="elec", granularity="h", lang="EN"):
    """Plot the spot price time series for one or several areas on the same axes.

    Parameters
    ----------
    spot_price : dict of xr.DataArray
        Keys: "elec", "CH4", "H2".  Each DataArray has dims [hour] or [area, hour].
    period : str
        Pandas partial-string period selector, e.g. "2050" or "2050-01".
    area : str or list of str
        Single country code or list of codes, e.g. ``["FR", "IT", "DE"]``.
        When a list is given all curves are overlaid on the same axes.
    vector : str
        Energy vector: "elec", "CH4", or "H2".
    granularity : str
        Time resolution: "h" (no resampling), "d" (daily), "w" (weekly), "m" (monthly).
    lang : str
        "EN" or "FR" for axis labels.

    Returns
    -------
    fig, ax
    """
    _resample_map = {"d": "D", "w": "W", "m": "ME"}
    _label_map = {
        "EN": {"elec": "Electricity spot price [€/MWh]", "CH4": "CH4 spot price [€/MWh]", "H2": "H2 spot price [€/MWh]"},
        "FR": {"elec": "Prix spot électricité [€/MWh]", "CH4": "Prix spot CH4 [€/MWh]", "H2": "Prix spot H2 [€/MWh]"},
    }

    areas = [area] if isinstance(area, str) else list(area)
    da = spot_price[vector]
    colors = [mpl.cm.tab10(i % 10) for i in range(len(areas))]

    fig, ax = plt.subplots(figsize=(14, 4))

    for i, a in enumerate(areas):
        da_a = da.sel(area=a) if "area" in da.dims else da
        ts = da_a.to_series()[period]
        if granularity != "h":
            ts = ts.resample(_resample_map.get(granularity, granularity)).mean()
        color = colors[i] if len(areas) > 1 else "#2196F3"
        ax.plot(ts.index, ts.values, linewidth=0.8, color=color, label=a)

    ax.set_ylabel(_label_map.get(lang, _label_map["EN"]).get(vector, f"{vector} spot price [€/MWh]"))
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    area_label = ", ".join(areas)
    title = f"{area_label} — {vector} spot price ({period})"
    ax.set_title(title)
    if len(areas) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_cost_breakdown(capacity, existing_capacity, energy_capacity, existing_energy_capacity,
                        generation_per_technology, annuities, storage_annuities, fOM, vOM,
                        area, nb_years, lang="EN"):
    """Stacked horizontal bar chart of annualised cost breakdown per technology.

    Components (all in M€/yr):
      - CAPEX annuity : new_cap × annuity
      - fOM           : total_cap × fOM
      - vOM           : generation × 1000 × vOM
      - Storage CAPEX : new_energy_cap × storage_annuity

    Technologies with zero total cost are excluded.  Bars are sorted descending by total.

    Parameters
    ----------
    capacity : xr.DataArray
        Optimised capacity [GW], dims [tech] or [tech, area].
    existing_capacity : xr.DataArray
        Pre-existing capacity [GW].
    energy_capacity : xr.DataArray
        Optimised storage energy capacity [GWh].
    existing_energy_capacity : xr.DataArray
        Pre-existing storage energy capacity [GWh].
    generation_per_technology : xr.DataArray
        Total generation over simulation [TWh], dims [tech] or [tech, area].
    annuities : xr.DataArray
        Power CAPEX annuity [M€/GW/yr], indexed by tech.
    storage_annuities : xr.DataArray
        Energy CAPEX annuity [M€/GWh/yr], indexed by tech.
    fOM : xr.DataArray
        Fixed O&M [M€/GW/yr], indexed by tech.
    vOM : xr.DataArray
        Variable O&M [M€/GWh], indexed by tech.
    area : str
        Country/area code to plot.
    nb_years : int
        Number of simulated years (used to annualise).
    lang : str
        "EN" or "FR".

    Returns
    -------
    fig, ax
    """
    def _sel_area(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    cap_a    = _sel_area(capacity, area)
    exist_a  = _sel_area(existing_capacity, area)
    ecap_a   = _sel_area(energy_capacity, area)
    eexist_a = _sel_area(existing_energy_capacity, area)
    gen_a    = _sel_area(generation_per_technology, area)

    new_cap_a  = xr.where(cap_a  >= exist_a,  cap_a  - exist_a,  0.0)
    new_ecap_a = xr.where(ecap_a >= eexist_a, ecap_a - eexist_a, 0.0)

    techs = cap_a.tech.values.tolist()

    _ann     = annuities.reindex(tech=techs, fill_value=0.0)
    _fom     = fOM.reindex(tech=techs, fill_value=0.0)
    _vom     = vOM.reindex(tech=techs, fill_value=0.0)
    _str_ann = storage_annuities.reindex(tech=ecap_a.tech.values.tolist(), fill_value=0.0)

    capex_v = (new_cap_a  * _ann).values
    fom_v   = (cap_a      * _fom).values
    if gen_a.size > 0 and "tech" in gen_a.coords:
        vom_raw = pd.Series(gen_a.values, index=gen_a.coords["tech"].values)
    else:
        vom_raw = pd.Series(dtype=float)
    vom_v   = np.array([vom_raw.get(t, 0.0) * 1000 * float(_vom.sel(tech=t).values) for t in techs])
    str_techs = ecap_a.tech.values.tolist()
    str_v_dict = {t: float(new_ecap_a.sel(tech=t).values) * float(_str_ann.sel(tech=t).values) for t in str_techs}
    str_v   = np.array([str_v_dict.get(t, 0.0) for t in techs])

    data = pd.DataFrame({
        "capex": capex_v,
        "fom":   fom_v,
        "vom":   vom_v,
        "str":   str_v,
    }, index=techs)

    data["total"] = data.sum(axis=1)
    data = data[data["total"] > 0.1].sort_values("total", ascending=True)

    _labels_en = {"capex": "CAPEX annuity", "fom": "Fixed O&M", "vom": "Variable O&M", "str": "Storage CAPEX"}
    _labels_fr = {"capex": "Annuité CAPEX", "fom": "Charges fixes", "vom": "Charges variables", "str": "CAPEX stockage"}
    labels = _labels_fr if lang == "FR" else _labels_en
    colors = {"capex": "#1f77b4", "fom": "#ff7f0e", "vom": "#2ca02c", "str": "#9467bd"}

    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.45)))
    y = np.arange(len(data))
    left = np.zeros(len(data))
    for comp in ["capex", "fom", "vom", "str"]:
        vals = data[comp].values
        ax.barh(y, vals, left=left, color=colors[comp], label=labels[comp], height=0.6)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    xlabel = "Coût annualisé [M€/an]" if lang == "FR" else "Annualised cost [M€/yr]"
    ax.set_xlabel(xlabel)
    title = f"{area} — {'Décomposition des coûts' if lang == 'FR' else 'Cost breakdown'}"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_cost_breakdown_per_mwh(capacity, existing_capacity, energy_capacity, existing_energy_capacity,
                                generation_per_technology, annuities, storage_annuities, fOM, vOM,
                                area, nb_years, hourly_balance, spot_price,
                                elec_balance, CH4_balance, H2_balance,
                                lang="EN", exclude_dummies=True,
                                vector="all", tech_sets=None):
    """Stacked horizontal bar chart of cost breakdown per MWh produced, per technology.

    Wraps :func:`compute_lrmc_per_tech` + :func:`plot_lrmc_per_tech` with the same
    signature as :func:`plot_cost_breakdown`.

    Parameters
    ----------
    hourly_balance : xr.Dataset
        Output of extract_hourly_balance, used to value storage charging cost.
    spot_price : dict of xr.DataArray
        Keys "elec", "CH4", "H2", used to value storage charging cost.
    elec_balance, CH4_balance, H2_balance : xr.DataArray of tech names
        Used to route each storage tech's charging cost to the correct vector.
    vector : str
        "all" (default) to show all technologies.  Pass "elec", "CH4", or "H2" to
        restrict the chart to one energy vector.  Requires ``tech_sets``.
    tech_sets : dict, optional
        Mapping from vector name to iterable of tech strings, e.g.
        ``{"elec": model.elec_balance.values, "CH4": model.CH4_balance.values}``.

    Returns
    -------
    fig, ax
    """
    lrmc_df = compute_lrmc_per_tech(
        capacity, existing_capacity, energy_capacity, existing_energy_capacity,
        generation_per_technology, annuities, storage_annuities, fOM, vOM,
        area, nb_years, hourly_balance, spot_price,
        elec_balance, CH4_balance, H2_balance, exclude_dummies=exclude_dummies,
    )
    return plot_lrmc_per_tech(lrmc_df, area, lang=lang, vector=vector, tech_sets=tech_sets)


def plot_lrmc_per_tech(lrmc_df, area, lang="EN", vector="all", tech_sets=None):
    """Stacked horizontal bar chart of LRMC components per technology [€/MWh].

    Technologies with zero generation (NaN LRMC) are displayed as a separate
    group at the bottom with only their fixed cost M€/yr annotated.

    Parameters
    ----------
    lrmc_df : pd.DataFrame
        Output of :func:`compute_lrmc_per_tech`.
    area : str
        Used only for the chart title.
    lang : str
        "EN" or "FR".
    vector : str
        "all" (default) to show all technologies.  Pass "elec", "CH4", or "H2" to
        restrict to one energy vector.  Requires ``tech_sets``.
    tech_sets : dict, optional
        Mapping from vector name to iterable of tech strings, e.g.
        ``{"elec": model.elec_balance.values, "CH4": model.CH4_balance.values}``.

    Returns
    -------
    fig, ax
    """
    if vector != "all" and tech_sets is not None and vector in tech_sets:
        keep = set(tech_sets[vector])
        lrmc_df = lrmc_df[lrmc_df.index.isin(keep)]

    _labels_en = {"capex_€/MWh": "CAPEX annuity", "fOM_€/MWh": "Fixed O&M",
                  "storage_capex_€/MWh": "Storage CAPEX", "charging_cost_€/MWh": "Storage charging cost",
                  "vOM_€/MWh": "Variable O&M"}
    _labels_fr = {"capex_€/MWh": "Annuité CAPEX", "fOM_€/MWh": "Charges fixes",
                  "storage_capex_€/MWh": "CAPEX stockage", "charging_cost_€/MWh": "Coût de recharge",
                  "vOM_€/MWh": "Charges variables"}
    labels = _labels_fr if lang == "FR" else _labels_en
    colors = {"capex_€/MWh": "#1f77b4", "fOM_€/MWh": "#ff7f0e",
              "storage_capex_€/MWh": "#9467bd", "charging_cost_€/MWh": "#8c564b",
              "vOM_€/MWh": "#2ca02c"}
    comps = ["capex_€/MWh", "fOM_€/MWh", "storage_capex_€/MWh", "charging_cost_€/MWh", "vOM_€/MWh"]

    has_gen  = lrmc_df[lrmc_df["lrmc_€/MWh"].notna()].copy()
    no_gen   = lrmc_df[lrmc_df["lrmc_€/MWh"].isna()].copy()

    has_gen = has_gen.sort_values("lrmc_€/MWh", ascending=True)

    n_rows = len(has_gen) + (1 if len(no_gen) > 0 else 0)
    fig, ax = plt.subplots(figsize=(11, max(4, n_rows * 0.45 + 1)))

    y = np.arange(len(has_gen))
    left = np.zeros(len(has_gen))
    for comp in comps:
        vals = has_gen[comp].fillna(0).values
        ax.barh(y, vals, left=left, color=colors[comp], label=labels[comp], height=0.6)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(has_gen.index)

    # Annotate lrmc total on each bar
    for j, (tech, row) in enumerate(has_gen.iterrows()):
        ax.text(row["lrmc_€/MWh"] + 1, j, f"{row['lrmc_€/MWh']:.0f}", va="center", fontsize=8)

    # Techs with 0 generation: show as text below
    if len(no_gen) > 0:
        caption_label = "Sans génération (coût fixe M€/an)" if lang == "FR" else "No generation (fixed cost M€/yr)"
        lines = [f"  {t}: {row['fixed_cost_M€/yr']:.1f} M€/yr  |  {row['cap_GW']:.2f} GW"
                 for t, row in no_gen.iterrows()]
        ax.text(0.01, -0.12 - 0.04 * len(lines),
                caption_label + "\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="top",
                color="gray", family="monospace")

    xlabel = "LRMC [€/MWh]"
    ax.set_xlabel(xlabel)
    _vec_suffix = f" ({vector})" if vector != "all" else ""
    title_txt = f"{area} — {'Coût marginal de long terme par technologie' if lang == 'FR' else 'LRMC per technology'}{_vec_suffix}"
    ax.set_title(title_txt)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_dual_vs_profit(duals, profits, area, lang="EN", exclude_dummies=True, atol=1e-3):
    """Side-by-side horizontal bar charts comparing, for each tech in a given area, the
    dual value of its max-capacity constraint against its total profit.

    Rationale: a technology whose capacity constraint is binding (non-zero dual) is being
    held back from further investment, which should show up as a positive profit (a
    scarcity rent) — the model would otherwise keep investing until profit ~ 0. This chart
    is meant to visually check that intuition. Techs with a binding constraint (|dual| >
    atol) have their tech label shown in bold in both panels for easy comparison.

    Parameters
    ----------
    duals : xr.DataArray or pd.DataFrame
        Capacity-constraint dual, e.g. output of :func:`compute_results.extract_capacity_duals`
        (dims ["tech"] or ["tech", "area"]), or a long-format DataFrame as produced by
        ``m.model.constraints.max_capacity_prod.dual.to_dataframe()`` (area/tech in the index,
        one value column).
    profits : xr.DataArray
        Output of extract_profit (e.g. ``m.profits``). Dims ["tech"] or ["tech", "area"].
    area : str
    lang : "EN" or "FR"
    exclude_dummies : bool
        If True, drop technologies whose name ends with "_dummy".
    atol : float
        Duals with |dual| below this threshold are considered "not binding" [M€/GW/yr].

    Returns
    -------
    fig, axes
    """
    def _to_series(da, a):
        if isinstance(da, xr.DataArray):
            d = da.sel(area=a) if "area" in da.dims else da
            return d.to_pandas()
        if isinstance(da, pd.DataFrame):
            if isinstance(da.index, pd.MultiIndex) and "area" in da.index.names:
                d = da.xs(a, level="area")
            elif a in da.index.get_level_values(0):
                d = da.loc[a]
            else:
                d = da
            return d.iloc[:, 0] if d.shape[1] == 1 else d.squeeze()
        return da  # already a Series

    dual_s = _to_series(duals, area).rename("dual")
    profit_s = _to_series(profits, area).rename("profit")

    df = pd.concat([dual_s, profit_s], axis=1)
    df["dual"] = df["dual"].fillna(0.0)
    df = df.dropna(subset=["profit"])

    if exclude_dummies:
        df = df[~df.index.str.endswith("_dummy")]
    # Drop technologies that are entirely absent (no capacity, no profit) from the chart
    df = df[(df["dual"].abs() > 1e-9) | (df["profit"].abs() > 1e-9)]
    df = df.sort_values("profit")

    n_rows = max(len(df), 1)
    # Not sharey=True: both axes get their own independent y-tick labels (same order/values on
    # each), since sharey silently no-ops set_yticklabels() on every axes but the first one.
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, n_rows * 0.35 + 1)))

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data" if lang == "EN" else "Aucune donnée",
                    ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig, axes

    y = np.arange(len(df))
    binding = df["dual"].abs() > atol

    axes[0].barh(y, df["dual"], color=["#d62728" if b else "#7f7f7f" for b in binding], height=0.6)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y)
    labels0 = axes[0].set_yticklabels(df.index)
    axes[0].set_xlabel("Dual max_capacity [M€/GW/yr]" if lang == "EN" else "Dual capacité max [M€/GW/an]")
    axes[0].set_title("Capacity constraint dual" if lang == "EN" else "Dual contrainte de capacité")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].barh(y, df["profit"], color=["#2ca02c" if v >= 0 else "#1f77b4" for v in df["profit"]], height=0.6)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y)
    labels1 = axes[1].set_yticklabels(df.index)
    axes[1].set_xlabel("Profit [M€/yr]" if lang == "EN" else "Profit [M€/an]")
    axes[1].set_title("Profit per tech" if lang == "EN" else "Profit par techno")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    # Use the Text objects returned directly by set_yticklabels (not a fresh get_yticklabels()
    # call) - with sharey=True, re-querying axes[1] for its tick labels can return a list that
    # doesn't match df's length, since the shared y-axis machinery doesn't always keep each
    # Axes' own label Artists in sync with what was just set.
    for i, b in enumerate(binding):
        if b:
            labels0[i].set_fontweight("bold")
            labels1[i].set_fontweight("bold")

    suptitle = f"{area} — Dual vs. profit"
    suptitle += " (bold = binding capacity constraint)" if lang == "EN" else " (gras = contrainte de capacité active)"
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def plot_price_setter(price_setter_df, area, period=None, granularity="annual", lang="EN"):
    """Visualise the marginal price-setting technology.

    Parameters
    ----------
    price_setter_df : pd.DataFrame  output of identify_price_setter
    area : str
    period : slice or None  e.g. slice("2050-01", "2050-03")
        If None, uses full dataset.
    granularity : {"annual", "monthly"}
        "annual" → horizontal bar chart of % hours over full period.
        "monthly" → stacked horizontal bar chart with one bar per month.
    lang : {"EN", "FR"}

    Returns
    -------
    fig, ax
    """
    _labels = _TECH_LABEL_EN if lang == "EN" else _TECH_LABEL_FR

    df = price_setter_df.copy()
    if period is not None:
        df = df.loc[period]

    # Map raw tech names to display labels (group e.g. battery_1h/2h/4h/8h)
    df["label"] = df["price_setter"].map(lambda t: _labels.get(t, t))

    if granularity == "annual":
        summary = (
            df.groupby("label")
            .agg(hours=("spot_elec", "count"), avg_spot=("spot_elec", "mean"))
            .assign(pct=lambda d: d["hours"] / len(df) * 100)
            .sort_values("pct")
        )

        fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.55 + 1.5)))
        cmap = mpl.cm.tab10
        colors = [cmap(i % 10) for i in range(len(summary))]
        bars = ax.barh(summary.index, summary["pct"], color=colors)
        ax.bar_label(bars, fmt="%.1f %%", padding=4, fontsize=9)
        ax.set_xlabel("% of hours" if lang == "EN" else "% des heures", fontsize=11)
        title = (f"{area} — Marginal price-setting technology"
                 if lang == "EN" else f"{area} — Technologie marginale par heure")
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        plt.show()
        return fig, ax

    elif granularity == "monthly":
        # Convert hour index to datetime for month extraction
        try:
            idx = pd.to_datetime(df.index, unit="h", origin="2050-01-01")
        except Exception:
            idx = df.index
        df = df.copy()
        df["month"] = pd.Series(idx, index=df.index).dt.month

        # Pivot: rows=month, cols=label, values=% hours
        pivot = (
            df.groupby(["month", "label"])
            .size()
            .unstack(fill_value=0)
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

        all_labels = list(pivot.columns)
        cmap = mpl.cm.tab10
        colors = {l: cmap(i % 10) for i, l in enumerate(all_labels)}

        month_names_en = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_names_fr = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        month_names = month_names_en if lang == "EN" else month_names_fr

        fig, ax = plt.subplots(figsize=(13, 5))
        bottom = np.zeros(len(pivot))
        for label in all_labels:
            vals = pivot[label].values
            ax.bar(pivot.index, vals, bottom=bottom, label=label,
                   color=colors[label], edgecolor="white", linewidth=0.4)
            bottom += vals

        ax.set_xticks(pivot.index)
        ax.set_xticklabels([month_names[m - 1] for m in pivot.index])
        ax.set_ylabel("% of hours" if lang == "EN" else "% des heures", fontsize=11)
        ax.set_ylim(0, 100)
        title = (f"{area} — Monthly marginal technology mix"
                 if lang == "EN" else f"{area} — Mix de techno marginale par mois")
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
        ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        plt.show()
        return fig, ax

    else:
        raise ValueError(f"granularity must be 'annual' or 'monthly', got '{granularity}'")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-year / cross-scenario comparison charts
# Fed by utils_batch.py's loading functions - see notebook_batch_comparison.ipynb.
# ══════════════════════════════════════════════════════════════════════════════

TECH_COLORS = {
    'offshore_float': '#1565C0', 'offshore_ground': '#1E88E5', 'onshore': '#64B5F6',
    'pv_ground': '#F9A825', 'pv_roof_com': '#FBC02D', 'pv_roof_indiv': '#FFD54F',
    'river': '#26C6DA', 'lake': '#0097A7', 'phs': '#006064',
    'nuclear': '#7B1FA2',
    'methanization': '#2E7D32', 'pyrogazification': '#388E3C', 'biogas_import': '#66BB6A',
    'natural_gas': '#B0BEC5', 'coal': '#37474F',
    'ch4_ccgt': '#E53935', 'ch4_ocgt': '#EF9A9A', 'h2_ccgt': '#FF6F00',
    'electrolysis': '#29B6F6', 'methanation': '#00897B',
    'battery_1h': '#FFF176', 'battery_2h': '#FFEE58',
    'battery_4h': '#F57F17', 'battery_8h': '#A04000',
    'ch4_reservoir': '#78909C', 'h2_saltcavern': '#90A4AE',
    'waste': '#6D4C41', 'marine': '#00BCD4',
    'biomass_coge': '#4CAF50', 'geothermal_coge': '#D32F2F', 'ocgt_coge': '#FFAB40',
    'H2_import': '#FF9800',
    'phs_input': '#006064',
    'battery_1h_input': '#FFF176', 'battery_2h_input': '#FFEE58',
    'battery_4h_input': '#F57F17', 'battery_8h_input': '#A04000',
    'electrolysis_input': '#29B6F6', 'methanation_input': '#00897B',
    'ch4_reservoir_input': '#78909C',
    'ch4_ccgt_input': '#E53935', 'ch4_ocgt_input': '#EF9A9A',
    'ocgt_coge_input': '#FFAB40',
    'h2_saltcavern_input': '#90A4AE', 'h2_ccgt_input': '#FF6F00',
    'demand': '#455A64', 'curtailment': '#CFD8DC',
    'Solaire': '#F9A825', 'Offshore': '#1565C0', 'Hydro': '#26C6DA',
}

TECH_LABELS_FR = {
    'offshore_float': 'Éolien offshore flottant', 'offshore_ground': 'Éolien offshore posé',
    'onshore': 'Éolien terrestre',
    'pv_ground': 'Solaire au sol', 'pv_roof_com': 'Solaire toiture com.', 'pv_roof_indiv': 'Solaire toiture indiv.',
    'river': "Hydro fil-de-l'eau", 'lake': 'Hydro lacs', 'phs': 'STEP',
    'nuclear': 'Nucléaire',
    'methanization': 'Méthanisation', 'pyrogazification': 'Pyrogazéification', 'biogas_import': 'Import biogaz',
    'natural_gas': 'Gaz naturel',
    'ch4_ccgt': 'CCGT CH₄', 'ch4_ocgt': 'OCGT CH₄', 'h2_ccgt': 'CCGT H₂',
    'electrolysis': 'Électrolyse', 'methanation': 'Méthanation',
    'battery_1h': 'Batterie 1h', 'battery_2h': 'Batterie 2h', 'battery_4h': 'Batterie 4h', 'battery_8h': 'Batterie 8h',
    'ch4_reservoir': 'Stockage CH₄', 'h2_saltcavern': 'Caverne H₂',
    'waste': 'Déchets', 'marine': 'Énergie marine',
    'biomass_coge': 'Biomasse cogén.', 'geothermal_coge': 'Géothermie cogén.', 'ocgt_coge': 'OCGT cogén.',
    'H2_import': 'Import H₂', 'coal': 'Charbon', 'lost_load': 'Délestage',
    'phs_input': 'STEP (charge)',
    'battery_1h_input': 'Batterie 1h (charge)', 'battery_2h_input': 'Batterie 2h (charge)',
    'battery_4h_input': 'Batterie 4h (charge)', 'battery_8h_input': 'Batterie 8h (charge)',
    'electrolysis_input': 'Électrolyse', 'methanation_input': 'Méthanation',
    'ch4_reservoir_input': 'Stockage CH₄ (charge)',
    'ch4_ccgt_input': 'CCGT CH₄', 'ch4_ocgt_input': 'OCGT CH₄', 'ocgt_coge_input': 'OCGT cogén.',
    'h2_saltcavern_input': 'Caverne H₂ (charge)', 'h2_ccgt_input': 'CCGT H₂',
    'demand': 'Demande finale', 'curtailment': 'Écrêtement',
    'Solaire': 'Solaire (total)', 'Offshore': 'Éolien offshore', 'Hydro': 'Hydro (fil + lacs)',
}

TECH_LABELS_EN = {
    'offshore_float': 'Offshore wind (floating)', 'offshore_ground': 'Offshore wind (bottom-fixed)',
    'onshore': 'Onshore wind',
    'pv_ground': 'Ground-mounted PV', 'pv_roof_com': 'Commercial rooftop PV', 'pv_roof_indiv': 'Residential rooftop PV',
    'river': 'Run-of-river hydro', 'lake': 'Hydro dams', 'phs': 'PHS',
    'nuclear': 'Nuclear',
    'methanization': 'Methanization', 'pyrogazification': 'Pyrogasification', 'biogas_import': 'Biogas import',
    'natural_gas': 'Natural gas',
    'ch4_ccgt': 'CCGT (CH4)', 'ch4_ocgt': 'OCGT (CH4)', 'h2_ccgt': 'H2 turbine',
    'electrolysis': 'Electrolysis', 'methanation': 'Methanation',
    'battery_1h': 'Battery 1h', 'battery_2h': 'Battery 2h', 'battery_4h': 'Battery 4h', 'battery_8h': 'Battery 8h',
    'ch4_reservoir': 'CH4 storage', 'h2_saltcavern': 'H2 salt cavern',
    'waste': 'Waste', 'marine': 'Marine energy',
    'biomass_coge': 'Biomass CHP', 'geothermal_coge': 'Geothermal CHP', 'ocgt_coge': 'OCGT CHP',
    'H2_import': 'H2 import', 'coal': 'Coal', 'lost_load': 'Unserved demand',
    'demand': 'Final demand', 'curtailment': 'Curtailment',
    'Solaire': 'Solar (total)', 'Offshore': 'Offshore wind', 'Hydro': 'Hydro (river + dams)',
}

# Placeholder / non-physical techs excluded by default from summary charts.
TECH_EXCLUDE_DEFAULT = {'rsv_dummy', 'str_dummy', 'lost_load', 'coal',
                        'H2_import', 'natural_gas', 'biomass_coge', 'geothermal_coge'}

# Vector assignment per tech, used by cost-by-vector and energy-flow charts.
TECH_VECTOR = {
    'offshore_float': 'elec', 'offshore_ground': 'elec', 'onshore': 'elec',
    'pv_ground': 'elec', 'pv_roof_com': 'elec', 'pv_roof_indiv': 'elec',
    'river': 'elec', 'lake': 'elec', 'nuclear': 'elec', 'waste': 'elec',
    'marine': 'elec', 'ch4_ccgt': 'elec', 'ch4_ocgt': 'elec', 'h2_ccgt': 'elec',
    'phs': 'elec', 'battery_1h': 'elec', 'battery_2h': 'elec',
    'battery_4h': 'elec', 'battery_8h': 'elec',
    'biomass_coge': 'elec', 'geothermal_coge': 'elec', 'ocgt_coge': 'elec',
    'methanization': 'CH4', 'pyrogazification': 'CH4', 'biogas_import': 'CH4',
    'natural_gas': 'CH4', 'ch4_reservoir': 'CH4', 'methanation': 'CH4',
    'electrolysis': 'H2', 'H2_import': 'H2', 'h2_saltcavern': 'H2',
}


def plot_min_mean_max(data_dict, title, unit, min_val=0.01, exclude=None,
                      tech_colors=None, tech_labels=None, figsize_w=13):
    """Horizontal bar chart of min/mean/max per technology, one group of 3 bars per
    scenario. Designed for a dict of DataFrames indexed by year (see
    utils_batch.load_across_years): {scenario_label: DataFrame(index=year, columns=tech)}.

    Numeric annotations ("mean [min-max]") are added to the right of each group.

    Parameters
    ----------
    data_dict : dict {label: pd.DataFrame(index=year, columns=tech)}
    title, unit : str
    min_val : float — techs whose max is below this are dropped.
    exclude : set of str, optional — tech names to always drop. Defaults to TECH_EXCLUDE_DEFAULT.
    tech_colors, tech_labels : dict, optional — default to TECH_COLORS / TECH_LABELS_FR.

    Returns
    -------
    fig or None (if no data)
    """
    exclude = TECH_EXCLUDE_DEFAULT if exclude is None else exclude
    tech_colors = TECH_COLORS if tech_colors is None else tech_colors
    tech_labels = TECH_LABELS_FR if tech_labels is None else tech_labels

    stats = {}
    for label, df in data_dict.items():
        if df.empty:
            continue
        df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        st = compute_stats_local(df_num)
        stats[label] = filter_techs_local(st, min_val=min_val, exclude=exclude)

    if not stats:
        print('No data to plot.')
        return None

    ref_st = list(stats.values())[0]
    tech_order = list(ref_st.sort_values('mean', ascending=True).index)
    for st in stats.values():
        for t in st.index:
            if t not in tech_order:
                tech_order.append(t)

    n_techs, n_scen = len(tech_order), len(stats)
    bar_h, n_bars = 0.20, 3
    group_h = n_bars * n_scen * bar_h + 0.30
    fig_h = max(5, n_techs * group_h * 0.9)

    fig, ax = plt.subplots(figsize=(figsize_w, fig_h))
    y_base = np.arange(n_techs) * group_h
    stat_keys, stat_alphas = ['max', 'mean', 'min'], [0.50, 1.00, 0.30]

    x_global_max = 0
    for si, (label, st) in enumerate(stats.items()):
        for bi, (key, alpha) in enumerate(zip(stat_keys, stat_alphas)):
            y_pos = y_base + (si * n_bars + bi) * bar_h
            for ti, t in enumerate(tech_order):
                v = max(0, st.loc[t, key]) if t in st.index else 0
                x_global_max = max(x_global_max, v)
                c = tech_colors.get(t, '#888')
                h = '///' if si == 1 else ''
                ax.barh(y_pos[ti], v, bar_h * 0.88, color=c, alpha=alpha, hatch=h,
                       edgecolor='white', linewidth=0.4)

    x_offset = x_global_max * 0.015 + 0.05
    for si, (label, st) in enumerate(stats.items()):
        y_pos_mean = y_base + (si * n_bars + 1) * bar_h
        for ti, t in enumerate(tech_order):
            if t not in st.index:
                continue
            v_max, v_mean, v_min = max(0, st.loc[t, 'max']), max(0, st.loc[t, 'mean']), max(0, st.loc[t, 'min'])
            if v_mean < min_val:
                continue
            fmt = '.1f' if v_max < 10 else '.0f'
            txt = f'{v_mean:{fmt}} [{v_min:{fmt}}–{v_max:{fmt}}]'
            ax.text(v_max + x_offset, y_pos_mean[ti], txt, va='center', ha='left', fontsize=7, color='#333333')

    ax.set_xlim(right=x_global_max * 1.35 + x_offset * 10)
    y_centers = y_base + (n_bars * n_scen * bar_h) / 2
    ax.set_yticks(y_centers)
    ax.set_yticklabels([tech_labels.get(t, t) for t in tech_order])
    ax.set_xlabel(unit)
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)

    leg = [
        mpatches.Patch(color='grey', alpha=1.0, label='Mean'),
        mpatches.Patch(facecolor='grey', alpha=0.50, edgecolor='grey', label='Max'),
        mpatches.Patch(color='grey', alpha=0.30, label='Min'),
    ]
    if n_scen > 1:
        labels_list = list(stats.keys())
        leg.append(mpatches.Patch(color='grey', alpha=0.9, hatch='', label=labels_list[0]))
        if len(labels_list) > 1:
            leg.append(mpatches.Patch(color='grey', alpha=0.9, hatch='///', label=labels_list[1]))
    ax.legend(handles=leg, loc='lower right', fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


# Small local helpers (avoid a hard dependency of utils_plots on utils_batch)
def compute_stats_local(df):
    return pd.DataFrame({'min': df.min(), 'mean': df.mean(), 'max': df.max()})


def filter_techs_local(stats_df, min_val=0.01, exclude=frozenset()):
    mask = (stats_df['max'] >= min_val) & (~stats_df.index.isin(exclude))
    return stats_df[mask].sort_values('mean', ascending=True)


def plot_category_timeseries(data_by_scenario_year, categories, category_colors=None,
                             unit='GW', title='', figsize=(14, 5)):
    """Line chart of aggregated categories across climate years, one subplot per scenario.

    Parameters
    ----------
    data_by_scenario_year : dict {label: DataFrame(index=year, columns=tech)}
        E.g. installed_power or generation loaded via utils_batch.load_across_years.
    categories : dict {category_name: [tech, ...]}
    category_colors : dict, optional
    unit : str — used for the y-axis label ('GW' for capacity, 'TWh/yr' for generation).
    title : str

    Returns
    -------
    fig
    """
    category_colors = category_colors or {}
    n_scen = len(data_by_scenario_year)
    fig, axes = plt.subplots(1, n_scen, figsize=figsize, sharey=True)
    if n_scen == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, data_by_scenario_year.items()):
        if df.empty:
            ax.set_title(f'{label}\n(no data)')
            continue
        df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        years_s = sorted(df_num.index)
        for cat, techs in categories.items():
            cols = [t for t in techs if t in df_num.columns]
            if not cols:
                continue
            ys = df_num.loc[years_s, cols].sum(axis=1).values
            ax.plot(years_s, ys, marker='o', markersize=4, linewidth=1.6,
                   label=cat, color=category_colors.get(cat, '#888'))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Climate year')
        ax.set_ylabel(unit)
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        ax.grid(alpha=0.25)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontweight='bold', fontsize=13)
    plt.tight_layout()
    return fig


def plot_energy_flows(supply_by_vector, usage_by_vector, title='', figsize=(18, 5.5)):
    """Offer <- | -> demand diagram, one panel per vector (elec / CH4 / H2).

    Parameters
    ----------
    supply_by_vector, usage_by_vector : dict {vector: list of (name, value_TWh, color)}
        Already-aggregated supply/usage components for one (scenario, year). Values <= 0
        are skipped automatically upstream by the caller if desired.
    title : str

    Returns
    -------
    fig
    """
    vectors = list(supply_by_vector.keys())
    fig, axes = plt.subplots(1, len(vectors), figsize=figsize)
    if len(vectors) == 1:
        axes = [axes]
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.98)

    for ax, vec in zip(axes, vectors):
        sup_agg = [(n, v, c) for n, v, c in supply_by_vector.get(vec, []) if v > 0.5]
        use_agg = [(n, v, c) for n, v, c in usage_by_vector.get(vec, []) if v > 0.5]

        if not sup_agg and not use_agg:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(vec, fontsize=13)
            continue

        x_max = max(sum(v for _, v, _ in sup_agg) if sup_agg else 0,
                    sum(v for _, v, _ in use_agg) if use_agg else 0) * 1.12
        x_max = x_max or 1

        bot = 0
        sup_patches = []
        for name, v, color in sup_agg:
            ax.barh(0, -v, left=-bot, height=0.5, color=color, edgecolor='white', linewidth=0.6)
            if v > x_max * 0.06:
                ax.text(-bot - v / 2, 0, f'{v:.0f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            sup_patches.append(mpatches.Patch(color=color, label=name))
            bot += v

        bot = 0
        use_patches = []
        for name, v, color in use_agg:
            ax.barh(0, v, left=bot, height=0.5, color=color, edgecolor='white', linewidth=0.6, alpha=0.88)
            if v > x_max * 0.06:
                ax.text(bot + v / 2, 0, f'{v:.0f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            use_patches.append(mpatches.Patch(color=color, alpha=0.88, label=name))
            bot += v

        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-0.6, 0.6)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_yticks([])
        ax.set_xlabel('TWh — <- Supply | Demand ->')
        ax.set_title(vec, fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.0f}'))

        leg_sup = ax.legend(handles=sup_patches, title='<- Supply', loc='upper left',
                            fontsize=7.5, title_fontsize=7.5, framealpha=0.85, ncol=1)
        ax.add_artist(leg_sup)
        ax.legend(handles=use_patches, title='Demand ->', loc='upper right',
                 fontsize=7.5, title_fontsize=7.5, framealpha=0.85, ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_lost_load(lost_load_pivot, focus_country=None, cap_pct=0.002, lang='FR'):
    """Two-panel lost-load chart: (1) mean/max per country against the model's cap,
    (2) year-by-year evolution for `focus_country` (defaults to the first column).

    Parameters
    ----------
    lost_load_pivot : pd.DataFrame  — output of utils_batch.load_lost_load (raw fraction,
        NOT %; this function multiplies by 100 internally for display).
    focus_country : str, optional
    cap_pct : float — model's lost-load cap, in % (default 0.002%).
    lang : "FR" or "EN"

    Returns
    -------
    fig
    """
    if lost_load_pivot.empty:
        print('No lost-load data to plot.')
        return None
    pct = lost_load_pivot * 100
    focus_country = focus_country or pct.columns[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    means, maxs = pct.mean(), pct.max()
    x = np.arange(len(pct.columns))
    ax.bar(x - 0.2, means.values, width=0.35, label='Mean' if lang == 'EN' else 'Moyenne', color='#4c9be8', alpha=0.85)
    ax.bar(x + 0.2, maxs.values, width=0.35, label='Max', color='#e8834c', alpha=0.85)
    ax.axhline(cap_pct, color='red', ls='--', lw=1.5, label=f'Cap ({cap_pct}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(pct.columns)
    ax.set_ylabel('Lost load [% of demand]' if lang == 'EN' else 'Délestage [% de la demande]')
    ax.set_title('Mean and max lost load per country' if lang == 'EN' else 'Délestage moyen et max par pays')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    if focus_country in pct.columns:
        s = pct[focus_country]
        ax.bar(s.index, s.values, color='#4c9be8', alpha=0.85)
        ax.axhline(cap_pct, color='red', ls='--', lw=1.5, label=f'Cap ({cap_pct}%)')
        ax.set_xlabel('Climate year' if lang == 'EN' else 'Année climatique')
        ax.set_ylabel(f'Lost load {focus_country} [%]' if lang == 'EN' else f'Délestage {focus_country} [%]')
        ax.set_title(f'{focus_country} lost load per year' if lang == 'EN' else f'Délestage {focus_country} par année')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def plot_curtailment_stats(curt_pct, lang='FR'):
    """Bar chart of curtailment min/mean/max [%] per country, across years."""
    if curt_pct.empty:
        print('No curtailment data to plot.')
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    countries = list(curt_pct.columns)
    x = np.arange(len(countries))
    means, mins, maxs = curt_pct.mean().values, curt_pct.min().values, curt_pct.max().values
    ax.bar(x, means, color='#4c9be8', alpha=0.85, label='Mean' if lang == 'EN' else 'Moyenne', zorder=2)
    ax.errorbar(x, means, yerr=[means - mins, maxs - means], fmt='none', color='#222', capsize=5, lw=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel('Curtailment [% of primary generation]' if lang == 'EN' else 'Écrêtement [% de la production primaire]')
    ax.set_title('Curtailment min/mean/max per country' if lang == 'EN' else 'Écrêtement min/moy/max par pays')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    plt.tight_layout()
    return fig


def plot_curtailment_heatmap(curt_pct, lang='FR'):
    """Heatmap of curtailment [%]: years (rows) x countries (columns)."""
    if curt_pct.empty:
        print('No curtailment data to plot.')
        return None
    countries = list(curt_pct.columns)
    data_hm = curt_pct.fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(4, len(data_hm) * 0.4)))
    im = ax.imshow(data_hm.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries)
    ax.set_yticks(range(len(data_hm.index)))
    ax.set_yticklabels(data_hm.index)
    plt.colorbar(im, ax=ax, label='Curtailment [%]')
    vmax = data_hm.values.max() if data_hm.size else 1
    for i in range(len(data_hm.index)):
        for j in range(len(countries)):
            val = data_hm.values[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8,
                   color='black' if val < vmax * 0.7 else 'white')
    ax.set_title('Curtailment [%] per country and year' if lang == 'EN' else 'Écrêtement [%] par pays et par année', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_trade_balance(trade_df, countries, lang='FR'):
    """Grouped bar chart of mean net trade [TWh/yr] per country, one bar per vector
    (elec, CH4+biogas, H2). `trade_df` is the output of utils_batch.load_trade_balance."""
    if trade_df.empty:
        print('No trade data to plot.')
        return None
    trade_mean = trade_df.groupby('country')[
        ['elec_net_TWh', 'CH4_net_TWh', 'biogas_import_TWh', 'CH4+biogas_net_TWh', 'H2_net_TWh']
    ].mean()
    trade_mean = trade_mean.loc[[c for c in countries if c in trade_mean.index]]

    vectors = [
        ('elec_net_TWh', 'Electricity' if lang == 'EN' else 'Électricité', '#4c9be8'),
        ('CH4+biogas_net_TWh', 'CH4 + biogas import' if lang == 'EN' else 'CH4 + import biogaz', '#e8834c'),
        ('H2_net_TWh', 'Hydrogen' if lang == 'EN' else 'Hydrogène', '#6cc644'),
    ]
    countries_avail = list(trade_mean.index)
    x = np.arange(len(countries_avail))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, label, color) in enumerate(vectors):
        ax.bar(x + (i - 1) * width, trade_mean.loc[countries_avail, col].values, width=width,
              label=label, color=color, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(countries_avail)
    ax.set_ylabel('Mean net trade [TWh/yr]\n(positive = net importer)' if lang == 'EN'
                 else 'Solde net moyen [TWh/an]\n(positif = importateur net)')
    ax.set_title('Mean energy balance per country' if lang == 'EN' else 'Bilan énergétique moyen par pays')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_trade_balance_detail(trade_df, country, lang='FR'):
    """Per-vector, per-year net trade for one country (3 side-by-side subplots)."""
    sub = trade_df[trade_df['country'] == country].set_index('year').sort_index()
    if sub.empty:
        print(f'No trade data for {country}.')
        return None
    vectors = [
        ('elec_net_TWh', 'Electricity' if lang == 'EN' else 'Électricité', '#4c9be8'),
        ('CH4+biogas_net_TWh', 'CH4 + biogas' if lang == 'EN' else 'CH4 + biogaz', '#e8834c'),
        ('H2_net_TWh', 'Hydrogen' if lang == 'EN' else 'Hydrogène', '#6cc644'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for ax, (col, label, color) in zip(axes, vectors):
        vals = sub[col].dropna()
        ax.bar(vals.index, vals.values, color=color, alpha=0.85)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{country} — {label}')
        ax.set_ylabel('TWh/yr')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle(f'Annual net trade — {country}', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_interconnection_utilization(sol, links_df, area='FR', lang='FR'):
    """Two figures describing how saturated an area's cross-border links are over the year:
    (1) boxplot of hourly utilisation % per link/direction, (2) stacked bar of the share of
    hours spent idle / partially used / saturated.

    Parameters
    ----------
    sol : the solved linopy model's solution (m.model.solution), must contain the
        'export' variable (absent for a single-country, no-interconnection run).
    links_df : pd.DataFrame — inputs/area_indexed/links.csv, index=area, columns=area.
    area : str — the area whose links are analysed (e.g. 'FR').
    lang : "FR" or "EN"

    Returns
    -------
    (fig1, fig2, summary_df) or (None, None, None) if no interconnection data available.
    """
    if 'export' not in sol:
        print(f"[!] 'export' variable absent (no-interconnection run?)")
        return None, None, None

    records = []
    for partner in links_df.columns:
        if partner == area:
            continue
        for area_from, area_to, direction in [(area, partner, 'exp'), (partner, area, 'imp')]:
            try:
                cap = float(links_df.loc[area_from, area_to])
            except (KeyError, ValueError):
                continue
            if not (pd.notna(cap) and cap > 0):
                continue
            try:
                flow = sol['export'].sel(area=area_from, area_bis=area_to).values.flatten()
            except Exception:
                continue
            util = flow / cap * 100
            n_h = len(flow)
            records.append({
                'label': f'{area_from}→{area_to}', 'partner': partner, 'dir': direction,
                'cap_gw': cap, 'util': util, 'flow': flow, 'mean_util': float(util.mean()),
                'pct_full': float((util >= 99).mean() * 100), 'pct_idle': float((util <= 1).mean() * 100),
                'pct_partial': float(((util > 1) & (util < 99)).mean() * 100),
                'energy_twh': float(flow.mean() * n_h / 1000),
            })

    if not records:
        print('[!] No interconnection data available.')
        return None, None, None

    partners_ordered = sorted({r['partner'] for r in records})
    color_exp, color_imp = '#e8834c', '#4c9be8'

    # Figure 1: boxplot with annotated mean
    fig1, ax1 = plt.subplots(figsize=(max(10, len(partners_ordered) * 2.5), 5))
    xtick_pos, xtick_labels = [], []
    for i, partner in enumerate(partners_ordered):
        base = i * 3
        recs_p = sorted([r for r in records if r['partner'] == partner], key=lambda x: x['dir'])
        for r in recs_p:
            pos = base + (0 if r['dir'] == 'exp' else 1)
            color = color_exp if r['dir'] == 'exp' else color_imp
            ax1.boxplot(r['util'], positions=[pos], widths=0.8, patch_artist=True, showfliers=False,
                       showmeans=True, boxprops=dict(facecolor=color, alpha=0.65),
                       medianprops=dict(color='white', lw=0),
                       meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=7, zorder=5),
                       whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
            ax1.text(pos, r['mean_util'] + 3, f"{r['mean_util']:.0f}%", ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='black')
        xtick_pos.append(base + 0.5)
        xtick_labels.append(partner)

    ax1.axhline(100, color='red', lw=1.2, ls='--', alpha=0.6)
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_labels, fontsize=11)
    ax1.set_ylabel("Utilisation rate [%]" if lang == 'EN' else "Taux d'utilisation [%]", fontsize=11)
    ax1.set_ylim(-3, 122)
    ax1.set_title(
        f"{area} interconnection usage — hourly distribution" if lang == 'EN'
        else f"Fréquentation des interconnexions {area} — Distribution horaire",
        fontweight='bold')
    ax1.legend(handles=[
        mpatches.Patch(facecolor=color_exp, alpha=0.75, label=f'{area} → country (export)' if lang == 'EN' else f'{area} → pays (export)'),
        mpatches.Patch(facecolor=color_imp, alpha=0.75, label=f'Country → {area} (import)' if lang == 'EN' else f'Pays → {area} (import)'),
    ], fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    # Figure 2: stacked bar of idle / partial / saturated hours
    all_dirs = sorted(records, key=lambda x: (x['partner'], x['dir']))
    bar_labels = [r['label'] for r in all_dirs]
    bar_colors = [color_exp if r['dir'] == 'exp' else color_imp for r in all_dirs]
    pct_idle = np.array([r['pct_idle'] for r in all_dirs])
    pct_partial = np.array([r['pct_partial'] for r in all_dirs])
    pct_full = np.array([r['pct_full'] for r in all_dirs])
    x = np.arange(len(all_dirs))

    fig2, ax2 = plt.subplots(figsize=(max(10, len(all_dirs) * 1.5), 5))
    ax2.bar(x, pct_idle, color='#cccccc', label='Idle (<=1%)' if lang == 'EN' else "À l'arrêt (≤ 1 %)")
    ax2.bar(x, pct_partial, bottom=pct_idle, color=bar_colors, alpha=0.45,
           label='Partial use (1-99%)' if lang == 'EN' else 'Utilisation partielle (1–99 %)')
    ax2.bar(x, pct_full, bottom=pct_idle + pct_partial, color=bar_colors,
           label='Saturated (>=99%)' if lang == 'EN' else 'Saturation (≥ 99 %)')
    for xi, (yi, yp, yf) in enumerate(zip(pct_idle, pct_partial, pct_full)):
        if yf >= 0.5:
            ax2.text(xi, yi + yp + yf + 0.8, f'{yf:.0f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=30, ha='right', fontsize=10)
    ax2.set_ylabel('% of hours in the year' if lang == 'EN' else "% des heures de l'année", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title(f'{area} interconnection usage breakdown' if lang == 'EN' else f'Répartition horaire — Interconnexions {area}',
                 fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    summary = pd.DataFrame([{
        'interco': r['label'], 'cap_gw': r['cap_gw'], 'mean_util_%': round(r['mean_util'], 1),
        'mean_flow_gw': round(float(r['flow'].mean()), 2), '% h saturated': round(r['pct_full'], 1),
        '% h idle': round(r['pct_idle'], 1), 'energy_TWh': round(r['energy_twh'], 2),
    } for r in sorted(records, key=lambda x: x['label'])]).set_index('interco')

    return fig1, fig2, summary


def plot_interconnection_diverging(stats_df, area, lang='FR', figsize_w=9):
    """Single 0-100% stacked bar per partner country summarising interconnection usage:
    import (partner -> area, red shades) and export (area -> partner, blue shades) segments
    combined into ONE bar per partner (not two, and not split around a zero axis).

    Assumes the link only flows one way at a time (true for a single physical
    interconnection): idle is derived as the complement of both directions' "active" share,
    not read directly from either direction's own pct_hours_idle - that stat also counts
    hours where the OTHER direction is active, so using it directly would double-count.

    Segment order (left to right): import saturated | import partial | idle | export
    partial | export saturated.

    Parameters
    ----------
    stats_df : pd.DataFrame with columns direction (e.g. "FR→DE"), partner, pct_hours_idle,
        pct_hours_at_full - e.g. utils_batch.load_interconnection_stats's output, or
        pd.read_csv("interconnection_stats.csv") for a single run.
    area : str — the area whose links are analysed (e.g. "FR").
    lang : "FR" or "EN"

    Returns
    -------
    (fig, ax) or (None, None) if stats_df is empty (e.g. a mono-country scenario with no
    interconnections, so interconnection_stats.csv was never written).
    """
    if stats_df is None or stats_df.empty or "partner" not in stats_df.columns:
        print(f"[!] No interconnection data for {area} (mono-country scenario?)")
        return None, None

    partners = sorted(stats_df["partner"].unique())
    idle_c = "#D9D9D9"
    imp_partial_c, imp_full_c = "#F4A9A0", "#C62828"
    exp_partial_c, exp_full_c = "#9FC5F8", "#1565C0"

    fig_h = max(3, 0.6 * len(partners) + 1.5)
    fig, ax = plt.subplots(figsize=(figsize_w, fig_h))
    kept = []

    for partner in partners:
        imp_row = stats_df[(stats_df["partner"] == partner) & (stats_df["direction"] == f"{partner}→{area}")]
        exp_row = stats_df[(stats_df["partner"] == partner) & (stats_df["direction"] == f"{area}→{partner}")]
        if imp_row.empty or exp_row.empty:
            continue

        imp_idle = float(imp_row["pct_hours_idle"].iloc[0])
        imp_full = float(imp_row["pct_hours_at_full"].iloc[0])
        imp_partial = max(0.0, 100 - imp_idle - imp_full)  # import active share
        exp_idle = float(exp_row["pct_hours_idle"].iloc[0])
        exp_full = float(exp_row["pct_hours_at_full"].iloc[0])
        exp_partial = max(0.0, 100 - exp_idle - exp_full)  # export active share
        idle = max(0.0, 100 - (imp_partial + imp_full) - (exp_partial + exp_full))

        i = len(kept)
        kept.append(partner)
        left = 0.0
        for val, color in [(imp_full, imp_full_c), (imp_partial, imp_partial_c), (idle, idle_c),
                           (exp_partial, exp_partial_c), (exp_full, exp_full_c)]:
            ax.barh(i, val, left=left, color=color, height=0.6)
            left += val

        if imp_full >= 3:
            ax.text(imp_full / 2, i, f"{imp_full:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        if exp_full >= 3:
            ax.text(100 - exp_full / 2, i, f"{exp_full:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_yticks(np.arange(len(kept)))
    ax.set_yticklabels(kept, fontsize=11)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("% des heures de l'année" if lang == "FR" else "% of hours in the year")
    ax.set_title(f"Répartition horaire par interconnexion — {area}" if lang == "FR"
                else f"Hourly utilization by interconnection — {area}", fontweight="bold")

    pays = "pays" if lang == "FR" else "country"
    patches = [
        mpatches.Patch(color=imp_full_c, label=f"Import sature ({pays} → {area})" if lang == "FR" else f"Import saturated ({pays} → {area})"),
        mpatches.Patch(color=imp_partial_c, label=f"Import partiel ({pays} → {area})" if lang == "FR" else f"Partial import ({pays} → {area})"),
        mpatches.Patch(color=idle_c, label="À l'arrêt" if lang == "FR" else "Idle"),
        mpatches.Patch(color=exp_partial_c, label=f"Export partiel ({area} → {pays})" if lang == "FR" else f"Partial export ({area} → {pays})"),
        mpatches.Patch(color=exp_full_c, label=f"Export saturé ({area} → {pays})" if lang == "FR" else f"Export saturated ({area} → {pays})"),
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.15),
             ncol=3, fontsize=8, framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_residual_vs_flexibility(residual_stats, flex_gen, flex_cap, area='FR', lang='FR'):
    """Two charts comparing residual-demand indicators against flexible-asset sizing,
    across climate years: (1) residual energy balance vs flexible generation,
    (2) residual peak vs installed flexible capacity.

    Parameters
    ----------
    residual_stats : pd.DataFrame — output of utils_batch.load_residual_load_stats
        (columns include positive_area_GWh, negative_area_GWh, peak_gw).
    flex_gen : pd.Series (index=year) — aggregated flexible-tech generation [TWh/yr].
    flex_cap : pd.Series (index=year) — aggregated flexible-tech installed capacity [GW].
    area : str — used only in chart titles.
    lang : "FR" or "EN"

    Returns
    -------
    (fig1, fig2)
    """
    if residual_stats.empty:
        print('No residual_load_stats data to plot.')
        return None, None

    df = residual_stats.copy()
    df['positive_TWh'] = df['positive_area_GWh'] / 1000
    df['negative_TWh'] = df['negative_area_GWh'] / 1000
    df['net_TWh'] = df['positive_TWh'] - df['negative_TWh']
    years_avail = sorted(df.index)

    fig1, ax = plt.subplots(figsize=(13, 5))
    ax.plot(years_avail, df.loc[years_avail, 'positive_TWh'], 'o-', color='#e8834c', lw=2,
           label='Dispatchable need (+ area) [TWh/yr]' if lang == 'EN' else 'Besoin pilotable (aire +) [TWh/an]')
    ax.plot(years_avail, df.loc[years_avail, 'negative_TWh'], 's-', color='#4c9be8', lw=2,
           label='VRE surplus (− area) [TWh/yr]' if lang == 'EN' else 'Excès ENR (aire −) [TWh/an]')
    ax.plot(years_avail, df.loc[years_avail, 'net_TWh'], 'D-', color='#c0392b', lw=2.2,
           label='Net balance [TWh/yr]' if lang == 'EN' else 'Bilan net [TWh/an]')
    fg = flex_gen.reindex(years_avail).dropna()
    if not fg.empty:
        ax.bar(fg.index, fg.values, alpha=0.22, color='#27ae60', width=0.7,
              label='Flexible generation [TWh/yr]' if lang == 'EN' else 'Génération flex [TWh/an]')
    ax.set_xlabel('Climate year' if lang == 'EN' else 'Année climatique')
    ax.set_ylabel('Energy [TWh/yr]' if lang == 'EN' else 'Énergie [TWh/an]')
    ax.set_title(f'Residual demand balance — {area}', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(years_avail, df.loc[years_avail, 'peak_gw'], 'o-', color='#e8834c', lw=2,
            label='Residual demand peak [GW]' if lang == 'EN' else 'Pic demande résiduelle [GW]')
    ax2.fill_between(years_avail, df.loc[years_avail, 'peak_gw'], alpha=0.12, color='#e8834c')
    fc = flex_cap.reindex(years_avail).dropna()
    if not fc.empty:
        ax2.plot(fc.index, fc.values, 's--', color='#27ae60', lw=2,
                label='Installed flexible capacity [GW]' if lang == 'EN' else 'Capacité flex installée [GW]')
        ax2.fill_between(fc.index, fc.values, alpha=0.12, color='#27ae60')
    ax2.set_xlabel('Climate year' if lang == 'EN' else 'Année climatique')
    ax2.set_ylabel('Power [GW]' if lang == 'EN' else 'Puissance [GW]')
    ax2.set_title(f'Residual peak vs flexible capacity — {area}', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    return fig1, fig2


def plot_capacity_duals_heatmap(duals_by_group, group_labels=None, lang='FR'):
    """Heatmaps (one per constraint group, typically 'prod'/'conv'/'str') of the mean
    |dual| of the max-capacity constraint, tech x area, averaged over climate years.

    Parameters
    ----------
    duals_by_group : dict {group: pd.DataFrame} — long-format DataFrames (columns
        tech, area, dual, year), e.g. output of utils_batch.load_capacity_duals per group.
    group_labels : dict {group: display label}, optional.
    lang : "FR" or "EN"

    Returns
    -------
    fig
    """
    group_labels = group_labels or {g: g for g in duals_by_group}
    fig, axes = plt.subplots(1, len(duals_by_group), figsize=(6.5 * len(duals_by_group), 7))
    if len(duals_by_group) == 1:
        axes = [axes]

    for ax, (grp, df) in zip(axes, duals_by_group.items()):
        grp_label = group_labels.get(grp, grp)
        if df.empty:
            ax.text(0.5, 0.5, 'No data' if lang == 'EN' else 'Aucune donnée', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(grp_label)
            continue
        mean_df = df.groupby(['tech', 'area'])['dual'].mean().unstack('area')
        mean_df = mean_df.loc[(mean_df.abs() > 1e-2).any(axis=1)]
        if mean_df.empty:
            ax.text(0.5, 0.5, 'All duals are zero' if lang == 'EN' else 'Tous duals nuls', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(grp_label)
            continue
        order = mean_df.abs().mean(axis=1).sort_values(ascending=True).index
        mean_df = mean_df.loc[order]
        vals = mean_df.abs().values
        vmax = vals[np.isfinite(vals)].max() if np.any(np.isfinite(vals)) else 1.0
        im = ax.imshow(vals, aspect='auto', cmap='Reds', vmin=0, vmax=vmax)
        n_years = df['year'].nunique()
        ax.set_title(f'{grp_label}\n{"mean over" if lang == "EN" else "moyenne sur"} {n_years} {"years" if lang == "EN" else "années"}',
                    fontweight='bold', fontsize=10)
        areas_cols = list(mean_df.columns)
        ax.set_xticks(range(len(areas_cols)))
        ax.set_xticklabels(areas_cols, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(mean_df.index)))
        ax.set_yticklabels(mean_df.index, fontsize=9)
        for i, tech in enumerate(mean_df.index):
            for j, area in enumerate(areas_cols):
                v = mean_df.loc[tech, area]
                if pd.notna(v) and abs(v) > 1e-2:
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7,
                           color='white' if abs(v) > vmax * 0.55 else 'black')
        plt.colorbar(im, ax=ax, label='|Mean dual| [M€/GW/yr]', shrink=0.75)

    fig.suptitle('Max-capacity constraint duals' if lang == 'EN' else 'Duals — contraintes de capacité maximale',
                fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_annual_duals_timeseries(duals_by_constraint, areas=None, lang='FR'):
    """Grid of small multiples: |dual| vs climate year, one subplot per annual constraint.

    Parameters
    ----------
    duals_by_constraint : dict {constraint_name: (pd.DataFrame, display_label, has_area)}
        DataFrame columns: year, dual, [area]. has_area=False for constraints without an
        area dimension (e.g. methanation_CO2, France-only).
    areas : list of str, optional — restrict to these areas when has_area=True.
    lang : "FR" or "EN"

    Returns
    -------
    fig
    """
    n = len(duals_by_constraint)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax_idx, (name, (df, label, has_area)) in enumerate(duals_by_constraint.items()):
        ax = axes_flat[ax_idx]
        if df.empty:
            ax.text(0.5, 0.5, 'No data' if lang == 'EN' else 'Aucune donnée', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            ax.set_visible(False)
            continue
        df = df.dropna(subset=['dual'])
        if has_area and 'area' in df.columns:
            area_list = sorted(df['area'].unique())
            if areas is not None:
                area_list = [a for a in area_list if a in areas]
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(area_list), 1)))
            any_nonzero = False
            for a, color in zip(area_list, colors):
                sub = df[df['area'] == a].sort_values('year')
                sub_nz = sub[sub['dual'].abs() > 1e-4]
                if sub_nz.empty:
                    continue
                any_nonzero = True
                ax.plot(sub_nz['year'], sub_nz['dual'].abs(), 'o-', lw=1.8, ms=5, label=a, color=color)
            if not any_nonzero:
                ax.text(0.5, 0.5, 'Inactive constraint\n(duals ≈ 0)' if lang == 'EN' else 'Contrainte non active\n(duals ≈ 0)',
                       ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=10)
            else:
                ax.legend(fontsize=8, ncol=2, loc='upper left')
        else:
            df_s = df.sort_values('year')
            nz = df_s[df_s['dual'].abs() > 1e-4]
            if nz.empty:
                ax.text(0.5, 0.5, 'Inactive constraint\n(dual ≈ 0)' if lang == 'EN' else 'Contrainte non active\n(dual ≈ 0)',
                       ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=10)
            else:
                ax.plot(nz['year'], nz['dual'].abs(), 'o-', color='#e8834c', lw=2, ms=6)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Climate year' if lang == 'EN' else 'Année climatique')
        ax.set_ylabel('|Dual| [M€/TWh]')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle('Annual-constraint duals' if lang == 'EN' else 'Duals — contraintes annuelles', fontweight='bold', fontsize=13)
    plt.tight_layout()
    return fig


def save_residual_load(m, output_dir):
    """Compute residual load duration curves and stats per country for a solved ModelEOLES
    instance, saving one PNG per country to {output_dir}/residual_load/ plus a combined
    residual_load_stats.csv. Shared by run_batch.py and example.py."""
    rl_dir = Path(output_dir) / "residual_load"
    rl_dir.mkdir(exist_ok=True)

    rows = []
    for area in m.countries:
        try:
            residual, demand, vre = compute_residual_demand(m.hourly_balance, area=area)

            pos_area_gwh = float(np.maximum(residual, 0).sum())
            neg_area_gwh = float(abs(np.minimum(residual, 0).sum()))

            fig, stats = plot_residual_load_duration(
                m.hourly_balance, area=area, lang="FR",
                save_path=rl_dir / f"residual_load_{area}.png",
                show=False,
            )
            plt.close(fig)

            rows.append({
                "area":               area,
                "peak_gw":            stats["peak_gw"],
                "min_gw":             stats["min_gw"],
                "mean_residual_gw":   stats["mean_residual_gw"],
                "mean_demand_gw":     stats["mean_demand_gw"],
                "mean_vre_gw":        stats["mean_vre_gw"],
                "vre_coverage_pct":   stats["vre_coverage_pct"],
                "hours_negative":     stats["hours_negative"],
                "pct_negative":       stats["pct_negative"],
                "peak_hour":          stats["peak_hour"],
                "positive_area_GWh":  pos_area_gwh,
                "negative_area_GWh":  neg_area_gwh,
            })
        except Exception as e:
            print(f"    [warn] residual_load({area}) failed: {e}")

    if rows:
        pd.DataFrame(rows).set_index("area").to_csv(Path(output_dir) / "residual_load_stats.csv")
    print(f"    Residual load saved -> {rl_dir}")
