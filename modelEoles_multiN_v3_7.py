#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import linopy
import xarray as xr

# Function used to read data
from utils_io import get_config, read_constant_xr, read_profile_xr, read_links

# Functions used to build the model
from utils_build import calculate_annuities_capex_xr, calculate_annuities_storage_capex_xr

# Functions used to process outputs
from utils_results import get_technical_cost, extract_curtailment, extract_hourly_balance, \
    extract_spot_price, extract_primary_gene, extract_annualized_costs_investment_new_capa, extract_CH4_to_power,\
    extract_power_to_CH4, extract_power_to_H2, extract_annualized_costs_investment_new_capa_nofOM, \
    extract_OM_cost, extract_carbon_value, extract_H2_to_power,\
    compute_costs, extract_summary, extract_balance, \
    extract_profit, extract_carbon_footprint

# Functions used to plot outputs
from utils_plots import plot_load_shift_week, plot_elec_balance_week, plot_elec_residual_balance_week, \
    plot_storage_state_year, plot_installed_power, plot_gene_per_tech, compare_operable_mix, \
    plot_CH4_balance, plot_H2_balance

from linopy import(
    Model
)


# # Model definition

class ModelEOLES():
    """Multi-node linear optimisation model of the European electricity, methane and hydrogen
    system (investment + hourly dispatch), built with linopy.

    One instance solves for a single scenario: a set of countries (self.countries, from
    links.csv, optionally restricted via `restricted_area`), a set of climate/demand years
    (`years_of_interest`), and a carbon policy (`carbon_constraint` + per-country
    `carbon_budget`). Electricity, CH4 and H2 balances are solved jointly, with cross-border
    electricity trade and annual CH4/H2 trade between countries.

    Not every country gets the same level of operational detail: only `detailed_countries`
    (France by default — see __init__) get demand-side load shifting, FCR/FRR reserves, and
    hydro-specific constraints (spillage, minimal volume, minimal daily outflow). Other
    countries are modelled with a simplified operational layer. See the `detailed_countries`
    parameter docstring below for the rationale and the one constraint (CO2/methanation) that
    stays France-only regardless.

    Typical usage::

        m = ModelEOLES(name="run1", config=get_config("config/config_multi_nodes.json"),
                       output_path="outputs/run1")
        m.build_model()
        status, termination = m.solve(solver_name="gurobi")
        m.extract_optimisation_results_linopy()
    """

    def __init__(self, name, config, output_path,
                 include_reserve=False,
                 existing_capacity=None, existing_energy_capacity=None,
                 existing_annualized_costs_elec=0,
                 existing_annualized_costs_CH4=0, existing_annualized_costs_H2=0,
                 restricted_area=None,
                 years_of_interest=None,
                 detailed_countries=None):
        """
        :param name: str
        :param config: dict
        :param output_path: str
        :param include_reserve: bool
        :param existing_capacity: xr.DataArray
        :param existing_energy_capacity: xr.DataArray
        :param existing_annualized_costs_elec: float
        :param existing_annualized_costs_CH4: float
        :param existing_annualized_costs_H2: float
        :param restricted_area: list of str or None — if provided, only the listed countries
            are modelled (subset of the full 7-country dataset).  Transmission links are
            filtered accordingly.  Example: ['FR', 'BE', 'DE'].
        :param detailed_countries: list of str or None — countries modelled with the full
            "France-style" level of detail: demand-side load shifting (DSM), FCR/FRR reserves,
            and hydro-specific constraints (lake spillage, minimal stored volume, minimal daily
            outflow). Defaults to ['FR'] when not provided. Countries outside this list get the
            simplified treatment (no DSM, no reserves, no hydro spillage/min-volume/min-outflow
            constraints) — consistent with the fact that those profiles are currently all-zero
            for non-FR areas. The CO2 balance for methanation stays France-specific regardless
            of this parameter: it relies on France-only scalar constants (CO2_industry_demand /
            CO2_industry_prod) that have no per-country breakdown yet.
        """
        self.name = name
        self.config = config
        self.output_path = output_path
        self.model = linopy.Model()

        self.include_reserve = include_reserve
        self.restricted_area = restricted_area
        self._years_override = years_of_interest
        self._detailed_countries_override = detailed_countries

        self.existing_capacity = existing_capacity # GW
        self.existing_energy_capacity = existing_energy_capacity # GWh
        self.existing_annualized_costs_elec = existing_annualized_costs_elec
        self.existing_annualized_costs_CH4 = existing_annualized_costs_CH4
        self.existing_annualized_costs_H2 = existing_annualized_costs_H2


    def load_inputs(self):
        """Read every input file listed in `self.config` and derive the constants used to
        build the model: simulation parameters, technology/cost/reserve parameters, capacities,
        time-indexed profiles (demand, VRE, hydro), the per-country detailed/carbon-budget
        flags, and pre-computed annuities. Must run before define_sets()."""

        #############################
        ### Simulation parameters ###
        #############################

        self.nb_years = self.config["nb_optimisation_years"]
        self.years_of_interest = self.config["years_of_interest"]
        if self._years_override is not None:
            self.years_of_interest = self._years_override
        self.year_simulated = self.config["year_simulated"]
        self.carbon_constraint = self.config["carbon_constraint"]

        # Reading the constants of the problem
        self.constants = read_constant_xr(self.config["constants"])
        self.costs = read_constant_xr(self.config["costs"], dims = ["tech", "parameter"])
        self.reserve = read_constant_xr(self.config["reserve"], dims = ["reserve_type", "parameter"])
        self.tech_parameters = read_constant_xr(self.config["tech_parameters"], dims = ["tech", "parameter"])
        self.fuel_prices = read_constant_xr(self.config["fuel_prices"], dims = ["fuel", "parameter"])

        # Reading the capacities
        if self.existing_capacity is None:
            self.existing_capacity = read_constant_xr(self.config["existing_capacity"], dims = ["tech", "area"])
            self.existing_capacity = self.existing_capacity.fillna(0.0)
        if self.existing_energy_capacity is None:
            self.existing_energy_capacity = read_constant_xr(self.config["existing_energy_capacity"], dims = ["tech", "area"])
            self.existing_energy_capacity = self.existing_energy_capacity.fillna(0.0)
        self.maximum_capacity = read_constant_xr(self.config["maximum_capacity"], dims = ["tech", "area"]).fillna(np.inf)
        self.maximum_energy_capacity = read_constant_xr(self.config["maximum_energy_capacity"], dims = ["tech", "area"]).fillna(np.inf)
        self.minimum_capacity = read_constant_xr(self.config["minimum_capacity"], dims = ["tech", "area"]).fillna(0.0)
        self.minimum_energy_capacity = read_constant_xr(self.config["minimum_energy_capacity"], dims = ["tech", "area"]).fillna(0.0)
        self.biogas_potential = read_constant_xr(self.config["biogas_potential"], dims = ["tech", "area"]).fillna(np.inf)

        # Reading the links — optionally restricted to a subset of countries
        self.cap_dict, countries_from_links = read_links(self.config["links"])
        if self.restricted_area is not None:
            # Preserve the ordering from links.csv, keep only requested countries
            self.countries = [c for c in countries_from_links if c in self.restricted_area]
            # Keep only transmission links where both endpoints are in the restricted set
            self.cap_dict = {(a, b): v for (a, b), v in self.cap_dict.items()
                             if a in self.restricted_area and b in self.restricted_area}
        else:
            self.countries = countries_from_links
        self.tr_loss = float(self.constants.loc["tr_loss"].value)

        # Detailed-modelling countries (DSM, FCR/FRR reserves, hydro spillage/min-volume/
        # min-outflow) — see the `detailed_countries` docstring in __init__. Defaults to ['FR'].
        _requested_detailed = self._detailed_countries_override if self._detailed_countries_override is not None else ["FR"]
        self.detailed_countries = [c for c in self.countries if c in _requested_detailed]
        self.non_detailed_countries = [c for c in self.countries if c not in self.detailed_countries]
        # France-specific flag, used only by the CO2/methanation balance (see define_constraints):
        # that constraint relies on France-only scalar constants and cannot be generalised.
        self.france_in_model = "FR" in self.countries


        #############################
        ### Time-indexed profiles ###
        #############################

        # Demand — elec_demand sets the reference hourly index for all subsequent profile reads.
        # years_of_interest selects which climate years to optimise over (independent of year_simulated).
        self.elec_demand = read_profile_xr(self.config["elec_demand"], time_scale="hourly",
                                           years=self.years_of_interest)
        _hourly_ref = self.elec_demand.indexes["hour"]
        # nb_years: if years_of_interest is specified, derive it from the selected years
        # so that the annualisation in the objective reflects the actual simulation horizon.
        if self.years_of_interest is not None:
            self.nb_years = len(self.years_of_interest)
        _daily_ref = pd.DatetimeIndex(_hourly_ref.normalize().unique())
        self.H2_demand = read_profile_xr(self.config["H2_demand"], time_scale="hourly", reference_index=_hourly_ref)
        self.CH4_demand = read_profile_xr(self.config["CH4_demand"], time_scale="hourly", reference_index=_hourly_ref)

        # CH4 and H2 demand profile flags: dict {area: bool}
        # True  → hourly adequacy constraint (constraint 'CH4_adequacy' / 'H2_adequacy')
        # False → annual adequacy constraint only (constraint 'CH4_adequacy_annual' / 'H2_adequacy_annual')
        # Default: detailed_countries (France by default) use the hourly profile, all other
        # areas use the annual balance only.
        self.CH4_demand_is_profile = {area: (area in self.detailed_countries) for area in self.countries}
        self.H2_demand_is_profile  = {area: (area in self.detailed_countries) for area in self.countries}
        # Override from config if provided (optional):
        # config["CH4_demand_is_profile"] = {"FR": true, "DE": false, ...}
        for flag_name, flag_dict in [("CH4_demand_is_profile", self.CH4_demand_is_profile),
                                      ("H2_demand_is_profile",  self.H2_demand_is_profile)]:
            if flag_name in self.config and isinstance(self.config[flag_name], dict):
                for area, val in self.config[flag_name].items():
                    if area in flag_dict:
                        flag_dict[area] = bool(val)

        # VRE profiles
        vre_list = ["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                    "river", "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge"]
        vre_profiles_list = []
        for tech in vre_list:
            profile = read_profile_xr(self.config[f"prod_profile_{tech}"], time_scale="hourly", reference_index=_hourly_ref)
            profile = profile.where(profile > 1e-4, 0)
            profile = profile.expand_dims(tech=[tech])
            vre_profiles_list.append(profile)
        self.vre_profiles = xr.concat(vre_profiles_list, dim="tech")
        self.vre_profiles = self.vre_profiles.transpose("area", "tech", "hour")

        # Other hydro profiles
        self.lake_inflows = read_profile_xr(self.config["lake_inflows"], time_scale="hourly", reference_index=_hourly_ref)  # GWh
        self.phs_inflows = read_profile_xr(self.config["phs_inflows"], time_scale="hourly", reference_index=_hourly_ref)    # GWh
        self.lake_spill = read_profile_xr(self.config["lake_spill"], time_scale="hourly", reference_index=_hourly_ref)
        self.lake_minimal_volume = read_profile_xr(self.config["lake_minimal_volume"], time_scale="hourly", reference_index=_hourly_ref)
        self.lake_minimal_outflow = read_profile_xr(self.config["lake_minimal_outflow"], time_scale="daily", reference_index=_daily_ref)


        ############################
        ### Exogeneous constants ###
        ############################

        self.capacity_factor_nuclear_yearly = self.constants.loc["capacity_factor_nuclear_yearly"].value

        # Carbon
        self.scc = float(self.constants.loc["social_cost_of_carbon"].value)
        # Per-country carbon budget (MtCO2/yr), from inputs/area_indexed/carbon_budget.csv
        # (row "carbon_budget" x area). Missing file/key or missing country → 0, ie only
        # CO2-neutral techs used, forcing zero-emission there when carbon_constraint=True.
        if "carbon_budget" in self.config:
            self.carbon_budget = (read_constant_xr(self.config["carbon_budget"], dims=["tech", "area"])
                                  .loc[{"tech": "carbon_budget"}]
                                  .reindex(area=self.countries, fill_value=0.0)
                                  .fillna(0.0))
        else:
            self.carbon_budget = xr.DataArray(np.zeros(len(self.countries)),
                                              coords={"area": self.countries}, dims=["area"])
        self.carbon_content = self.tech_parameters.loc[{"parameter":"carbon_content"}]
        self.carbon_footprint = self.tech_parameters.loc[{"parameter":"carbon_footprint"}]

        # CO2 balance constants (used for methanation constraint, France only)
        self.CO2_industry_demand = float(self.constants.loc["CO2_industry_demand"].value)
        self.CO2_industry_prod = float(self.constants.loc["CO2_industry_prod"].value)
        self.CO2_usable_industry = float(self.constants.loc["CO2_usable_industry"].value)
        self.CO2_fraction = self.tech_parameters.loc[{"parameter":"co2_fraction"}].dropna(dim='tech')
        self.CO2_usable = self.tech_parameters.loc[{"parameter":"co2_usable"}].dropna(dim='tech')

        # Conversion parameters and efficiency
        self.h2_saltcavern_charge_to_storage_ratio = float(self.constants.loc["h2_saltcavern_charge_to_storage_ratio"].value)
        self.h2_saltcavern_discharge_to_storage_ratio = float(self.constants.loc["h2_saltcavern_discharge_to_storage_ratio"].value)
        self.ch4_reservoir_charge_to_discharge_ratio = float(self.constants.loc["ch4_reservoir_charge_to_discharge_ratio"].value)
        self.phs_charge_to_discharge_ratio = float(self.constants.loc["phs_charge_to_discharge_ratio"].value)
        self.efficiency_storage_in = self.tech_parameters.loc[{"parameter":"efficiency_storage_in"}].fillna(1.0)
        self.efficiency_storage_out = self.tech_parameters.loc[{"parameter":"efficiency_storage_out"}].fillna(1.0)
        self.conversion_efficiency = self.tech_parameters.loc[{"parameter":"conversion_efficiency"}].dropna(dim="tech")

        # Reserves
        self.fcr_requirement = float(self.constants.loc["fcr_requirement"].value)
        self.frr_requirements = self.tech_parameters.loc[{"parameter":"frr_requirements"}].fillna(0.0)
        self.reserve_activation_rate = self.reserve.loc[{"parameter":"reserve_activation_rate"}]
        self.reserve_activation_time = self.reserve.loc[{"parameter":"reserve_activation_time"}]

        # Load shifting (only for France; load_shift_maximum_power has area dimension but only FR is used)
        self.load_shift_maximum_power = float(self.constants.loc["load_shift_maximum_power"].value)*self.elec_demand.sum(dim="hour")/self.nb_years
        self.load_shift_period = float(self.constants.loc["load_shift_period"].value)
        self.load_uncertainty = float(self.constants.loc["load_uncertainty"].value)
        self.load_variation = float(self.constants.loc["load_variation"].value)

        # Ramping rate
        self.ramp_rate = self.tech_parameters.loc[{"parameter":"ramp_rate"}].dropna(dim="tech")

        # Costs parameters
        self.fOM = self.costs.loc[{"parameter":"fOM"}]
        self.vOM = self.costs.loc[{"parameter":"vOM"}]
        self.capex = self.costs.loc[{"parameter":"capex"}]
        self.storage_capex = self.costs.loc[{"parameter":"storage_capex"}].dropna(dim="tech")
        self.discount_rate = self.costs.loc[{"parameter":"discount_rate"}]
        self.lifetime = self.costs.loc[{"parameter":"lifetime"}]
        self.construction_time = self.costs.loc[{"parameter":"construction_time"}]

        # Annuities calculation
        self.annuities = calculate_annuities_capex_xr(self.discount_rate, self.capex, self.construction_time, self.lifetime)
        self.storage_annuities = calculate_annuities_storage_capex_xr(self.discount_rate, self.storage_capex, self.construction_time, self.lifetime)

        # Fuel prices
        self.fuel_prices_bis = self.fuel_prices.loc[{"parameter":"fuel_prices_2020"}] * (1+self.fuel_prices.loc[{"parameter":"fuel_prices_rate"}])**(self.year_simulated - 2020)
        self.vOM.loc["natural_gas"] = self.fuel_prices_bis.loc["natural_gas"]
        self.vOM.loc["coal"] = self.fuel_prices_bis.loc["coal"]

        # Extra biogas import
        self.limit_biogas_extra_import = float(self.constants.loc["limit_biogas_extra_import"].value)



    def define_sets(self):
        """Build the xarray coordinate arrays used throughout the model: the hourly index and
        its day/year groupings, the load-shift pair index, the technology-family sets (prod_tech,
        conversion_tech, str, elec_balance, CH4_balance, H2_balance, ...), and the country
        coordinate arrays (self.areas plus self.detailed_area for detailed_countries)."""
        # Range of hour
        self.hours = self.elec_demand.coords["hour"]
        self.hours_array = self.hours.values

        # Arrays to attribute each time step to a period (day or year)
        self.day = self.hours.dt.floor("D")
        self.year = self.hours.dt.year

        # Pairs for load shifting (Zerrahn et al. 2015)
        n = len(self.hours_array)
        L = self.load_shift_period
        h = np.repeat(np.arange(n), 2*L+1).astype(int)
        offsets = np.tile(np.arange(-L, L+1), n).astype(int)
        hh = (h + offsets) % n
        hh = hh.astype(int)
        self.pairs = xr.Dataset({
            "hour": ("pair", self.hours_array[h]),
            "hh":   ("pair", self.hours_array[hh]),
        })

        ### Technologies ###

        self.all_tech = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                      "river", "lake", "nuclear", "methanization", "pyrogazification", "ocgt_coge",
                                      "natural_gas", "biogas_import", "H2_import", "coal", "biomass_coge", "geothermal_coge", "waste", "marine", "rsv_dummy",
                                      "ch4_ocgt", "ch4_ccgt", "h2_ccgt", "electrolysis", "methanation",
                                      "phs", "battery_1h", "battery_2h", "battery_4h", "battery_8h", "h2_saltcavern", "ch4_reservoir", "str_dummy", "lost_load"], dims=["tech"])

        self.prod_tech = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                       "river", "lake", "nuclear", "methanization", "pyrogazification",
                                       "natural_gas", "biogas_import", "H2_import", "coal", "biomass_coge", "geothermal_coge", "waste", "marine", "rsv_dummy", "lost_load"], dims=["tech"])

        self.vre = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                 "river", "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge"], dims=["tech"])

        self.solar = xr.DataArray(["pv_ground", "pv_roof_com", "pv_roof_indiv"], dims=["tech"])

        self.elec_balance = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                          "river", "lake", "nuclear", "phs",
                                          "battery_1h", "battery_2h", "battery_4h", "battery_8h", "ch4_ocgt", "ch4_ccgt", "h2_ccgt", "coal",
                                          "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge",
                                          "str_dummy", "lost_load"], dims=["tech"])

        self.reserve = xr.DataArray(["lake", "phs", "ch4_ocgt", "ch4_ccgt", "nuclear", "h2_ccgt", "coal",
                                     "battery_1h", "battery_2h", "battery_4h", "battery_8h", "rsv_dummy", "lost_load"], dims=["tech"])
        self.reserve_ramp_limited = xr.DataArray(["ch4_ocgt", "ch4_ccgt", "nuclear", "h2_ccgt", "coal"], dims=["tech"])

        self.elec_primary_prod = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                               "river", "lake", "nuclear", "coal", "biomass_coge", "geothermal_coge", "waste", "marine"], dims=["tech"])
        self.elec_prod = xr.DataArray(["offshore_float", "offshore_ground", "onshore", "pv_ground", "pv_roof_com", "pv_roof_indiv",
                                       "river", "lake", "nuclear", "coal", "biomass_coge", "geothermal_coge", "waste", "marine",
                                       "ch4_ocgt", "ocgt_coge", "ch4_ccgt", "h2_ccgt"], dims=["tech"])
        self.CH4_primary_prod = xr.DataArray(["methanization", "pyrogazification", "biogas_import", "natural_gas"], dims=["tech"])
        self.CH4_prod = xr.DataArray(["methanization", "pyrogazification", "methanation", "biogas_import", "natural_gas"], dims=["tech"])
        self.H2_prod = xr.DataArray(["electrolysis", "H2_import"], dims=["tech"])

        self.use_elec = xr.DataArray(["phs", "battery_1h", "battery_2h", "battery_4h", "battery_8h", "electrolysis", "methanation", "str_dummy"], dims=["tech"])
        self.use_CH4 = xr.DataArray(["ch4_reservoir", "ch4_ocgt", "ch4_ccgt", "ocgt_coge"], dims=["tech"])
        self.use_H2 = xr.DataArray(["h2_saltcavern", "h2_ccgt"], dims=["tech"])

        self.CH4_balance = xr.DataArray(["methanization", "pyrogazification", "biogas_import", "natural_gas", "methanation", "ch4_reservoir"], dims=["tech"])
        self.CH4_balance_biogas = xr.DataArray(["methanization", "pyrogazification", "methanation", "biogas_import"], dims=["tech"])
        self.H2_balance = xr.DataArray(["electrolysis", "h2_saltcavern", "H2_import"], dims=["tech"])

        self.conversion_tech = xr.DataArray(["ch4_ocgt", "ch4_ccgt", "h2_ccgt", "electrolysis", "methanation", "ocgt_coge"], dims=["tech"])
        self.from_elec_to_CH4 = xr.DataArray(["methanation"], dims=["tech"])
        self.from_elec_to_H2 = xr.DataArray(["electrolysis"], dims=["tech"])
        self.from_CH4_to_elec = xr.DataArray(["ch4_ocgt", "ch4_ccgt"], dims=["tech"])
        self.from_H2_to_elec = xr.DataArray(["h2_ccgt"], dims=["tech"])

        self.str = xr.DataArray(["phs", "battery_1h", "battery_2h", "battery_4h", "battery_8h", "h2_saltcavern", "ch4_reservoir", "str_dummy"], dims=["tech"])
        self.str_elec = xr.DataArray(["phs", "battery_1h", "battery_2h", "battery_4h", "battery_8h", "str_dummy"], dims=["tech"])
        self.battery = xr.DataArray(["battery_1h", "battery_2h", "battery_4h", "battery_8h"], dims=["tech"])
        self.str_CH4 = xr.DataArray(["ch4_reservoir"], dims=["tech"])
        self.str_H2 = xr.DataArray(["h2_saltcavern"], dims=["tech"])

        ### Countries ###

        self.areas = xr.DataArray(self.countries, dims=["area"])
        self.areas_bis = xr.DataArray(self.countries, dims=["area_bis"])

        # self.detailed_countries / self.non_detailed_countries / self.france_in_model are
        # computed in load_inputs() (they only depend on self.countries, known earlier).
        # Coordinate array for the detailed-modelling subset (DSM, reserves, hydro specifics).
        if self.detailed_countries:
            self.detailed_area = xr.DataArray(self.detailed_countries, dims=["area"])

        self.cap_interco = xr.DataArray(
            np.zeros((len(self.areas), len(self.areas_bis))),
            coords={"area": self.areas, "area_bis": self.areas_bis},
            dims=("area", "area_bis")
        )
        for (a, b), val in self.cap_dict.items():
            self.cap_interco.loc[dict(area=a, area_bis=b)] = val


    def define_variables(self):
        """Declare every linopy decision variable: hourly generation/storage/conversion output,
        installed capacities, storage energy capacities, FCR/FRR reserves and DSM load-shifting
        (both restricted to detailed_countries), cross-border import/export flows, and annual
        CH4/H2 trade volumes."""

        # Hourly energy output in GW
        self.gene = self.model.add_variables(coords=[self.areas, self.prod_tech, self.hours], lower=0, name='gene')
        self.str_output = self.model.add_variables(coords=[self.areas, self.str, self.hours], lower=0, name='str_output')
        self.conv_output = self.model.add_variables(coords=[self.areas, self.conversion_tech, self.hours], lower=0, name='conv_output')

        # Installed capacity in GW
        self.nominal_power = self.model.add_variables(coords=[self.areas, self.prod_tech], lower=0, name='nominal_power')

        self.output_power = self.model.add_variables(coords=[self.areas, self.conversion_tech], lower=0, name='output_power')

        self.discharging_power = self.model.add_variables(coords=[self.areas, self.str], lower=0, name='discharging_power')

        self.charging_power = self.model.add_variables(coords=[self.areas, self.str], lower=0, name='charging_power')

        self.energy_capacity = self.model.add_variables(coords=[self.areas, self.str], lower=0, name='energy_capacity')

        self.str_input = self.model.add_variables(coords=[self.areas, self.str, self.hours], lower=0, name='str_input')
        self.state_of_charge = self.model.add_variables(coords=[self.areas, self.str, self.hours], lower=0, name='state_of_charge')
        self.lake_stored = self.model.add_variables(coords=[self.areas, self.hours], lower=0, name='lake_stored')

        if self.detailed_countries:
            self.fcr = self.model.add_variables(coords=[self.detailed_area, self.reserve, self.hours], lower=0, name='fcr')
            self.frr = self.model.add_variables(coords=[self.detailed_area, self.reserve, self.hours], lower=0, name='frr')

        # DSM variables: detailed_countries only — avoids spurious shifts in countries without DSM data/policy
        if self.detailed_countries:
            self.dsm_up = self.model.add_variables(
                coords=[self.detailed_area, self.hours],
                lower=0,
                name='dsm_up'
            )
            self.dsm_down = self.model.add_variables(
                coords={"area": self.detailed_area, "pair": self.pairs.pair},
                lower=0,
                dims=["area", "pair"],
                name="dsm_down"
            )

        # Interconnection variables
        self.imp = self.model.add_variables(coords=[self.areas, self.areas_bis, self.hours], lower=0, name='import')
        self.exp = self.model.add_variables(coords=[self.areas, self.areas_bis, self.hours], lower=0, name='export')

        # Gas import and export annual variables
        # self.year has dim=hour with duplicate values → use unique years as coordinate
        _year_unique = xr.DataArray(sorted(np.unique(self.hours.dt.year.values)), dims=["year"])
        self.CH4_imp_annual = self.model.add_variables(coords=[self.areas, _year_unique], lower=0, name="gas annual import")
        self.CH4_exp_annual = self.model.add_variables(coords=[self.areas, _year_unique], lower=0, name="gas annual export")

        self.H2_imp_annual = self.model.add_variables(coords=[self.areas, _year_unique], lower=0, name="H2 annual import")
        self.H2_exp_annual = self.model.add_variables(coords=[self.areas, _year_unique], lower=0, name="H2 annual export")


    def define_constraints(self):
        """Declare every constraint of the model, organised by section (general capacity bounds,
        electricity/CH4/H2 adequacy, interconnections, load shifting, VRE, storage, batteries,
        nuclear, hydro). Sections gated on `self.detailed_countries` add the France-style
        operational detail (reserves, DSM, hydro specifics) for that subset of countries; the
        CO2/methanation balance is the one exception, gated on `self.france_in_model` instead
        (see its section below for why).

        Scaling factors (*10, *100, /1000 etc.) are intentional for numerical conditioning.
        Target: matrix coefficients in [1e-3, 1e3], RHS in [1e-2, 1e2]."""

        ###############
        ### GENERAL ###
        ###############

        # Capacity bounds given by maximum capacity and minimum capacity.
        # Existing capacity is also considered as lower bound for capacity.
        low_cap = self.minimum_capacity.reindex(tech=self.all_tech, area=self.areas).fillna(0.0)
        exist_cap = self.existing_capacity.reindex(tech=self.all_tech, area=self.areas).fillna(0.0)
        up_cap = self.maximum_capacity.reindex(tech=self.all_tech, area=self.areas).fillna(np.inf)

        self.model.add_constraints(self.nominal_power.loc[{"tech":self.prod_tech, "area":self.areas}] >= low_cap.loc[{"tech": self.prod_tech, "area":self.areas}], name="min_capacity_prod")
        self.model.add_constraints(self.nominal_power.loc[{"tech":self.prod_tech, "area":self.areas}] >= exist_cap.loc[{"tech": self.prod_tech, "area":self.areas}], name="exist_capacity_prod")
        self.model.add_constraints(self.nominal_power.loc[{"tech":self.prod_tech, "area":self.areas}] <= up_cap.loc[{"tech": self.prod_tech, "area":self.areas}], name="max_capacity_prod")

        self.model.add_constraints(self.output_power.loc[{"tech":self.conversion_tech, "area":self.areas}] >= low_cap.loc[{"tech": self.conversion_tech, "area":self.areas}], name="min_capacity_conv")
        self.model.add_constraints(self.output_power.loc[{"tech":self.conversion_tech, "area":self.areas}] >= exist_cap.loc[{"tech": self.conversion_tech, "area":self.areas}], name="exist_capacity_conv")
        self.model.add_constraints(self.output_power.loc[{"tech":self.conversion_tech, "area":self.areas}] <= up_cap.loc[{"tech": self.conversion_tech, "area":self.areas}], name="max_capacity_conv")

        self.model.add_constraints(self.discharging_power.loc[{"tech":self.str, "area":self.areas}] >= low_cap.loc[{"tech": self.str, "area":self.areas}], name="min_capacity_str")
        self.model.add_constraints(self.discharging_power.loc[{"tech":self.str, "area":self.areas}] >= exist_cap.loc[{"tech": self.str, "area":self.areas}], name="exist_capacity_str")
        self.model.add_constraints(self.discharging_power.loc[{"tech":self.str, "area":self.areas}] <= up_cap.loc[{"tech": self.str, "area":self.areas}], name="max_capacity_str")

        # Capacity bounds for storage
        low_ec = self.minimum_energy_capacity.reindex(tech=self.str, area=self.areas).fillna(0.0)
        exist_ec = self.existing_energy_capacity.reindex(tech=self.str, area=self.areas).fillna(0.0)
        up_ec = self.maximum_energy_capacity.reindex(tech=self.str, area=self.areas).fillna(np.inf)

        self.model.add_constraints(self.energy_capacity.loc[{"tech":self.str, "area":self.areas}] >= low_ec.loc[{"tech": self.str, "area":self.areas}], name="min_en_capacity_str")
        self.model.add_constraints(self.energy_capacity.loc[{"tech":self.str, "area":self.areas}] >= exist_ec.loc[{"tech": self.str, "area":self.areas}], name="exist_en_capacity_str")
        self.model.add_constraints(self.energy_capacity.loc[{"tech":self.str, "area":self.areas}] <= up_ec.loc[{"tech": self.str, "area":self.areas}], name="max_en_capacity_str")




        # Generation is always lower than installed capacity
        self.model.add_constraints(self.nominal_power >= self.gene, name="installed_power_prod")
        self.model.add_constraints(self.output_power >= self.conv_output, name="installed_power_conversion")
        self.model.add_constraints(self.discharging_power >= self.str_output, name="installed_power_storage")

        # Carbon budget, per country (self.carbon_budget: 0 by default — see load_inputs)
        if self.carbon_constraint:
            hourly_carbon = (self.gene.loc[{"tech": 'coal'}] * self.carbon_content.loc[{"tech": 'coal'}] +
                             self.gene.loc[{"tech": 'natural_gas'}] * self.carbon_content.loc[{"tech": 'natural_gas'}])
            if self.detailed_countries:
                hourly_carbon = hourly_carbon + (
                    self.frr.loc[{"tech": 'coal'}] * self.carbon_content.loc[{"tech": 'coal'}] * self.reserve_activation_time.loc[{"reserve_type": 'frr'}] +
                    self.fcr.loc[{"tech": 'coal'}] * self.carbon_content.loc[{"tech": 'coal'}] * self.reserve_activation_time.loc[{"reserve_type": 'fcr'}])
            yearly_carbon = hourly_carbon.groupby(self.year).sum(dim="hour")
            # Written as (linopy expression) <= (plain array) rather than the reverse: with an
            # xr.DataArray on the left, xarray's own comparison operator intercepts the call
            # before linopy gets a chance to, corrupting the resulting constraint's shape.
            self.model.add_constraints(yearly_carbon/10 <= self.carbon_budget*100, name='carbon_budget')


        ###################
        ### ELECTRICITY ###
        ###################

        self.vre_prod_tech = self.vre[self.vre.isin(self.prod_tech)]

        if self.detailed_countries:
            reserve_prod = self.reserve[self.reserve.isin(self.prod_tech)]
            reserve_conversion = self.reserve[self.reserve.isin(self.conversion_tech)]
            reserve_str = self.reserve[self.reserve.isin(self.str)]

            # Reserve (FCR+FRR) stacks on top of actual dispatch: the two combined can't
            # exceed installed capacity, for each tech category (prod/conversion/storage).
            self.model.add_constraints(self.gene.loc[{'tech': reserve_prod, "area": self.detailed_countries}] + self.fcr.loc[{'tech': reserve_prod}] + self.frr.loc[{'tech': reserve_prod}] <= self.nominal_power.loc[{'tech': reserve_prod, "area": self.detailed_countries}],
                                       name="reserve_power_prod")
            self.model.add_constraints(self.conv_output.loc[{'tech': reserve_conversion, "area": self.detailed_countries}] + self.fcr.loc[{'tech': reserve_conversion}] + self.frr.loc[{'tech': reserve_conversion}] <= self.output_power.loc[{'tech': reserve_conversion, "area": self.detailed_countries}],
                                       name="reserve_power_conversion")
            self.model.add_constraints(self.str_output.loc[{'tech': reserve_str, "area": self.detailed_countries}] + self.fcr.loc[{'tech': reserve_str}] + self.frr.loc[{'tech': reserve_str}] <= self.discharging_power.loc[{'tech': reserve_str, "area": self.detailed_countries}],
                                       name="reserve_power_storage")

            # NB: fcr_requirement/frr provisioning below use a single global scalar/profile
            # applied identically to every area in detailed_countries — a first-order
            # approximation when detailed_countries has more than one country (reserve needs
            # are not currently broken down per country in the input data).
            self.model.add_constraints(self.fcr.sum(dim='tech') == self.fcr_requirement * int(self.include_reserve), name='fcr_provision')

            reserve_ramp_limited_prod = self.reserve_ramp_limited[self.reserve_ramp_limited.isin(self.prod_tech)]
            reserve_ramp_limited_output = self.reserve_ramp_limited[self.reserve_ramp_limited.isin(self.conversion_tech)]
            reserve_ramp_limited_discharging = self.reserve_ramp_limited[self.reserve_ramp_limited.isin(self.str)]

            # Ramp-limited techs (thermal/nuclear) can't provide more FCR than they could
            # actually ramp to within the FCR activation window (reserve_activation_time).
            self.model.add_constraints(self.fcr.loc[{'tech': reserve_ramp_limited_prod}] * 10 <=
                                       self.nominal_power.loc[{'tech': reserve_ramp_limited_prod, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_prod}] * self.reserve_activation_time.loc["fcr"] * 10,
                                       name="fcr_limited_ramp_rate_prod")
            self.model.add_constraints(self.fcr.loc[{'tech': reserve_ramp_limited_output}] * 10 <=
                                       self.output_power.loc[{'tech': reserve_ramp_limited_output, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_output}] * self.reserve_activation_time.loc["fcr"] * 10,
                                       name="fcr_limited_ramp_rate_conversion")
            self.model.add_constraints(self.fcr.loc[{'tech': reserve_ramp_limited_discharging}] * 10 <=
                                       self.discharging_power.loc[{'tech': reserve_ramp_limited_discharging, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_discharging}] * self.reserve_activation_time.loc["fcr"] * 10,
                                       name="fcr_limited_ramp_rate_storage")

            # Total FRR provided must cover both demand forecast uncertainty (load_req) and
            # VRE forecast uncertainty (res_req, proportional to installed VRE capacity —
            # this is why VRE techs are charged an implicit reserve cost in extract_profit,
            # even though they never provide FRR themselves).
            res_req = (self.nominal_power.loc[{"tech": self.vre_prod_tech, "area": self.detailed_countries}] * self.frr_requirements.loc[{'tech': self.vre_prod_tech}]).sum(dim="tech")
            load_req = self.elec_demand.loc[{"area": self.detailed_countries}] * self.load_uncertainty * (1 + self.load_variation)
            frr_sum = self.frr.sum(dim="tech")

            self.model.add_constraints((frr_sum - res_req * int(self.include_reserve)) * 10 == load_req * int(self.include_reserve) * 10, name='frr_provision')

            # Same ramp-limited logic as FCR above, applied to FRR (longer activation window).
            self.model.add_constraints(self.frr.loc[{'tech': reserve_ramp_limited_prod}] * 10 <=
                                       self.nominal_power.loc[{'tech': reserve_ramp_limited_prod, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_prod}] * self.reserve_activation_time.loc["frr"] * 10,
                                       name="frr_limited_ramp_rate_prod")
            self.model.add_constraints(self.frr.loc[{'tech': reserve_ramp_limited_output}] * 10 <=
                                       self.output_power.loc[{'tech': reserve_ramp_limited_output, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_output}] * self.reserve_activation_time.loc["frr"] * 10,
                                       name="frr_limited_ramp_rate_conversion")
            self.model.add_constraints(self.frr.loc[{'tech': reserve_ramp_limited_discharging}] * 10 <=
                                       self.discharging_power.loc[{'tech': reserve_ramp_limited_discharging, "area": self.detailed_countries}] * self.ramp_rate.loc[{'tech': reserve_ramp_limited_discharging}] * self.reserve_activation_time.loc["frr"] * 10,
                                       name="frr_limited_ramp_rate_storage")

        # Electricity adequacy
        prod_elec_balance = self.elec_balance[self.elec_balance.isin(self.prod_tech)]
        conv_elec_balance = self.elec_balance[self.elec_balance.isin(self.conversion_tech)]
        str_elec_balance = self.elec_balance[self.elec_balance.isin(self.str)]

        storage = (self.str_input.loc[{'tech':self.str_elec}]).sum(dim="tech")
        gene_from_elec = (self.conv_output.loc[{"tech":'electrolysis'}] / self.conversion_efficiency.loc['electrolysis']
                          + self.conv_output.loc[{"tech":'methanation'}] / self.conversion_efficiency.loc['methanation'])
        prod_elec = ((self.gene.loc[{'tech':prod_elec_balance}]).sum(dim="tech")
                     + (self.conv_output.loc[{'tech':conv_elec_balance}]).sum(dim="tech")
                     + (self.str_output.loc[{'tech':str_elec_balance}]).sum(dim="tech"))

        imports = self.imp.sum(dim="area_bis")
        exports = self.exp.sum(dim="area_bis")
        # Net electricity available = generation - storage charge - elec-to-gas conversion
        # input + net imports. Compared against demand (+ DSM net shift) in the adequacy
        # constraints below.
        balance = prod_elec - storage - gene_from_elec + imports - exports

        # DSM is limited to detailed_countries: compute net shift there, apply to their adequacy
        if self.detailed_countries:
            down_shift = self.dsm_down.groupby(self.pairs.hh).sum().rename({"hh": "hour"})
            net_load_shift_up = self.dsm_up - down_shift  # dims: [area=detailed_countries, hour]

            # 'adequacy' covers detailed_countries (with DSM) — name kept for extract_spot_price compatibility
            self.adequacy = self.model.add_constraints(
                balance.loc[{"area": self.detailed_countries}] - net_load_shift_up >= self.elec_demand.loc[{"area": self.detailed_countries}],
                name='adequacy'
            )
        # adequacy_non_fr covers all other (non-detailed) countries (no DSM)
        # NOTE: to get spot prices for non-detailed countries, use model.constraints.adequacy_non_fr.dual
        # When detailed_countries is empty, this constraint covers all active countries.
        if self.non_detailed_countries:
            self.adequacy_non_fr = self.model.add_constraints(
                balance.loc[{"area": self.non_detailed_countries}] >= self.elec_demand.loc[{"area": self.non_detailed_countries}],
                name='adequacy_non_fr'
            )
        elif not self.detailed_countries:
            # Edge case: no countries at all — should not occur in practice
            pass

        # Lost-load limit. If needed, below is provided the code to keep lost load under RTE criterion.
        # We keep RTE criteria : 0.002% of total elec consumption
        #self.lost_load_limit = self.model.add_constraints(self.gene.loc[{"tech":"lost_load", "area":self.areas}].groupby(self.year).sum(dim="hour") <= self.elec_demand.loc[{"area":self.areas}].groupby(self.year).sum(dim="hour")*0.00002)


        ########################
        ### INTERCONNECTIONS ###
        ########################

        # On each line between countries, imports must equal exports, and must be below the interconnection capacity.
        # We incroporate a slight loss self.tr_loss to prevent simultaneous export and import between two countries.
        self.transmission_adequacy = self.model.add_constraints(self.imp == self.exp.rename(area="area_bis", area_bis="area")*(1-self.tr_loss), name="transmission_adequacy")
        self.transmission_limits = self.model.add_constraints(self.exp <= self.cap_interco, name="transmission_limits")


        #####################
        ### LOAD SHIFTING ###
        ### (detailed_countries only) ###
        #####################

        if self.detailed_countries:
            # dsm_up must equal the total of all compensating downward shifts
            sum_down = self.dsm_down.groupby(self.pairs.hour).sum()
            self.model.add_constraints(self.dsm_up == sum_down, name="dsm_up_equal_sum_down")

            # Upward shift bounded by maximum shiftable power
            self.model.add_constraints(
                self.dsm_up <= self.load_shift_maximum_power.loc[{"area": self.detailed_countries}],
                name="limited_dsm_up"
            )

            # Downward shift bounded by maximum shiftable power
            self.model.add_constraints(
                self.dsm_down.groupby(self.pairs.hh).sum() <= self.load_shift_maximum_power.loc[{"area": self.detailed_countries}],
                name="limited_dsm_down"
            )

            # Simultaneous up and down shift bounded by maximum shiftable power
            self.model.add_constraints(
                self.dsm_up + down_shift <= self.load_shift_maximum_power.loc[{"area": self.detailed_countries}],
                name='limit_simultaneous_dsm'
            )


        ###########
        ### VRE ###
        ###########

        self.vre_conversion_tech = self.vre[self.vre.isin(self.conversion_tech)]

        # VRE production at each hour equals the capacity factor given with by the vre_profiles.
        self.model.add_constraints(
            self.gene.loc[{'tech': self.vre_prod_tech, "area":self.countries}]
            - self.nominal_power.loc[{'tech': self.vre_prod_tech, "area":self.countries}] * self.vre_profiles.loc[{'tech': self.vre_prod_tech, "area":self.countries}] == 0,
            name='vre_generation_prod'
        )
        self.model.add_constraints(
            self.conv_output.loc[{'tech': self.vre_conversion_tech, "area":self.countries}]
            - self.output_power.loc[{'tech': self.vre_conversion_tech, "area":self.countries}] * self.vre_profiles.loc[{'tech': self.vre_conversion_tech, "area":self.countries}] == 0,
            name='vre_generation_conversion'
        )


        ###############
        ### STORAGE ###
        ###############

        # Charging power is lower than discharging power for storage assets
        self.model.add_constraints(self.charging_power - self.discharging_power <= 0, name="charging_lower_discharging")

        # Storage energy balance: charge (input, minus round-trip losses) minus discharge
        # (output, grossed up for losses), including reserve activation for storage techs
        # that provide FCR/FRR (detailed_countries only).
        reserve_str = self.reserve[self.reserve.isin(self.str)]
        str_not_in_reserve = self.str[~self.str.isin(self.reserve)]

        charge = self.str_input * self.efficiency_storage_in.loc[self.str]
        charge += self.phs_inflows.loc[{"area": self.countries}] * (self.str_input.data.tech == "phs")
        discharge = self.str_output / self.efficiency_storage_out.loc[self.str]

        flux_no_res = charge - discharge

        # State-of-charge recursion: SOC(h) = SOC(h-1) + net flux(h-1) (x100 for numerical
        # conditioning). Reserve-eligible storage techs in detailed_countries also lose energy
        # to FCR/FRR activation (discharge_fcr/discharge_frr); split into 3 constraints below
        # (reserve-eligible in detailed_countries / reserve-eligible elsewhere / not
        # reserve-eligible) since only detailed_countries has fcr/frr variables at all.
        if self.detailed_countries:
            discharge_fcr = self.fcr.loc[{"tech": reserve_str}] * self.reserve_activation_rate.loc['fcr'] / self.efficiency_storage_out.loc[reserve_str]
            discharge_frr = self.frr.loc[{"tech": reserve_str}] * self.reserve_activation_rate.loc['frr'] / self.efficiency_storage_out.loc[reserve_str]
            flux_res_fr = charge.loc[{"area": self.detailed_countries}] - discharge.loc[{"area": self.detailed_countries}] - discharge_fcr - discharge_frr
            self.model.add_constraints(
                self.state_of_charge.loc[{"tech": reserve_str, "area": self.detailed_countries}] * 100
                == self.state_of_charge.loc[{"tech": reserve_str, "area": self.detailed_countries}].roll(hour=1) * 100
                + flux_res_fr.loc[{"tech": reserve_str}].roll(hour=1) * 100,
                name="state_of_charge_eq_rsv"
            )
            if self.non_detailed_countries:
                self.model.add_constraints(
                    self.state_of_charge.loc[{"tech": reserve_str, "area": self.non_detailed_countries}] * 100
                    == self.state_of_charge.loc[{"tech": reserve_str, "area": self.non_detailed_countries}].roll(hour=1) * 100
                    + flux_no_res.loc[{"tech": reserve_str, "area": self.non_detailed_countries}].roll(hour=1) * 100,
                    name="state_of_charge_eq_rsv_non_fr"
                )
        else:
            self.model.add_constraints(
                self.state_of_charge.loc[{"tech": reserve_str}] * 100
                == self.state_of_charge.loc[{"tech": reserve_str}].roll(hour=1) * 100
                + flux_no_res.loc[{"tech": reserve_str}].roll(hour=1) * 100,
                name="state_of_charge_eq_rsv"
            )

        # Non-reserve-eligible storage techs (e.g. batteries in a non-detailed country): same
        # SOC recursion, no reserve-activation loss term.
        self.model.add_constraints(self.state_of_charge.loc[{"tech":str_not_in_reserve}]*100 == self.state_of_charge.loc[{"tech":str_not_in_reserve}].roll(hour=1)*100 + flux_no_res.loc[{"tech":str_not_in_reserve}].roll(hour=1)*100, name="state_of_charge_eq_no_rsv")

        # Stored energy can't exceed installed energy capacity.
        self.model.add_constraints(self.state_of_charge/1000 <= self.energy_capacity/1000, name="energy_capacity_constraint")
        # Actual charging rate can't exceed installed charging power.
        self.model.add_constraints(self.str_input <= self.charging_power, name="charging_power_constraint")


        #################
        ### BATTERIES ###
        #################

        # For each type of battery, the energy power ratio is known
        self.model.add_constraints(self.discharging_power.loc[{"tech":"battery_1h"}] == self.energy_capacity.loc[{"tech":"battery_1h"}], name="battery_1h_capacity")
        self.model.add_constraints(self.discharging_power.loc[{"tech":"battery_2h"}] == self.energy_capacity.loc[{"tech":"battery_2h"}]/2, name="battery_2h_capacity")
        self.model.add_constraints(self.discharging_power.loc[{"tech":"battery_4h"}] == self.energy_capacity.loc[{"tech":"battery_4h"}]/4, name="battery_4h_capacity")
        self.model.add_constraints(self.discharging_power.loc[{"tech":"battery_8h"}] == self.energy_capacity.loc[{"tech":"battery_8h"}]/8, name="battery_8h_capacity")
        # Batteries have symmetric charge/discharge power (single inverter rating).
        self.model.add_constraints(self.discharging_power.loc[{'tech':self.battery}] == self.charging_power.loc[{'tech':self.battery}], name="battery_charging_discharging")
        # Can't charge and discharge at the same time beyond the inverter's rated power.
        self.model.add_constraints(self.str_input.loc[{'tech':self.battery}] + self.str_output.loc[{'tech':self.battery}] <= self.discharging_power.loc[{'tech':self.battery}], name='battery_simultaneous_functioning')


        ###############
        ### METHANE ###
        ###############

        # Constraint on biomethane import in Europe
        total_biogas_import = self.gene.loc[{"tech" : "biogas_import"}].groupby(self.year).sum(dim="hour")
        # For each country, import is lower than a given potential
        self.model.add_constraints(total_biogas_import.loc[{"area": self.areas}]/1000 + self.CH4_imp_annual.loc[{"area": self.areas}] <= self.biogas_potential.loc[{"tech":"biogas_import", "area": self.areas}], name="annual_biogas_import")
        # There is also a maximum potential of imports from outside over the whole region
        self.model.add_constraints(total_biogas_import.sum(dim = "area")/1000 <= self.limit_biogas_extra_import, name = "limit_extra_biogas")

        # Constraint on gas exchange
        ## Annual CH4 imports and exports 
        self.model.add_constraints(self.CH4_exp_annual.sum(dim="area") == self.CH4_imp_annual.sum(dim="area"), name = "balance_CH4_trade")

        # Annual biomass potential constraints (all areas)
        yearly_methanization = self.gene.loc[{"tech": "methanization"}].groupby(self.year).sum(dim="hour")
        self.model.add_constraints(yearly_methanization.loc[{"area": self.areas}]/1000 <= self.biogas_potential.loc[{"tech":"methanization", "area": self.areas}], name="annual_methanization")

        yearly_pyrogazification = self.gene.loc[{"tech": "pyrogazification"}].groupby(self.year).sum(dim="hour")
        self.model.add_constraints(yearly_pyrogazification.loc[{"area": self.areas}]/1000 <= self.biogas_potential.loc[{"tech":"pyrogazification", "area": self.areas}], name="annual_pyrogazification")

        # CO2 balance for methanation — France only, deliberately NOT tied to detailed_countries.
        # CO2_industry_demand/CO2_industry_prod (constants.csv) are global scalars with no area
        # dimension, so this constraint cannot be generalised to another country without adding
        # a per-country breakdown of those two inputs first.
        # Follows v3_2 structure: CO2 fraction and usable fraction from tech_parameters
        if self.france_in_model:
            methanation_co2_fr = (self.conv_output.loc[{'tech': 'methanation', 'area': 'FR'}]
                                  .groupby(self.year).sum(dim="hour") / self.conversion_efficiency.loc["methanation"])
            methanization_co2_fr = (self.gene.loc[{'tech': 'methanization', 'area': 'FR'}]
                                    .groupby(self.year).sum(dim="hour")
                                    * self.CO2_fraction.loc["methanization"]
                                    / (1 - self.CO2_fraction.loc["methanization"])
                                    * self.CO2_usable.loc["methanization"])
            pyrogazification_co2_fr = (self.gene.loc[{'tech': 'pyrogazification', 'area': 'FR'}]
                                       .groupby(self.year).sum(dim="hour")
                                       * self.CO2_fraction.loc["pyrogazification"]
                                       / (1 - self.CO2_fraction.loc["pyrogazification"])
                                       * self.CO2_usable.loc["pyrogazification"])
            industry_co2 = self.CO2_industry_prod * self.CO2_usable_industry - self.CO2_industry_demand
            self.model.add_constraints(methanation_co2_fr <= methanization_co2_fr + pyrogazification_co2_fr + industry_co2, name="methanation_CO2")

        # Methane balance — supply and usage
        prod_tech_ch4 = self.CH4_balance[self.CH4_balance.isin(self.prod_tech)]
        conversion_tech_ch4 = self.CH4_balance[self.CH4_balance.isin(self.conversion_tech)]
        str_ch4 = self.CH4_balance[self.CH4_balance.isin(self.str)]
        conversion_tech_use_ch4 = self.use_CH4[self.use_CH4.isin(self.conversion_tech)]
        str_use_ch4 = self.use_CH4[self.use_CH4.isin(self.str)]
        reserve_use_ch4 = self.use_CH4[self.use_CH4.isin(self.reserve)]

        supply_h = (self.gene.loc[{'tech': prod_tech_ch4}].sum(dim='tech')
                    + self.conv_output.loc[{'tech': conversion_tech_ch4}].sum(dim='tech')
                    + self.str_output.loc[{'tech': str_ch4}].sum(dim='tech'))
        usage_h = (self.str_input.loc[{'tech': str_use_ch4}].sum(dim='tech')
                   + (self.conv_output.loc[{'tech': conversion_tech_use_ch4}] / self.conversion_efficiency.loc[{'tech': conversion_tech_use_ch4}]).sum(dim='tech'))
        if self.detailed_countries:
            usage_h = usage_h + (
                (self.frr.loc[{'tech': reserve_use_ch4}] / self.conversion_efficiency.loc[{'tech': reserve_use_ch4}] * self.reserve_activation_rate.loc['frr']).sum(dim='tech')
                + (self.fcr.loc[{'tech': reserve_use_ch4}] / self.conversion_efficiency.loc[{'tech': reserve_use_ch4}] * self.reserve_activation_rate.loc['fcr']).sum(dim='tech'))

        self.ch4_hourly = [c for c in self.countries if self.CH4_demand_is_profile.get(c, True)]
        self.ch4_annual = [c for c in self.countries if not self.CH4_demand_is_profile.get(c, True)]

        # Hourly CH4 adequacy (default: France) — dual used by extract_spot_price
        if self.ch4_hourly:
            self.model.add_constraints(
                supply_h.loc[{"area": self.ch4_hourly}] - usage_h.loc[{"area": self.ch4_hourly}] >= self.CH4_demand.loc[{"area": self.ch4_hourly}],
                name="CH4_adequacy"
            )

        # Annual CH4 balance — applies to all countries (hourly-profile ones get a dedicated name).
        # Import/export enter here: annual variables align with groupby(self.year).sum() results.
        if self.ch4_hourly:
            annual_supply_fr = supply_h.loc[{"area": self.ch4_hourly}].groupby(self.year).sum(dim="hour")
            annual_usage_fr  = usage_h.loc[{"area": self.ch4_hourly}].groupby(self.year).sum(dim="hour")
            annual_demand_fr = self.CH4_demand.loc[{"area": self.ch4_hourly}].groupby(self.year).sum(dim="hour")
            self.model.add_constraints(
                annual_supply_fr/100 - annual_usage_fr/100 + self.CH4_imp_annual.loc[{"area": self.ch4_hourly}]*10
                >= self.CH4_exp_annual.loc[{"area": self.ch4_hourly}]*10 + annual_demand_fr/100,
                name="CH4_adequacy_annual_fr"
            )

        # Annual CH4 adequacy (default: non-FR countries)
        # NOTE: to get non-FR CH4 spot price use model.constraints.CH4_adequacy_annual.dual
        if self.ch4_annual:
            annual_supply_ch4 = supply_h.loc[{"area": self.ch4_annual}].groupby(self.year).sum(dim="hour")
            annual_usage_ch4  = usage_h.loc[{"area": self.ch4_annual}].groupby(self.year).sum(dim="hour")
            annual_demand_ch4 = self.CH4_demand.loc[{"area": self.ch4_annual}].groupby(self.year).sum(dim="hour")
            self.model.add_constraints(
                annual_supply_ch4/100 - annual_usage_ch4/100 + self.CH4_imp_annual.loc[{"area": self.ch4_annual}]*10
                >= self.CH4_exp_annual.loc[{"area": self.ch4_annual}]*10 + annual_demand_ch4/100,
                name="CH4_adequacy_annual"
            )
    


        ################
        ### HYDROGEN ###
        ################

        # H2 salt cavern discharge/charge power bounded by fixed ratios of its energy capacity
        # (technical compressor/turbine sizing relative to cavern volume).
        self.model.add_constraints(self.discharging_power.loc[{"tech":"h2_saltcavern"}]*10 <= self.energy_capacity.loc[{"tech":"h2_saltcavern"}]*self.h2_saltcavern_discharge_to_storage_ratio*10, name="H2_saltcavern_discharge")
        self.model.add_constraints(self.charging_power.loc[{"tech":"h2_saltcavern"}]*10 <= self.energy_capacity.loc[{"tech":"h2_saltcavern"}]*self.h2_saltcavern_charge_to_storage_ratio*10, name="H2_saltcavern_charge")

        # H2_import is an exogenous import tech, capped per-country by biogas_potential for H2
        total_H2_import = self.gene.loc[{"tech" : "H2_import"}].groupby(self.year).sum(dim="hour")
        self.model.add_constraints(total_H2_import.loc[{"area": self.areas}]/1000 <= self.biogas_potential.loc[{"tech":"H2_import", "area": self.areas}], name="annual_H2_import")

        # Constraint on H2 exchange between countries over a year
        self.model.add_constraints(self.H2_exp_annual.sum(dim="area") == self.H2_imp_annual.sum(dim="area"), name = "balance_H2_trade")


        prod_tech_h2 = self.H2_balance[self.H2_balance.isin(self.prod_tech)]
        conversion_tech_h2 = self.H2_balance[self.H2_balance.isin(self.conversion_tech)]
        str_h2 = self.H2_balance[self.H2_balance.isin(self.str)]
        conversion_tech_use_h2 = self.use_H2[self.use_H2.isin(self.conversion_tech)]
        str_use_h2 = self.use_H2[self.use_H2.isin(self.str)]
        reserve_use_h2 = self.use_H2[self.use_H2.isin(self.reserve)]

        supply_h2 = (self.gene.loc[{'tech': prod_tech_h2}].sum(dim='tech')
                     + self.conv_output.loc[{'tech': conversion_tech_h2}].sum(dim='tech')
                     + self.str_output.loc[{'tech': str_h2}].sum(dim='tech'))
        usage_h2 = (self.str_input.loc[{'tech': str_use_h2}].sum(dim='tech')
                    + (self.conv_output.loc[{'tech': conversion_tech_use_h2}] / self.conversion_efficiency.loc[{'tech': conversion_tech_use_h2}]).sum(dim='tech'))
        if self.detailed_countries:
            usage_h2 = usage_h2 + (
                (self.frr.loc[{'tech': reserve_use_h2}] / self.conversion_efficiency.loc[{'tech': reserve_use_h2}] * self.reserve_activation_rate.loc['frr']).sum(dim='tech')
                + (self.fcr.loc[{'tech': reserve_use_h2}] / self.conversion_efficiency.loc[{'tech': reserve_use_h2}] * self.reserve_activation_rate.loc['fcr']).sum(dim='tech'))

        h2_hourly = [c for c in self.countries if self.H2_demand_is_profile.get(c, True)]
        h2_annual = [c for c in self.countries if not self.H2_demand_is_profile.get(c, True)]

        # Hourly H2 adequacy (default: France) — dual used by extract_spot_price
        # Trade handled in annual constraint below (mirrors CH4 structure).
        if h2_hourly:
            self.model.add_constraints(
                supply_h2.loc[{"area": h2_hourly}] - usage_h2.loc[{"area": h2_hourly}]
                >= self.H2_demand.loc[{"area": h2_hourly}],
                name="H2_adequacy"
            )
            # Annual H2 adequacy with trade for hourly-profile countries (FR).
            # Trade in TWh, hourly sums in GWh → ×10 = ×1000/100.
            annual_supply_h2_fr = supply_h2.loc[{"area": h2_hourly}].groupby(self.year).sum(dim="hour")
            annual_usage_h2_fr  = usage_h2.loc[{"area": h2_hourly}].groupby(self.year).sum(dim="hour")
            annual_demand_h2_fr = self.H2_demand.loc[{"area": h2_hourly}].groupby(self.year).sum(dim="hour")
            self.model.add_constraints(
                annual_supply_h2_fr/100 - annual_usage_h2_fr/100 + self.H2_imp_annual.loc[{"area": h2_hourly}]*10
                >= self.H2_exp_annual.loc[{"area": h2_hourly}]*10 + annual_demand_h2_fr/100,
                name="H2_adequacy_annual_fr"
            )
        # Annual H2 adequacy (default: non-FR countries)
        # NOTE: to get non-FR H2 spot price use model.constraints.H2_adequacy_annual.dual
        if h2_annual:
            annual_supply_h2 = supply_h2.loc[{"area": h2_annual}].groupby(self.year).sum(dim="hour")
            annual_usage_h2 = usage_h2.loc[{"area": h2_annual}].groupby(self.year).sum(dim="hour")
            annual_demand_h2 = self.H2_demand.loc[{"area": h2_annual}].groupby(self.year).sum(dim="hour")
            self.model.add_constraints(
                annual_supply_h2/100 - annual_usage_h2/100 + self.H2_imp_annual.loc[{"area": h2_annual}]*10
                >= self.H2_exp_annual.loc[{"area": h2_annual}]*10 + annual_demand_h2/100,
                name="H2_adequacy_annual"
            )


        ###############
        ### NUCLEAR ###
        ###############

        # Average annual generation can't exceed the planned/technical nuclear capacity factor
        # (accounts for scheduled outages/maintenance, unlike the hourly capacity bound).
        yearly_nuc = self.gene.loc[{'tech':'nuclear'}].groupby(self.year).sum(dim="hour")
        self.model.add_constraints(yearly_nuc / 8760 <= self.capacity_factor_nuclear_yearly * self.nominal_power.loc[{'tech':'nuclear'}], name="nuclear_yearly_CF")

        # Hour-to-hour ramp-rate limit (both directions); in detailed_countries, reserve
        # activation (FCR+FRR) counts towards the ramp too, since it also changes output.
        nuclear_gene = self.gene.loc[{"tech": "nuclear"}]
        ramp_limit_nuc = self.ramp_rate.loc["nuclear"] * self.nominal_power.loc[{"tech": "nuclear"}]
        if self.detailed_countries:
            nuclear_h_fr = nuclear_gene.loc[{"area": self.detailed_countries}] + self.frr.loc[{"tech": "nuclear"}] + self.fcr.loc[{"tech": "nuclear"}]
            self.model.add_constraints(nuclear_h_fr - nuclear_h_fr.roll(hour=1) - ramp_limit_nuc.loc[{"area": self.detailed_countries}] <= 0, name="ramp_up_nuclear")
            self.model.add_constraints(nuclear_h_fr.roll(hour=1) - nuclear_h_fr - ramp_limit_nuc.loc[{"area": self.detailed_countries}] <= 0, name="ramp_down_nuclear")
            if self.non_detailed_countries:
                nuclear_h_non_fr = nuclear_gene.loc[{"area": self.non_detailed_countries}]
                self.model.add_constraints(nuclear_h_non_fr - nuclear_h_non_fr.roll(hour=1) - ramp_limit_nuc.loc[{"area": self.non_detailed_countries}] <= 0, name="ramp_up_nuclear_non_fr")
                self.model.add_constraints(nuclear_h_non_fr.roll(hour=1) - nuclear_h_non_fr - ramp_limit_nuc.loc[{"area": self.non_detailed_countries}] <= 0, name="ramp_down_nuclear_non_fr")
        else:
            self.model.add_constraints(nuclear_gene - nuclear_gene.roll(hour=1) - ramp_limit_nuc <= 0, name="ramp_up_nuclear")
            self.model.add_constraints(nuclear_gene.roll(hour=1) - nuclear_gene - ramp_limit_nuc <= 0, name="ramp_down_nuclear")


        #############
        ### HYDRO ###
        #############

        # PHS charging (pumping) power bounded by a fixed ratio of discharging (turbine) power.
        self.model.add_constraints(self.charging_power.loc[{"tech":'phs'}] <= self.discharging_power.loc[{"tech":'phs'}] * self.phs_charge_to_discharge_ratio, name="phs_charging")

        # Lake reservoir stored volume can't exceed its maximum energy capacity, when defined.
        if 'lake' in self.maximum_energy_capacity.coords['tech']:
            cap = self.maximum_energy_capacity.loc[{"tech": 'lake'}]
            if not cap.isnull().all():
                self.model.add_constraints(self.lake_stored/100 <= cap/100, name='lake_storage_capacity')

        # Spillage constraint — detailed_countries only (spill data is zero for other
        # countries → constraint trivially satisfied there; restricting avoids useless LP rows).
        # NOTE: lake_spill is kept with self.countries in the outflow formula below so that
        # lake_state_of_charge covers all areas (spill=0 for non-detailed areas has no
        # mathematical effect).
        if self.detailed_countries:
            gene_lake_fr = self.gene.loc[{"tech": 'lake', "area": self.detailed_countries}]
            self.model.add_constraints(gene_lake_fr >= self.lake_spill.loc[{"area": self.detailed_countries}], name='lake_spillage')

        # Lake water balance recursion (same idea as the storage SOC above): stored volume(h)
        # = stored volume(h-1) + inflows(h-1) - outflow(h-1), where outflow = generation (net
        # of spillage) grossed up for turbine losses, plus reserve activation in
        # detailed_countries.
        outflow_base = (self.gene.loc[{"tech": 'lake'}] - self.lake_spill.loc[{"area": self.countries}]) / self.efficiency_storage_out.loc["lake"]
        if self.detailed_countries:
            outflow_fr = (outflow_base.loc[{"area": self.detailed_countries}]
                          + self.frr.loc[{"tech": 'lake'}] * self.reserve_activation_rate.loc["frr"] / self.efficiency_storage_out.loc["lake"]
                          + self.fcr.loc[{"tech": 'lake'}] * self.reserve_activation_rate.loc["fcr"] / self.efficiency_storage_out.loc["lake"])
            self.model.add_constraints(
                self.lake_stored.loc[{"area": self.detailed_countries}] == self.lake_stored.loc[{"area": self.detailed_countries}].roll(hour=1) + self.lake_inflows.loc[{"area": self.detailed_countries}].roll(hour=1) - outflow_fr.roll(hour=1),
                name='lake_state_of_charge'
            )
            if self.non_detailed_countries:
                self.model.add_constraints(
                    self.lake_stored.loc[{"area": self.non_detailed_countries}] == self.lake_stored.loc[{"area": self.non_detailed_countries}].roll(hour=1) + self.lake_inflows.loc[{"area": self.non_detailed_countries}].roll(hour=1) - outflow_base.loc[{"area": self.non_detailed_countries}].roll(hour=1),
                    name='lake_state_of_charge_non_fr'
                )
        else:
            self.model.add_constraints(
                self.lake_stored == self.lake_stored.roll(hour=1) + self.lake_inflows.loc[{"area": self.countries}].roll(hour=1) - outflow_base.roll(hour=1),
                name='lake_state_of_charge'
            )

        if self.detailed_countries:
            # Minimal stored volume — detailed_countries only
            self.model.add_constraints(self.lake_stored.loc[{"area": self.detailed_countries}]/100 >= self.lake_minimal_volume.loc[{"area": self.detailed_countries}]/100, name='lake_minimal_volume')

            # Minimal daily outflow — detailed_countries only (non-detailed outflow data is zero → useless LP rows otherwise)
            daily_lake = self.gene.loc[{'tech':'lake'}].groupby(self.day).sum(dim="hour")
            daily_lake = daily_lake.rename({daily_lake.dims[0]: "day"})
            self.model.add_constraints(daily_lake.loc[{"area": self.detailed_countries}] >= self.lake_minimal_outflow.loc[{"area": self.detailed_countries}], name="lake_minimal_outflow")


    def define_objective(self):
        """Build the total system cost objective (minimised): CAPEX annuities on new power and
        storage-energy capacity, fixed and variable O&M (including reserve activation), and a
        small DSM friction cost. `self.obj_constant` (existing-capacity annuities, invariant to
        the optimisation) is subtracted before handing the objective to linopy and added back
        in extract_optimisation_results_linopy() to report the true total cost."""
        # DSM friction cost: small value to prevent spurious load shifts at zero cost.
        # Units: M€/GWh = €/kWh. Adjust as needed. Default: 1 €/MWh = 0.001 M€/GWh.
        DSM_VOM = 0.001

        power_annuities = (((self.nominal_power.loc[{'tech': self.prod_tech}] - self.existing_capacity.loc[{'tech': self.prod_tech}])*self.annuities.loc[{'tech': self.prod_tech}]).sum(dim='tech')
                           + ((self.output_power.loc[{'tech': self.conversion_tech}] - self.existing_capacity.loc[{'tech': self.conversion_tech}])*self.annuities.loc[{'tech': self.conversion_tech}]).sum(dim='tech')
                           + ((self.discharging_power.loc[{'tech': self.str}] - self.existing_capacity.loc[{'tech': self.str}])*self.annuities.loc[{'tech': self.str}]).sum(dim='tech')) * self.nb_years

        storage_cap_annuities = (((self.energy_capacity.loc[{'tech': self.str}] - self.existing_energy_capacity.loc[{'tech': self.str}])
                                   * self.storage_annuities.loc[{'tech': self.str}]).sum(dim="tech")) * self.nb_years

        fixed_OM = ((self.nominal_power.loc[{'tech': self.prod_tech}] * self.fOM.loc[{'tech': self.prod_tech}]).sum(dim="tech")
                    + (self.output_power.loc[{'tech': self.conversion_tech}] * self.fOM.loc[{'tech': self.conversion_tech}]).sum(dim="tech")
                    + (self.discharging_power.loc[{'tech': self.str}] * self.fOM.loc[{'tech': self.str}]).sum(dim="tech")) * self.nb_years

        variable_OM = (((self.gene.loc[{'tech': self.prod_tech}] * self.vOM.loc[{'tech': self.prod_tech}]).sum(dim='hour')).sum(dim="tech")
                       + ((self.conv_output.loc[{'tech': self.conversion_tech}] * self.vOM.loc[{'tech': self.conversion_tech}]).sum(dim='hour')).sum(dim="tech")
                       + ((self.str_output.loc[{'tech': self.str}] * self.vOM.loc[{'tech': self.str}]).sum(dim='hour')).sum(dim="tech"))

        if self.detailed_countries:
            reserve_variable_OM = (((self.frr.loc[{'tech': self.reserve}] * self.reserve_activation_rate.loc['frr']
                                      + self.fcr.loc[{'tech': self.reserve}] * self.reserve_activation_rate.loc['fcr'])
                                     * self.vOM.loc[{'tech': self.reserve}]).sum(dim="tech")).sum(dim="hour")
        else:
            reserve_variable_OM = 0

        dsm_cost = self.dsm_up.sum(dim="hour").sum(dim="area") * DSM_VOM if self.detailed_countries else 0

        obj = power_annuities + storage_cap_annuities + fixed_OM + reserve_variable_OM + variable_OM + dsm_cost

        self.obj_constant = obj.const
        obj = obj - self.obj_constant

        self.model.add_objective(obj, sense="min")


    def build_model(self):
        """Convenience wrapper running the full model-building sequence in order:
        load_inputs -> define_sets -> define_variables -> define_constraints ->
        define_objective. Call solve() next."""
        self.load_inputs()
        self.define_sets()
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def solve(self, solver_name="gurobi", infeasible_value=1000):
        """Solve the LP with `solver_name` (Gurobi barrier method, no crossover — see
        solver_options below), falling back to HiGHS if the primary solver raises an exception
        (e.g. missing license). Sets `self.system_social_cost = infeasible_value` if the
        solve did not reach an optimal/acceptable-suboptimal status. Returns (status,
        termination_condition) as reported by linopy."""
        print(f"Solving EOLES model using {solver_name}")
        print("Barrier algorithm only (no crossover). Solution may not be basic.")
        solver_options = {
            "Method": 2,
            "Crossover": 0,
            "NumericFocus": 3,
            "BarConvTol": 1e-8,
            "Presolve": 2,
            "LogFile": f"{self.output_path}/logfile_{self.name}.txt"
        }

        try:
            status, termination_condition = self.model.solve(solver_name=solver_name, **solver_options)
        except Exception as e:
            print("Primary solver failed:", e)
            print("Retrying with HiGHS...")
            status, termination_condition = self.model.solve(solver_name="highs")

        if status == "ok" and termination_condition == "optimal":
            print("Optimization successful")
        elif status == "warning" and termination_condition == "other":
            print("WARNING: Optimization might be sub-optimal. Writing output anyway")
        else:
            print(f"Optimisation failed with status {status} and terminal condition {termination_condition}")
            self.system_social_cost = infeasible_value
        return status, termination_condition

    def extract_optimisation_results_linopy(self):
        """Post-process the solved model into ready-to-use results: system cost breakdown,
        hourly balances per vector, spot prices, installed capacities/generation per
        technology, annualised investment costs, per-country summary table (utils_results.
        extract_summary), profits per technology, and carbon footprint. Must run after solve().
        Populates the many `self.*` attributes consumed by run_batch.py's save_results()."""
        # obj_constant was stripped from the objective per-area (add_objective rejects any
        # nonzero constant), so its true contribution to the total cost is the sum across
        # all its remaining dims (e.g. "area") — not a broadcasted add of a scalar to an array.
        obj_constant_scalar = float(self.obj_constant.sum().values)
        self.system_cost = self.model.objective.value + obj_constant_scalar
        self.nom_power = self.nominal_power.solution
        self.all = self.model.solution

        self.system_social_cost = self.model.objective.value + obj_constant_scalar

        if self.carbon_constraint:
            self.system_technical_cost, self.emissions = get_technical_cost(self.all, self.system_social_cost, scc=0, nb_years=self.nb_years, carbon_content=self.carbon_content)
        else:
            self.system_technical_cost, self.emissions = get_technical_cost(self.all, self.system_social_cost, self.scc, self.nb_years, self.carbon_content)

        self.hourly_balance = extract_hourly_balance(self.all, self.elec_demand, self.H2_demand, self.CH4_demand,
                                                     self.conversion_efficiency, self.efficiency_storage_in, self.efficiency_storage_out,
                                                     self.load_shift_period, self.pairs,
                                                     self.prod_tech, self.conversion_tech, self.str, self.reserve)

        self.curtailment = extract_curtailment(self.hourly_balance, self.elec_balance, self.use_elec)
        self.frr_results = (self.hourly_balance[[f"{tech}_frr" for tech in self.reserve.values]].to_array("tech").sum(dim="hour")/1000)
        self.fcr_results = (self.hourly_balance[[f"{tech}_fcr" for tech in self.reserve.values]].to_array("tech").sum(dim="hour")/1000)

        self.spot_price = extract_spot_price(self.model, year_coords=self.year)
        self.carbon_value = extract_carbon_value(self.model, self.carbon_constraint, self.scc)

        self.installed_power = xr.concat(
            [self.all["nominal_power"], self.all["output_power"], self.all["discharging_power"]],
            dim="tech"
        ).dropna(dim='tech')
        self.energy_capacity = self.all["energy_capacity"].loc[{"tech": self.str}]
        self.charging_power = self.all["charging_power"].loc[{"tech": self.str}]

        self.primary_generation = extract_primary_gene(self.hourly_balance, self.nb_years, self.prod_tech)
        self.CH4_to_power_generation = extract_CH4_to_power(self.from_CH4_to_elec, self.conversion_efficiency, self.nb_years, self.hourly_balance)
        self.H2_to_power_generation = extract_H2_to_power(self.from_H2_to_elec, self.conversion_efficiency, self.nb_years, self.hourly_balance)
        self.power_to_CH4_generation = extract_power_to_CH4(self.from_elec_to_CH4, self.conversion_efficiency, self.nb_years, self.hourly_balance)
        self.power_to_H2_generation = extract_power_to_H2(self.from_elec_to_H2, self.conversion_efficiency, self.nb_years, self.hourly_balance)

        self.new_capacity_annualized_costs, self.new_energy_capacity_annualized_costs = \
            extract_annualized_costs_investment_new_capa(self.installed_power, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity,
                                                         self.annuities, self.storage_annuities, self.fOM)

        self.elec_supply, self.elec_usage = extract_balance("elec", self.elec_balance, self.use_elec, self.elec_demand, self.conversion_efficiency, self.hourly_balance)
        self.CH4_supply, self.CH4_usage = extract_balance("CH4", self.CH4_balance, self.use_CH4, self.CH4_demand, self.conversion_efficiency, self.hourly_balance, demand_is_profile=any(self.CH4_demand_is_profile.values()))
        self.H2_supply, self.H2_usage = extract_balance("H2", self.H2_balance, self.use_H2, self.H2_demand, self.conversion_efficiency, self.hourly_balance, demand_is_profile=any(self.H2_demand_is_profile.values()))

        self.summary, self.generation_per_technology = \
            extract_summary(self.system_social_cost, self.model, self.elec_demand,
                            self.H2_demand, any(self.H2_demand_is_profile.values()),
                            self.CH4_demand, any(self.CH4_demand_is_profile.values()),
                            self.installed_power, self.existing_capacity,
                            self.energy_capacity, self.existing_energy_capacity, self.annuities,
                            self.storage_annuities, self.fOM, self.vOM, self.conversion_efficiency,
                            self.scc, self.nb_years, self.carbon_constraint, self.carbon_content,
                            self.hourly_balance, self.spot_price,
                            self.all_tech, self.elec_primary_prod, self.elec_prod, self.elec_balance, self.str,
                            self.CH4_primary_prod, self.CH4_prod, self.CH4_balance,
                            self.H2_balance,
                            self.from_CH4_to_elec, self.from_H2_to_elec,
                            self.from_elec_to_CH4, self.from_elec_to_H2, self.countries)

        self.load_factor = self.generation_per_technology*1000/(self.installed_power*8760*self.nb_years)*100

        self.profits = extract_profit(self.hourly_balance, self.spot_price, self.vOM,
                                      self.new_capacity_annualized_costs, self.new_energy_capacity_annualized_costs,
                                      self.frr_requirements, self.fcr_requirement, self.reserve_activation_rate, self.conversion_efficiency,
                                      self.installed_power,
                                      self.all_tech, self.elec_balance, self.str, self.from_CH4_to_elec, self.from_H2_to_elec, self.reserve, self.vre,
                                      self.CH4_balance, self.H2_balance, self.from_elec_to_CH4, self.from_elec_to_H2)

        # calculate_lcoe_per_tech uses pandas single-node patterns (.at[tech], hourly_balance.loc[:, ...])
        # incompatible with multi-area xarray DataArrays — disabled until rewritten for multi-node.
        self.lcoe_per_tech = None

        self.footprint = extract_carbon_footprint(self, self.generation_per_technology, self.carbon_footprint, self.nb_years)
        if isinstance(self.summary, pd.DataFrame):
            for area in self.summary.columns:
                self.summary.at["footprint [MtCO2eq/yr]", area] = self.footprint.at["TOTAL"]
        else:
            self.summary.at["footprint [MtCO2eq/yr]"] = self.footprint.at["TOTAL"]

        self.new_capacity_annualized_costs_nofOM, self.new_energy_capacity_annualized_costs_nofOM = \
            extract_annualized_costs_investment_new_capa_nofOM(self.installed_power, self.energy_capacity,
                                                               self.existing_capacity, self.existing_energy_capacity,
                                                               self.annuities, self.storage_annuities)

        # Per tech (and per area, for multi-node runs) — no area pre-aggregation, so
        # OM_costs_M€yr.csv reports a genuine per-country breakdown, like the investment costs.
        gen_for_om = self.generation_per_technology * 1000  # TWh -> GWh
        self.OM_cost, self.carbon_cost = extract_OM_cost(self.installed_power, self.fOM, self.vOM, gen_for_om,
                                       self.scc, self.carbon_content,
                                       carbon_constraint=self.carbon_constraint, nb_years=self.nb_years)
