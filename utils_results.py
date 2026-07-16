"""
Post-processing of a solved ModelEOLES instance: cost accounting, hourly
balances, capacity/generation summaries, price-setter analysis, and CSV
export helpers. Pure data functions - no plotting here (see utils_plots.py).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr


def get_technical_cost(solution, objective, scc, nb_years, carbon_content):
    """Returns technical cost (social cost without CO2 emissions-related cost)"""
    gene_ngas = solution.gene.loc[{"tech": "natural_gas"}].sum(dim="hour")  # GWh
    gene_ncoal = solution.gene.loc[{"tech": "coal"}].sum(dim="hour")  # GWh
    total_emissions_gas = gene_ngas * carbon_content.loc['natural_gas'] / 1000
    total_emissions_coal = gene_ncoal * carbon_content.loc['coal'] / 1000
    emissions = xr.concat(
        [total_emissions_gas, total_emissions_coal],
        dim="tech"
    ).assign_coords(tech=["natural_gas", "coal"])
    total_emissions = emissions.sum("tech")
    technical_cost = objective - total_emissions * scc

    return technical_cost, total_emissions


def extract_carbon_footprint(model_eoles, gene_per_tech, carbon_footprint, nb_years):  # MtCO2eq/yr
    """Calculates the yearly and per energy carbon footprint of each technology.
    Now works with linopy/xarray and requires the ModelEOLES instance (not model) to access technology sets."""
    footprint = pd.Series(dtype=float)
    
    # Extract tech names from the xarray DataArrays
    all_tech_names = model_eoles.all_tech.values
    ch4_balance_biogas = model_eoles.CH4_balance_biogas.values if hasattr(model_eoles, 'CH4_balance_biogas') else []
    elec_balance = model_eoles.elec_balance.values if hasattr(model_eoles, 'elec_balance') else []
    elec_primary_prod = model_eoles.elec_primary_prod.values if hasattr(model_eoles, 'elec_primary_prod') else []
    
    # Convert gene_per_tech to Series — collapse area dim first if multi-node
    if isinstance(gene_per_tech, xr.DataArray) and "area" in gene_per_tech.dims:
        gene_per_tech = gene_per_tech.sum("area")
    gene_per_tech_series = gene_per_tech.to_pandas() if hasattr(gene_per_tech, 'to_pandas') else gene_per_tech

    # Convert carbon_footprint to Series if it's a DataArray
    if hasattr(carbon_footprint, 'to_pandas'):
        carbon_footprint = carbon_footprint.to_pandas()
    
    # Calculate footprint for each technology
    for tech in all_tech_names:
        if tech not in ch4_balance_biogas:
            if tech in gene_per_tech_series.index and tech in carbon_footprint.index:
                footprint.at[tech] = gene_per_tech_series.at[tech] * carbon_footprint.at[tech] / nb_years
            else:
                footprint.at[tech] = 0.0
    
    # Add biogas footprint
    biogas_gen = sum(gene_per_tech_series.at[tech] for tech in ch4_balance_biogas if tech in gene_per_tech_series.index)
    footprint.at["biogas"] = biogas_gen * carbon_footprint.at["biogas"] / nb_years if "biogas" in carbon_footprint.index else 0.0
    
    # Add network footprint
    elec_gen = sum(gene_per_tech_series.at[tech] for tech in elec_balance if tech in gene_per_tech_series.index)
    footprint.at["elec_network"] = elec_gen * carbon_footprint.at["network"] / nb_years if "network" in carbon_footprint.index else 0.0
    
    elec_primary_gen = sum(gene_per_tech_series.at[tech] for tech in elec_primary_prod if tech in gene_per_tech_series.index)
    footprint.at["elec_network_primarygene"] = elec_primary_gen * carbon_footprint.at["network"] / nb_years if "network" in carbon_footprint.index else 0.0

    footprint.at["TOTAL"] = footprint.sum() - footprint.at["elec_network_primarygene"]

    return footprint


def extract_hourly_balance(solution, elec_demand, H2_demand, CH4_demand, conversion_efficiency, eta_in, eta_out, load_shift_period, pairs,
                           prod_tech, conversion_tech, str, reserve):
    """Extracts hourly defined data, including demand, generation and storage
    Returns a dataset with hourly generation for each hour.
    Using this function, you limit the number of times model output values are extracted, which is costly in computational power.
    You can then manipulate xarray Datasets which are much faster."""

    hourly_balance = xr.Dataset()
    hourly_balance["elec_demand"] = elec_demand
    hourly_balance["H2_demand"] = H2_demand
    hourly_balance["CH4_demand"] = CH4_demand
    # DSM is France-only in v3_5; if FR is absent (restricted_area run), variables
    # don't exist in the solution — fall back to zeros for all areas.
    all_areas = elec_demand.coords["area"]
    if 'dsm_up' in solution and 'dsm_down' in solution:
        load_shift_up = solution.dsm_up.reindex(area=all_areas, fill_value=0)
        load_shift_down = (solution.dsm_down.groupby(pairs.hh).sum()
                           .rename({"hh": "hour"})
                           .reindex(area=all_areas, fill_value=0))
    else:
        zeros = xr.DataArray(0.0, coords={"area": all_areas, "hour": elec_demand.coords["hour"]}, dims=["area", "hour"])
        load_shift_up = zeros
        load_shift_down = zeros
    hourly_balance["load_shift_up"] = load_shift_up
    hourly_balance["load_shift_down"] = load_shift_down
    _shift_raw = elec_demand + load_shift_up - load_shift_down
    hourly_balance["elec_demand_w_shift"] = _shift_raw.where(_shift_raw.notnull(), elec_demand)
    for tech in prod_tech.values:
        hourly_balance[tech] = solution.gene.loc[{"tech": tech}]  # GW
    for tech in conversion_tech.values:
        hourly_balance[tech] = solution.conv_output.loc[{"tech": tech}]
        # We add technologies which include a conversion parameter, to express their hourly generation in GWh of the input vector
        hourly_balance[f'{tech}_input'] = solution.conv_output.loc[{"tech": tech}] / conversion_efficiency.loc[{"tech": tech}]
    for tech in str.values:
        hourly_balance[tech] = solution.str_output.loc[{"tech": tech}]
        hourly_balance[f'{tech}_input'] = solution.str_input.loc[{"tech": tech}]
        hourly_balance[f'{tech}_state_charge'] = solution.state_of_charge.loc[{"tech": tech}]
    # FCR/FRR are France-only in v3_7; if FR is absent (restricted_area run), variables
    # don't exist in the solution — fall back to zeros for all areas.
    if 'fcr' in solution and 'frr' in solution:
        for tech in reserve.values:
            hourly_balance[f'{tech}_fcr'] = solution.fcr.loc[{"tech": tech}]
            hourly_balance[f'{tech}_frr'] = solution.frr.loc[{"tech": tech}]
    else:
        zeros_reserve = xr.DataArray(0.0, coords={"area": all_areas, "hour": elec_demand.coords["hour"]}, dims=["area", "hour"])
        for tech in reserve.values:
            hourly_balance[f'{tech}_fcr'] = zeros_reserve
            hourly_balance[f'{tech}_frr'] = zeros_reserve
    hourly_balance["lake_state_charge"] = solution.lake_stored
    hourly_balance["storage_input_losses"] = (hourly_balance[[f'{tech}_input' for tech in str.values]].to_array("tech").assign_coords(tech=str.values) * (1 - eta_in)).sum("tech")
    hourly_balance["storage_output_losses"] = (hourly_balance[[tech for tech in str.values]].to_array("tech").assign_coords(tech=str.values) * (1 / eta_out - 1)).sum("tech")

    # Interconnection flows: sum over counterpart dimension to get net per area per hour
    if 'import' in solution and 'export' in solution:
        hourly_balance["imports"] = solution['import'].sum("area_bis")  # GW, dims [area, hour]
        hourly_balance["exports"] = solution['export'].sum("area_bis")  # GW, dims [area, hour]

    return hourly_balance


def extract_curtailment(hourly_balance, elec_balance, use_elec):
    """Calculates and returns hourly curtailed electricity.
    Also adds it to hourly_balance"""

    curtailment = xr.Dataset()
    elec_supply = sum(hourly_balance[tech] for tech in elec_balance.values)
    elec_usage = sum(hourly_balance[tech + "_input"] for tech in use_elec.values)
    curtailment = elec_supply + hourly_balance["imports"] - elec_usage - hourly_balance["exports"] - hourly_balance["elec_demand_w_shift"]

    hourly_balance["curtailment"] = curtailment
    mask = np.isclose(curtailment, 0, atol=1e-9)
    curtailment = curtailment.where(~mask, 0)
    return curtailment


def extract_spot_price(model, year_coords=None):
    """Extracts spot prices in €/MWh for all areas.

    Unit conversion: objective is in M€, constraints in GWh → dual in M€/GWh = €/kWh.
    Multiply by 1e3 to get €/MWh.

    Multi-node specifics (v3_5+):
    - Electricity: 'adequacy' covers FR; 'adequacy_non_fr' covers other areas (both hourly,
      same unit). Merged along 'area'.
    - CH4/H2: 'CH4_adequacy'/'H2_adequacy' cover FR (hourly, unscaled → × 1e3).
      'CH4_adequacy_annual'/'H2_adequacy_annual' cover non-FR (annual, /100-scaled →
      dual × 10 = /100 × 1e3). Annual duals are broadcast from [area, year] to [area, hour]
      via year_coords (DataArray with dim='hour' containing the year for each hour).
    """
    spot_price = xr.Dataset()

    # --- Electricity ---
    # 'adequacy' exists only when FR is in the model; fall back to adequacy_non_fr otherwise
    elec_parts = []
    if hasattr(model.constraints, 'adequacy'):
        elec_parts.append(model.constraints.adequacy.dual * 1e3)
    if hasattr(model.constraints, 'adequacy_non_fr'):
        elec_parts.append(model.constraints.adequacy_non_fr.dual * 1e3)
    if len(elec_parts) > 1:
        spot_price["elec"] = xr.concat(elec_parts, dim="area")
    elif len(elec_parts) == 1:
        spot_price["elec"] = elec_parts[0]
    else:
        spot_price["elec"] = xr.DataArray(0.0)

    # --- CH4 ---
    # FR uses the annual trade constraint (CH4_adequacy_annual_fr) so that its price
    # is computed on the same basis as non-FR countries (annual dual, same /100 scaling).
    # The hourly adequacy dual (CH4_adequacy) is near-zero for zero-vOM technologies
    # like methanization and is not comparable across countries.
    ch4_parts = []
    if hasattr(model.constraints, 'CH4_adequacy_annual_fr') and year_coords is not None:
        # /100-scaled → multiply by 10; broadcast [area, year] → [area, hour]
        ch4_parts.append((model.constraints.CH4_adequacy_annual_fr.dual * 10
                          ).sel(year=year_coords).drop_vars("year"))
    elif hasattr(model.constraints, 'CH4_adequacy'):
        # fallback for runs where CH4_adequacy_annual_fr is absent
        ch4_parts.append(model.constraints.CH4_adequacy.dual * 1e3)
    if hasattr(model.constraints, 'CH4_adequacy_annual') and year_coords is not None:
        ch4_annual_hourly = (model.constraints.CH4_adequacy_annual.dual * 10
                             ).sel(year=year_coords).drop_vars("year")
        ch4_parts.append(ch4_annual_hourly)
    if len(ch4_parts) > 1:
        spot_price["CH4"] = xr.concat(ch4_parts, dim="area")
    elif len(ch4_parts) == 1:
        spot_price["CH4"] = ch4_parts[0]
    else:
        spot_price["CH4"] = model.constraints.CH4_adequacy.dual * 1e3  # fallback

    # --- H2 ---
    # Same logic as CH4: use H2_adequacy_annual_fr for FR (annual dual, /100 scaling).
    h2_parts = []
    if hasattr(model.constraints, 'H2_adequacy_annual_fr') and year_coords is not None:
        h2_parts.append((model.constraints.H2_adequacy_annual_fr.dual * 10
                         ).sel(year=year_coords).drop_vars("year"))
    elif hasattr(model.constraints, 'H2_adequacy'):
        h2_parts.append(model.constraints.H2_adequacy.dual * 1e3)
    if hasattr(model.constraints, 'H2_adequacy_annual') and year_coords is not None:
        h2_annual_hourly = (model.constraints.H2_adequacy_annual.dual * 10
                            ).sel(year=year_coords).drop_vars("year")
        h2_parts.append(h2_annual_hourly)
    if len(h2_parts) > 1:
        spot_price["H2"] = xr.concat(h2_parts, dim="area")
    elif len(h2_parts) == 1:
        spot_price["H2"] = h2_parts[0]
    else:
        spot_price["H2"] = model.constraints.H2_adequacy.dual * 1e3  # fallback

    # fcr_provision/frr_provision only exist when FR is in the model (restricted_area runs without FR don't have them)
    # No negation here, matching the elec/CH4/H2 convention above (dual used directly): both
    # fcr_provision ("fcr_sum == fcr_requirement") and frr_provision ("frr_sum - res_req ==
    # load_req") are adequacy-style equality constraints structurally analogous to "adequacy"
    # ("balance - shift >= elec_demand") — providing reserve shares the same capacity as
    # providing energy (see reserve_power_prod/_conversion/_storage: gene+fcr+frr<=capacity),
    # so at the margin the reserve price should track the electricity price with a *positive*
    # correlation (same opportunity cost). An earlier version negated the dual here, which
    # produced a near-perfect *negative* correlation with spot_price["elec"] instead (verified
    # empirically: corrcoef ~= -0.98) — the mirror image of the expected sign, confirming the
    # negation was a bug rather than a deliberate correction.
    # fcr_provision ("fcr_sum == fcr_requirement * int(include_reserve)") carries no scaling
    # factor, so no correction is needed beyond the standard GWh->MWh *1e3 conversion.
    if hasattr(model.constraints, 'fcr_provision'):
        spot_price["fcr"] = model.constraints.fcr_provision.dual * 1e3
    else:
        spot_price["fcr"] = xr.DataArray(0.0)
    # frr_provision is written as "(frr_sum - res_req)*10 == load_req*10" — a uniform *10
    # scaling of the whole constraint for numerical conditioning (see define_constraints'
    # docstring). Per LP sensitivity: if the solver sees RHS_solver = 10*load_req, the reported
    # dual is d(Obj)/d(RHS_solver), and d(Obj)/d(load_req) = dual * 10 (chain rule) — so the raw
    # dual must be multiplied by 10 to recover the true per-unit price, on top of the standard
    # *1e3 unit conversion. This mirrors CH4/H2's annual adequacy constraints, which apply the
    # same net correction (there via a /100 scaling on demand: dual*10 = dual/100*1e3). Missing
    # this factor was verified empirically: a technology's revenue-minus-cost-minus-annuity
    # balanced to ~0 only once the FRR cost term was scaled by this same *10.
    if hasattr(model.constraints, 'frr_provision'):
        spot_price["frr"] = model.constraints.frr_provision.dual * 10 * 1e3
    else:
        spot_price["frr"] = xr.DataArray(0.0)
    return spot_price


def extract_carbon_value(model, carbon_constraint, scc):
    """Extracts the social value of carbon in the considered model. Corresponds to the given SCC or to the shadow price of the carbon constraint if the latter is used.

    The carbon_budget constraint is written as "yearly_carbon/10 <= carbon_budget*100"
    (see ModelEOLES.define_constraints — numerical-conditioning scaling, per-country
    carbon_budget in MtCO2/yr). Since carbon_budget's own coefficient in the constraint is
    *100 (i.e. the solver's RHS is 100x the true parameter), the raw dual must be multiplied
    by 100 to recover the true marginal value per unit of carbon_budget — same reasoning as
    frr_provision's *10 correction in extract_spot_price. M€/MtCO2 = €/tCO2 directly, no
    further unit conversion needed.
    """
    if carbon_constraint:
        carbon_value = model.constraints.carbon_budget.dual[0] * 100
    else:
        carbon_value = scc
    return carbon_value


def extract_primary_gene(hourly_balance, nb_years, prod_tech):
    """Extracts yearly primary energy generation per source of energy in TWh"""
    primary_generation = (hourly_balance[[tech for tech in prod_tech.values]]
                          .to_array("tech")
                          .sum("hour") / 1000 / nb_years)  # TWh

    return primary_generation


def extract_CH4_to_power(from_CH4_to_elec, conversion_efficiency, nb_years, hourly_balance):
    """Extracts CH4 used to produce electricity in TWh"""
    gas_to_power_input = (hourly_balance[[f"{tech}_input" for tech in from_CH4_to_elec.values]]
                          .to_array("tech")
                          .sum("hour") / 1000 / nb_years)  # TWh

    return gas_to_power_input


def extract_H2_to_power(from_H2_to_elec, conversion_efficiency, nb_years, hourly_balance):
    """Extracts H2 used to produce electricity in TWh"""
    H2_to_power_input = (hourly_balance[[f"{tech}_input" for tech in from_H2_to_elec.values]]
                          .to_array("tech")
                          .sum("hour") / 1000 / nb_years)  # TWh

    return H2_to_power_input


def extract_power_to_CH4(from_elec_to_CH4, conversion_efficiency, nb_years, hourly_balance):
    """Extracts electricity generation necessary to produce CH4 in TWh"""
    power_to_CH4_input = (hourly_balance[[f"{tech}_input" for tech in from_elec_to_CH4.values]]
                          .to_array("tech")
                          .sum("hour") / 1000 / nb_years)  # TWh

    return power_to_CH4_input


def extract_power_to_H2(from_elec_to_H2, conversion_efficiency, nb_years, hourly_balance):
    """Extracts electricity used to produce H2 in TWh"""
    power_to_H2_input = (hourly_balance[[f"{tech}_input" for tech in from_elec_to_H2.values]]
                          .to_array("tech")
                          .sum("hour") / 1000 / nb_years)  # TWh

    return power_to_H2_input


def extract_balance(vector, vector_balance, use_vector, demand, conversion_efficiency, hourly_balance, demand_is_profile=True):
    """Extracts total supply and usage (including demand) of the given vector (elec, CH4 or H2) in TWh
    :param vector: string
        Should be one of : 'elec', 'electricity', 'methane', 'CH4', 'hydrogen', 'H2'
    :param vector_balance: dict of production techs for this vector
    :param use_vector: dict of consumption techs for this vector
    :param demand: xr.DataArray
    :param conversion_efficiency: xr.DataArray
    :param hourly_balance: xr.Dataset
    :param demand_is_profile: bool
    :return: tuple (supply, usage) in TWh
    """

    if vector == "elec" or vector == "electricity" or vector == "methane" or vector == "CH4" or vector == "hydrogen" or vector == "H2":
        supply_list = vector_balance
        usage_list = use_vector
    else:
        raise ValueError(f"{vector} is not a recognized vector")

    supply = (hourly_balance[[tech for tech in supply_list.values]]
              .to_array("tech")
              .sum("hour") / 1000)  # TWh

    usage = (hourly_balance[[f"{tech}_input" for tech in usage_list.values]]
              .to_array("tech")
              .sum("hour") / 1000)

    if demand_is_profile:
        d = demand.sum("hour") / 1000  # TWh
    else:
        d = demand / 1000  # TWh
    d = d.assign_coords(tech="demand")
    usage = xr.concat([usage, d], dim="tech")

    if vector == "elec" or vector == "electricity":
        c = hourly_balance["curtailment"].sum("hour") / 1000
        c = c.assign_coords(tech="curtailment")
        usage = xr.concat([usage, c], dim="tech")

    return supply, usage


def extract_storage_losses(elec_str_losses, CH4_str_losses, H2_str_losses):
    str_losses = pd.concat([elec_str_losses, CH4_str_losses, H2_str_losses])
    str_losses.loc[np.isclose(str_losses, 0)] = 0
    str_losses.at["TOTAL"] = str_losses.sum()
    return str_losses


def extract_annualized_costs_investment_new_capa(capacities, energy_capacities, existing_capacities, existing_energy_capacities,
                                                 annuities, storage_annuities, fOM):
    """
    Returns the annualized costs coming from newly invested capacities and energy capacities. This includes annualized CAPEX + fOM.
    Unit: 1e6€/yr
    """
    new_capacity = capacities - existing_capacities
    annualized_costs = new_capacity * (annuities + fOM)

    new_storage_capacity = energy_capacities - existing_energy_capacities
    annualized_storage_costs = new_storage_capacity * storage_annuities

    return annualized_costs, annualized_storage_costs


def extract_annualized_costs_investment_new_capa_nofOM(capacities, energy_capacities, existing_capacities, existing_energy_capacities,
                                                       annuities, storage_annuities):
    """
    Returns the annualized investment coming from newly invested capacities and energy capacities, without fOM. Unit: 1e6€/yr
    Works with both xr.DataArray (multi-node) and pandas Series (single-node).
    """
    if isinstance(capacities, xr.DataArray):
        new_capacity = xr.where(capacities >= existing_capacities, capacities - existing_capacities, 0.0)
        annualized_costs = new_capacity * annuities
        new_storage_capacity = xr.where(energy_capacities >= existing_energy_capacities,
                                        energy_capacities - existing_energy_capacities, 0.0)
        annualized_storage_costs = new_storage_capacity * storage_annuities
        return annualized_costs, annualized_storage_costs
    else:
        # Legacy pandas path (single-node)
        new_capacity = capacities - existing_capacities
        costs_new_capacity = pd.concat([new_capacity, annuities], axis=1, ignore_index=True).rename(
            columns={0: "new_capacity", 1: "annuities"})
        costs_new_capacity = costs_new_capacity.dropna()
        costs_new_capacity.loc[:, "annualized_costs"] = costs_new_capacity.loc[:, "new_capacity"] * costs_new_capacity.loc[:, "annuities"]
        new_storage_capacity = energy_capacities - existing_energy_capacities
        costs_new_energy_capacity = pd.concat([new_storage_capacity, storage_annuities], axis=1, ignore_index=True).rename(
            columns={0: "new_capacity", 1: "storage_annuities"})
        costs_new_energy_capacity = costs_new_energy_capacity.dropna()
        costs_new_energy_capacity.loc[:, "annualized_costs"] = costs_new_energy_capacity.loc[:, "new_capacity"] * costs_new_energy_capacity.loc[:, "storage_annuities"]
        return costs_new_capacity.loc[:, "annualized_costs"], costs_new_energy_capacity.loc[:, "annualized_costs"]


def update_vom_costs_scc(vOM_init, scc, emission_rate):
    """Add emission cost related to social cost of carbon to fossil vectors vOM costs.
    :param vOM_init: float
        Initial vOM in M€/GW = €/kW
    :param scc: int
        €/tCO2
    :param emission_rate: float
        tCO2/MWh.

    Returns
    vOM in M€/GW(h)  = €/kW(h)
    """
    return vOM_init + scc * emission_rate / 1000


def extract_OM_cost(capacities, fOM, vOM, generation, scc, carbon_content, carbon_constraint=True, nb_years=1):
    """Returns operation and maintenance costs (fOM + vOM), per technology — and per area for
    multi-node runs, since capacities/generation carry an "area" dim while fOM/vOM/carbon_content
    (tech-only) broadcast across it. Unit: 1e6€/yr.

    IMPORTANT REMARK: generation is divided by nb_years to get the average yearly generation.

    Returns
    -------
    OM_cost : xr.DataArray, dims ["tech"] or ["tech", "area"]
    carbon_cost : xr.DataArray or None
        Only returned (non-None) when carbon_constraint is False: the SCC-driven share of
        vOM, summed over "tech" (kept per "area" if present).
    """
    gen_per_yr = generation / nb_years

    if not carbon_constraint:  # ie optimization with a given social cost of carbon
        # we remove the SCC in this vOM
        vOM_no_scc = vOM.copy()
        vOM_no_scc.loc["natural_gas"] = update_vom_costs_scc(vOM_no_scc.loc["natural_gas"], scc=(-scc),
                                                              emission_rate=carbon_content.loc["natural_gas"])  # €/kWh
        vOM_no_scc.loc["coal"] = update_vom_costs_scc(vOM_no_scc.loc["coal"], scc=(-scc),
                                                      emission_rate=carbon_content.loc["coal"])  # €/kWh

        # variable cost only due to actual scc, not anticipated scc
        vOM_scc_only = vOM - vOM_no_scc

        OM_cost = capacities * fOM + gen_per_yr * vOM_no_scc
        carbon_cost_per_tech = gen_per_yr * vOM_scc_only
        carbon_cost = carbon_cost_per_tech.sum("tech") if hasattr(carbon_cost_per_tech, "sum") else carbon_cost_per_tech.sum()
        return OM_cost, carbon_cost

    OM_cost = capacities * fOM + gen_per_yr * vOM
    return OM_cost, None


def compute_costs(annuities, fOM, vOM, storage_annuities, gene_per_tech, capacity, existing_capacity,
                energy_capacity, existing_energy_capacity, nb_years, elec_balance, storage_techs, CH4_balance, H2_balance,
                per_area=False):
    # Total cost (not LCOE of new investments only):
    #   CAPEX annuity on new capacity, fOM on total capacity, vOM on total generation.
    # By default, aggregate across areas before computing (multi-node and single-node compatible).
    # per_area=True skips that aggregation, so the returned costs keep the 'area' dimension
    # (used for genuine per-country LCOE instead of a single system-wide value).
    def _sum_area(da):
        if per_area:
            return da
        return da.sum("area") if isinstance(da, xr.DataArray) and "area" in da.dims else da

    gene_per_tech     = _sum_area(gene_per_tech)
    capacity          = _sum_area(capacity)
    existing_capacity = _sum_area(existing_capacity)
    energy_capacity          = _sum_area(energy_capacity)
    existing_energy_capacity = _sum_area(existing_energy_capacity)

    g = gene_per_tech.sel(tech=elec_balance)
    c = capacity.sel(tech=elec_balance)
    e = existing_capacity.sel(tech=elec_balance)

    costs_elec = ((c - e) * annuities.sel(tech=elec_balance) * nb_years
                  + c * fOM.sel(tech=elec_balance) * nb_years
                  + g * 1000 * vOM.sel(tech=elec_balance)).sum("tech")
    sel_res = g.tech[g.tech.isin(storage_techs)]
    costs_elec += ((energy_capacity.sel(tech=sel_res) - existing_energy_capacity.sel(tech=sel_res)) * storage_annuities.sel(tech=sel_res) * nb_years).sum("tech")

    g = gene_per_tech.sel(tech=CH4_balance)
    c = capacity.sel(tech=CH4_balance)
    e = existing_capacity.sel(tech=CH4_balance)

    costs_CH4 = ((c - e) * annuities.sel(tech=CH4_balance) * nb_years
                 + c * fOM.sel(tech=CH4_balance) * nb_years
                 + g * 1000 * vOM.sel(tech=CH4_balance)).sum("tech")
    sel_res = g.tech[g.tech.isin(storage_techs)]
    costs_CH4 += ((energy_capacity.sel(tech=sel_res) - existing_energy_capacity.sel(tech=sel_res)) * storage_annuities.sel(tech=sel_res) * nb_years).sum("tech")

    g = gene_per_tech.sel(tech=H2_balance)
    c = capacity.sel(tech=H2_balance)
    e = existing_capacity.sel(tech=H2_balance)

    costs_H2 = ((c - e) * annuities.sel(tech=H2_balance) * nb_years
                + c * fOM.sel(tech=H2_balance) * nb_years
                + g * 1000 * vOM.sel(tech=H2_balance)).sum("tech")
    sel_res = g.tech[g.tech.isin(storage_techs)]
    costs_H2 += ((energy_capacity.sel(tech=sel_res) - existing_energy_capacity.sel(tech=sel_res)) * storage_annuities.sel(tech=sel_res) * nb_years).sum("tech")

    return costs_elec, costs_CH4, costs_H2


def compute_lcoe(costs_elec, costs_CH4, costs_H2, G2P_bought, P2G_CH4_bought, P2G_H2_bought, sumgene_elec, sumgene_CH4, sumgene_H2):
    """Compute LCOE by using the costs of buying electricity / CH4 / H2. Parameters sumgene_elec, sumgene_CH4 and
    sumgene_H2 refer to the total production from each system (which can be used either to satisfy final demand, or for
     vector coupling). Inputs may be plain scalars (system-wide) or xr.DataArray carrying an 'area' dimension
     (per-country); in the latter case the division is broadcast per country. Returns NaN (rather than a sentinel
     string) wherever the corresponding production is zero, so results stay usable as plain numbers per area."""
    lcoe_elec = xr.where(sumgene_elec != 0, (costs_elec + G2P_bought) / xr.where(sumgene_elec != 0, sumgene_elec, 1.0), np.nan)
    lcoe_CH4 = xr.where(sumgene_CH4 != 0, (costs_CH4 + P2G_CH4_bought) / xr.where(sumgene_CH4 != 0, sumgene_CH4, 1.0), np.nan)
    lcoe_H2 = xr.where(sumgene_H2 != 0, (costs_H2 + P2G_H2_bought) / xr.where(sumgene_H2 != 0, sumgene_H2, 1.0), np.nan)
    return lcoe_elec, lcoe_CH4, lcoe_H2


def compute_lcoe_volumetric(gene_per_tech, conversion_efficiency, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot,
                            from_CH4_to_elec, from_H2_to_elec, from_elec_to_H2, from_elec_to_CH4):
    """Computes a volumetric LCOE, where costs of each system (respectively, electricity, methane and hydrogen) are distributed across the different systems based on volumes (eg, volume of demand versus volume of gas used for the electricity system).
    Inputs may carry an 'area' dimension (per-country); .sum("tech") only collapses the tech axis so 'area', if present, is preserved throughout and the result is a per-country LCOE."""
    gene_from_CH4_to_elec = (gene_per_tech.loc[from_CH4_to_elec] / conversion_efficiency.loc[from_CH4_to_elec]).sum("tech")  # TWh
    gene_from_H2_to_elec = (gene_per_tech.loc[from_H2_to_elec] / conversion_efficiency.loc[from_H2_to_elec]).sum("tech")  # TWh
    gene_from_elec_to_CH4 = (gene_per_tech.loc[from_elec_to_CH4] / conversion_efficiency.loc[from_elec_to_CH4]).sum("tech")  # TWh
    gene_from_elec_to_H2 = (gene_per_tech.loc[from_elec_to_H2] / conversion_efficiency.loc[from_elec_to_H2]).sum("tech")  # TWh

    costs_CH4_to_demand = costs_CH4 * CH4_demand_tot / (CH4_demand_tot + gene_from_CH4_to_elec)  # 1e6 €
    costs_CH4_to_elec = costs_CH4 * gene_from_CH4_to_elec / (CH4_demand_tot + gene_from_CH4_to_elec)
    costs_H2_to_demand = costs_H2 * H2_demand_tot / (H2_demand_tot + gene_from_H2_to_elec)
    costs_H2_to_elec = costs_H2 * gene_from_H2_to_elec / (H2_demand_tot + gene_from_H2_to_elec)
    costs_elec_to_demand = costs_elec * elec_demand_tot / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_CH4 = costs_elec * gene_from_elec_to_CH4 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_H2 = costs_elec * gene_from_elec_to_H2 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)

    lcoe_elec_volume = xr.where(elec_demand_tot != 0, (costs_CH4_to_elec + costs_H2_to_elec + costs_elec_to_demand) / xr.where(elec_demand_tot != 0, elec_demand_tot, 1.0), np.nan)  # € / MWh
    lcoe_CH4_volume = xr.where(CH4_demand_tot != 0, (costs_elec_to_CH4 + costs_CH4_to_demand) / xr.where(CH4_demand_tot != 0, CH4_demand_tot, 1.0), np.nan)  # € / MWh
    lcoe_H2_volume = xr.where(H2_demand_tot != 0, (costs_elec_to_H2 + costs_H2_to_demand) / xr.where(H2_demand_tot != 0, H2_demand_tot, 1.0), np.nan)  # € / MWh
    return lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume


def compute_lcoe_value(hourly_balance, costs_elec, costs_CH4, costs_H2, elec_demand_tot, CH4_demand_tot, H2_demand_tot,
                       elec_demand, CH4_demand, H2_demand, spot_price,
                       from_CH4_to_elec, from_H2_to_elec, from_elec_to_CH4, from_elec_to_H2):

    # .sum(["tech", "hour"]) only collapses those two axes, so 'area' (if present) is preserved
    # throughout and the result is a per-country value (works for both single- and multi-node)
    gene_from_CH4_to_elec_value = (hourly_balance[[f'{tech}_input' for tech in from_CH4_to_elec.values]].to_array("tech") * spot_price["CH4"]).sum(["tech", "hour"])
    gene_from_H2_to_elec_value = (hourly_balance[[f'{tech}_input' for tech in from_H2_to_elec.values]].to_array("tech") * spot_price["H2"]).sum(["tech", "hour"])
    gene_from_elec_to_CH4_value = (hourly_balance[[f'{tech}_input' for tech in from_elec_to_CH4.values]].to_array("tech") * spot_price["elec"]).sum(["tech", "hour"])
    gene_from_elec_to_H2_value = (hourly_balance[[f'{tech}_input' for tech in from_elec_to_H2.values]].to_array("tech") * spot_price["elec"]).sum(["tech", "hour"])

    elec_demand_tot_value = (elec_demand * spot_price["elec"]).sum("hour")
    CH4_demand_tot_value = (CH4_demand * spot_price["CH4"]).sum("hour")
    H2_demand_tot_value = (H2_demand * spot_price["H2"]).sum("hour")

    # 1e6 €
    costs_CH4_to_demand_value = costs_CH4 * CH4_demand_tot_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    costs_CH4_to_elec_value = costs_CH4 * gene_from_CH4_to_elec_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    costs_H2_to_demand_value = costs_H2 * H2_demand_tot_value / (
            H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_H2_to_elec_value = costs_H2 * gene_from_H2_to_elec_value / (
            H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_elec_to_demand_value = costs_elec * elec_demand_tot_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_CH4_value = costs_elec * gene_from_elec_to_CH4_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_H2_value = costs_elec * gene_from_elec_to_H2_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)

    lcoe_elec_value = xr.where(elec_demand_tot != 0, (costs_CH4_to_elec_value + costs_H2_to_elec_value + costs_elec_to_demand_value) / xr.where(elec_demand_tot != 0, elec_demand_tot, 1.0), np.nan)  # € / MWh
    lcoe_CH4_value = xr.where(CH4_demand_tot != 0, (costs_elec_to_CH4_value + costs_CH4_to_demand_value) / xr.where(CH4_demand_tot != 0, CH4_demand_tot, 1.0), np.nan)  # € / MWh
    lcoe_H2_value = xr.where(H2_demand_tot != 0, (costs_elec_to_H2_value + costs_H2_to_demand_value) / xr.where(H2_demand_tot != 0, H2_demand_tot, 1.0), np.nan)  # € / MWh
    return lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value


def extract_profit(hourly_balance, spot_price, vOM, new_annuities, new_str_annuities, frr_requirements, fcr_requirement, reserve_activation_rate, conversion_efficiency, capacity,
                   all_tech, elec_balance, str, from_CH4_to_elec, from_H2_to_elec, reserve, vre,
                   CH4_balance, H2_balance, from_elec_to_CH4, from_elec_to_H2):
    """Extracts profit collected by each tech. This profit should be null except for rare goods (ie for techs with a limiting potential), so this output
    is used to identify issues"""

    # Helper: select by tech dim — works whether da has (tech,) or (area, tech) shape.
    # .loc[DataArray] indexes the first dim, which may be 'area' in multi-node → KeyError.
    # .sel(tech=...) is always correct.
    def _st(da, tech_arr):
        if isinstance(da, xr.DataArray) and "tech" in da.dims:
            return da.sel(tech=tech_arr)
        return da.loc[tech_arr]  # fallback for pandas Series (single-node)

    # Helper: select f"{tech}_{suffix}" columns from hourly_balance and relabel the resulting
    # "tech" dim with the BASE tech names. Dataset.to_array("tech") sets the new dim's
    # coordinate to the literal (suffixed) variable names ("phs_input", not "phs") — left as
    # is, any later arithmetic against a base-tech-indexed array (vOM, annuities,
    # conversion_efficiency...) or reindex(tech=all_tech) silently aligns on ZERO overlapping
    # labels, collapsing to an empty array that reindex/fillna(0.0) turns into all-zeros. This
    # was previously the case here (and was the root cause of storage charging costs, CH4/H2
    # conversion fuel costs, and reserve revenue/cost all being silently dropped from profits).
    def _hb(suffix, tech_arr):
        da = hourly_balance[[f"{tech}_{suffix}" for tech in tech_arr.values]].to_array("tech")
        return da.assign_coords(tech=tech_arr.values)

    sp_frr = spot_price.get("frr", 0)
    sp_fcr = spot_price.get("fcr", 0)

    # Profits for tech in elec_balance
    profits_el = (hourly_balance[[tech for tech in elec_balance.values]].to_array("tech") * (spot_price["elec"] / 1000 - vOM.loc[elec_balance])).sum("hour") - _st(new_annuities, elec_balance)

    elec_str = elec_balance[elec_balance.isin(str)]
    profits_str = -(_hb("input", elec_str) * spot_price["elec"] / 1000).sum("hour") - _st(new_str_annuities, elec_str)

    profits_CH4_el = -(_hb("input", from_CH4_to_elec) * spot_price["CH4"] / 1000).sum("hour")

    profits_H2_el = -(_hb("input", from_H2_to_elec) * spot_price["H2"] / 1000).sum("hour")

    elec_res = elec_balance[elec_balance.isin(reserve)]
    profits_reserve = (_hb("frr", elec_res) * sp_frr / 1000).sum("hour") \
                    - (_hb("frr", elec_res) * reserve_activation_rate.loc["frr"] * vOM.loc[elec_res] / 1000).sum("hour") \
                    + (_hb("fcr", elec_res) * sp_fcr / 1000).sum("hour") \
                    - (_hb("fcr", elec_res) * reserve_activation_rate.loc["fcr"] * vOM.loc[elec_res] / 1000).sum("hour")

    elec_res_CH4 = elec_res[elec_res.isin(from_CH4_to_elec)]
    elec_res_H2 = elec_res[elec_res.isin(from_H2_to_elec)]
    profits_reserve_CH4 = -(_hb("frr", elec_res_CH4) / conversion_efficiency.loc[elec_res_CH4] * reserve_activation_rate.loc["frr"] * spot_price["CH4"] / 1000).sum("hour") \
                        - (_hb("fcr", elec_res_CH4) / conversion_efficiency.loc[elec_res_CH4] * reserve_activation_rate.loc["fcr"] * spot_price["CH4"] / 1000).sum("hour")
    profits_reserve_H2 = -(_hb("frr", elec_res_H2) / conversion_efficiency.loc[elec_res_H2] * reserve_activation_rate.loc["frr"] * spot_price["H2"] / 1000).sum("hour") \
                        - (_hb("fcr", elec_res_H2) / conversion_efficiency.loc[elec_res_H2] * reserve_activation_rate.loc["fcr"] * spot_price["H2"] / 1000).sum("hour")

    elec_vre = elec_balance[elec_balance.isin(vre)]
    # sp_frr only carries an "hour" dim when FR (and thus frr_provision) is in the model;
    # otherwise it's a scalar 0.0 (int/float default, or the xr.DataArray(0.0) fallback from
    # extract_spot_price), and frr_requirements/capacity are never hourly, so there's no
    # "hour" dim anywhere in the product to sum over.
    if not (isinstance(sp_frr, xr.DataArray) and "hour" in sp_frr.dims):
        profits_vre = xr.DataArray(0.0)
    else:
        profits_vre = -(frr_requirements.loc[elec_vre] * _st(capacity, elec_vre) * sp_frr / 1000).sum("hour")

    # Profits for tech in CH4_balance
    profits_CH4 = (hourly_balance[[tech for tech in CH4_balance.values]].to_array("tech") * (spot_price["CH4"] / 1000 - vOM.loc[CH4_balance])).sum("hour") - _st(new_annuities, CH4_balance)
    CH4_str = CH4_balance[CH4_balance.isin(str)]
    profits_str_CH4 = -(_hb("input", CH4_str) * spot_price["CH4"] / 1000).sum("hour") - _st(new_str_annuities, CH4_str)
    profits_el_CH4 = -(_hb("input", from_elec_to_CH4) * spot_price["elec"] / 1000).sum("hour")

    # Profits for tech in H2 balance
    profits_H2 = (hourly_balance[[tech for tech in H2_balance.values]].to_array("tech") * (spot_price["H2"] / 1000 - vOM.loc[H2_balance])).sum("hour") - _st(new_annuities, H2_balance)
    H2_str = H2_balance[H2_balance.isin(str)]
    profits_str_H2 = -(_hb("input", H2_str) * spot_price["H2"] / 1000).sum("hour") - _st(new_str_annuities, H2_str)
    profits_el_H2 = -(_hb("input", from_elec_to_H2) * spot_price["elec"] / 1000).sum("hour")

    profits = profits_el.reindex(tech=all_tech).fillna(0.0) \
             + profits_str.reindex(tech=all_tech).fillna(0.0) \
             + profits_CH4_el.reindex(tech=all_tech).fillna(0.0) + profits_H2_el.reindex(tech=all_tech).fillna(0.0) \
             + profits_reserve.reindex(tech=all_tech).fillna(0.0) + profits_reserve_CH4.reindex(tech=all_tech).fillna(0.0) + profits_reserve_H2.reindex(tech=all_tech).fillna(0.0) \
             + profits_vre.reindex(tech=all_tech).fillna(0.0) \
             + profits_CH4.reindex(tech=all_tech).fillna(0.0) + profits_str_CH4.reindex(tech=all_tech).fillna(0.0) + profits_el_CH4.reindex(tech=all_tech).fillna(0.0) \
             + profits_H2.reindex(tech=all_tech).fillna(0.0) + profits_str_H2.reindex(tech=all_tech).fillna(0.0) + profits_el_H2.reindex(tech=all_tech).fillna(0.0)

    profits = profits.where(~np.isclose(profits, 0, atol=5e-4), 0)

    return profits


def extract_summary(objective, model, elec_demand, H2_demand, H2_demand_is_profile, CH4_demand, CH4_demand_is_profile, capacity, existing_capacity,
                    energy_capacity, existing_energy_capacity, annuities,
                    storage_annuities, fOM, vOM, conversion_efficiency,
                    scc, nb_years, carbon_constraint, carbon_content, hourly_balance, spot_price,
                    all_tech, elec_primary_prod, elec_prod, elec_balance, storage_techs,
                    CH4_primary_prod, CH4_prod, CH4_balance,
                    H2_balance,
                    from_CH4_to_elec, from_H2_to_elec, from_elec_to_CH4, from_elec_to_H2,
                    countries):
    """This function compiles different general statistics of the electricity mix by area/country, including in particular LCOE.
    Returns a DataFrame indexed by area if multiple areas, or a Series if single area."""

    # Get area dimension from one of the inputs
    areas = list(countries)
    is_multi_area = len(areas) > 1
    
    # Initialize output as dict of Series, one per area
    summary_dict = {}
    
    for area in areas:
        summary = pd.Series(dtype=float)  # final dictionary for output
        
        # Objective cost - handle scalar extraction
        obj_val = objective.values if hasattr(objective, 'values') else objective
        obj_scalar = float(np.atleast_1d(obj_val)[0]) if hasattr(obj_val, '__iter__') else float(obj_val)
        summary.at["total system cost [1e9€]"] = obj_scalar / 1000

        # Total demands - keep area dimension
        if "area" in elec_demand.dims:
            elec_dem = elec_demand.loc[{"area": area}]
        else:
            elec_dem = elec_demand
        elec_demand_tot = float(elec_dem.sum("hour").values / 1000)  # electricity demand in TWh
        summary.at["elec_demand_tot [TWh]"] = elec_demand_tot

        # H2_demand/CH4_demand are always stored at hourly resolution regardless of the
        # is_profile flag (which only controls how the adequacy constraint aggregates them),
        # so the total here must always sum over "hour" when that dim is present.
        if isinstance(H2_demand, xr.DataArray):
            h2_dem = H2_demand.loc[{"area": area}] if "area" in H2_demand.dims else H2_demand
            H2_demand_tot = float(h2_dem.sum("hour").values / 1000) if "hour" in h2_dem.dims else float(h2_dem.values / 1000)
        else:
            H2_demand_tot = float(H2_demand / 1000)  # H2 demand in TWh
        summary.at["H2_demand_tot [TWh]"] = H2_demand_tot

        if isinstance(CH4_demand, xr.DataArray):
            ch4_dem = CH4_demand.loc[{"area": area}] if "area" in CH4_demand.dims else CH4_demand
            CH4_demand_tot = float(ch4_dem.sum("hour").values / 1000) if "hour" in ch4_dem.dims else float(ch4_dem.values / 1000)
        else:
            CH4_demand_tot = float(CH4_demand / 1000)  # CH4 demand in TWh
        summary.at["CH4_demand_tot [TWh]"] = CH4_demand_tot
        
        # Prices weighted by hourly demand (ie average price paid by consumers)
        elec_dem = elec_demand.loc[{"area": area}] if "area" in elec_demand.dims else elec_demand
        spot_elec = spot_price["elec"].loc[{"area": area}] if "area" in spot_price["elec"].dims else spot_price["elec"]
        if elec_demand_tot != 0:
            weighted_elec_price_demand = (spot_elec * elec_dem).sum("hour") / (elec_demand_tot * 1e3)
            summary.at["elec_price_weighted_by_demand [€/MWh]"] = float(weighted_elec_price_demand.values)
        else:
            summary.at["elec_price_weighted_by_demand [€/MWh]"] = 0

        ch4_dem = CH4_demand.loc[{"area": area}] if "area" in CH4_demand.dims and CH4_demand_is_profile else CH4_demand
        spot_ch4 = spot_price["CH4"].loc[{"area": area}] if "area" in spot_price["CH4"].dims else spot_price["CH4"]
        if CH4_demand_tot != 0 and CH4_demand_is_profile:
            weighted_CH4_price_demand = (spot_ch4 * ch4_dem).sum("hour") / (CH4_demand_tot * 1e3)
            summary.at["CH4_price_weighted_by_demand [€/MWh]"] = float(weighted_CH4_price_demand.values)
        else:
            summary.at["CH4_price_weighted_by_demand [€/MWh]"] = 0

        h2_dem = H2_demand.loc[{"area": area}] if "area" in H2_demand.dims and H2_demand_is_profile else H2_demand
        spot_h2 = spot_price["H2"].loc[{"area": area}] if "area" in spot_price["H2"].dims else spot_price["H2"]
        if H2_demand_tot != 0 and H2_demand_is_profile:
            weighted_H2_price_demand = (spot_h2 * h2_dem).sum("hour") / (H2_demand_tot * 1e3)
            summary.at["H2_price_weighted_by_demand [€/MWh]"] = float(weighted_H2_price_demand.values)
        else:
            summary.at["H2_price_weighted_by_demand [€/MWh]"] = 0

        # Load shifting - area-specific
        hb_area = hourly_balance.loc[{"area": area}] if "area" in hourly_balance.dims else hourly_balance
        load_shift_val = float(hb_area["load_shift_up"].sum("hour").values / 1000)
        summary.at["load_shifted [TWh]"] = load_shift_val
        summary.at["load_shifted [%]"] = (load_shift_val / elec_demand_tot * 100) if elec_demand_tot > 0 else 0

        # Energy generated by technology for this area
        techs_in_balance = [tech for tech in all_tech.values if tech in hb_area.data_vars]
        if techs_in_balance:
            gene_per_tech_area = (hb_area[techs_in_balance].to_array("tech").sum("hour") / 1000)
            gene_per_tech_area = gene_per_tech_area.where(~np.isclose(gene_per_tech_area, 0, atol=5e-4), 0)
        else:
            gene_per_tech_area = xr.DataArray([], dims="tech")

        # Per-country attributed cost (CAPEX annuity + fOM + vOM + storage CAPEX)
        cap_a    = capacity.sel(area=area)    if isinstance(capacity,                xr.DataArray) and "area" in capacity.dims                else capacity
        exist_a  = existing_capacity.sel(area=area) if isinstance(existing_capacity, xr.DataArray) and "area" in existing_capacity.dims      else existing_capacity
        ecap_a   = energy_capacity.sel(area=area)   if isinstance(energy_capacity,   xr.DataArray) and "area" in energy_capacity.dims        else energy_capacity
        eexist_a = existing_energy_capacity.sel(area=area) if isinstance(existing_energy_capacity, xr.DataArray) and "area" in existing_energy_capacity.dims else existing_energy_capacity
        new_cap_a  = xr.where(cap_a  >= exist_a,  cap_a  - exist_a,  0.0)
        new_ecap_a = xr.where(ecap_a >= eexist_a, ecap_a - eexist_a, 0.0)
        _ct      = cap_a.tech.values
        _ann     = annuities.reindex(tech=_ct, fill_value=0.0)
        _fom     = fOM.reindex(tech=_ct, fill_value=0.0)
        _vom     = vOM.reindex(tech=_ct, fill_value=0.0)
        capex_m  = float((new_cap_a * _ann).sum("tech").values) * nb_years
        fom_m    = float((cap_a * _fom).sum("tech").values) * nb_years
        vom_m    = float((gene_per_tech_area.reindex(tech=_ct, fill_value=0.0) * 1000 * _vom).sum("tech").values) if gene_per_tech_area.size > 0 else 0.0
        _str_ann = storage_annuities.reindex(tech=ecap_a.tech.values, fill_value=0.0)
        str_m    = float((new_ecap_a * _str_ann).sum("tech").values) * nb_years
        if "imports" in hb_area.data_vars and "exports" in hb_area.data_vars and spot_price is not None:
            sp_a = spot_price["elec"].loc[{"area": area}] if "area" in spot_price["elec"].dims else spot_price["elec"]
            trade_M = float(((hb_area["imports"] - hb_area["exports"]) * sp_a / 1000).sum("hour").values)
        else:
            trade_M = 0.0
        summary.at["attributed_cost [1e9€]"] = (capex_m + fom_m + vom_m + str_m + trade_M) / 1000

        # Electricity generation metrics
        elec_primary = [t for t in elec_primary_prod.values if t in gene_per_tech_area.tech.values]
        if elec_primary:
            primary_gene_elec = gene_per_tech_area.loc[elec_primary].sum().item() if gene_per_tech_area.size > 0 else 0
            summary.at["primary_gene_elec [TWh]"] = primary_gene_elec
        else:
            summary.at["primary_gene_elec [TWh]"] = 0

        elec_prod_list = [t for t in elec_prod.values if t in gene_per_tech_area.tech.values]
        if elec_prod_list:
            gene_elec = gene_per_tech_area.loc[elec_prod_list].sum().item() if gene_per_tech_area.size > 0 else 0
            summary.at["gene_elec [TWh]"] = gene_elec
        else:
            summary.at["gene_elec [TWh]"] = 0

        elec_bal = [t for t in elec_balance.values if t in hb_area.data_vars]
        if elec_bal and capacity is not None:
            cap_area = capacity.loc[{"area": area}] if "area" in capacity.dims else capacity
            exist_cap_area = existing_capacity.loc[{"area": area}] if "area" in existing_capacity.dims else existing_capacity
            g_elec = gene_per_tech_area.sel(tech=elec_bal) if any(t in gene_per_tech_area.tech.values for t in elec_bal) else xr.DataArray(0)
            c = cap_area.sel(tech=elec_bal) if any(t in cap_area.tech.values for t in elec_bal) else xr.DataArray(0)
            e = exist_cap_area.sel(tech=elec_bal) if any(t in exist_cap_area.tech.values for t in elec_bal) else xr.DataArray(0)
            gene_elec_new = (g_elec * (c - e) / c).where(c != 0, 0).sum("tech") if g_elec.size > 0 else 0
        else:
            gene_elec_new = 0
        
        curtail_val = float(hb_area["curtailment"].sum("hour").values / 1000) if "curtailment" in hb_area.data_vars else 0
        summary.at["gene_curtailed [TWh]"] = curtail_val
        summary.at["gene_curtailed [%]"] = (curtail_val / summary.at["primary_gene_elec [TWh]"] * 100) if summary.at["primary_gene_elec [TWh]"] > 0 else 0

        # Interconnection flows — total per area over the simulation period
        if "imports" in hb_area.data_vars:
            summary.at["imports [TWh/yr]"] = float(hb_area["imports"].sum("hour").values / 1000 / nb_years)
            summary.at["exports [TWh/yr]"] = float(hb_area["exports"].sum("hour").values / 1000 / nb_years)
            summary.at["net_imports [TWh/yr]"] = summary.at["imports [TWh/yr]"] - summary.at["exports [TWh/yr]"]
        else:
            summary.at["imports [TWh/yr]"] = 0.0
            summary.at["exports [TWh/yr]"] = 0.0
            summary.at["net_imports [TWh/yr]"] = 0.0

        # Lost load (unserved electricity demand) — capped near-zero by the lost_load_limit
        # constraint, used as a feasibility safety valve / infeasibility diagnostic.
        if "lost_load" in hb_area.data_vars:
            lost_load_val = float(hb_area["lost_load"].sum("hour").values / 1000 / nb_years)
        else:
            lost_load_val = 0.0
        summary.at["lost_load [TWh/yr]"] = lost_load_val
        summary.at["lost_load [% of demand]"] = (lost_load_val / elec_demand_tot * 100) if elec_demand_tot > 0 else 0.0

        # CH4 generation metrics
        ch4_prim = [t for t in CH4_primary_prod.values if t in gene_per_tech_area.tech.values]
        if ch4_prim:
            primary_gene_CH4 = gene_per_tech_area.loc[ch4_prim].sum().item() if gene_per_tech_area.size > 0 else 0
            summary.at["primary_gene_CH4 [TWh]"] = primary_gene_CH4
        else:
            summary.at["primary_gene_CH4 [TWh]"] = 0

        ch4_prod_list = [t for t in CH4_prod.values if t in gene_per_tech_area.tech.values]
        if ch4_prod_list:
            gene_CH4 = gene_per_tech_area.loc[ch4_prod_list].sum().item() if gene_per_tech_area.size > 0 else 0
            summary.at["gene_CH4 [TWh]"] = gene_CH4
        else:
            summary.at["gene_CH4 [TWh]"] = 0

        ch4_bal = [t for t in CH4_balance.values if t in hb_area.data_vars]
        if ch4_bal:
            gene_CH4_new = 0  # simplified, would need capacity calc similar to elec
        else:
            gene_CH4_new = 0

        # H2 generation metrics  
        h2_bal_list = [t for t in H2_balance.values if t in gene_per_tech_area.tech.values]
        if h2_bal_list:
            gene_H2 = gene_per_tech_area.loc[h2_bal_list].sum().item() if gene_per_tech_area.size > 0 else 0
            summary.at["gene_H2 [TWh]"] = gene_H2
        else:
            summary.at["gene_H2 [TWh]"] = 0

        gene_H2_new = 0  # simplified

        # Biogas import
        if "biogas_import" in hb_area.data_vars:
            biogas_val = float(hb_area["biogas_import"].sum("hour").values / 1000 / nb_years)
        else:
            biogas_val = 0.0
        summary.at["biogas_import [TWh/yr]"] = biogas_val

        # H2 import (exogenous import-facility tech, capped per area by biogas_potential["H2_import"])
        if "H2_import" in hb_area.data_vars:
            h2_import_val = float(hb_area["H2_import"].sum("hour").values / 1000 / nb_years)
        else:
            h2_import_val = 0.0
        summary.at["H2_import [TWh/yr]"] = h2_import_val

        # CH4 inter-country trade: CH4_imp_annual/CH4_exp_annual are annual variables
        # (coords [area, year]), not part of hourly_balance — read directly from the solution.
        # linopy stores variables under the string passed to add_variables(name=...), which
        # here is "gas annual import"/"gas annual export" rather than the python attribute name.
        ch4_imp_key, ch4_exp_key = "gas annual import", "gas annual export"
        if ch4_imp_key in model.solution and ch4_exp_key in model.solution:
            ch4_imp_a = model.solution[ch4_imp_key]
            ch4_exp_a = model.solution[ch4_exp_key]
            ch4_imp_a = ch4_imp_a.sel(area=area) if "area" in ch4_imp_a.dims else ch4_imp_a
            ch4_exp_a = ch4_exp_a.sel(area=area) if "area" in ch4_exp_a.dims else ch4_exp_a
            ch4_import_val = float(ch4_imp_a.sum().values) / nb_years
            ch4_export_val = float(ch4_exp_a.sum().values) / nb_years
        else:
            ch4_import_val = 0.0
            ch4_export_val = 0.0
        summary.at["CH4_imports [TWh/yr]"] = ch4_import_val
        summary.at["CH4_exports [TWh/yr]"] = ch4_export_val
        summary.at["CH4_net_imports [TWh/yr]"] = ch4_import_val - ch4_export_val

        # Vector conversion costs (monetary value) - area specific
        if "area" in hourly_balance.dims:
            hb_conv = hourly_balance.loc[{"area": area}]
            sp_elec = spot_price["elec"].loc[{"area": area}] if "area" in spot_price["elec"].dims else spot_price["elec"]
            sp_ch4 = spot_price["CH4"].loc[{"area": area}] if "area" in spot_price["CH4"].dims else spot_price["CH4"]
            sp_h2 = spot_price["H2"].loc[{"area": area}] if "area" in spot_price["H2"].dims else spot_price["H2"]
        else:
            hb_conv = hourly_balance
            sp_elec = spot_price["elec"]
            sp_ch4 = spot_price["CH4"]
            sp_h2 = spot_price["H2"]
        
        # G2P (gas to power) costs
        ch4_to_elec_techs = [f"{tech}_input" for tech in from_CH4_to_elec.values if f"{tech}_input" in hb_conv.data_vars]
        h2_to_elec_techs = [f"{tech}_input" for tech in from_H2_to_elec.values if f"{tech}_input" in hb_conv.data_vars]
        if ch4_to_elec_techs:
            G2P_CH4_bought = float((sp_ch4 * hb_conv[ch4_to_elec_techs].to_array("tech").sum("tech")).sum("hour").values / 1e3)
        else:
            G2P_CH4_bought = 0.0
        if h2_to_elec_techs:
            G2P_H2_bought = float((sp_h2 * hb_conv[h2_to_elec_techs].to_array("tech").sum("tech")).sum("hour").values / 1e3)
        else:
            G2P_H2_bought = 0.0
        G2P_bought = G2P_CH4_bought + G2P_H2_bought
        
        # P2G (power to gas) costs
        elec_to_ch4_techs = [f"{tech}_input" for tech in from_elec_to_CH4.values if f"{tech}_input" in hb_conv.data_vars]
        elec_to_h2_techs = [f"{tech}_input" for tech in from_elec_to_H2.values if f"{tech}_input" in hb_conv.data_vars]
        if elec_to_ch4_techs:
            P2G_CH4_bought = float((sp_elec * hb_conv[elec_to_ch4_techs].to_array("tech").sum("tech")).sum("hour").values / 1e3)
        else:
            P2G_CH4_bought = 0.0
        if elec_to_h2_techs:
            P2G_H2_bought = float((sp_elec * hb_conv[elec_to_h2_techs].to_array("tech").sum("tech")).sum("hour").values / 1e3)
        else:
            P2G_H2_bought = 0.0

        # Store LCOE values to add after loop (need global gene_per_tech and costs)
        summary.at["lcoe_elec [€/MWh]"] = None  # Will be calculated globally
        summary.at["lcoe_CH4 [€/MWh]"] = None
        summary.at["lcoe_H2 [€/MWh]"] = None
        summary.at["lcoe_elec_volume [€/MWh]"] = None
        summary.at["lcoe_CH4_volume [€/MWh]"] = None
        summary.at["lcoe_H2_volume [€/MWh]"] = None
        summary.at["lcoe_elec_value [€/MWh]"] = None
        summary.at["lcoe_CH4_value [€/MWh]"] = None
        summary.at["lcoe_H2_value [€/MWh]"] = None
        
        if not carbon_constraint:
            summary.at["lcoe_CH4_noSCC [€/MWh]"] = None
            summary.at["lcoe_CH4_volume_noSCC [€/MWh]"] = None

        summary.at["transport_and_distrib_lcoe [€/yr/MWh]"] = None

        # Store computed values for this area
        summary_dict[area] = summary

    # Calculate per-country LCOE metrics
    # - These are computed once, vectorized across all areas at once (not re-looped per area)
    # - gene_per_tech_global keeps the 'area' dimension (only "hour" is summed away), despite its name,
    #   which just reflects that it's computed once for the whole extract_summary call rather than per-area.

    # hourly_balance/elec_demand/CH4_demand/H2_demand are built from the full area-indexed input
    # files, so their 'area' dim can be wider than the countries actually modelled in this run
    # (e.g. a restricted_area=["FR","DE"] run still carries all countries from links.csv in these
    # inputs). capacity/energy_capacity etc. (the solved variables) only cover the modelled
    # countries, so any quantity computed straight from hourly_balance without first going through
    # an arithmetic op against one of those (which auto-aligns via inner join) must be explicitly
    # restricted to `areas` here, or it will silently carry phantom extra countries and cause an
    # xarray AlignmentError once combined with costs_elec/CH4/H2 below.
    # Beyond just the country set, solution-derived arrays (capacity, energy_capacity...) come
    # back from linopy/xarray internal merges sorted alphabetically by area, while
    # hourly_balance/spot_price/demand keep the links.csv order — same countries, different
    # order. xr.where's strict ("exact") alignment treats that as a mismatch, so every per-area
    # array feeding compute_lcoe* must be explicitly sorted the same way, not just restricted.
    def _restrict_area(obj):
        return obj.sel(area=areas).sortby("area") if "area" in getattr(obj, "dims", ()) else obj

    def _sort_area(obj):
        return obj.sortby("area") if "area" in getattr(obj, "dims", ()) else obj

    hb_g = _restrict_area(hourly_balance)
    sp_g = {k: _restrict_area(v) for k, v in spot_price.items()}
    elec_demand_g = _restrict_area(elec_demand)
    CH4_demand_g = _restrict_area(CH4_demand)
    H2_demand_g = _restrict_area(H2_demand)

    # Generate per technology (per area)
    techs_in_balance_all = [tech for tech in all_tech.values if tech in hb_g.data_vars]
    if techs_in_balance_all:
        gene_per_tech_global = (hb_g[techs_in_balance_all].to_array("tech").sum("hour") / 1000)
        gene_per_tech_global = gene_per_tech_global.where(~np.isclose(gene_per_tech_global, 0, atol=5e-4), 0)
    else:
        gene_per_tech_global = xr.DataArray([], dims="tech")

    # Calculate costs for LCOE computation - per country (keeps the 'area' dimension throughout)
    costs_elec, costs_CH4, costs_H2 = compute_costs(annuities, fOM, vOM, storage_annuities,
                                                    gene_per_tech_global, capacity, existing_capacity,
                                                    energy_capacity, existing_energy_capacity, nb_years,
                                                    elec_balance, storage_techs, CH4_balance, H2_balance,
                                                    per_area=True)
    costs_elec, costs_CH4, costs_H2 = _sort_area(costs_elec), _sort_area(costs_CH4), _sort_area(costs_H2)

    # Calculate vector conversion costs - per country (sum only over "hour", keep "area")
    ch4_to_elec_techs_global = [f"{tech}_input" for tech in from_CH4_to_elec.values if f"{tech}_input" in hb_g.data_vars]
    h2_to_elec_techs_global = [f"{tech}_input" for tech in from_H2_to_elec.values if f"{tech}_input" in hb_g.data_vars]
    elec_to_ch4_techs_global = [f"{tech}_input" for tech in from_elec_to_CH4.values if f"{tech}_input" in hb_g.data_vars]
    elec_to_h2_techs_global = [f"{tech}_input" for tech in from_elec_to_H2.values if f"{tech}_input" in hb_g.data_vars]

    if ch4_to_elec_techs_global:
        G2P_CH4_bought_global = (sp_g["CH4"] * hb_g[ch4_to_elec_techs_global].to_array("tech").sum("tech")).sum("hour") / 1e3
    else:
        G2P_CH4_bought_global = 0.0
    if h2_to_elec_techs_global:
        G2P_H2_bought_global = (sp_g["H2"] * hb_g[h2_to_elec_techs_global].to_array("tech").sum("tech")).sum("hour") / 1e3
    else:
        G2P_H2_bought_global = 0.0
    G2P_bought_global = G2P_CH4_bought_global + G2P_H2_bought_global

    if elec_to_ch4_techs_global:
        P2G_CH4_bought_global = (sp_g["elec"] * hb_g[elec_to_ch4_techs_global].to_array("tech").sum("tech")).sum("hour") / 1e3
    else:
        P2G_CH4_bought_global = 0.0
    if elec_to_h2_techs_global:
        P2G_H2_bought_global = (sp_g["elec"] * hb_g[elec_to_h2_techs_global].to_array("tech").sum("tech")).sum("hour") / 1e3
    else:
        P2G_H2_bought_global = 0.0

    # Calculate energy generation per sector, per country (sum only over "tech", keep "area")
    elec_primary = [t for t in elec_primary_prod.values if t in gene_per_tech_global.tech.values]
    sumgene_elec = gene_per_tech_global.sel(tech=elec_primary).sum("tech") if elec_primary and gene_per_tech_global.size > 0 else 0.0

    ch4_prod_global = [t for t in CH4_prod.values if t in gene_per_tech_global.tech.values]
    sumgene_CH4 = gene_per_tech_global.sel(tech=ch4_prod_global).sum("tech") if ch4_prod_global and gene_per_tech_global.size > 0 else 0.0

    h2_prod_global = [t for t in H2_balance.values if t in gene_per_tech_global.tech.values]
    sumgene_H2 = gene_per_tech_global.sel(tech=h2_prod_global).sum("tech") if h2_prod_global and gene_per_tech_global.size > 0 else 0.0

    # Compute per-country LCOEs
    lcoe_elec, lcoe_CH4, lcoe_H2 = compute_lcoe(costs_elec, costs_CH4, costs_H2,
                                                 G2P_bought_global, P2G_CH4_bought_global, P2G_H2_bought_global,
                                                 sumgene_elec, sumgene_CH4, sumgene_H2)

    # Per-country demand totals (keep "area" if present) — used by volumetric/value LCOE.
    # H2_demand/CH4_demand are always stored at hourly resolution regardless of the
    # is_profile flag, so always sum over "hour" when that dim is present (see extract_summary above).
    elec_dem_tot = elec_demand_g.sum("hour") / 1000 if hasattr(elec_demand_g, 'sum') else float(elec_demand_g) / 1000
    if isinstance(H2_demand_g, xr.DataArray):
        h2_dem_tot = H2_demand_g.sum("hour") / 1000 if "hour" in H2_demand_g.dims else H2_demand_g / 1000
    else:
        h2_dem_tot = float(H2_demand_g) / 1000
    if isinstance(CH4_demand_g, xr.DataArray):
        ch4_dem_tot = CH4_demand_g.sum("hour") / 1000 if "hour" in CH4_demand_g.dims else CH4_demand_g / 1000
    else:
        ch4_dem_tot = float(CH4_demand_g) / 1000

    # Volumetric LCOE, per country
    lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume = compute_lcoe_volumetric(
        gene_per_tech_global, conversion_efficiency, costs_elec, costs_CH4, costs_H2,
        elec_dem_tot, ch4_dem_tot, h2_dem_tot,
        from_CH4_to_elec, from_H2_to_elec, from_elec_to_H2, from_elec_to_CH4)

    # Value-weighted LCOE, per country
    lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value = compute_lcoe_value(
        hb_g, costs_elec, costs_CH4, costs_H2,
        elec_dem_tot, ch4_dem_tot, h2_dem_tot,
        elec_demand_g, CH4_demand_g, H2_demand_g, sp_g,
        from_CH4_to_elec, from_H2_to_elec, from_elec_to_CH4, from_elec_to_H2)

    # Optional: calculate noSCC versions if no carbon constraint
    if not carbon_constraint:
        vOM_noSCC = vOM.copy()
        vOM_noSCC.loc["natural_gas"] = update_vom_costs_scc(vOM_noSCC.loc["natural_gas"], scc=(-scc), emission_rate=carbon_content.loc['natural_gas'])
        vOM_noSCC.loc["coal"] = update_vom_costs_scc(vOM_noSCC.loc["coal"], scc=(-scc), emission_rate=carbon_content.loc['coal'])

        costs_CH4_noSCC, _, _ = compute_costs(annuities, fOM, vOM_noSCC, storage_annuities,
                                              gene_per_tech_global, capacity, existing_capacity,
                                              energy_capacity, existing_energy_capacity, nb_years,
                                              elec_balance, str, CH4_balance, H2_balance,
                                              per_area=True)
        costs_CH4_noSCC = _sort_area(costs_CH4_noSCC)

        lcoe_CH4_noSCC, _, _ = compute_lcoe(costs_elec, costs_CH4_noSCC, costs_H2,
                                            G2P_bought_global, P2G_CH4_bought_global, P2G_H2_bought_global,
                                            sumgene_elec, sumgene_CH4, sumgene_H2)

        lcoe_CH4_volume_noSCC, _, _ = compute_lcoe_volumetric(
            gene_per_tech_global, conversion_efficiency, costs_elec, costs_CH4_noSCC, costs_H2,
            elec_dem_tot, ch4_dem_tot, h2_dem_tot,
            from_CH4_to_elec, from_H2_to_elec, from_elec_to_H2, from_elec_to_CH4)

    # Extract a per-area scalar from a value that may be a per-country xr.DataArray (with an
    # "area" dim), a scalar xr.DataArray (single-node), or a plain float.
    def _area_scalar(da, a):
        if isinstance(da, xr.DataArray) and "area" in da.dims:
            return float(da.sel(area=a).values)
        try:
            return float(da)
        except (TypeError, ValueError):
            return np.nan

    # Fill in per-country LCOE values
    for area in areas:
        if area in summary_dict:
            summary_dict[area].at["lcoe_elec [€/MWh]"] = _area_scalar(lcoe_elec, area)
            summary_dict[area].at["lcoe_CH4 [€/MWh]"] = _area_scalar(lcoe_CH4, area)
            summary_dict[area].at["lcoe_H2 [€/MWh]"] = _area_scalar(lcoe_H2, area)
            summary_dict[area].at["lcoe_elec_volume [€/MWh]"] = _area_scalar(lcoe_elec_volume, area)
            summary_dict[area].at["lcoe_CH4_volume [€/MWh]"] = _area_scalar(lcoe_CH4_volume, area)
            summary_dict[area].at["lcoe_H2_volume [€/MWh]"] = _area_scalar(lcoe_H2_volume, area)
            summary_dict[area].at["lcoe_elec_value [€/MWh]"] = _area_scalar(lcoe_elec_value, area)
            summary_dict[area].at["lcoe_CH4_value [€/MWh]"] = _area_scalar(lcoe_CH4_value, area)
            summary_dict[area].at["lcoe_H2_value [€/MWh]"] = _area_scalar(lcoe_H2_value, area)

            if not carbon_constraint:
                summary_dict[area].at["lcoe_CH4_noSCC [€/MWh]"] = _area_scalar(lcoe_CH4_noSCC, area)
                summary_dict[area].at["lcoe_CH4_volume_noSCC [€/MWh]"] = _area_scalar(lcoe_CH4_volume_noSCC, area)


    # After loop: return appropriate format
    if is_multi_area:
        # Combine all areas into DataFrame
        df_summary = pd.DataFrame(summary_dict)
        return df_summary, gene_per_tech_global
    else:
        # Single area: return Series for backward compatibility
        return summary_dict[areas[0]], gene_per_tech_global


def export_hourly_dispatch(hourly_balance, area, output_path, nb_years=1, spot_price=None, filename=None):
    """Export hourly dispatch for a given country to CSV.

    Columns include all production, storage in/out, demand, imports, exports, and curtailment.
    Reserve (fcr/frr) and state-of-charge variables are excluded.
    If spot_price is provided, columns ``spot_elec_€/MWh`` (and ``spot_CH4_€/MWh`` for FR)
    are appended.

    Parameters
    ----------
    hourly_balance : xr.Dataset
        Output of extract_hourly_balance.
    area : str
        Country code to export (e.g. "FR").
    output_path : str or Path
        Directory where the CSV will be written.
    nb_years : int
        Number of simulated years (not used for values, kept for consistency).
    spot_price : dict of xr.DataArray, optional
        Dict with keys "elec", "CH4", "H2" (output of extract_spot_price or similar).
        If provided, spot price columns are appended.
    filename : str, optional
        Override the default filename ``hourly_dispatch_{area}.csv``.

    Returns
    -------
    pd.DataFrame
    """
    hb = hourly_balance.sel(area=area) if "area" in hourly_balance.dims else hourly_balance
    exclude_suffixes = ("_fcr", "_frr", "_state_charge")
    cols = [v for v in hb.data_vars if not any(v.endswith(s) for s in exclude_suffixes)]
    df = hb[cols].to_dataframe()
    df.index.name = "datetime"

    if spot_price is not None:
        sp_elec = spot_price["elec"]
        sp_elec_a = sp_elec.sel(area=area).values if "area" in sp_elec.dims else sp_elec.values
        df["spot_elec_€/MWh"] = sp_elec_a

        if area == "FR" and "CH4" in spot_price:
            sp_ch4 = spot_price["CH4"]
            sp_ch4_a = sp_ch4.sel(area=area).values if "area" in sp_ch4.dims else sp_ch4.values
            df["spot_CH4_€/MWh"] = sp_ch4_a

    if filename is None:
        filename = f"hourly_dispatch_{area}.csv"
    filepath = Path(output_path) / filename
    df.to_csv(filepath)
    print(f"Saved hourly dispatch ({area}) -> {filepath}")
    return df


def export_tech_summary(installed_power, energy_capacity, generation_per_technology,
                        hourly_balance, profits, area, output_path, nb_years=1, filename=None):
    """Export a per-technology summary CSV for a given country.

    Columns: installed_power_GW, energy_capacity_GWh, annual_generation_TWh_yr,
             storage_loss_TWh_yr, annual_profit_M€_yr.

    Technologies with zero installed capacity and zero generation are excluded.

    Parameters
    ----------
    installed_power : xr.DataArray
        Optimised capacities, dims [tech, area] or [tech].
    energy_capacity : xr.DataArray
        Optimised energy capacities (storage), dims [tech, area] or [tech].
    generation_per_technology : xr.DataArray
        Total generation over the simulation, dims [tech, area] or [tech].  Unit: TWh.
    hourly_balance : xr.Dataset
        Output of extract_hourly_balance — used to derive storage losses.
    profits : xr.DataArray or None
        Profits per tech, dims [tech, area] or [tech].  Unit: M€.
    area : str
        Country code (e.g. "FR").
    output_path : str or Path
        Directory where the CSV will be written.
    nb_years : int
        Number of simulated years — used to annualise generation, losses, and profit.
    filename : str, optional
        Override the default filename ``tech_summary_{area}.csv``.

    Returns
    -------
    pd.DataFrame
    """
    def _sel(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    ip  = _sel(installed_power, area)
    ec  = _sel(energy_capacity, area)
    gen = _sel(generation_per_technology, area)
    hb  = hourly_balance.sel(area=area) if "area" in hourly_balance.dims else hourly_balance

    rows = []
    for tech in ip.tech.values:
        row = {"technology": tech}
        row["installed_power_GW"] = float(ip.sel(tech=tech).values)

        if tech in ec.tech.values:
            ec_val = float(ec.sel(tech=tech).values)
            row["energy_capacity_GWh"] = None if np.isnan(ec_val) else ec_val
        else:
            row["energy_capacity_GWh"] = None

        if gen.size > 0 and tech in gen.tech.values:
            row["annual_generation_TWh_yr"] = float(gen.sel(tech=tech).values) / nb_years
        else:
            row["annual_generation_TWh_yr"] = 0.0

        out_key = tech
        in_key  = f"{tech}_input"
        if out_key in hb.data_vars and in_key in hb.data_vars:
            out_twh = float(hb[out_key].sum("hour").values) / 1000
            in_twh  = float(hb[in_key].sum("hour").values) / 1000
            row["storage_loss_TWh_yr"] = (in_twh - out_twh) / nb_years
        else:
            row["storage_loss_TWh_yr"] = None

        if profits is not None:
            prf = _sel(profits, area)
            row["annual_profit_M€_yr"] = float(prf.sel(tech=tech).values) / nb_years if tech in prf.tech.values else None
        else:
            row["annual_profit_M€_yr"] = None

        rows.append(row)

    df = pd.DataFrame(rows).set_index("technology")
    df = df[(df["installed_power_GW"] > 1e-4) | (df["annual_generation_TWh_yr"].fillna(0) > 1e-4)]

    if filename is None:
        filename = f"tech_summary_{area}.csv"
    filepath = Path(output_path) / filename
    df.to_csv(filepath)
    print(f"Saved tech summary ({area}) -> {filepath}")
    return df


def compute_lrmc_per_tech(capacity, existing_capacity, energy_capacity, existing_energy_capacity,
                          generation_per_technology, annuities, storage_annuities, fOM, vOM,
                          area, nb_years, hourly_balance, spot_price,
                          elec_balance, CH4_balance, H2_balance, exclude_dummies=True):
    """Compute the Long-Run Marginal Cost (LRMC) per technology for a given area.

    LRMC = (CAPEX_annuity + fOM + storage_CAPEX_annuity + storage_charging_cost) / gen_per_year  +  vOM × 1000
    All costs are in €/MWh.  Technologies with zero generation have their fixed costs
    reported separately (lrmc_total = NaN).

    Storage techs (those with an energy_capacity entry) also pay for the energy used to
    recharge them, valued at the spot price of whichever vector they draw from (elec for
    phs/battery_*, CH4 for ch4_reservoir, H2 for h2_saltcavern — routed via elec_balance/
    CH4_balance/H2_balance). This mirrors the accounting already used in extract_profit,
    which nets discharge revenue against charging cost — without it, LRMC previously only
    reflected the discharge side of storage.

    Unit reminder
    -------------
    - annuities, fOM, storage_annuities : M€/GW/yr or M€/GWh/yr
    - vOM                               : M€/GWh  → ×1000 → €/MWh
    - generation_per_technology         : TWh over full simulation  → /nb_years → TWh/yr
    - hourly_balance[f"{tech}_input"] × spot_price : GWh × €/MWh → k€ → /1000/nb_years → M€/yr
    - fixed costs / gen [M€/yr / TWh/yr = M€/TWh = €/MWh]

    Parameters
    ----------
    capacity, existing_capacity : xr.DataArray  [GW], dims [tech] or [tech, area]
    energy_capacity, existing_energy_capacity : xr.DataArray  [GWh]
    generation_per_technology : xr.DataArray  [TWh over simulation], dims [tech] or [tech, area]
    annuities : xr.DataArray  [M€/GW/yr], indexed by tech
    storage_annuities : xr.DataArray  [M€/GWh/yr], indexed by tech
    fOM : xr.DataArray  [M€/GW/yr], indexed by tech
    vOM : xr.DataArray  [M€/GWh], indexed by tech
    area : str
    nb_years : int
    hourly_balance : xr.Dataset
        Output of extract_hourly_balance.  Used to read each storage tech's f"{tech}_input"
        charging flow.  Must have dims [area, hour] (or [hour] for single-node).
    spot_price : dict of xr.DataArray
        Keys "elec", "CH4", "H2".  Dims [area, hour] or [hour].  Units €/MWh.
    elec_balance, CH4_balance, H2_balance : xr.DataArray of tech names
        Used to route each storage tech's charging cost to the correct vector's spot price.
    exclude_dummies : bool
        If True, drop technologies whose name ends with '_dummy'.

    Returns
    -------
    pd.DataFrame
        Indexed by tech, columns:
        cap_GW, gen_TWh_yr, capex_€/MWh, fOM_€/MWh, storage_capex_€/MWh, charging_cost_€/MWh,
        vOM_€/MWh, lrmc_€/MWh, fixed_cost_M€/yr
    """
    def _sel_area(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    cap_a    = _sel_area(capacity, area)
    exist_a  = _sel_area(existing_capacity, area)
    ecap_a   = _sel_area(energy_capacity, area)
    eexist_a = _sel_area(existing_energy_capacity, area)
    gen_a    = _sel_area(generation_per_technology, area)
    hb_area  = hourly_balance.sel(area=area) if "area" in hourly_balance.dims else hourly_balance
    sp_elec  = _sel_area(spot_price["elec"], area)
    sp_CH4   = _sel_area(spot_price["CH4"], area)
    sp_H2    = _sel_area(spot_price["H2"], area)

    new_cap_a  = xr.where(cap_a  >= exist_a,  cap_a  - exist_a,  0.0)
    new_ecap_a = xr.where(ecap_a >= eexist_a, ecap_a - eexist_a, 0.0)

    techs = cap_a.tech.values.tolist()
    storage_techs = ecap_a.tech.values.tolist()

    _ann     = annuities.reindex(tech=techs, fill_value=0.0)
    _fom     = fOM.reindex(tech=techs, fill_value=0.0)
    _vom     = vOM.reindex(tech=techs, fill_value=0.0)
    _str_ann = storage_annuities.reindex(tech=storage_techs, fill_value=0.0)

    # Gen indexed by tech name → per year
    if gen_a.size > 0 and "tech" in gen_a.coords:
        gen_pd = pd.Series(gen_a.values, index=gen_a.coords["tech"].values) / nb_years
    else:
        gen_pd = pd.Series(0.0, index=techs)

    str_ann_dict = {t: float(_str_ann.sel(tech=t).values) for t in storage_techs}

    # Charging cost (M€/yr) per storage tech, valued at the spot price of the vector it
    # draws from — elec for phs/battery_*, CH4 for ch4_reservoir, H2 for h2_saltcavern.
    charge_cost_dict = {}
    for t in storage_techs:
        input_var = f"{t}_input"
        if input_var not in hb_area.data_vars:
            continue
        if t in elec_balance.values:
            sp_vec = sp_elec
        elif t in CH4_balance.values:
            sp_vec = sp_CH4
        elif t in H2_balance.values:
            sp_vec = sp_H2
        else:
            continue
        charge_cost_dict[t] = float((hb_area[input_var] * sp_vec).sum("hour").values) / 1000 / nb_years  # M€/yr

    rows = []
    for i, t in enumerate(techs):
        cap_v     = float(cap_a.sel(tech=t).values)
        new_cap_v = float(new_cap_a.sel(tech=t).values)
        gen_v     = gen_pd.get(t, 0.0)          # TWh/yr

        capex_m    = new_cap_v * float(_ann.sel(tech=t).values)   # M€/yr
        fom_m      = cap_v     * float(_fom.sel(tech=t).values)    # M€/yr
        new_ecap_v = float(new_ecap_a.sel(tech=t).values) if t in storage_techs else 0.0
        str_m      = new_ecap_v * str_ann_dict.get(t, 0.0)         # M€/yr
        charge_m   = charge_cost_dict.get(t, 0.0)                  # M€/yr
        vom_rate   = float(_vom.sel(tech=t).values) * 1000          # €/MWh

        fixed_m = capex_m + fom_m + str_m + charge_m  # M€/yr

        if gen_v > 1e-4:
            capex_mwh  = capex_m  / gen_v    # M€/TWh = €/MWh
            fom_mwh    = fom_m    / gen_v
            str_mwh    = str_m    / gen_v
            charge_mwh = charge_m / gen_v
            lrmc       = capex_mwh + fom_mwh + str_mwh + charge_mwh + vom_rate
        else:
            capex_mwh  = np.nan
            fom_mwh    = np.nan
            str_mwh    = np.nan
            charge_mwh = np.nan
            lrmc       = np.nan

        if cap_v < 1e-4 and gen_v < 1e-4:
            continue

        rows.append({
            "tech":                t,
            "cap_GW":              cap_v,
            "gen_TWh_yr":          gen_v,
            "capex_€/MWh":         capex_mwh,
            "fOM_€/MWh":           fom_mwh,
            "storage_capex_€/MWh": str_mwh,
            "charging_cost_€/MWh": charge_mwh,
            "vOM_€/MWh":           vom_rate,
            "lrmc_€/MWh":          lrmc,
            "fixed_cost_M€/yr":    fixed_m,
        })

    df = pd.DataFrame(rows).set_index("tech")

    if exclude_dummies:
        df = df[~df.index.str.endswith("_dummy")]

    return df


def compute_country_cost_summary(hourly_balance, spot_price, capacity, existing_capacity,
                                  energy_capacity, existing_energy_capacity,
                                  generation_per_technology, annuities, storage_annuities,
                                  fOM, vOM, nb_years, areas):
    """Per-country annual cost breakdown (M€/yr).

    Parameters
    ----------
    hourly_balance : xr.Dataset
        Output of extract_hourly_balance.  Must have dims [area, hour].
    spot_price : dict of xr.DataArray
        Keys "elec" (and optionally "CH4", "H2").  Electricity spot price used to
        value imports/exports.  Dims [area, hour] or [hour].  Units: €/MWh.
    capacity, existing_capacity : xr.DataArray  [GW], dims [tech, area] or [tech]
    energy_capacity, existing_energy_capacity : xr.DataArray  [GWh]
    generation_per_technology : xr.DataArray
        Total generation over the full simulation [TWh], dims [tech, area] or [tech].
    annuities : xr.DataArray  [M€/GW/yr], indexed by tech
    storage_annuities : xr.DataArray  [M€/GWh/yr], indexed by tech
    fOM : xr.DataArray  [M€/GW/yr], indexed by tech
    vOM : xr.DataArray  [M€/GWh], indexed by tech
    nb_years : int
        Number of simulated years (used to annualise generation and trade flows).
    areas : list of str
        Country codes to include.

    Returns
    -------
    pd.DataFrame
        Index: area.  Columns (all M€/yr):
        capex_M€/yr, fOM_M€/yr, vOM_M€/yr, storage_capex_M€/yr,
        import_cost_M€/yr, export_revenue_M€/yr, net_cost_M€/yr.

    Notes
    -----
    - CAPEX annuity applies only to *new* capacity (installed − existing).
    - fOM applies to *total* installed capacity.
    - vOM is computed from annual generation (total generation / nb_years).
    - import_cost  = Σ_h (imports_h  × spot_elec_h) / 1000 / nb_years
    - export_revenue = Σ_h (exports_h × spot_elec_h) / 1000 / nb_years
    - net_cost = capex + fOM + vOM + storage_capex + import_cost − export_revenue
    """
    def _sel(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    rows = {}
    for area in areas:
        hb_a     = hourly_balance.sel(area=area) if "area" in hourly_balance.dims else hourly_balance
        cap_a    = _sel(capacity, area)
        exist_a  = _sel(existing_capacity, area)
        ecap_a   = _sel(energy_capacity, area)
        eexist_a = _sel(existing_energy_capacity, area)
        gen_a    = _sel(generation_per_technology, area)

        new_cap_a  = xr.where(cap_a  >= exist_a,  cap_a  - exist_a,  0.0)
        new_ecap_a = xr.where(ecap_a >= eexist_a, ecap_a - eexist_a, 0.0)

        techs = cap_a.tech.values.tolist()
        _ann     = annuities.reindex(tech=techs, fill_value=0.0)
        _fom     = fOM.reindex(tech=techs, fill_value=0.0)
        _vom     = vOM.reindex(tech=techs, fill_value=0.0)
        _str_ann = storage_annuities.reindex(tech=ecap_a.tech.values.tolist(), fill_value=0.0)

        capex_m = float((new_cap_a * _ann).sum("tech").values)   # M€/yr
        fom_m   = float((cap_a     * _fom).sum("tech").values)   # M€/yr

        if gen_a.size > 0 and "tech" in gen_a.coords:
            gen_pd = pd.Series(gen_a.values, index=gen_a.coords["tech"].values) / nb_years
        else:
            gen_pd = pd.Series(0.0, index=techs)
        vom_m = float(sum(gen_pd.get(t, 0.0) * 1000.0 * float(_vom.sel(tech=t).values) for t in techs))

        str_m = float((new_ecap_a * _str_ann).sum("tech").values)  # M€/yr

        if ("imports" in hb_a.data_vars and "exports" in hb_a.data_vars
                and spot_price is not None):
            sp_a = (spot_price["elec"].sel(area=area)
                    if "area" in spot_price["elec"].dims else spot_price["elec"])
            import_m  = float((hb_a["imports"] * sp_a / 1000).sum("hour").values) / nb_years
            export_m  = float((hb_a["exports"] * sp_a / 1000).sum("hour").values) / nb_years
        else:
            import_m = export_m = 0.0

        rows[area] = {
            "capex_M€/yr":          capex_m,
            "fOM_M€/yr":            fom_m,
            "vOM_M€/yr":            vom_m,
            "storage_capex_M€/yr":  str_m,
            "import_cost_M€/yr":    import_m,
            "export_revenue_M€/yr": export_m,
            "net_cost_M€/yr":       capex_m + fom_m + vom_m + str_m + import_m - export_m,
        }

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# Marginal price-setter identification
# ─────────────────────────────────────────────────────────────────────────────

_NON_TECH_VARS = frozenset({
    "elec_demand", "H2_demand", "CH4_demand",
    "load_shift_up", "load_shift_down", "elec_demand_w_shift",
    "imports", "exports", "curtailment",
    "storage_input_losses", "storage_output_losses", "lake_state_charge",
})


def identify_price_setter(
    spot_price,
    hourly_balance,
    vOM,
    conversion_efficiency,
    area,
    elec_prod_techs=None,
    ch4_techs=None,
    h2_techs=None,
    storage_techs=None,
    str_dummy_name="str_dummy",
    gen_threshold=0.01,
    match_tol=30.0,
    # Optional LRMC parameters — when provided, mc = vOM + CAPEX/gen + fOM/gen
    annuities=None,
    fOM=None,
    capacity=None,
    existing_capacity=None,
    generation_per_technology=None,
    nb_years=1,
    renewable_surplus_tol=0.5,
    # Optional LP dual parameters (v3_6+)
    dual_nuclear_CF=None,
    nuclear_tech="nuclear",
    dual_max_capacity=None,
    max_cap_filter_techs=None,
    max_cap_filter_eps=1e-3,
):
    """For each hour, identify the technology that sets the marginal electricity price.

    In a linear program, the spot price equals the variable cost of the marginal unit.
    The price-setter is the dispatching technology whose effective marginal cost
    is closest to spot_price['elec'].

    Parameters
    ----------
    spot_price : dict-like  {"elec": DataArray(area?,hour), "CH4": ..., "H2": ...}
        Shadow prices from extract_spot_price.  Units: €/MWh.
    hourly_balance : xr.Dataset  [dims area, hour per variable]
        Output of extract_hourly_balance.
    vOM : xr.DataArray  [M€/GWh, indexed by tech]
        Variable O&M costs.  Internally converted to €/MWh (× 1000).
    conversion_efficiency : xr.DataArray  [indexed by tech]
        Fuel-to-electricity efficiency for thermal techs (e.g., 0.6 for CCGT).
        Only used for ch4_techs and h2_techs.
    area : str
    elec_prod_techs : list of str, optional
        Variables in hourly_balance that produce electricity (EXCLUDING CH4/H2 producers
        such as methanization, electrolysis — those are not electricity outputs).
        If None, auto-detected by excluding known non-tech variables and suffixes,
        but this may include non-electricity techs → prefer explicit list.
        Example: ["nuclear","onshore","offshore_ground","offshore_float","river","lake",
                  "phs","battery_1h","ch4_ccgt","ch4_ocgt","h2_ccgt","str_dummy"]
    ch4_techs : list of str, optional
        Subset of elec_prod_techs that burn CH4 (e.g. ["ch4_ccgt", "ch4_ocgt"]).
        mc[h] = lrmc_fixed[t] + spot_CH4[h] / eta
    h2_techs : list of str, optional
        Subset that burn H2 (e.g. ["h2_ccgt"]).
        mc[h] = lrmc_fixed[t] + spot_H2[h] / eta
    storage_techs : list of str, optional
        Storage discharge techs (e.g. ["phs", "battery_1h", "battery_4h"]).
        When the best mc-match has match_diff > match_tol AND a storage tech is
        discharging, the hour is labelled "storage" (opportunity cost not captured
        by LRMC).  Defaults to ["phs","battery_1h","battery_2h","battery_4h","battery_8h"].
    str_dummy_name : str
        Name of the unserved-demand variable.  Assigned mc = 10 000 €/MWh.
    gen_threshold : float  [GW]
        Minimum generation to consider a technology as dispatching.
    match_tol : float  [€/MWh]
        Maximum acceptable match_diff.  Hours exceeding this threshold are
        re-labelled: "storage" if a storage tech is discharging, else "unresolved".
        Default 30 €/MWh.  Raise to 200+ to disable the override.
    annuities : xr.DataArray  [M€/GW/yr, indexed by tech], optional
        Annualised investment cost.  Applied only to NEW capacity (installed − existing).
        When provided together with fOM, capacity, existing_capacity and
        generation_per_technology, the effective mc becomes the full LRMC:
            lrmc_fixed[t] = vOM[t]*1e3
                          + (annuities[t]*new_cap[t] + fOM[t]*cap[t]) / gen_twh_yr[t] * 1e3
        This is essential for technologies with zero existing capacity (e.g. new nuclear)
        where CAPEX dominates the marginal price.
    fOM : xr.DataArray  [M€/GW/yr, indexed by tech], optional
        Fixed O&M.  Applied to total installed capacity.
    capacity : xr.DataArray  [GW], optional
        Total installed capacity, dims (tech,) or (tech, area).
    existing_capacity : xr.DataArray  [GW], optional
        Pre-existing capacity (annuity-free), dims (tech,) or (tech, area).
    generation_per_technology : xr.DataArray  [TWh], optional
        Total generation over the simulation period, dims (tech,) or (tech, area).
    nb_years : int
        Number of simulated years (used to annualise generation_per_technology).
    renewable_surplus_tol : float  [€/MWh]
        Hours with spot_elec below this threshold are labelled "renewable_surplus"
        (curtailment/zero-price period) without mc-matching.  Default 0.5 €/MWh.
    dual_nuclear_CF : xr.DataArray, optional  [M€/GW, dims: (area,) or (area, year)]
        Dual of the nuclear_yearly_CF constraint, from
        ``model.constraints.nuclear_yearly_CF.dual``.
        When provided, the effective mc for *nuclear_tech* is computed exactly from
        the LP KKT stationarity condition::

            mc_nuclear [€/MWh] = (vOM_nuclear + |dual_CF| / 8760) × 1000

        This replaces the approximate LRMC for nuclear and eliminates the need for
        annuities/capacity/generation when nuclear is the only concern.
        ``abs()`` is applied because linopy may report the dual with a negative sign
        depending on the constraint direction (≤ vs ≥) and solver convention — the
        correction is always a positive addition to vOM.
        If *dual_nuclear_CF* has a ``year`` dimension, the mean across years is taken.
    nuclear_tech : str
        Name of the nuclear technology in *elec_prod_techs*.  Default ``"nuclear"``.
    dual_max_capacity : xr.DataArray, optional  [M€/GW/yr, dims: (area, tech)]
        Dual of the max_capacity_prod constraint, from
        ``model.constraints.max_capacity_prod.dual``.
        Used jointly with *max_cap_filter_techs*: for each listed technology, if
        ``|dual_max_capacity[area, tech]| > max_cap_filter_eps`` the technology is
        treated as investment-constrained (always fully dispatched, not the
        marginal price-setter) and its generation is masked out of the matching.
        Hours that would have been labelled by a filtered tech fall through to
        "storage" or "unresolved".
    max_cap_filter_techs : list of str, optional
        Techs to exclude from price-setter candidates when investment-constrained.
        Typically VRE that are always at full profile output:
        ``["offshore_float","offshore_ground","onshore",
           "pv_ground","pv_roof_com","pv_roof_indiv","marine"]``.
        Default None (no filtering).
    max_cap_filter_eps : float  [M€/GW/yr]
        Minimum absolute dual value to consider the max_capacity constraint binding.
        Default 1e-3.

    Returns
    -------
    pd.DataFrame  indexed by hour, columns:
        price_setter [str], mc_setter [€/MWh], spot_elec [€/MWh], match_diff [€/MWh]

    Notes
    -----
    - Storage (PHS, batteries) opportunity cost = price at charge time / round-trip η,
      which equals the SOC constraint dual in the LP.  This is NOT captured by LRMC.
      Storage hours are identified via match_tol fallback, not mc-matching.
    - Existing capacity (lake, river, existing PHS) has no annuity → their lrmc_fixed
      equals vOM only.  Their water-value opportunity cost is also not in lrmc_fixed,
      so they may still fall under the match_tol/storage override.
    - Imports are added automatically when present (single-node models have none).
    - With Crossover=0, primal solutions may not be at a vertex; mc-matching is
      more robust than checking partial loading directly.
    """
    def _sel(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    sp_elec = _sel(spot_price["elec"], area)
    sp_ch4  = _sel(spot_price["CH4"],  area) if "CH4" in spot_price else None
    sp_h2   = _sel(spot_price["H2"],   area) if "H2"  in spot_price else None

    hours   = sp_elec.coords["hour"].values
    n_hours = len(hours)
    sp_arr  = sp_elec.values.astype(float)

    ch4_set = set(ch4_techs or [])
    h2_set  = set(h2_techs  or [])
    _default_storage = ["phs", "battery_1h", "battery_2h", "battery_4h", "battery_8h"]
    storage_set = set(storage_techs if storage_techs is not None else _default_storage)

    # Whether LRMC mode is active (all required params provided)
    _use_lrmc = all(p is not None for p in [
        annuities, fOM, capacity, existing_capacity, generation_per_technology
    ])

    def _lrmc_fixed(tech):
        """Fixed LRMC component [€/MWh]: vOM + (annuity×new_cap + fOM×total_cap)/gen."""
        vom_val = (float(vOM.sel(tech=tech).values) * 1000.0
                   if "tech" in vOM.coords and tech in vOM.coords["tech"].values else 0.0)

        # Nuclear CF dual override: exact LP KKT correction replaces LRMC approximation.
        # KKT stationarity: spot[h] = vOM + |dual_CF| / 8760  [M€/GWh]
        if dual_nuclear_CF is not None and tech == nuclear_tech:
            da = dual_nuclear_CF
            if "area" in da.dims:
                da = da.sel(area=area)
            if "year" in da.dims:
                da = da.mean(dim="year")
            return vom_val + abs(float(da.values)) / 8760 * 1000

        if not _use_lrmc:
            return vom_val
        cap_t   = float(_sel(capacity,           area).sel(tech=tech).values
                        if tech in capacity.coords["tech"].values           else 0.0)
        exist_t = float(_sel(existing_capacity,  area).sel(tech=tech).values
                        if tech in existing_capacity.coords["tech"].values  else 0.0)
        new_t   = max(cap_t - exist_t, 0.0)
        ann_t   = (float(annuities.sel(tech=tech).values)
                   if tech in annuities.coords["tech"].values else 0.0)
        fom_t   = (float(fOM.sel(tech=tech).values)
                   if tech in fOM.coords["tech"].values else 0.0)
        gen_t   = (float(_sel(generation_per_technology, area).sel(tech=tech).values)
                   if tech in generation_per_technology.coords["tech"].values else 0.0)
        gen_twh_yr = gen_t / nb_years
        if gen_twh_yr < 1e-6:
            return vom_val  # avoid division by zero for non-generating techs
        # M€/yr / (TWh/yr) = M€/TWh = 10⁶€/10⁶MWh = 1 €/MWh  (no ×1000)
        capex_per_mwh = (ann_t * new_t + fom_t * cap_t) / gen_twh_yr
        return vom_val + capex_per_mwh

    # Auto-detect elec producing techs if not supplied
    if elec_prod_techs is None:
        elec_prod_techs = [
            v for v in hourly_balance.data_vars
            if (v not in _NON_TECH_VARS
                and not v.endswith("_input")
                and not v.endswith("_state_charge")
                and not v.endswith("_fcr")
                and not v.endswith("_frr"))
        ]

    # Build candidate arrays
    labels     = []
    gen_arrays = []
    mc_arrays  = []

    for tech in elec_prod_techs:
        if tech not in hourly_balance.data_vars:
            continue
        gen_arr   = _sel(hourly_balance[tech], area).values.astype(float)
        lrmc_base = _lrmc_fixed(tech)

        if tech == str_dummy_name:
            mc_arr = np.full(n_hours, 10_000.0)
        elif tech in ch4_set and sp_ch4 is not None:
            eta    = float(conversion_efficiency.sel(tech=tech).values)
            mc_arr = lrmc_base + sp_ch4.values.astype(float) / eta
        elif tech in h2_set and sp_h2 is not None:
            eta    = float(conversion_efficiency.sel(tech=tech).values)
            mc_arr = lrmc_base + sp_h2.values.astype(float) / eta
        else:
            mc_arr = np.full(n_hours, lrmc_base)

        labels.append(tech)
        gen_arrays.append(gen_arr)
        mc_arrays.append(mc_arr)

    # Imports: mc ≈ spot_elec (added last so thermal techs take priority on ties)
    if "imports" in hourly_balance.data_vars:
        gen_arr = _sel(hourly_balance["imports"], area).values.astype(float)
        labels.append("imports")
        gen_arrays.append(gen_arr)
        mc_arrays.append(sp_arr.copy())

    if not labels:
        return pd.DataFrame(columns=["price_setter", "mc_setter", "spot_elec", "match_diff"])

    gen_matrix  = np.array(gen_arrays)                        # (n_cands, n_hours)
    mc_matrix   = np.array(mc_arrays)                         # (n_cands, n_hours)

    # Mask investment-constrained VRE: their hourly mc is vOM≈0 but they are always
    # fully dispatched and do not set the marginal price.
    if dual_max_capacity is not None and max_cap_filter_techs:
        for i, tech in enumerate(labels):
            if tech not in max_cap_filter_techs:
                continue
            try:
                da = _sel(dual_max_capacity, area)
                tech_dual = float(da.sel(tech=tech).values) if tech in da.coords["tech"].values else 0.0
            except Exception:
                tech_dual = 0.0
            if abs(tech_dual) > max_cap_filter_eps:
                gen_matrix[i, :] = 0.0  # treat as non-dispatching for price-setting

    diff_matrix = np.abs(mc_matrix - sp_arr[np.newaxis, :])   # (n_cands, n_hours)
    diff_matrix[gen_matrix < gen_threshold] = np.inf          # mask non-dispatching

    best_idx       = np.argmin(diff_matrix, axis=0)           # (n_hours,)
    all_inf        = np.all(np.isinf(diff_matrix), axis=0)    # hours with no dispatch
    best_diff_arr  = diff_matrix[best_idx, np.arange(n_hours)]

    # match_tol override: if best match is poor, check storage before giving up
    exceeds_tol = (~all_inf) & (best_diff_arr > match_tol)
    # Build storage dispatch mask (any storage tech discharging above threshold)
    _str_arrs = [
        _sel(hourly_balance[t], area).values.astype(float)
        for t in storage_set if t in hourly_balance.data_vars
    ]
    if _str_arrs:
        storage_dispatching = np.any(
            np.array(_str_arrs) >= gen_threshold, axis=0
        )
    else:
        storage_dispatching = np.zeros(n_hours, dtype=bool)

    reclassify_storage    = exceeds_tol & storage_dispatching
    reclassify_unresolved = exceeds_tol & ~storage_dispatching
    # Zero/negative spot hours: renewable surplus or must-run overflow → no mc-matching
    renewable_surplus = sp_arr < renewable_surplus_tol

    price_setter = []
    for h in range(n_hours):
        if renewable_surplus[h]:
            price_setter.append("renewable_surplus")
        elif all_inf[h] or reclassify_unresolved[h]:
            price_setter.append("unresolved")
        elif reclassify_storage[h]:
            price_setter.append("storage")
        else:
            price_setter.append(labels[best_idx[h]])

    mc_setter   = np.where(all_inf, np.nan, mc_matrix[best_idx, np.arange(n_hours)])
    match_diff  = np.where(all_inf, np.nan, best_diff_arr)
    match_diff  = np.where(np.isinf(match_diff), np.nan, match_diff)

    return pd.DataFrame({
        "price_setter": price_setter,
        "mc_setter":    mc_setter,
        "spot_elec":    sp_arr,
        "match_diff":   match_diff,
    }, index=pd.Index(hours, name="hour"))


def summarize_price_setter(price_setter_df):
    """Aggregate identify_price_setter output: % of hours and avg prices per tech.

    Returns
    -------
    pd.DataFrame  indexed by price_setter, sorted by pct_hours descending.
        Columns: hours, pct_hours [%], avg_spot_€MWh, avg_mc_€MWh, avg_match_diff_€MWh
    """
    n = len(price_setter_df)
    return (
        price_setter_df
        .groupby("price_setter", sort=False)
        .agg(
            hours          =("spot_elec",   "count"),
            avg_spot_MWh  =("spot_elec",   "mean"),
            avg_mc_MWh    =("mc_setter",   "mean"),
            avg_match_diff =("match_diff",  "mean"),
        )
        .assign(pct_hours=lambda df: df["hours"] / n * 100)
        [["hours", "pct_hours", "avg_spot_MWh", "avg_mc_MWh", "avg_match_diff"]]
        .sort_values("pct_hours", ascending=False)
    )


# Label grouping for cleaner charts
_TECH_LABEL_EN = {
    "nuclear": "Nuclear", "onshore": "Wind - Onshore",
    "offshore_ground": "Wind - Offshore", "offshore_float": "Wind - Offshore",
    "river": "Hydro - Run-of-river", "lake": "Hydro - Dams",
    "phs": "Hydro - PHS",
    "ch4_ccgt": "CCGT (CH4)", "ch4_ocgt": "OCGT (CH4)",
    "h2_ccgt": "H2 turbine",
    "battery_1h": "Battery", "battery_2h": "Battery",
    "battery_4h": "Battery", "battery_8h": "Battery",
    "methanization": "Methanization", "methanation": "Methanation",
    "imports": "Imports",
    "storage": "Storage (opportunity cost)",
    "renewable_surplus": "Renewable surplus",
    "str_dummy": "Unserved demand",
    "unresolved": "Unresolved",
}
_TECH_LABEL_FR = {
    "nuclear": "Nucléaire", "onshore": "Eolien terrestre",
    "offshore_ground": "Eolien en mer", "offshore_float": "Eolien en mer",
    "river": "Hydraulique fil de l'eau", "lake": "Hydraulique barrages",
    "phs": "Hydraulique STEP",
    "ch4_ccgt": "CCGT (CH4)", "ch4_ocgt": "OCGT (CH4)",
    "h2_ccgt": "Turbine H2",
    "battery_1h": "Batterie", "battery_2h": "Batterie",
    "battery_4h": "Batterie", "battery_8h": "Batterie",
    "methanization": "Méthanisation", "methanation": "Méthanation",
    "imports": "Imports",
    "storage": "Stockage (coût d'opportunité)",
    "renewable_surplus": "Surplus renouvelable",
    "str_dummy": "Demande non servie",
    "unresolved": "Non résolu",
}


def extract_capacity_duals(model):
    """Combine the max-capacity dual values across all tech categories (production,
    conversion, storage) into a single xr.DataArray indexed by tech (and area for
    multi-node runs).

    Each of the three underlying constraints (max_capacity_prod, max_capacity_conv,
    max_capacity_str) only applies to its own disjoint tech subset (prod_tech,
    conversion_tech, str — see ModelEOLES.define_sets), so concatenating along "tech"
    does not double-count anything. A non-zero (negative) dual means that tech's
    capacity constraint is binding — the model would invest more if the cap were
    relaxed — which is the natural counterpart to check against extract_profit's
    output (a binding cap is expected to coincide with a positive scarcity profit).

    Parameters
    ----------
    model : linopy.Model
        The solved model, i.e. ModelEOLES.model (not the ModelEOLES wrapper itself).

    Returns
    -------
    xr.DataArray, dims ["tech"] or ["tech", "area"]
    """
    parts = []
    for name in ("max_capacity_prod", "max_capacity_conv", "max_capacity_str"):
        con = getattr(model.constraints, name, None)
        if con is not None:
            parts.append(con.dual)
    if not parts:
        return xr.DataArray([], dims="tech")
    return xr.concat(parts, dim="tech")


def check_profit_dual_consistency(profits, capacity_duals, capacity, existing_capacity,
                                   annuities, fOM, area, atol=1e-3):
    """Numerically verify the LP-duality identity linking extract_profit and
    extract_capacity_duals: at the optimum, for every tech governed by a max-capacity
    constraint (production, conversion, storage discharge power),

        profit_t  ==  (annuity_t + fOM_t) * existing_capacity_t  -  dual_t * total_capacity_t

    Derivation sketch: differentiate the objective (CAPEX annuity on *new* capacity, fOM on
    *total* capacity, vOM on generation — see ModelEOLES.define_objective) w.r.t. installed
    capacity. Complementary slackness on the hourly generation<=capacity limit lets the
    resulting shadow-value sum be replaced by (price - vOM) x actual dispatch, i.e. exactly
    the energy-market margin that extract_profit computes. Everything cancels except the
    value of existing (already-amortised) capacity plus the max-capacity dual's scarcity
    rent on total capacity — this is the standard "zero (economic) profit at the margin,
    positive rent on binding capacity" result for capacity-expansion LPs. See the notebook's
    "Dual vs. profit" section for the intuition this makes precise.

    A non-zero residual can come from: (i) storage techs, where profit also nets the storage
    *energy*-capacity annuity, which this check does not include (only the power-capacity
    duals from extract_capacity_duals cover that); (ii) small numerical noise from the barrier
    solver run without crossover (ModelEOLES.solve's default), which does not guarantee an
    exactly basic/vertex solution.

    Parameters
    ----------
    profits : xr.DataArray, dims ["tech"] or ["tech", "area"] — output of extract_profit.
    capacity_duals : xr.DataArray, dims ["tech"] or ["area", "tech"] — output of extract_capacity_duals.
    capacity : xr.DataArray [GW], total installed capacity (e.g. m.installed_power).
    existing_capacity : xr.DataArray [GW], pre-existing capacity (e.g. m.existing_capacity).
    annuities, fOM : xr.DataArray [M€/GW/yr], indexed by tech.
    area : str
    atol : float
        Residuals with |residual| below this (M€/yr) are flagged as consistent.

    Returns
    -------
    pd.DataFrame indexed by tech, columns: profit_M€/yr, predicted_M€/yr, residual_M€/yr, consistent
    """
    def _sel_area(da, a):
        return da.sel(area=a) if isinstance(da, xr.DataArray) and "area" in da.dims else da

    profit_a = _sel_area(profits, area)
    dual_a = _sel_area(capacity_duals, area)
    cap_a = _sel_area(capacity, area)
    exist_a = _sel_area(existing_capacity, area)

    techs = [t for t in profit_a.tech.values if t in dual_a.tech.values and t in cap_a.tech.values]
    _ann = annuities.reindex(tech=techs, fill_value=0.0)
    _fom = fOM.reindex(tech=techs, fill_value=0.0)

    rows = []
    for t in techs:
        profit_v = float(profit_a.sel(tech=t).values)
        dual_v = float(dual_a.sel(tech=t).values)
        cap_v = float(cap_a.sel(tech=t).values)
        exist_v = float(exist_a.sel(tech=t).values) if t in exist_a.tech.values else 0.0
        ann_v = float(_ann.sel(tech=t).values)
        fom_v = float(_fom.sel(tech=t).values)

        predicted = (ann_v + fom_v) * exist_v - dual_v * cap_v
        residual = profit_v - predicted

        rows.append({
            "tech": t,
            "profit_M€/yr": profit_v,
            "predicted_M€/yr": predicted,
            "residual_M€/yr": residual,
            "consistent": abs(residual) <= atol,
        })

    return pd.DataFrame(rows).set_index("tech")


def compute_residual_demand(hourly_balance, area=None):
    """
    Calcule la demande résiduelle horaire = demande - production ENR variable.

    Technologies non pilotables soustraites :
        solaire (pv_ground, pv_roof_com, pv_roof_indiv), éolien (onshore, offshore_ground, offshore_float), fil de l'eau (river, marine).
    Barrages (lake), nucléaire et autres technologies pilotables sont conservés.

    Parameters
    ----------
    hourly_balance : xr.Dataset  — m.hourly_balance
    area           : str or None — None = somme toutes zones

    Returns
    -------
    residual : np.ndarray  demande résiduelle [GW]
    demand   : np.ndarray  demande totale [GW]
    vre      : np.ndarray  production ENR variable totale [GW]
    """
    if area is not None and "area" in hourly_balance.dims:
        hb = hourly_balance.sel(area=area)
    elif "area" in hourly_balance.dims:
        hb = hourly_balance.sum("area")
    else:
        hb = hourly_balance

    n_hours = len(hb[list(hb.dims)[0]])

    def _g(var):
        if var in hb.data_vars:
            return hb[var].values.flatten()
        return np.zeros(n_hours)

    demand = _g("elec_demand_w_shift") if "elec_demand_w_shift" in hb.data_vars else _g("elec_demand")

    vre_techs = [
        "pv_ground", "pv_roof_com", "pv_roof_indiv",
        "onshore", "offshore_ground", "offshore_float",
        "river", "marine",
    ]
    vre = sum(_g(t) for t in vre_techs)
    return demand - vre, demand, vre


def check_vector_balance(vector, hourly_balance, areas, str_elec=None, elec_balance=None, elec_primary_prod=None,
                         trade_net_twh=None):
    """Sanity-check the supply/demand balance for one energy vector, per area.

    Sums up every hourly_balance term that should account for a vector's inputs and outputs
    (production, conversion, storage losses, imports/exports, curtailment) and returns
    'net_result' [TWh]. For elec, this should be ~0 up to solver tolerance (curtailment is
    tracked explicitly, so nothing should be left over). For CH4/H2, the adequacy constraint
    is an inequality (supply >= demand), so a small POSITIVE net_result (a percent or so of
    total demand) is expected and benign - it reflects free "implicit curtailment" that isn't
    tracked as its own variable (unlike elec), amplified by the barrier-without-crossover
    solver method (see ModelEOLES.solve), which does not return a basic/vertex solution and
    can leave small slack in inequality constraints. A LARGE residual (double-digit % of
    demand) or a NEGATIVE one (apparent deficit) is the signal to actually investigate -
    typically a hourly_balance term missing here (e.g. a new tech added to the model but not
    to this check).

    For CH4/H2 in a multi-country run with active trade, pass `trade_net_twh` (see below) -
    otherwise cross-border annual trade volumes are not part of hourly_balance and net_result
    will be off for areas that trade, even though the model itself balances correctly.

    :param vector: "elec", "CH4" or "H2"
    :param hourly_balance: xr.Dataset, output of extract_hourly_balance
    :param areas: iterable of str
    :param str_elec: xr.DataArray, storage techs part of the elec balance (only for vector="elec")
    :param elec_balance: xr.DataArray (only for vector="elec")
    :param elec_primary_prod: xr.DataArray (only for vector="elec")
    :param trade_net_twh: pd.Series indexed by area, net annual trade [TWh] (imports - exports),
        only for vector="CH4"/"H2". E.g. (m.model.solution["gas annual import"]
        - m.model.solution["gas annual export"]).sum("year").to_pandas().
    :return: pd.DataFrame indexed by area
    """
    df = pd.DataFrame(index=list(areas))

    if vector == "H2":
        for area in areas:
            demand_for_h2 = float(hourly_balance["H2_demand"].loc[{"area": area}].sum().values) / 1000
            demand_for_elec = float(hourly_balance["h2_ccgt_input"].loc[{"area": area}].sum().values) / 1000
            prod_from_electrolysis = float(hourly_balance["electrolysis"].loc[{"area": area}].sum().values) / 1000
            prod_from_import = float(hourly_balance["H2_import"].loc[{"area": area}].sum().values) / 1000
            trade_net = float(trade_net_twh.get(area, 0.0)) if trade_net_twh is not None else 0.0
            storage_losses = (float(hourly_balance["h2_saltcavern_input"].loc[{"area": area}].sum().values) / 1000
                              - float(hourly_balance["h2_saltcavern"].loc[{"area": area}].sum().values) / 1000)
            df.loc[area, "demand_for_h2"] = demand_for_h2
            df.loc[area, "demand_for_elec"] = demand_for_elec
            df.loc[area, "prod_from_electrolysis"] = prod_from_electrolysis
            df.loc[area, "prod_from_import"] = prod_from_import
            df.loc[area, "trade_net"] = trade_net
            df.loc[area, "storage_losses"] = storage_losses
            df.loc[area, "net_result"] = (demand_for_elec + demand_for_h2 + storage_losses
                                          - prod_from_electrolysis - prod_from_import - trade_net)

    elif vector == "CH4":
        for area in areas:
            demand_for_ch4 = float(hourly_balance["CH4_demand"].loc[{"area": area}].sum().values) / 1000
            demand_for_elec = (float(hourly_balance["ch4_ocgt_input"].loc[{"area": area}].sum().values) / 1000
                               + float(hourly_balance["ch4_ccgt_input"].loc[{"area": area}].sum().values) / 1000
                               + float(hourly_balance["ocgt_coge_input"].loc[{"area": area}].sum().values) / 1000)
            prod_from_methanation = float(hourly_balance["methanation"].loc[{"area": area}].sum().values) / 1000
            prod_from_methanization = float(hourly_balance["methanization"].loc[{"area": area}].sum().values) / 1000
            prod_from_pyrogazification = float(hourly_balance["pyrogazification"].loc[{"area": area}].sum().values) / 1000
            prod_from_biogas_import = float(hourly_balance["biogas_import"].loc[{"area": area}].sum().values) / 1000
            prod_from_natural_gas = float(hourly_balance["natural_gas"].loc[{"area": area}].sum().values) / 1000
            trade_net = float(trade_net_twh.get(area, 0.0)) if trade_net_twh is not None else 0.0
            storage_losses = (float(hourly_balance["ch4_reservoir_input"].loc[{"area": area}].sum().values) / 1000
                              - float(hourly_balance["ch4_reservoir"].loc[{"area": area}].sum().values) / 1000)
            df.loc[area, "demand_for_ch4"] = demand_for_ch4
            df.loc[area, "demand_for_elec"] = demand_for_elec
            df.loc[area, "prod_from_methanation"] = prod_from_methanation
            df.loc[area, "prod_from_methanization"] = prod_from_methanization
            df.loc[area, "prod_from_pyrogazification"] = prod_from_pyrogazification
            df.loc[area, "prod_from_biogas_import"] = prod_from_biogas_import
            df.loc[area, "prod_from_natural_gas"] = prod_from_natural_gas
            df.loc[area, "trade_net"] = trade_net
            df.loc[area, "storage_losses"] = storage_losses
            df.loc[area, "net_result"] = (demand_for_elec + demand_for_ch4 + storage_losses
                                          - prod_from_methanation - prod_from_methanization
                                          - prod_from_pyrogazification - prod_from_biogas_import
                                          - prod_from_natural_gas - trade_net)

    elif vector == "elec":
        if str_elec is None or elec_balance is None or elec_primary_prod is None:
            raise ValueError("vector='elec' requires str_elec, elec_balance and elec_primary_prod")
        storage_techs = np.intersect1d(str_elec, elec_balance)
        for area in areas:
            demand_for_elec = float(hourly_balance["elec_demand_w_shift"].loc[{"area": area}].sum().values) / 1000
            demand_for_CH4 = float(hourly_balance["methanation_input"].loc[{"area": area}].sum().values) / 1000
            demand_for_H2 = float(hourly_balance["electrolysis_input"].loc[{"area": area}].sum().values) / 1000
            exports = float(hourly_balance["exports"].loc[{"area": area}].sum().values) / 1000
            curtailment = float(hourly_balance["curtailment"].loc[{"area": area}].sum().values) / 1000
            imports = float(hourly_balance["imports"].loc[{"area": area}].sum().values) / 1000

            demand_for_storage = sum(float(hourly_balance[f"{tech}_input"].loc[{"area": area}].sum().values) / 1000 for tech in storage_techs)
            prod_from_storage = sum(float(hourly_balance[f"{tech}"].loc[{"area": area}].sum().values) / 1000 for tech in storage_techs)
            primary_prod = sum(float(hourly_balance[f"{tech}"].loc[{"area": area}].sum().values) / 1000 for tech in elec_primary_prod.values)
            prod_from_CH4 = (float(hourly_balance["ch4_ocgt"].loc[{"area": area}].sum().values) / 1000
                             + float(hourly_balance["ch4_ccgt"].loc[{"area": area}].sum().values) / 1000
                             + float(hourly_balance["ocgt_coge"].loc[{"area": area}].sum().values) / 1000)
            prod_from_H2 = float(hourly_balance["h2_ccgt"].loc[{"area": area}].sum().values) / 1000
            losses_storage = demand_for_storage - prod_from_storage

            df.loc[area, "demand_for_elec"] = demand_for_elec
            df.loc[area, "demand_for_CH4"] = demand_for_CH4
            df.loc[area, "demand_for_H2"] = demand_for_H2
            df.loc[area, "exports"] = exports
            df.loc[area, "imports"] = imports
            df.loc[area, "curtailment"] = curtailment
            df.loc[area, "primary_prod"] = primary_prod
            df.loc[area, "prod_from_CH4"] = prod_from_CH4
            df.loc[area, "prod_from_H2"] = prod_from_H2
            df.loc[area, "losses_storage"] = losses_storage
            df.loc[area, "net_result"] = (primary_prod + prod_from_CH4 + prod_from_H2 + imports
                                          - demand_for_elec - demand_for_CH4 - demand_for_H2
                                          - curtailment - exports - losses_storage)
    else:
        raise ValueError(f"Unknown vector {vector!r}: expected 'elec', 'CH4' or 'H2'")

    demand_col = {"elec": "demand_for_elec", "CH4": "demand_for_ch4", "H2": "demand_for_h2"}[vector]
    for area in areas:
        net_twh = df.loc[area, "net_result"]
        demand_twh = df.loc[area, demand_col]
        pct_of_demand = (net_twh / demand_twh * 100) if demand_twh > 1e-6 else 0.0
        if vector == "elec":
            # Curtailment is tracked explicitly, so net_result should be ~0 regardless of sign.
            flag = abs(net_twh * 1e6) > 1.0
        else:
            # net_result here = usage - supply - trade: positive means usage > supply (a real
            # deficit - always worth investigating), negative means supply > usage (implicit,
            # untracked curtailment allowed by the ">=" adequacy constraint - expected up to a
            # few % of demand, especially with the barrier-without-crossover solver).
            deficit = net_twh > 1e-4
            large_surplus = pct_of_demand < -5
            flag = deficit or large_surplus
        if flag:
            print(f"[warn] {vector} balance mismatch in {area}: {net_twh * 1e6:.3f} MWh "
                 f"({pct_of_demand:.2f}% of demand)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Batch/single-run export helpers — shared by run_batch.py and example.py so both
# write the exact same on-disk layout (see utils_batch.py for reading it back).
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_DUAL_CONSTRAINTS = [
    "max_capacity_prod",
    "max_capacity_conv",
    "max_capacity_str",
    "min_capacity_prod",
    "min_capacity_conv",
    "min_capacity_str",
    "annual_biogas_import",
    "annual_methanization",
    "annual_pyrogazification",
    "methanation_CO2",
    "annual_H2_import",
]


def to_csv(obj, path):
    """Save a pandas or xarray object to CSV, skipping None (missing/optional result)."""
    if obj is None:
        return
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_csv(path)
    elif isinstance(obj, xr.DataArray):
        obj.to_pandas().to_csv(path)
    elif isinstance(obj, xr.Dataset):
        obj.to_dataframe().to_csv(path)


def save_results(m, output_dir, export_hourly=False):
    """Persist all main results for one solved ModelEOLES instance: summary, installed
    power/energy capacity, generation, load factor, vector balances (incl. trade),
    investment/O&M/vector costs, carbon footprint, per-area tech summaries, and
    (optionally) hourly dispatch. Call after extract_optimisation_results_linopy()."""
    p = Path(output_dir)

    # ── 1. Summary (scalar indicators per country) ───────────────────────────
    to_csv(m.summary, p / "summary.csv")

    # ── 2. Installed power [GW] — tech × area ────────────────────────────────
    to_csv(m.installed_power, p / "installed_power_GW.csv")

    # ── 3. Energy capacity [GWh] — tech × area ───────────────────────────────
    to_csv(m.energy_capacity, p / "energy_capacity_GWh.csv")

    # ── 4. Generation per technology [TWh/yr] — tech × area ──────────────────
    to_csv(m.generation_per_technology, p / "generation_per_tech_TWh.csv")

    # ── 5. Load factor [%] ───────────────────────────────────────────────────
    to_csv(m.load_factor, p / "load_factor_pct.csv")

    # ── 6. Vector balances — supply and usage in TWh ─────────────────────────
    sol = m.model.solution
    _trade = {
        "CH4": ("gas annual import", "gas annual export"),
        "H2":  ("H2 annual import",  "H2 annual export"),
    }
    for vector, supply, usage in [
        ("elec", m.elec_supply, m.elec_usage),
        ("CH4",  m.CH4_supply,  m.CH4_usage),
        ("H2",   m.H2_supply,   m.H2_usage),
    ]:
        if vector in _trade:
            # CH4/H2 trade is annual (uncapped, lossless system-wide balance — see
            # ModelEOLES.define_constraints), tracked by separate "annual import"/"export"
            # variables rather than in hourly_balance.
            var_imp, var_exp = _trade[vector]
            try:
                if var_imp in sol and var_exp in sol:
                    imp = sol[var_imp].sum("year").expand_dims({"tech": [f"{vector}_trade_imports"]})
                    exp = sol[var_exp].sum("year").expand_dims({"tech": [f"{vector}_trade_exports"]})
                    supply = xr.concat([supply, imp], dim="tech")
                    usage  = xr.concat([usage,  exp], dim="tech")
                    to_csv(sol[var_imp].to_pandas(), p / f"balance_{vector}_trade_import_TWh.csv")
                    to_csv(sol[var_exp].to_pandas(), p / f"balance_{vector}_trade_export_TWh.csv")
            except Exception as e:
                print(f"    [warn] {vector} trade integration failed: {e}")
        elif vector == "elec" and "imports" in m.hourly_balance.data_vars and "exports" in m.hourly_balance.data_vars:
            # Electricity trade is hourly and capacity-limited (see links.csv), already tracked
            # as "imports"/"exports" data_vars in hourly_balance — fold it in the same way as
            # CH4/H2 trade above, so it shows up in balance_elec_supply/usage_TWh.csv too.
            try:
                imp_annual = m.hourly_balance["imports"].sum("hour") / 1000
                exp_annual = m.hourly_balance["exports"].sum("hour") / 1000
                imp = imp_annual.expand_dims({"tech": ["elec_trade_imports"]})
                exp = exp_annual.expand_dims({"tech": ["elec_trade_exports"]})
                supply = xr.concat([supply, imp], dim="tech")
                usage  = xr.concat([usage,  exp], dim="tech")
                to_csv(imp_annual.to_pandas(), p / "balance_elec_trade_import_TWh.csv")
                to_csv(exp_annual.to_pandas(), p / "balance_elec_trade_export_TWh.csv")
            except Exception as e:
                print(f"    [warn] elec trade integration failed: {e}")
        to_csv(supply, p / f"balance_{vector}_supply_TWh.csv")
        to_csv(usage,  p / f"balance_{vector}_usage_TWh.csv")

    # ── 7. Annualised investment costs [M€/yr] ────────────────────────────────
    to_csv(m.new_capacity_annualized_costs,        p / "investment_costs_annualized_M€yr.csv")
    to_csv(m.new_energy_capacity_annualized_costs, p / "investment_costs_energy_annualized_M€yr.csv")

    # ── 8. O&M costs [M€/yr] ─────────────────────────────────────────────────
    to_csv(m.OM_cost, p / "OM_costs_M€yr.csv")
    if getattr(m, "carbon_cost", None) is not None:
        to_csv(m.carbon_cost, p / "carbon_cost_M€yr.csv")

    # ── 9. Cost decomposition by vector ───────────────────────────────────────
    try:
        costs_elec, costs_CH4, costs_H2 = compute_costs(
            m.annuities, m.fOM, m.vOM, m.storage_annuities,
            m.generation_per_technology, m.installed_power, m.existing_capacity,
            m.energy_capacity, m.existing_energy_capacity,
            m.nb_years, m.elec_balance, m.str, m.CH4_balance, m.H2_balance,
        )
        cost_by_vector = pd.Series({
            "costs_elec_M€": float(costs_elec),
            "costs_CH4_M€":  float(costs_CH4),
            "costs_H2_M€":   float(costs_H2),
        })
        cost_by_vector.to_csv(p / "cost_by_vector_M€.csv", header=["value"])
    except Exception as e:
        print(f"    [warn] compute_costs failed: {e}")

    # ── 10. Carbon footprint [MtCO2eq/yr] ────────────────────────────────────
    to_csv(m.footprint, p / "carbon_footprint.csv")

    # ── 11. Per-area technology summary (capacity + generation + losses) ──────
    for area in m.countries:
        try:
            export_tech_summary(
                m.installed_power, m.energy_capacity, m.generation_per_technology,
                m.hourly_balance, m.profits, area, output_dir, nb_years=m.nb_years,
            )
        except Exception as e:
            print(f"    [warn] export_tech_summary({area}) failed: {e}")

    # ── 12. Hourly dispatch (optional — large files) ──────────────────────────
    if export_hourly:
        for area in m.countries:
            try:
                export_hourly_dispatch(
                    m.hourly_balance, area, output_dir,
                    nb_years=m.nb_years, spot_price=m.spot_price,
                )
            except Exception as e:
                print(f"    [warn] export_hourly_dispatch({area}) failed: {e}")

    print(f"    Results saved -> {output_dir}")


def save_interconnection_stats(m, output_dir, links_csv="inputs/area_indexed/links.csv", ic_area="FR"):
    """Export hourly interconnection utilization stats for `ic_area`'s cross-border links
    to {output_dir}/interconnection_stats.csv. No-op if there are no interconnections
    (single-country run) or if links_csv is missing."""
    links_path = Path(links_csv)
    if not links_path.exists():
        print(f"    [warn] links.csv not found at {links_path}")
        return
    sol = m.model.solution
    if "export" not in sol:
        return  # single-country run: no interconnections

    links = pd.read_csv(links_path, index_col=0)
    rows = []
    for partner in links.columns:
        if partner == ic_area:
            continue
        for area_from, area_to in [(ic_area, partner), (partner, ic_area)]:
            try:
                cap = float(links.loc[area_from, area_to])
            except (KeyError, ValueError):
                continue
            if not (pd.notna(cap) and cap > 0):
                continue
            try:
                flow = sol["export"].sel(area=area_from, area_bis=area_to).values.flatten()
                util = flow / cap * 100
                n_h  = len(flow)
                rows.append({
                    "direction":         f"{area_from}→{area_to}",
                    "partner":           partner,
                    "cap_gw":            cap,
                    "mean_util_pct":     float(util.mean()),
                    "mean_flow_gw":      float(flow.mean()),
                    "pct_hours_at_full": float((util >= 99).mean() * 100),
                    "pct_hours_idle":    float((util <= 1).mean() * 100),
                    "energy_twh":        float(flow.mean() * n_h / 1000),
                })
            except Exception as e:
                print(f"    [warn] interconnect {area_from}->{area_to}: {e}")

    if rows:
        pd.DataFrame(rows).set_index("direction").to_csv(
            Path(output_dir) / "interconnection_stats.csv"
        )
        print(f"    Interconnection stats saved -> {output_dir}/interconnection_stats.csv")


def save_duals(m, output_dir, dual_constraints=None):
    """Export dual variables for key constraints, one CSV per constraint in {output_dir}/duals/.
    Defaults to DEFAULT_DUAL_CONSTRAINTS; constraints not present in the model are skipped."""
    dual_constraints = DEFAULT_DUAL_CONSTRAINTS if dual_constraints is None else dual_constraints
    duals_dir = Path(output_dir) / "duals"
    duals_dir.mkdir(exist_ok=True)
    for cname in dual_constraints:
        try:
            dual = m.model.constraints[cname].dual
            if dual is None:
                continue
            dual.to_series().dropna().to_csv(duals_dir / f"dual_{cname}.csv", header=["dual"])
        except Exception as e:
            print(f"    [warn] dual export for '{cname}' failed: {e}")
    print(f"    Duals saved -> {duals_dir}")


