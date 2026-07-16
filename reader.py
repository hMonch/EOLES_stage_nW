""" The present file defines a function that reads the excel scenario file
    and translate it into the inputs """

"""The excel file must follow the same structure as mentioned here:
    Constants, Costs, tech_parameters, reserve, Capacities"""

from pathlib import Path
import pandas as pd


def excel_to_inputs(path, inputs_path, single_node=True):
    xl = pd.ExcelFile(path)

    costs = pd.read_excel(xl, 'Costs', index_col=0)
    costs.to_csv(f'{inputs_path}/costs.csv')

    tech_param = pd.read_excel(xl, 'tech_parameters', index_col=0)
    tech_param.to_csv(f'{inputs_path}/tech_parameters.csv')

    reserve = pd.read_excel(xl, 'reserve', index_col=0)
    reserve.to_csv(f"{inputs_path}/reserve.csv")

    fuel_prices = pd.read_excel(xl, 'fuel_prices', index_col=0)
    fuel_prices.to_csv(f"{inputs_path}/fuel_prices.csv")

    if single_node:
        capacities = pd.read_excel(xl, 'Capacities', index_col=0)
        capacities.to_csv(f"{inputs_path}/single-node/capacities.csv")

    else:  # More than one node
        existing_capacity = pd.read_excel(xl, 'exist_cap', index_col=0)
        existing_capacity.to_csv(f"{inputs_path}/area_indexed/existing_capacity.csv")
        existing_energy_capacity = pd.read_excel(xl, 'exist_en_cap', index_col=0)
        existing_energy_capacity.to_csv(f"{inputs_path}/area_indexed/existing_energy_capacity.csv")

        maximum_capacity = pd.read_excel(xl, 'max_cap', index_col=0)
        maximum_capacity.to_csv(f"{inputs_path}/area_indexed/maximum_capacity.csv")
        maximum_energy_capacity = pd.read_excel(xl, 'max_en_cap', index_col=0)
        maximum_energy_capacity.to_csv(f"{inputs_path}/area_indexed/maximum_energy_capacity.csv")

        minimum_capacity = pd.read_excel(xl, 'min_cap', index_col=0)
        minimum_capacity.to_csv(f"{inputs_path}/area_indexed/minimum_capacity.csv")
        minimum_energy_capacity = pd.read_excel(xl, 'min_en_cap', index_col=0)
        minimum_energy_capacity.to_csv(f"{inputs_path}/area_indexed/minimum_energy_capacity.csv")

        links = pd.read_excel(xl, 'links', index_col=0)
        links.to_csv(f"{inputs_path}/area_indexed/links.csv")

        biogas_potential = pd.read_excel(xl, 'biogas_potential', index_col=0)
        biogas_potential.to_csv(f"{inputs_path}/area_indexed/biogas_potential.csv")

        # Optional sheet: per-country carbon budget (MtCO2/yr, one row "carbon_budget" x area).
        # Not part of the original workbook layout — if you haven't added this sheet yet,
        # inputs/area_indexed/carbon_budget.csv keeps whatever value is already there
        # (defaults to 0 for every country, see ModelEOLES.load_inputs).
        if 'carbon_budget' in xl.sheet_names:
            carbon_budget = pd.read_excel(xl, 'carbon_budget', index_col=0)
            carbon_budget.to_csv(f"{inputs_path}/area_indexed/carbon_budget.csv")
        else:
            print("Note: no 'carbon_budget' sheet found in the Excel file — "
                  "keeping the existing inputs/area_indexed/carbon_budget.csv as is.")

    constants = pd.read_excel(xl, 'Constants', index_col=0)
    constants.to_csv(f"{inputs_path}/constants.csv")


if __name__ == "__main__":
    _here = Path(__file__).parent
    excel_to_inputs(_here / "Scenario_data_EUR_plus.xlsx", inputs_path=_here / "inputs", single_node=False)
