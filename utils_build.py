"""
Small helper functions called while the model is being built (annuities).
"""


def calculate_annuities_capex_xr(discount_rate, capex, construction_time, lifetime):
    annuities = discount_rate * capex * ((1+discount_rate)**construction_time) / (1 - (1 + discount_rate)**(-lifetime))
    return annuities

def calculate_annuities_storage_capex_xr(discount_rate, storage_capex, construction_time, lifetime):
    annuities = discount_rate * storage_capex * ((1+discount_rate)**construction_time) / (1 - (1 + discount_rate)**(-lifetime))
    return annuities


