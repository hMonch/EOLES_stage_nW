"""
Reading of the raw model inputs (config file, constants, time-indexed profiles,
transmission links) into the pandas / xarray objects consumed by ModelEOLES.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import xarray as xr


def get_config(path) -> dict:
    with open(Path(__file__).parent / path) as file:
        return json.load(file)
    

def read_links(path):
    """Function that loads : 1) the countries simulated and 2) the interconnection capacity"""
    cap_dict = {}
    links = pd.read_csv(Path(__file__).parent / path, index_col=0)
    countries = list(links.index)
    for area_export in links.index:
        for area_import in links.index:
            cap = links[area_import][area_export]
            if not np.isnan(cap):
                cap_dict[(area_export, area_import)] = cap
    
    return cap_dict, countries

def read_constant_xr(path, dims=None):
    df = pd.read_csv(Path(__file__).parent / path, index_col=0)

    if dims is None:
        return df

    elif len(dims) == 1:
        df.index.name = dims[0] 
        
        return xr.DataArray(
            df.iloc[:, 0].values,
            dims=[dims[0]],
            coords={dims[0]: df.index}
        )

    else:  
        df.index.name = dims[0]
        df.columns.name = dims[1]

        return xr.DataArray(
            df.values,
            dims=dims,
            coords={
                dims[0]: df.index,
                dims[1]: df.columns
            }
        )


def read_profile_xr(path, time_scale="hourly", reference_index=None, years=None):
    """
    Read a time-indexed CSV as an xr.DataArray with dims [hour/day, area].

    Parameters
    ----------
    reference_index : pd.DatetimeIndex, optional
        If provided, the returned DataArray is selected to this exact index
        (df.loc[reference_index]).  This lets the elec_demand read set the
        temporal scope and all subsequent reads automatically align to it.
    years : list of int, optional
        If provided, filter timestamps to only the listed calendar years before
        any reference_index selection.  Example: years=[2050] or years=[2045, 2055].
    """
    df = pd.read_csv(Path(__file__).parent / path, index_col=0)
    if time_scale == "hourly":
        df.index = pd.to_datetime(df.index)
        df.index.name = "hour"
        if years is not None:
            df = df[df.index.year.isin(years)]
        if reference_index is not None:
            df = df.loc[reference_index]
        return xr.DataArray(df.values,
                            dims=["hour", "area"],
                            coords={"hour": df.index, "area": df.columns})
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "day"
        if years is not None:
            df = df[df.index.year.isin(years)]
        if reference_index is not None:
            reference_index.name = "day"
            df = df.loc[reference_index]
        return xr.DataArray(df.values,
                            dims=["day", "area"],
                            coords={"day": df.index, "area": df.columns})







'''[[[ Functions used when defining the model ]]]'''


