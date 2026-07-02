import pandas as pd
import numpy as np
import pandas_gbq as pgbq
import os
import pandas_gbq
import warnings
import datetime as dt
from datetime import date
from openpyxl import load_workbook



def targets_contratos(df, var_target, dt_ref, months):

    df["dt_ref"] = dt_ref
    df["var_target"] = df[var_target]

    # Calculate time2comp: difference in months between com_comp and activated_at, only if com_comp is not null
    mask_target_notnull = df["var_target"].notnull()
    time2target = pd.Series(np.nan, index=df.index)
    if mask_target_notnull.any():
        time2target_period = (
            pd.to_datetime(df.loc[mask_target_notnull, "var_target"]).dt.to_period("M")
            - pd.to_datetime(df.loc[mask_target_notnull, "activated_at"]).dt.to_period("M")
        )
        time2target.loc[mask_target_notnull] = time2target_period.apply(lambda x: x.n if pd.notnull(x) else np.nan)
    df["time2target"] = time2target

    # Calculate current_maturity: difference in months between dt_ref and activated_at/requested_at
    act_or_req = pd.to_datetime(df["activated_at"].fillna(df["requested_at"]))
    current_maturity = (
        pd.to_datetime(df["dt_ref"]).dt.to_period("M")
        - act_or_req.dt.to_period("M")
    )
    df["current_maturity"] = current_maturity.apply(lambda x: x.n if pd.notnull(x) else np.nan)

    # Calculate target_comp_3 following the business logic
    activated_notnull = df['activated_at'].notnull()
    time2target_notnull = df['time2target'].notnull()
    time2target_float = df['time2target'].astype(float)
    current_maturity_float = df['current_maturity'].astype(float)

    cond1 = activated_notnull & time2target_notnull & (time2target_float <= months) & (current_maturity_float >= months)
    cond2 = (~activated_notnull) | (current_maturity_float < months)

    name_target = f"target_{var_target}_{months}"

    df[name_target] = np.select(
        [cond1, cond2],
        [1, np.nan],
        default = 0
    )

    return df