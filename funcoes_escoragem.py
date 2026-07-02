import pandas as pd
import numpy as np

def prepare_blend4_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Blend4 variable treatment: income commitment, null flags,
    median imputation, and Robust Scaler normalization.
    """
    df = df.copy()

    # Income commitment
    df["income_commitment"] = df["rental_value"] / df["income"]
    df["income_commitment"] = df["income_commitment"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Robust Scaler statistics (median and IQR)
    scaler_stats = {
        "score_proposto__bvs": (500.0, 190.0),
        "SERASA_CHSV5": (468.0, 343.0),
        "age": (34.0, 20.0),
        "qtde_restricoes__consulta_realizada": (0.0, 1.0),
        "income_commitment": (0.4299448103970091, 0.45271472659925316),
        "agency_pc4_mais_100_contratos__pc_categorias": (0.12931034482758633, 1.0),
        "city_pc4_mais_100_contratos__pc_categorias": (0.10233029381965561, 0.037128293376093524),
    }

    fill_dict_blend4_1 = {
        col: median
        for col, (median, _) in scaler_stats.items()
    }
    fill_dict_blend4_1.update({
        "property_type": 1,
        "flag_tem__contratos_anteriores": 0,
        "flag_teve_boleto_atrasado__contratos_anteriores": 0,
    })

    # Null flags (before imputation)
    cols_flag = [
        "city_pc4_mais_100_contratos__pc_categorias",
        "agency_pc4_mais_100_contratos__pc_categorias",
    ]
    
    for col in cols_flag:
        df[f"{col}_is_null"] = df[col].isna().astype(int)

    # Median imputation
    df = df.fillna(fill_dict_blend4_1)

    # Robust Scaler (property_type and flags are not scaled)
    for col, (median, iqr) in scaler_stats.items():
        df[f"{col}__normalized4_1"] = (df[col] - median) / iqr

    return df

def predict_blend4_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Blend4 logistic regression, convert logit to probability,
    and transform probability into score (0-1000).
    """
    df = df.copy()

    df = df.assign(
        pred_blend4_1=lambda x: (
            0.28358622828722
            + (-0.567971655779376) * x["score_proposto__bvs__normalized4_1"]
            + (-0.597721591905483) * x["SERASA_CHSV5__normalized4_1"]
            + (-0.00318366325712087) * x["age__normalized4_1"]
            + (-0.0993505715207921) * x["property_type"]
            + (0.0298515619451761) * x["qtde_restricoes__consulta_realizada__normalized4_1"]
            + (0.0930821921992222) * x["income_commitment__normalized4_1"]
            + (0.0464401507537389)
            * x["agency_pc4_mais_100_contratos__pc_categorias__normalized4_1"]
            + (0.140206274527869)
            * x["city_pc4_mais_100_contratos__pc_categorias__normalized4_1"]
            + (-0.0509328445500735) * x["flag_tem__contratos_anteriores"]
            + (0.0806250431650827) * x["flag_teve_boleto_atrasado__contratos_anteriores"]
            + (-0.139807231965555)
            * x["agency_pc4_mais_100_contratos__pc_categorias_is_null"]
            + (-0.0814593999153949)
            * x["city_pc4_mais_100_contratos__pc_categorias_is_null"]
        )
    ).assign(
        pred_blend4_1=lambda x: 1 / (1 + np.exp(-x["pred_blend4_1"].astype(float)))
    )

    df["pred_blend4_1_to_score"] = round(
        (1 - df["pred_blend4_1"]) * 1000, 0
    )

    return df

def prepare_blend3_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Blend3.2 variable treatment: income commitment, null flags,
    median imputation, and Robust Scaler normalization.
    """
    df = df.copy()

    # Income commitment
    df["income_commitment"] = df["rental_value"] / df["income"]
    df["income_commitment"] = df["income_commitment"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Robust Scaler statistics (median and IQR)
    scaler_stats = {
        "SCRCRDPNMGRLPFLGBCLFCREDPGV1": (700.0, 182.0),
        "SERASA_HVA4": (668.0, 246.0),
        "age": (38.0, 21.0),
        "qtde_restricoes__consulta_realizada": (0.0, 1.0),
        "income_commitment": (0.33291792771942325, 0.35937397251265873),
        "agency_pc4_mais_100_contratos__pc_categorias": (0.1238938053097345, 1.0),
        "city_pc4_mais_100_contratos__pc_categorias": (0.10323574730354394, 0.03233543652391829),
    }

    fill_dict = {
        col: median
        for col, (median, _) in scaler_stats.items()
    }
    fill_dict.update({
        "property_type": 1,
        "flag_tem__contratos_anteriores": 0,
        "flag_teve_boleto_atrasado__contratos_anteriores": 0,
    })

    # Null flags (before imputation)
    cols_flag = [
        "city_pc4_mais_100_contratos__pc_categorias",
        "agency_pc4_mais_100_contratos__pc_categorias",
    ]
    for col in cols_flag:
        df[f"{col}_is_null"] = df[col].isna().astype(int)

    # Median imputation
    df = df.fillna(fill_dict)

    # Robust Scaler (property_type and flags are not scaled)
    for col, (median, iqr) in scaler_stats.items():
        df[f"{col}__normalized3_2"] = (df[col] - median) / iqr

    return df


def predict_blend3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Blend3.2 logistic regression, convert logit to probability,
    and transform probability into score (0-1000).
    """
    df = df.copy()

    df = df.assign(
        predict_blend3_2=lambda x: (
            -0.127297036249813
            + (-0.297212783378539) * x["SCRCRDPNMGRLPFLGBCLFCREDPGV1__normalized3_2"]
            + (-0.408969122005144) * x["SERASA_HVA4__normalized3_2"]
            + (0.292732320546562) * x["age__normalized3_2"]
            + (-0.134202392009538) * x["property_type"]
            + (0.0890819567329526) * x["qtde_restricoes__consulta_realizada__normalized3_2"]
            + (0.235331129159451) * x["income_commitment__normalized3_2"]
            + (0.0855288113746228)
            * x["agency_pc4_mais_100_contratos__pc_categorias__normalized3_2"]
            + (0.228478398386125)
            * x["city_pc4_mais_100_contratos__pc_categorias__normalized3_2"]
            + (-0.0822948528344378) * x["flag_tem__contratos_anteriores"]
            + (0.151958223727146) * x["flag_teve_boleto_atrasado__contratos_anteriores"]
            + (-0.149942669861252)
            * x["agency_pc4_mais_100_contratos__pc_categorias_is_null"]
            + (-0.0655542165633087)
            * x["city_pc4_mais_100_contratos__pc_categorias_is_null"]
        )
    ).assign(
        predict_blend3_2=lambda x: 1 / (1 + np.exp(-x["predict_blend3_2"].astype(float)))
    )

    df["predict_blend3_2_to_score"] = round(
        (1 - df["predict_blend3_2"]) * 1000, 0
    )

    return df