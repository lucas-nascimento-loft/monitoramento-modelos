import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PSI_EPS = 1e-6
PSI_STATUS = {
    "stable": 0.10,
    "moderate": 0.25,
}

# Blend4 model variables that must always use categories [0, 1]
BINARY_MODEL_VARS = [
    "property_type",
    "flag_tem__contratos_anteriores",
    "flag_teve_boleto_atrasado__contratos_anteriores",
    "agency_pc4_mais_100_contratos__pc_categorias_is_null",
    "city_pc4_mais_100_contratos__pc_categorias_is_null",
]


def _psi_status(psi_value: float) -> str:
    if pd.isna(psi_value):
        return "missing"
    if psi_value < PSI_STATUS["stable"]:
        return "stable"
    if psi_value < PSI_STATUS["moderate"]:
        return "moderate"
    return "unstable"


def _distribution_from_counts(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    dist = counts / total
    return dist.clip(lower=PSI_EPS)


def calculate_psi(expected_counts: pd.Series, actual_counts: pd.Series) -> float:
    """
    PSI = sum((actual - expected) * ln(actual / expected))
    Both series must share the same index (bin/category labels).
    """
    aligned = pd.concat(
        [
            _distribution_from_counts(expected_counts),
            _distribution_from_counts(actual_counts),
        ],
        axis=1,
        keys=["expected", "actual"],
    ).fillna(PSI_EPS)

    expected = aligned["expected"]
    actual = aligned["actual"]

    return float(((actual - expected) * np.log(actual / expected)).sum())


def _is_forced_binary(variable: str, binary_vars: List[str]) -> bool:
    return variable in binary_vars


def _quantile_bin_labels(edges: List[float]) -> List[str]:
    edges_arr = np.asarray(edges, dtype=float)
    return [
        f"({edges_arr[i]:.6f}, {edges_arr[i + 1]:.6f}]"
        for i in range(len(edges_arr) - 1)
    ]


def _count_binary(series: pd.Series, categories: List[int] = None) -> pd.Series:
    categories = categories or [0, 1]
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    counts = values.value_counts()
    return counts.reindex(categories, fill_value=0).astype(int)


def _count_quantile(series: pd.Series, edges: List[float]) -> pd.Series:
    edges_arr = np.asarray(edges, dtype=float)
    bin_labels = _quantile_bin_labels(edges_arr.tolist())

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.Series(0, index=bin_labels, dtype=int)

    labels = pd.cut(
        values,
        bins=edges_arr,
        labels=bin_labels,       # ← fix: usa os mesmos labels do artifact
        include_lowest=True,
        duplicates="drop",
    )

    counts = labels.value_counts()
    return counts.reindex(bin_labels, fill_value=0).astype(int)


def _build_quantile_spec(
    series: pd.Series,
    n_bins: int = 10,
) -> Optional[Dict[str, Any]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(clean.quantile(quantiles).values)

    if len(edges) < 2:
        edges = np.array([clean.min(), clean.max() + PSI_EPS])

    edges = edges.tolist()
    expected_counts = _count_quantile(clean, edges)

    return {
        "type": "quantile",
        "edges": edges,
        "bin_labels": _quantile_bin_labels(edges),
        "expected_counts": expected_counts.to_dict(),
    }


def _build_binary_spec(
    series: pd.Series,
    categories: List[int] = None,
) -> Optional[Dict[str, Any]]:
    categories = categories or [0, 1]
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None

    expected_counts = _count_binary(clean, categories=categories)

    return {
        "type": "binary",
        "categories": categories,
        "expected_counts": expected_counts.to_dict(),
    }


def build_psi_reference_artifact(
    df_reference: pd.DataFrame,
    variables: List[str],
    n_bins: int = 10,
    binary_vars: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a single PSI reference artifact from the train dataset.

    Stores, per variable:
      - type: 'binary' or 'quantile'
      - bins/categories
      - expected_counts from train

    This artifact is enough for daily monitoring without df_train.
    """
    binary_vars = binary_vars or BINARY_MODEL_VARS
    variable_specs: Dict[str, Dict[str, Any]] = {}

    for col in variables:
        if col not in df_reference.columns:
            continue

        if _is_forced_binary(col, binary_vars):
            spec = _build_binary_spec(df_reference[col], categories=[0, 1])
        else:
            spec = _build_quantile_spec(df_reference[col], n_bins=n_bins)

        if spec is not None:
            variable_specs[col] = spec

    artifact = {
        "metadata": metadata or {},
        "variables": variable_specs,
    }
    return artifact


def count_from_spec(series: pd.Series, spec: Dict[str, Any]) -> pd.Series:
    """Count observations using the stored reference spec."""
    spec_type = spec["type"]

    if spec_type == "binary":
        categories = spec.get("categories", [0, 1])
        return _count_binary(series, categories=categories)

    if spec_type == "quantile":
        return _count_quantile(series, spec["edges"])

    raise ValueError(f"Unsupported spec type: {spec_type}")


def calculate_psi_from_reference(
    actual_series: pd.Series,
    variable_spec: Dict[str, Any],
) -> float:
    expected_counts = pd.Series(variable_spec["expected_counts"], dtype=float)
    actual_counts = count_from_spec(actual_series, variable_spec)

    # Align indexes explicitly
    if variable_spec["type"] == "binary":
        index = [str(x) for x in variable_spec.get("categories", [0, 1])]
        expected_counts.index = expected_counts.index.astype(str)
        actual_counts.index = actual_counts.index.astype(str)
        expected_counts = expected_counts.reindex(index, fill_value=0)
        actual_counts = actual_counts.reindex(index, fill_value=0)
    else:
        index = variable_spec["bin_labels"]
        expected_counts = expected_counts.reindex(index, fill_value=0)
        actual_counts = actual_counts.reindex(index, fill_value=0)

    return calculate_psi(expected_counts, actual_counts)


def calculate_psi_table_from_reference(
    df_actual: pd.DataFrame,
    artifact: Dict[str, Any],
    comparison_label: str,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute PSI using only the saved artifact + actual data.
    Does NOT require df_train.
    """
    variable_specs = artifact["variables"]
    variables = variables or list(variable_specs.keys())

    rows = []
    for col in variables:
        if col not in variable_specs or col not in df_actual.columns:
            continue

        spec = variable_specs[col]
        psi_value = calculate_psi_from_reference(df_actual[col], spec)

        rows.append(
            {
                "variable": col,
                "comparison": comparison_label,
                "psi": round(psi_value, 6),
                "status": _psi_status(psi_value),
                "n_actual": int(df_actual[col].notna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def calculate_development_psi_baselines(
    df: pd.DataFrame,
    variables: List[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
    oot_mask: pd.Series,
    n_bins: int = 10,
    binary_vars: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute PSI baselines:
      - test vs train
      - oot vs train

    Returns:
      baseline table + full PSI reference artifact
    """
    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()
    df_oot = df.loc[oot_mask].copy()

    artifact = build_psi_reference_artifact(
        df_reference=df_train,
        variables=variables,
        n_bins=n_bins,
        binary_vars=binary_vars,
        metadata=metadata,
    )

    psi_test = calculate_psi_table_from_reference(
        df_actual=df_test,
        artifact=artifact,
        comparison_label="test_vs_train",
        variables=variables,
    )

    psi_oot = calculate_psi_table_from_reference(
        df_actual=df_oot,
        artifact=artifact,
        comparison_label="oot_vs_train",
        variables=variables,
    )

    baseline = pd.concat([psi_test, psi_oot], ignore_index=True)
    return baseline, artifact


def pivot_psi_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    """One row per variable with reference PSI values from development."""
    return (
        baseline.pivot(index="variable", columns="comparison", values="psi")
        .rename(
            columns={
                "test_vs_train": "psi_test_vs_train_ref",
                "oot_vs_train": "psi_oot_vs_train_ref",
            }
        )
        .reset_index()
    )


def save_psi_reference(
    artifact: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_psi_reference(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def monitor_daily_psi(
    df_production: pd.DataFrame,
    artifact: Dict[str, Any],
    date_col: str = "requested_at",
    baseline_ref: Optional[pd.DataFrame] = None,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Daily PSI of production vs train reference stored in artifact.
    Does NOT require df_train.
    """
    work = df_production.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["monitoring_day"] = work[date_col].dt.strftime("%Y-%m-%d")

    rows = []
    for day, group in work.groupby("monitoring_day"):
        daily_psi = calculate_psi_table_from_reference(
            df_actual=group,
            artifact=artifact,
            comparison_label="production_vs_train",
            variables=variables,
        )
        daily_psi["monitoring_day"] = day
        rows.append(daily_psi)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    if baseline_ref is not None:
        result = result.merge(baseline_ref, on="variable", how="left")
        result["delta_vs_test_ref"] = result["psi"] - result["psi_test_vs_train_ref"]
        result["delta_vs_oot_ref"] = result["psi"] - result["psi_oot_vs_train_ref"]

    return result.sort_values(["monitoring_day", "psi"], ascending=[True, False])

def monitor_weekly_psi(
    df_production: pd.DataFrame,
    artifact: Dict[str, Any],
    week_col: str = "year_week",
    baseline_ref: Optional[pd.DataFrame] = None,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Weekly PSI of production vs train reference stored in artifact.
    Expects df_production to already contain year_week (and week_start for ordering).
    Does NOT require df_train.
    """
    required = {week_col}
    missing = required - set(df_production.columns)
    if missing:
        raise ValueError(
            f"Missing columns for weekly PSI: {sorted(missing)}. "
            "Run prepare_week_columns before calling monitor_weekly_psi."
        )

    work = df_production.copy()

    sort_col = "week_start" if "week_start" in work.columns else week_col
    week_order = (
        work[[week_col, sort_col]]
        .drop_duplicates()
        .sort_values(sort_col)[week_col]
        .tolist()
    )

    rows = []
    for week in week_order:
        group = work[work[week_col] == week]
        if group.empty:
            continue

        weekly_psi = calculate_psi_table_from_reference(
            df_actual=group,
            artifact=artifact,
            comparison_label="production_vs_train",
            variables=variables,
        )
        weekly_psi["monitoring_week"] = week
        rows.append(weekly_psi)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    if baseline_ref is not None:
        result = result.merge(baseline_ref, on="variable", how="left")
        result["delta_vs_test_ref"] = result["psi"] - result["psi_test_vs_train_ref"]
        result["delta_vs_oot_ref"] = result["psi"] - result["psi_oot_vs_train_ref"]

    return result.sort_values(["monitoring_week", "psi"], ascending=[True, False])

def calculate_development_psi_baselines_final(
    df: pd.DataFrame,
    variables: List[str],
    n_bins: int = 10,
    binary_vars: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute PSI baselines:

    Returns:
      baseline table + full PSI reference artifact
    """

    artifact = build_psi_reference_artifact(
        df_reference=df,
        variables=variables,
        n_bins=n_bins,
        binary_vars=binary_vars,
        metadata=metadata,
    )

    return artifact

import matplotlib.pyplot as plt


def _pct_from_counts(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    return (counts / total * 100).round(2)


def _align_distribution_counts(
    expected_counts: pd.Series,
    actual_counts: pd.Series,
    variable_spec: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, List[str]]:
    spec_type = variable_spec["type"]

    if spec_type == "binary":
        index = [str(x) for x in variable_spec.get("categories", [0, 1])]
        expected_counts = expected_counts.copy()
        expected_counts.index = expected_counts.index.astype(str)
        actual_counts = actual_counts.copy()
        actual_counts.index = actual_counts.index.astype(str)
    else:
        index = variable_spec["bin_labels"]

    expected_counts = expected_counts.reindex(index, fill_value=0)
    actual_counts = actual_counts.reindex(index, fill_value=0)
    return expected_counts, actual_counts, index


def build_distribution_comparison(
    actual_series: pd.Series,
    variable_spec: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compare expected (reference/train) vs actual distributions
    using the frozen bins/categories from psi_reference_artifact.

    Returns one row per bin/category with counts and percentages.
    """
    expected_counts = pd.Series(variable_spec["expected_counts"], dtype=float)
    actual_counts = count_from_spec(actual_series, variable_spec)

    expected_counts, actual_counts, index = _align_distribution_counts(
        expected_counts, actual_counts, variable_spec
    )

    expected_pct = _pct_from_counts(expected_counts)
    actual_pct = _pct_from_counts(actual_counts)

    rows = []
    for label in index:
        rows.append(
            {
                "bin_label": label,
                "expected_count": int(expected_counts.loc[label]),
                "actual_count": int(actual_counts.loc[label]),
                "expected_pct": float(expected_pct.loc[label]),
                "actual_pct": float(actual_pct.loc[label]),
                "delta_pct": round(
                    float(actual_pct.loc[label] - expected_pct.loc[label]), 2
                ),
            }
        )

    return pd.DataFrame(rows)

def plot_distribution_comparison(
    comparison_df: pd.DataFrame,
    variable: str,
    spec_type: str,
    title: str = "Distribuição: referência vs observado",
    figsize: Tuple[int, int] = (12, 5),
    show_labels: bool = True,
    show_counts: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Continuous (quantile): grouped bars per bin.
    Binary: grouped bars for categories 0 and 1.
    """
    if comparison_df is None or comparison_df.empty:
        raise ValueError("comparison_df is empty — no data to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    if spec_type == "binary":
        x_labels = comparison_df["bin_label"].astype(str).tolist()
    else:
        x_labels = [f"B{i+1}" for i in range(len(comparison_df))]

    x = np.arange(len(comparison_df))
    width = 0.38

    bars_expected = ax.bar(
        x - width / 2,
        comparison_df["expected_pct"],
        width,
        label="Esperado (referência)",
        color="#94A3B8",
        alpha=0.95,
    )
    bars_actual = ax.bar(
        x + width / 2,
        comparison_df["actual_pct"],
        width,
        label="Observado (produção)",
        color="#2563EB",
        alpha=0.95,
    )

    if show_labels:
        for bars in (bars_expected, bars_actual):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.8,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    if show_counts:
        for xi, row in zip(x, comparison_df.itertuples()):
            ax.text(
                xi,
                -4,
                f"n_ref={row.expected_count}\nn_obs={row.actual_count}",
                ha="center",
                va="top",
                fontsize=7,
                color="#475569",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45 if spec_type != "binary" else 0, ha="right")
    ax.set_ylabel("Proporção (%)")
    ax.set_xlabel("Categoria" if spec_type == "binary" else "Bin (mesmos cortes do PSI)")
    ax.set_title(f"{title}\n{variable}")
    ax.set_ylim(bottom=-8 if show_counts else 0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax

def plot_variable_distribution(
    df_actual: pd.DataFrame,
    artifact: Dict[str, Any],
    variable: str,
    title: str = "Distribuição: referência vs observado",
    show_labels: bool = True,
    show_counts: bool = False,
    show: bool = True,
) -> pd.DataFrame:
    """
    Build and plot expected vs actual distribution for one variable
    using psi_reference_artifact.
    """
    if variable not in artifact["variables"]:
        raise KeyError(f"Variable not found in artifact: {variable}")
    if variable not in df_actual.columns:
        raise KeyError(f"Variable not found in dataframe: {variable}")

    spec = artifact["variables"][variable]
    comparison_df = build_distribution_comparison(df_actual[variable], spec)

    plot_distribution_comparison(
        comparison_df,
        variable=variable,
        spec_type=spec["type"],
        title=title,
        show_labels=show_labels,
        show_counts=show_counts,
    )

    if show:
        plt.show()

    return comparison_df

def plot_top_psi_distributions(
    df_actual: pd.DataFrame,
    artifact: Dict[str, Any],
    variables: Optional[List[str]] = None,
    top_n: int = 5,
    title_prefix: str = "Distribuição: referência vs observado",
) -> pd.DataFrame:
    """
    Plot distributions for the top-N variables by PSI.
    Useful after calculate_psi_table_from_reference or monitor_daily_psi.
    """
    psi_table = calculate_psi_table_from_reference(
        df_actual=df_actual,
        artifact=artifact,
        comparison_label="production_vs_train",
        variables=variables,
    ).sort_values("psi", ascending=False)

    selected = psi_table.head(top_n)["variable"].tolist()

    all_tables = []
    for variable in selected:
        comparison_df = plot_variable_distribution(
            df_actual=df_actual,
            artifact=artifact,
            variable=variable,
            title=title_prefix,
            show=True,
        )
        comparison_df.insert(0, "variable", variable)
        all_tables.append(comparison_df)

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)