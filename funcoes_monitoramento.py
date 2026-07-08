import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, List, Dict


def _sunday_week_start(dates: pd.Series) -> pd.Series:
    """Return the Sunday that starts the week (Sun–Sat) containing each date."""
    dates = pd.to_datetime(dates)
    days_since_sunday = (dates.dt.dayofweek + 1) % 7
    return (dates - pd.to_timedelta(days_since_sunday, unit="D")).dt.normalize()


def _week_label_from_date(date) -> str:
    """Format a date as its Sunday-week label (YYYY-MM-DD)."""
    ts = pd.to_datetime(date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot compute week label from invalid date: {date!r}")
    week_start = _sunday_week_start(pd.Series([ts])).iloc[0]
    if pd.isna(week_start):
        raise ValueError(f"Cannot compute week label from invalid date: {date!r}")
    return week_start.strftime("%Y-%m-%d")


def _assign_week_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add week_start, year_week, year and week_of_year (Sun–Sat weeks)."""
    df["week_start"] = _sunday_week_start(df[date_col])
    df["year_week"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df["year"] = df["week_start"].dt.year
    first_sunday_of_year = _sunday_week_start(
        pd.to_datetime(df["year"].astype(str) + "-01-01")
    )
    df["week_of_year"] = (
        (df["week_start"] - first_sunday_of_year).dt.days // 7 + 1
    ).astype(int)
    return df


def _ordered_week_labels(df: pd.DataFrame, week_col: str = "year_week") -> List[str]:
    sort_col = "week_start" if "week_start" in df.columns else week_col
    return (
        df[[week_col, sort_col]]
        .drop_duplicates()
        .sort_values(sort_col)[week_col]
        .tolist()
    )


def prepare_week_columns(df: pd.DataFrame, date_col: str = "requested_at") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return _assign_week_columns(df, date_col)


def compute_category_mix_by_week(
    df: pd.DataFrame,
    category_col: str,
    date_col: str = "requested_at",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns percentage mix by week and chronological week order.
    """
    work_df = prepare_week_columns(
        df[[date_col, category_col]].dropna(subset=[category_col]),
        date_col,
    )

    pct_df = (
        pd.crosstab(work_df["year_week"], work_df[category_col], normalize="index")
        .mul(100)
        .fillna(0)
    )

    week_order = _ordered_week_labels(work_df)
    pct_df = pct_df.reindex(week_order).fillna(0)
    return pct_df, week_order



def plot_stacked_category_mix(
    pct_df: pd.DataFrame,
    title: str = "Categoria distribuida por semana",
    xlabel: str = "Semana",
    ylabel: str = "Proporção (%)",
    legend_title: str = "Categoria",
    min_label_pct: float = 1.0,
    figsize: Tuple[int, int] = (14, 6),
    category_order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = False,
    reverse_legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 100% stacked bar chart from pct_df (rows=weeks, cols=categories).

    Parameters
    ----------
    category_order : list, optional
        Logical order of categories (e.g. ["A", "B", "C", "D", "E"]).
        Used for column ordering and legend when reverse_legend=True.
    reverse_stack : bool, default False
        If False, first category in category_order is drawn at the bottom.
        If True, last category in category_order is drawn at the bottom
        (e.g. E at base, A on top).
    reverse_legend : bool, default True
        If True and reverse_stack=True, legend keeps logical category_order
        instead of following matplotlib's draw order.
    """
    if pct_df.empty:
        raise ValueError("pct_df is empty — no data to plot.")

    if category_order:
        pct_df = pct_df.reindex(columns=category_order, fill_value=0)
    else:
        category_order = pct_df.columns.tolist()

    stack_order = category_order[::-1] if reverse_stack else category_order

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pct_df))
    bottom = np.zeros(len(pct_df))
    default_colors = plt.cm.tab10.colors

    for i, category in enumerate(stack_order):
        values = pct_df[category].values
        color = (
            color_map.get(str(category))
            if color_map
            else default_colors[i % len(default_colors)]
        )
        label_color = (
            text_color_map.get(str(category), "white")
            if text_color_map
            else "white"
        )

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=str(category),
            color=color,
            width=0.8,
        )

        for bar, val in zip(bars, values):
            if val >= min_label_pct:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=label_color,
                )

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pct_df.index, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)

    if reverse_stack and reverse_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            labels[::-1],
            title=legend_title,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    else:
        ax.legend(
            title=legend_title,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    plt.tight_layout()
    return fig, ax

def _format_volume(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)



def compute_category_mix(
    df: pd.DataFrame,
    category_col: str,
    date_col: str = "requested_at",
    time_grain: Literal["day", "week"] = "week",
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    work_df = df[[date_col, category_col]].dropna(subset=[category_col]).copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])

    if time_grain == "day":
        work_df["period"] = work_df[date_col].dt.strftime("%Y-%m-%d")
        period_order = sorted(work_df["period"].unique())
    else:
        work_df = _assign_week_columns(work_df, date_col)
        work_df["period"] = work_df["year_week"]
        period_order = _ordered_week_labels(work_df, week_col="period")

    pct_df = (
        pd.crosstab(work_df["period"], work_df[category_col], normalize="index")
        .mul(100)
        .fillna(0)
    )

    pct_df = pct_df.reindex(period_order).fillna(0)
    volume = work_df.groupby("period").size().reindex(period_order, fill_value=0)

    return pct_df, period_order, volume

def filter_monitoring_window(
    df: pd.DataFrame,
    date_col: str = "requested_at",
    n_days: int = 6,
    deploy_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: Literal["post_deploy", "rolling"] = "post_deploy",
) -> pd.DataFrame:
    """
    post_deploy: deploy_date até deploy_date + n_days - 1
    rolling: últimos n_days até end_date (default = max date do df, D-1 na prática)
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    if end_date is None:
        end_date = out[date_col].max()
    else:
        end_date = pd.to_datetime(end_date)

    if mode == "post_deploy":
        if deploy_date is None:
            raise ValueError("deploy_date is required when mode='post_deploy'")
        start_date = pd.to_datetime(deploy_date)
        end_date = start_date + pd.Timedelta(days=n_days - 1)
    else:
        start_date = end_date - pd.Timedelta(days=n_days - 1)

    mask = (out[date_col] >= start_date) & (out[date_col] <= end_date)
    return out.loc[mask].copy()

def plot_daily_mix(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    category_order: List[str],
    color_map: Dict[str, str],
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = False,
    legend_title: str = "Categoria",
    min_label_pct: float = 3.0,  # no diário, suba o threshold (volume menor)
    show_volume: bool = True,
):
    pct_df, _, volume = compute_category_mix(
        df, category_col=category_col, time_grain="day"
    )

    if pct_df.empty:
        print(f"[skip] Sem dados para: {title}")
        return

    if show_volume:
        fig, axes = plot_mix_and_volume(
            pct_df,
            volume,
            category_order=category_order,
            color_map=color_map,
            text_color_map=text_color_map,
            reverse_stack=reverse_stack,
            legend_title=legend_title,
            title=title,
            xlabel="Dia",
            min_label_pct=min_label_pct,
            figsize=(12, 8),
        )
    else:
        fig, ax = plot_stacked_category_mix(
            pct_df,
            category_order=category_order,
            color_map=color_map,
            text_color_map=text_color_map,
            reverse_stack=reverse_stack,
            legend_title=legend_title,
            title=title,
            xlabel="Dia",
            min_label_pct=min_label_pct,
            figsize=(12, 6),
        )

    plt.show()
    # return pct_df, volume

def compute_category_mix_value_by_week(
    df: pd.DataFrame,
    category_col: str,
    date_col: str = "requested_at",
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    work_df = prepare_week_columns(
        df[[date_col, category_col]].dropna(subset=[category_col]),
        date_col,
    )

    pct_df = (
        pd.crosstab(work_df["year_week"], work_df[category_col], normalize="index")
        .mul(100)
        .fillna(0)
    )

    week_order = _ordered_week_labels(work_df)
    pct_df = pct_df.reindex(week_order).fillna(0)

    volume = (
        work_df.groupby("year_week")
        .size()
        .reindex(week_order, fill_value=0)
    )

    return pct_df, week_order, volume

def plot_mix_and_volume(
    pct_df: pd.DataFrame,
    volume: pd.Series,
    title: str = "Mix e volume por Semana",
    xlabel: str = "Semana",
    mix_ylabel: str = "Proporção (%)",
    volume_ylabel: str = "Requisições (n)",
    figsize: Tuple[int, int] = (14, 9),
    volume_height_ratio: float = 0.35,
    show_volume: bool = True,
    show_volume_labels: bool = True,
    min_label_pct: float = 1.0,
    category_order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = False,
    reverse_legend: bool = True,
    legend_title: str = "Category",
) -> Tuple[plt.Figure, np.ndarray]:
    if pct_df.empty:
        raise ValueError("pct_df is empty — no data to plot.")

    if category_order:
        pct_df = pct_df.reindex(columns=category_order, fill_value=0)
    else:
        category_order = pct_df.columns.tolist()

    volume = volume.reindex(pct_df.index, fill_value=0)
    stack_order = category_order[::-1] if reverse_stack else category_order

    if show_volume:
        fig, axes = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1 - volume_height_ratio, volume_height_ratio]},
        )
        ax_mix, ax_vol = axes
    else:
        mix_figsize = (figsize[0], figsize[1] * (1 - volume_height_ratio))
        fig, ax_mix = plt.subplots(figsize=mix_figsize)
        axes = np.array([ax_mix])

    x = np.arange(len(pct_df))
    bottom = np.zeros(len(pct_df))
    default_colors = plt.cm.tab10.colors

    for i, category in enumerate(stack_order):
        values = pct_df[category].values
        color = (
            color_map.get(str(category))
            if color_map
            else default_colors[i % len(default_colors)]
        )
        label_color = (
            text_color_map.get(str(category), "white")
            if text_color_map
            else "white"
        )

        bars = ax_mix.bar(
            x, values, bottom=bottom,
            label=str(category), color=color, width=0.8,
        )

        for bar, val in zip(bars, values):
            if val >= min_label_pct:
                ax_mix.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color=label_color,
                )

        bottom += values

    ax_mix.set_ylabel(mix_ylabel)
    ax_mix.set_ylim(0, 100)
    ax_mix.set_title(title)
    ax_mix.grid(axis="y", alpha=0.2)

    if reverse_stack and reverse_legend:
        handles, labels = ax_mix.get_legend_handles_labels()
        ax_mix.legend(
            handles[::-1], labels[::-1],
            title=legend_title,
            bbox_to_anchor=(1.02, 1), loc="upper left",
        )
    else:
        ax_mix.legend(
            title=legend_title,
            bbox_to_anchor=(1.02, 1), loc="upper left",
        )

    if show_volume:
        vol_bars = ax_vol.bar(
            x, volume.values, color="#64748B", width=0.8, alpha=0.85
        )

        if show_volume_labels:
            for bar, n in zip(vol_bars, volume.values):
                if n > 0:
                    ax_vol.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        _format_volume(int(n)),
                        ha="center", va="bottom",
                        fontsize=8, color="#334155",
                    )

        ax_vol.set_xticks(x)
        ax_vol.set_xticklabels(pct_df.index, rotation=45, ha="right")
        ax_vol.set_xlabel(xlabel)
        ax_vol.set_ylabel(volume_ylabel)
        ax_vol.grid(axis="y", alpha=0.2)
    else:
        ax_mix.set_xticks(x)
        ax_mix.set_xticklabels(pct_df.index, rotation=45, ha="right")
        ax_mix.set_xlabel(xlabel)

    plt.tight_layout()
    return fig, axes

def plot_weekly_mix(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    category_order: List[str],
    color_map: Dict[str, str],
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = False,
    legend_title: str = "Categoria",
    min_label_pct: float = 3.0,
    show_volume: bool = True,
):
    pct_df, _, volume = compute_category_mix(df, category_col=category_col, time_grain="week")
    if pct_df.empty:
        print(f"[skip] Sem dados para: {title}")
        return

    if show_volume:
        fig, axes = plot_mix_and_volume(
            pct_df,
            volume,
            category_order=category_order,
            color_map=color_map,
            text_color_map=text_color_map,
            reverse_stack=reverse_stack,
            legend_title=legend_title,
            title=title,
            xlabel="Semana",
            min_label_pct=min_label_pct,
            figsize=(12, 8),
        )
    else:
        fig, ax = plot_stacked_category_mix(
            pct_df,
            category_order=category_order,
            color_map=color_map,
            text_color_map=text_color_map,
            reverse_stack=reverse_stack,
            legend_title=legend_title,
            title=title,
            xlabel="Semana",
            min_label_pct=min_label_pct,
            figsize=(12, 6),
        )

    plt.show()
    # return pct_df, volume

def build_daily_summary_table(
    df: pd.DataFrame,
    category_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    work = df.copy()
    work["day"] = pd.to_datetime(work["requested_at"]).dt.strftime("%Y-%m-%d")

    if group_col:
        rows = []
        for (day, group), g in work.groupby(["day", group_col]):
            n = len(g)
            dist = g[category_col].value_counts(normalize=True).mul(100).round(1)
            row = {"day": day, group_col: group, "volume": n}
            row.update(dist.to_dict())
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["day", group_col])
    else:
        rows = []
        for day, g in work.groupby("day"):
            n = len(g)
            dist = g[category_col].value_counts(normalize=True).mul(100).round(1)
            row = {"day": day, "volume": n}
            row.update(dist.to_dict())
            rows.append(row)
        return pd.DataFrame(rows).sort_values("day")

def prepare_binary_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Keeps only rows with 0/1 and casts to int.
    NaN rows are excluded from proportion calculation.
    """
    out = df.copy()
    out = out[out[col].notna()]
    out[col] = out[col].astype(int)
    out = out[out[col].isin([0, 1])]
    return out

BINARY_METRICS = [
    "lead_elegivel_pre_analysis",
    "sent",
    "activated",
]

BINARY_ORDER = [0, 1]

BINARY_COLOR_MAP = {
    0: "#CBD5E1",  # muted gray
    1: "#86B8B8",  # muted teal (same family as APROVAR)
}

BINARY_TEXT_COLOR_MAP = {
    0: "white",
    1: "white",
}

BINARY_LABELS = {
    "lead_elegivel_pre_analysis": {0: "Não elegível", 1: "Elegível"},
    "sent": {0: "Não enviado", 1: "Enviado"},
    "activated": {0: "Não ativado", 1: "Ativado"},
}

METRIC_TITLES = {
    "lead_elegivel_pre_analysis": "Lead elegível (pré-análise)",
    "sent": "Enviado (sent)",
    "activated": "Ativado",
}

MODEL_ORDER = [
    "BLEND4",
    "BLEND3_3",
    "BLEND_REGRESSAO_2026",
    "BVS_CUSTOM",
    "HVA3",
    "HVA4",
]

def plot_binary_mix(
    df: pd.DataFrame,
    metric_col: str,
    time_grain: Literal["day", "week"] = "week",
    title_suffix: str = "Geral",
    min_label_pct: float = 3.0,
    figsize: Tuple[int, int] = (12, 8),
):
    df_bin = prepare_binary_column(df, metric_col)
    pct_df, _, volume = compute_category_mix(
        df_bin,
        category_col=metric_col,
        time_grain=time_grain,
    )

    if pct_df.empty:
        print(f"[skip] Sem dados: {metric_col} | {title_suffix}")
        return None

    # Ensure columns 0 and 1 exist
    pct_df = pct_df.reindex(columns=BINARY_ORDER, fill_value=0)

    period_label = "Dia" if time_grain == "day" else "Semana"
    metric_name = METRIC_TITLES.get(metric_col, metric_col)

    fig, axes = plot_mix_and_volume(
        pct_df,
        volume,
        category_order=BINARY_ORDER,
        color_map=BINARY_COLOR_MAP,
        text_color_map=BINARY_TEXT_COLOR_MAP,
        reverse_stack=True,          # 1 on top
        reverse_legend=True,
        legend_title=metric_name,
        title=f"{metric_name} — {title_suffix} ({period_label})",
        xlabel=period_label,
        min_label_pct=min_label_pct,
        figsize=figsize,
    )

    # Rename legend labels to friendly names
    labels = BINARY_LABELS.get(metric_col, {0: "0", 1: "1"})
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        [labels[0], labels[1]],
        title=metric_name,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.show()
    return pct_df, volume

def run_binary_monitoring(
    metrics: List[str] = BINARY_METRICS,
    models: Optional[List[str]] = None,
    time_grains: List[Literal["day", "week"]] = ["day"],
    df: Optional[pd.DataFrame] = None,
    df_daily: Optional[pd.DataFrame] = None,
    model_col: str = "message_decisao",
    min_label_pct_general: Optional[Dict[str, float]] = None,
    min_label_pct_model: Optional[Dict[str, float]] = None,
):
    """
    Execution order:
      for each time_grain:
        for each metric (pré-análise → sent → activated):
          1) Geral
          2) Por modelo (MODEL_ORDER)
    """
    if models is None:
        models = MODEL_ORDER

    default_general = {"day": 5.0, "week": 3.0}
    default_model = {"day": 8.0, "week": 5.0}

    for time_grain in time_grains:
        # Day uses df_daily when provided; week always uses df
        if time_grain == "day":
            base_df = df_daily if df_daily is not None else df
        else:
            base_df = df

        if base_df is None or base_df.empty:
            print(f"[skip] Base vazia para grain='{time_grain}'")
            continue

        period_label = "Dia" if time_grain == "day" else "Semana"
        print(f"\n{'='*60}")
        print(f"MONITORAMENTO BINÁRIO — {period_label}")
        print(f"{'='*60}")

        for metric in metrics:
            metric_name = METRIC_TITLES.get(metric, metric)

            # 1) GERAL primeiro
            print(f"\n→ {metric_name} | Geral")
            plot_binary_mix(
                base_df,
                metric_col=metric,
                time_grain=time_grain,
                title_suffix="Geral",
                min_label_pct=(min_label_pct_general or default_general)[time_grain],
            )

            # 2) POR MODELO depois
            for model in models:
                df_model = base_df[base_df[model_col] == model]
                if df_model.empty:
                    print(f"  [skip] {model}: sem volume")
                    continue

                print(f"  → {metric_name} | {model}")
                plot_binary_mix(
                    df_model,
                    metric_col=metric,
                    time_grain=time_grain,
                    title_suffix=model,
                    min_label_pct=(min_label_pct_model or default_model)[time_grain],
                )

def filter_monitoring_week_window(
    df: pd.DataFrame,
    date_col: str = "requested_at",
    n_weeks: int = 4,
    deploy_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: Literal["post_deploy", "rolling"] = "rolling",
) -> pd.DataFrame:
    out = prepare_week_columns(df, date_col)
    out = out[out[date_col].notna()].copy()
    if out.empty:
        raise ValueError(
            f"No rows with valid '{date_col}' found; cannot compute week window."
        )

    if end_date is None:
        end_date = out[date_col].max()
    else:
        parsed_end_date = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(parsed_end_date):
            raise ValueError(f"Invalid end_date: {end_date!r}")
        end_date = parsed_end_date

    week_table = (
        out[["year_week", "week_start"]]
        .drop_duplicates()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    if mode == "post_deploy":
        if deploy_date is None:
            raise ValueError("deploy_date is required when mode='post_deploy'")
        deploy_yw = _week_label_from_date(deploy_date)
        pos = week_table.index[week_table["year_week"] == deploy_yw]
        if pos.empty:
            raise ValueError(f"deploy week {deploy_yw} not found in data")
        start_pos = pos[0]
        selected_weeks = week_table.loc[start_pos : start_pos + n_weeks - 1, "year_week"]
    else:
        end_yw = _week_label_from_date(end_date)
        pos = week_table.index[week_table["year_week"] == end_yw]
        if pos.empty:
            raise ValueError(f"end week {end_yw} not found in data")
        end_pos = pos[-1]
        start_pos = max(0, end_pos - n_weeks + 1)
        selected_weeks = week_table.loc[start_pos:end_pos, "year_week"]

    return out[out["year_week"].isin(selected_weeks)].copy()


def run_binary_batch(
    df: pd.DataFrame,
    time_grain: Literal["day", "week"],
    scope: Literal["general", "by_model"],
    metrics: List[str] = BINARY_METRICS,
    models: Optional[List[str]] = None,
    model_col: str = "message_decisao",
    min_label_pct_general: float = 5.0,
    min_label_pct_model: float = 8.0,
):
    """
    scope='general': one chart per metric (Geral)
    scope='by_model': for each metric, one chart per model
    """
    if models is None:
        models = MODEL_ORDER

    if df is None or df.empty:
        print(f"[skip] DataFrame vazio ({scope} | {time_grain})")
        return

    period_label = "Dia" if time_grain == "day" else "Semana"
    scope_label = "Geral" if scope == "general" else "Por Modelo"

    if scope == "general":
        for metric in metrics:
            metric_name = METRIC_TITLES.get(metric, metric)
            # print(f"\n→ {metric_name}")
            plot_binary_mix(
                df,
                metric_col=metric,
                time_grain=time_grain,
                title_suffix="Geral",
                min_label_pct=min_label_pct_general if time_grain == "day" else 3.0,
            )
    else:
        for metric in metrics:
            metric_name = METRIC_TITLES.get(metric, metric)
            # print(f"\n→ {metric_name}")
            for model in models:
                df_model = df[df[model_col] == model]
                if df_model.empty:
                    print(f"  [skip] {model}: sem volume")
                    continue
                # print(f"  → {model}")
                plot_binary_mix(
                    df_model,
                    metric_col=metric,
                    time_grain=time_grain,
                    title_suffix=model,
                    min_label_pct=min_label_pct_model if time_grain == "day" else 5.0,
                )

# ---------------------------------------------------------------------------
# Funnel monitoring (single line chart with 3 rates)
# ---------------------------------------------------------------------------

FUNNEL_COL_ELEGIVEL = "lead_elegivel_pre_analysis"
FUNNEL_COL_SENT = "sent"
FUNNEL_COL_ACTIVATED = "activated"

FUNNEL_RATE_COLUMNS = [
    "elegivel_pct_total",
    "sent_pct_elegivel",
    "activated_pct_total",
]

FUNNEL_RATE_LABELS = {
    "elegivel_pct_total": "Elegível (% do total)",
    "sent_pct_elegivel": "Enviado (% dos elegíveis)",
    "activated_pct_total": "Ativado (% do total)",
}

FUNNEL_RATE_COLORS = {
    "elegivel_pct_total": "#60A5FA",   # soft blue
    "sent_pct_elegivel": "#F59E0B",    # soft amber
    "activated_pct_total": "#34D399",  # soft green
}


def _prepare_funnel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast funnel flags to int, treating NaN as 0."""
    out = df.copy()
    for col in [FUNNEL_COL_ELEGIVEL, FUNNEL_COL_SENT, FUNNEL_COL_ACTIVATED]:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")
        out[col] = out[col].fillna(0).astype(int)
    return out


def _build_period_column(
    df: pd.DataFrame,
    date_col: str,
    time_grain: Literal["day", "week"],
) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    if time_grain == "day":
        work["period"] = work[date_col].dt.strftime("%Y-%m-%d")
        period_order = sorted(work["period"].unique())
    else:
        work = _assign_week_columns(work, date_col)
        work["period"] = work["year_week"]
        period_order = _ordered_week_labels(work, week_col="period")
    return work, period_order


def compute_funnel_rates(
    df: pd.DataFrame,
    time_grain: Literal["day", "week"] = "week",
    date_col: str = "requested_at",
) -> pd.DataFrame:
    """
    Compute funnel rates by period.

    Rates:
      - elegivel_pct_total: eligible / total * 100
      - sent_pct_elegivel: sent among eligible / eligible * 100
      - activated_pct_total: activated / total * 100
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = _prepare_funnel_columns(df)
    work, period_order = _build_period_column(work, date_col, time_grain)

    rows = []
    for period, group in work.groupby("period", sort=False):
        total = len(group)
        elegivel = int(group[FUNNEL_COL_ELEGIVEL].eq(1).sum())
        sent_among_elegivel = int(
            group.loc[group[FUNNEL_COL_ELEGIVEL].eq(1), FUNNEL_COL_SENT].eq(1).sum()
        )
        activated = int(group[FUNNEL_COL_ACTIVATED].eq(1).sum())

        rows.append(
            {
                "period": period,
                "volume": total,
                "elegivel_n": elegivel,
                "sent_n": sent_among_elegivel,
                "activated_n": activated,
                "elegivel_pct_total": (elegivel / total * 100) if total else np.nan,
                "sent_pct_elegivel": (sent_among_elegivel / elegivel * 100) if elegivel else np.nan,
                "activated_pct_total": (activated / total * 100) if total else np.nan,
            }
        )

    rates_df = pd.DataFrame(rows)
    if rates_df.empty:
        return rates_df

    rates_df["period"] = pd.Categorical(
        rates_df["period"], categories=period_order, ordered=True
    )
    return rates_df.sort_values("period").reset_index(drop=True)


def plot_funnel_rates(
    rates_df: pd.DataFrame,
    title: str = "Funil",
    xlabel: str = "Período",
    ylabel: str = "Proporção (%)",
    figsize: Tuple[int, int] = (14, 6),
    show_labels: bool = True,
    ylim: Tuple[float, float] = (0, 100),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 3 funnel rates in a single line chart."""
    if rates_df is None or rates_df.empty:
        raise ValueError("rates_df is empty — no data to plot.")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rates_df))
    periods = rates_df["period"].astype(str).tolist()

    for col in FUNNEL_RATE_COLUMNS:
        values = rates_df[col].values
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            color=FUNNEL_RATE_COLORS[col],
            label=FUNNEL_RATE_LABELS[col],
        )

        if show_labels:
            for xi, val in zip(x, values):
                if pd.notna(val):
                    ax.text(
                        xi,
                        val + 1.5,
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=FUNNEL_RATE_COLORS[col],
                        fontweight="bold",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Métrica", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


def run_funnel_monitoring(
    df: pd.DataFrame,
    df_daily: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_col: str = "message_decisao",
    show_plot: bool = True,
) -> None:
    """
    Plot funnel charts in this order:
      1) Daily — general
      2) Daily — by model
      3) Weekly — general
      4) Weekly — by model
    """
    if models is None:
        models = MODEL_ORDER

    def _plot_one(
        data: pd.DataFrame,
        time_grain: Literal["day", "week"],
        title_suffix: str,
    ):
        rates_df = compute_funnel_rates(data, time_grain=time_grain)
        if rates_df.empty:
            print(f"[skip] Sem dados: {title_suffix} | {time_grain}")
            return None

        period_label = "Dia" if time_grain == "day" else "Semana"
        title = f"Funil — {title_suffix} ({period_label})"
        xlabel = f"{period_label} (requested_at)"

        if show_plot:
            plot_funnel_rates(rates_df, title=title, xlabel=xlabel)
            plt.show()

        return rates_df

    # 1) Daily — general
    print("\n=== Funil Diário — Geral ===")
    _plot_one(df_daily, "day", "Geral")

    # 2) Daily — by model
    print("\n=== Funil Diário — Por Modelo ===")
    for model in models:
        df_model = df_daily[df_daily[model_col] == model]
        if df_model.empty:
            print(f"[skip] {model}: sem volume (day)")
            continue
        print(f"→ {model}")
        _plot_one(df_model, "day", model)

    # 3) Weekly — general
    print("\n=== Funil Semanal — Geral ===")
    _plot_one(df, "week", "Geral")

    # 4) Weekly — by model
    print("\n=== Funil Semanal — Por Modelo ===")
    for model in models:
        df_model = df[df[model_col] == model]
        if df_model.empty:
            print(f"[skip] {model}: sem volume (week)")
            continue
        print(f"→ {model}")
        _plot_one(df_model, "week", model)


def _sort_periods(periods: List[str], time_grain: Literal["day", "week"]) -> List[str]:
    return sorted(periods, key=lambda p: pd.to_datetime(p))


def _align_rating_mix(
    pct_left: pd.DataFrame,
    vol_left: pd.Series,
    pct_right: pd.DataFrame,
    vol_right: pd.Series,
    time_grain: Literal["day", "week"],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    period_order = _sort_periods(
        list(set(pct_left.index) | set(pct_right.index)),
        time_grain=time_grain,
    )

    pct_left = pct_left.reindex(period_order, fill_value=0)
    pct_right = pct_right.reindex(period_order, fill_value=0)
    vol_left = vol_left.reindex(period_order, fill_value=0)
    vol_right = vol_right.reindex(period_order, fill_value=0)

    return pct_left, vol_left, pct_right, vol_right, period_order


def _draw_rating_mix_panel(
    ax_mix: plt.Axes,
    ax_vol: plt.Axes,
    pct_df: pd.DataFrame,
    volume: pd.Series,
    *,
    panel_title: str,
    xlabel: str,
    category_order: List[str],
    color_map: Dict[str, str],
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = True,
    min_label_pct: float = 1.0,
    show_legend: bool = False,
    legend_title: str = "Rating",
) -> None:
    pct_df = pct_df.reindex(columns=category_order, fill_value=0)
    volume = volume.reindex(pct_df.index, fill_value=0)
    stack_order = category_order[::-1] if reverse_stack else category_order

    x = np.arange(len(pct_df))
    bottom = np.zeros(len(pct_df))
    default_colors = plt.cm.tab10.colors

    for i, category in enumerate(stack_order):
        values = pct_df[category].values
        color = color_map.get(str(category), default_colors[i % len(default_colors)])
        label_color = (
            text_color_map.get(str(category), "white")
            if text_color_map
            else "white"
        )

        bars = ax_mix.bar(
            x,
            values,
            bottom=bottom,
            label=str(category),
            color=color,
            width=0.8,
        )

        for bar, val in zip(bars, values):
            if val >= min_label_pct:
                ax_mix.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=label_color,
                )

        bottom += values

    ax_mix.set_title(panel_title)
    ax_mix.set_ylabel("Proporção (%)")
    ax_mix.set_ylim(0, 100)
    ax_mix.grid(axis="y", alpha=0.2)

    if show_legend:
        handles, labels = ax_mix.get_legend_handles_labels()
        ax_mix.legend(
            handles[::-1],
            labels[::-1],
            title=legend_title,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    vol_bars = ax_vol.bar(x, volume.values, color="#64748B", width=0.8, alpha=0.85)
    for bar, n in zip(vol_bars, volume.values):
        if n > 0:
            ax_vol.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_volume(int(n)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
            )

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(pct_df.index, rotation=45, ha="right")
    ax_vol.set_xlabel(xlabel)
    ax_vol.set_ylabel("Requisições (n)")
    ax_vol.grid(axis="y", alpha=0.2)


def _plot_rating_comparison(
    df: pd.DataFrame,
    *,
    time_grain: Literal["day", "week"],
    title: str,
    production_rating_col: str = "message_classificacao",
    simulated_rating_col: str = "rating_blend4",
    category_order: List[str],
    color_map: Dict[str, str],
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = True,
    min_label_pct: float = 1.0,
    figsize: Tuple[int, int] = (22, 8),
) -> None:
    pct_prod, _, vol_prod = compute_category_mix(
        df, category_col=production_rating_col, time_grain=time_grain
    )
    pct_sim, _, vol_sim = compute_category_mix(
        df, category_col=simulated_rating_col, time_grain=time_grain
    )

    if pct_prod.empty and pct_sim.empty:
        print(f"[skip] Sem dados para: {title}")
        return

    pct_prod, vol_prod, pct_sim, vol_sim, _ = _align_rating_mix(
        pct_prod, vol_prod, pct_sim, vol_sim, time_grain=time_grain
    )

    period_label = "Dia" if time_grain == "day" else "Semana"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    _draw_rating_mix_panel(
        axes[0, 0],
        axes[1, 0],
        pct_prod,
        vol_prod,
        panel_title="Produção (message_classificacao)",
        xlabel=period_label,
        category_order=category_order,
        color_map=color_map,
        text_color_map=text_color_map,
        reverse_stack=reverse_stack,
        min_label_pct=min_label_pct,
        show_legend=False,
    )

    _draw_rating_mix_panel(
        axes[0, 1],
        axes[1, 1],
        pct_sim,
        vol_sim,
        panel_title="Simulado (rating_blend4)",
        xlabel=period_label,
        category_order=category_order,
        color_map=color_map,
        text_color_map=text_color_map,
        reverse_stack=reverse_stack,
        min_label_pct=min_label_pct,
        show_legend=True,
    )

    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_daily_rating_comparison(
    df: pd.DataFrame,
    title: str,
    production_rating_col: str = "message_classificacao",
    simulated_rating_col: str = "rating_blend4",
    category_order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = True,
    min_label_pct: float = 3.0,
    figsize: Tuple[int, int] = (22, 8),
) -> None:
    """
    Side-by-side daily rating mix:
    production (message_classificacao) vs simulated (rating_blend4).
    """
    if category_order is None:
        category_order = ["A", "B", "C", "D", "E"]
    if color_map is None:
        color_map = {
            "A": "#7DD3FC",
            "B": "#5EEAD4",
            "C": "#93C5FD",
            "D": "#818CF8",
            "E": "#334155",
        }

    _plot_rating_comparison(
        df,
        time_grain="day",
        title=title,
        production_rating_col=production_rating_col,
        simulated_rating_col=simulated_rating_col,
        category_order=category_order,
        color_map=color_map,
        text_color_map=text_color_map,
        reverse_stack=reverse_stack,
        min_label_pct=min_label_pct,
        figsize=figsize,
    )


def plot_weekly_rating_comparison(
    df: pd.DataFrame,
    title: str,
    production_rating_col: str = "message_classificacao",
    simulated_rating_col: str = "rating_blend4",
    category_order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    text_color_map: Optional[Dict[str, str]] = None,
    reverse_stack: bool = True,
    min_label_pct: float = 1.0,
    figsize: Tuple[int, int] = (22, 8),
) -> None:
    """
    Side-by-side weekly rating mix:
    production (message_classificacao) vs simulated (rating_blend4).
    """
    if category_order is None:
        category_order = ["A", "B", "C", "D", "E"]
    if color_map is None:
        color_map = {
            "A": "#7DD3FC",
            "B": "#5EEAD4",
            "C": "#93C5FD",
            "D": "#818CF8",
            "E": "#334155",
        }

    _plot_rating_comparison(
        df,
        time_grain="week",
        title=title,
        production_rating_col=production_rating_col,
        simulated_rating_col=simulated_rating_col,
        category_order=category_order,
        color_map=color_map,
        text_color_map=text_color_map,
        reverse_stack=reverse_stack,
        min_label_pct=min_label_pct,
        figsize=figsize,
    )

def build_rating_mismatch_table(
    df: pd.DataFrame,
    production_rating_col: str = "message_classificacao",
    simulated_rating_col: str = "rating_blend4",
    model_col: Optional[str] = "message_decisao",
    rating_order: Optional[List[str]] = None,
    only_mismatch: bool = False,
) -> pd.DataFrame:
    """
    Build a transition table between production and simulated ratings.

    Returns one row per (production, simulated) pair with volume and percentages.
    """
    if rating_order is None:
        rating_order = ["A", "B", "C", "D", "E"]

    cols = [production_rating_col, simulated_rating_col]
    if model_col:
        cols.append(model_col)

    work = df[cols].copy()
    work = work.dropna(subset=[production_rating_col, simulated_rating_col])

    for col in [production_rating_col, simulated_rating_col]:
        work[col] = pd.Categorical(work[col], categories=rating_order, ordered=True)

    group_cols = [production_rating_col, simulated_rating_col]
    if model_col:
        group_cols = [model_col] + group_cols

    table = (
        work.groupby(group_cols, observed=False)
        .size()
        .reset_index(name="volume")
    )

    table["is_match"] = (
        table[production_rating_col] == table[simulated_rating_col]
    )

    if model_col:
        table["pct_model"] = (
            table.groupby(model_col)["volume"]
            .transform(lambda s: s / s.sum() * 100)
            .round(2)
        )
        table["pct_mismatch_model"] = (
            table.loc[~table["is_match"]]
            .groupby(model_col)["volume"]
            .transform(lambda s: s / s.sum() * 100)
            .round(2)
        )
    else:
        total = table["volume"].sum()
        mismatch_total = table.loc[~table["is_match"], "volume"].sum()

        table["pct_total"] = (table["volume"] / total * 100).round(2)
        table["pct_mismatch"] = np.where(
            table["is_match"],
            0.0,
            (table["volume"] / mismatch_total * 100).round(2),
        )

    table["rating_delta"] = (
        table[simulated_rating_col].cat.codes
        - table[production_rating_col].cat.codes
    )

    table["transition"] = (
        table[production_rating_col].astype(str)
        + " → "
        + table[simulated_rating_col].astype(str)
    )

    if only_mismatch:
        table = table[~table["is_match"]].copy()

    sort_cols = (
        [model_col, production_rating_col, simulated_rating_col]
        if model_col
        else [production_rating_col, simulated_rating_col]
    )

    return table.sort_values(sort_cols, ascending=[True, True, True]).reset_index(drop=True)


def build_rating_match_summary(
    df: pd.DataFrame,
    production_rating_col: str = "message_classificacao",
    simulated_rating_col: str = "rating_blend4",
    model_col: Optional[str] = "message_decisao",
) -> pd.DataFrame:
    """
    High-level summary of rating agreement between production and simulation.
    """
    mismatch_table = build_rating_mismatch_table(
        df,
        production_rating_col=production_rating_col,
        simulated_rating_col=simulated_rating_col,
        model_col=model_col,
        only_mismatch=False,
    )

    if model_col:
        summary = (
            mismatch_table.groupby(model_col, observed=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "volume_total": g["volume"].sum(),
                        "volume_match": g.loc[g["is_match"], "volume"].sum(),
                        "volume_mismatch": g.loc[~g["is_match"], "volume"].sum(),
                        "pct_match": round(
                            g.loc[g["is_match"], "volume"].sum() / g["volume"].sum() * 100, 2
                        ),
                        "pct_mismatch": round(
                            g.loc[~g["is_match"], "volume"].sum() / g["volume"].sum() * 100, 2
                        ),
                        "transitions_distinct": g.loc[~g["is_match"], "transition"].nunique(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        total = mismatch_table["volume"].sum()
        match = mismatch_table.loc[mismatch_table["is_match"], "volume"].sum()
        mismatch = total - match

        summary = pd.DataFrame(
            [{
                "volume_total": total,
                "volume_match": match,
                "volume_mismatch": mismatch,
                "pct_match": round(match / total * 100, 2),
                "pct_mismatch": round(mismatch / total * 100, 2),
                "transitions_distinct": mismatch_table.loc[
                    ~mismatch_table["is_match"], "transition"
                ].nunique(),
            }]
        )

    return summary

def build_score_match_summary(
    df: pd.DataFrame,
    production_score_col: str = "message_scores_BLEND3_3",
    simulated_score_col: str = "pred_blend4_1_to_score",
    model_col: Optional[str] = "message_decisao",
    tolerances: Optional[List[int]] = None,
    time_grain: Optional[Literal["day", "week"]] = None,
    date_col: str = "requested_at",
) -> pd.DataFrame:
    """
    Summary of score agreement between production and simulated scores.

    - time_grain=None (default): aggregate by model (or overall if model_col=None)
    - time_grain="day" | "week": one row per period (+ model if model_col is set)
    """
    if tolerances is None:
        tolerances = [0, 5, 10, 25, 50]

    work = _prepare_score_match_work(df, production_score_col, simulated_score_col)

    if work.empty:
        return pd.DataFrame()

    def _summarize(g: pd.DataFrame) -> pd.Series:
        row = {
            "volume_total": len(g),
            "score_prod_mean": round(g["score_prod"].mean(), 1),
            "score_sim_mean": round(g["score_sim"].mean(), 1),
            "score_diff_mean": round(g["score_diff"].mean(), 1),
            "score_abs_diff_mean": round(g["score_abs_diff"].mean(), 1),
            "score_abs_diff_median": round(g["score_abs_diff"].median(), 1),
            "score_abs_diff_p90": round(g["score_abs_diff"].quantile(0.90), 1),
            "corr_pearson": round(g["score_prod"].corr(g["score_sim"]), 4),
        }

        for tol in tolerances:
            row[f"pct_match_abs_le_{tol}"] = round(
                (g["score_abs_diff"] <= tol).mean() * 100, 2
            )

        return pd.Series(row)

    if time_grain is None:
        if model_col:
            return (
                work.groupby(model_col, observed=False)
                .apply(_summarize, include_groups=False)
                .reset_index()
            )
        return _summarize(work).to_frame().T

    work, period_order = _build_period_column(work, date_col, time_grain)
    group_cols = ["period"]
    if model_col:
        group_cols.append(model_col)

    summary = (
        work.groupby(group_cols, observed=False)
        .apply(_summarize, include_groups=False)
        .reset_index()
    )

    summary["period"] = pd.Categorical(
        summary["period"], categories=period_order, ordered=True
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def build_score_match_table(
    df: pd.DataFrame,
    production_score_col: str = "message_scores_BLEND3_3",
    simulated_score_col: str = "pred_blend4_1_to_score",
    model_col: Optional[str] = "message_decisao",
    diff_bins: Optional[List[int]] = None,
    only_mismatch: bool = False,
) -> pd.DataFrame:
    """
    Detailed table by absolute score difference buckets.

    Example buckets:
      0, 1-5, 6-10, 11-25, 26-50, 51-100, >100
    """
    if diff_bins is None:
        diff_bins = [0, 5, 10, 25, 50, 100, np.inf]

    labels = []
    for i in range(len(diff_bins) - 1):
        left, right = diff_bins[i], diff_bins[i + 1]
        if left == 0 and right == 0:
            labels.append("0 (match exato)")
        elif np.isinf(right):
            labels.append(f">{int(left)}")
        else:
            labels.append(f"{int(left)+1 if left == 0 else int(left)}-{int(right)}")

    work = df.copy()
    work["score_prod"] = pd.to_numeric(work[production_score_col], errors="coerce")
    work["score_sim"] = pd.to_numeric(work[simulated_score_col], errors="coerce")
    work = work.dropna(subset=["score_prod", "score_sim"])

    work["score_diff"] = work["score_sim"] - work["score_prod"]
    work["score_abs_diff"] = work["score_diff"].abs()

    work["abs_diff_bucket"] = pd.cut(
        work["score_abs_diff"],
        bins=diff_bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    group_cols = ["abs_diff_bucket"]
    if model_col:
        group_cols = [model_col, "abs_diff_bucket"]

    table = (
        work.groupby(group_cols, observed=False)
        .agg(
            volume=("score_abs_diff", "size"),
            score_prod_mean=("score_prod", "mean"),
            score_sim_mean=("score_sim", "mean"),
            score_diff_mean=("score_diff", "mean"),
            score_abs_diff_mean=("score_abs_diff", "mean"),
        )
        .reset_index()
    )

    if model_col:
        table["pct_model"] = (
            table.groupby(model_col)["volume"]
            .transform(lambda s: s / s.sum() * 100)
            .round(2)
        )
    else:
        table["pct_total"] = (table["volume"] / table["volume"].sum() * 100).round(2)

    if only_mismatch:
        table = table[~table["abs_diff_bucket"].astype(str).str.contains("match exato")]

    return table

def _prepare_score_match_work(
    df: pd.DataFrame,
    production_score_col: str,
    simulated_score_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work["score_prod"] = pd.to_numeric(work[production_score_col], errors="coerce")
    work["score_sim"] = pd.to_numeric(work[simulated_score_col], errors="coerce")
    work = work.dropna(subset=["score_prod", "score_sim"])
    work["score_diff"] = work["score_sim"] - work["score_prod"]
    work["score_abs_diff"] = work["score_diff"].abs()
    return work

def plot_score_match_summary(
    summary_df: pd.DataFrame,
    match_tolerance: int = 10,
    title: str = "Match de score (prod vs sim)",
    xlabel: str = "Período",
    ylabel: str = "Match (%)",
    model_col: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    show_labels: bool = True,
    show_volume: bool = True,
    ylim: Tuple[float, float] = (0, 100),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Line chart of pct_match_abs_le_{match_tolerance} over period.
    Expects output of build_score_match_summary with time_grain set.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty — no data to plot.")

    pct_col = f"pct_match_abs_le_{match_tolerance}"
    if pct_col not in summary_df.columns:
        raise KeyError(f"Column not found: {pct_col}")

    periods = summary_df["period"].astype(str).tolist()
    x = np.arange(len(periods))

    if show_volume:
        fig, axes = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [0.65, 0.35]},
        )
        ax_pct, ax_vol = axes
    else:
        fig, ax_pct = plt.subplots(figsize=figsize)
        axes = np.array([ax_pct])

    if model_col and model_col in summary_df.columns:
        for model, g in summary_df.groupby(model_col, observed=False):
            g = g.sort_values("period")
            xi = [periods.index(str(p)) for p in g["period"].astype(str)]
            ax_pct.plot(
                xi,
                g[pct_col].values,
                marker="o",
                linewidth=2,
                label=str(model),
            )
            if show_labels:
                for xj, val in zip(xi, g[pct_col].values):
                    if pd.notna(val):
                        ax_pct.text(
                            xj, val + 1.5, f"{val:.1f}%",
                            ha="center", va="bottom", fontsize=8,
                        )
        ax_pct.legend(title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left")
        volume_series = (
            summary_df.groupby("period", observed=False)["volume_total"]
            .sum()
            .reindex(summary_df["period"].cat.categories, fill_value=0)
        )
    else:
        values = summary_df[pct_col].values
        ax_pct.plot(
            x, values, marker="o", linewidth=2, color="#2563EB", label=f"|Δ| ≤ {match_tolerance}",
        )
        if show_labels:
            for xi, val in zip(x, values):
                if pd.notna(val):
                    ax_pct.text(
                        xi, val + 1.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8, color="#2563EB",
                    )
        ax_pct.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        volume_series = summary_df.set_index("period")["volume_total"]

    ax_pct.set_ylabel(ylabel)
    ax_pct.set_title(f"{title} (|Δ| ≤ {match_tolerance})")
    ax_pct.set_ylim(*ylim)
    ax_pct.grid(axis="y", alpha=0.25)

    if show_volume:
        vol_values = volume_series.reindex(summary_df["period"].cat.categories, fill_value=0).values
        ax_vol.bar(x, vol_values, color="#64748B", width=0.8, alpha=0.85)
        for bar, n in zip(ax_vol.bar(x, vol_values, color="#64748B", width=0.8, alpha=0.85), vol_values):
            if n > 0:
                ax_vol.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    _format_volume(int(n)),
                    ha="center", va="bottom", fontsize=8, color="#334155",
                )
        ax_vol.set_xticks(x)
        ax_vol.set_xticklabels(periods, rotation=45, ha="right")
        ax_vol.set_xlabel(xlabel)
        ax_vol.set_ylabel("Volume (n)")
        ax_vol.grid(axis="y", alpha=0.2)
    else:
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(periods, rotation=45, ha="right")
        ax_pct.set_xlabel(xlabel)

    plt.tight_layout()
    return fig, axes


def plot_score_match_monitoring(
    df: pd.DataFrame,
    production_score_col: str = "message_scores_BLEND3_3",
    simulated_score_col: str = "pred_blend4_1_to_score",
    model_col: Optional[str] = None,
    match_tolerance: int = 10,
    title_prefix: str = "Match de score (prod vs sim)",
    show_volume: bool = True,
    ylim: Tuple[float, float] = (0, 110),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Graph 1: daily match %.
    Graph 2: weekly match %.
    """
    daily_summary = build_score_match_summary(
        df,
        production_score_col=production_score_col,
        simulated_score_col=simulated_score_col,
        model_col=model_col,
        time_grain="day",
    )
    weekly_summary = build_score_match_summary(
        df,
        production_score_col=production_score_col,
        simulated_score_col=simulated_score_col,
        model_col=model_col,
        time_grain="week",
    )

    if daily_summary.empty:
        print("[skip] Sem dados para match diário de score")
    else:
        plot_score_match_summary(
            daily_summary,
            match_tolerance=match_tolerance,
            title=f"{title_prefix} — Diário",
            xlabel="Dia",
            model_col=model_col,
            show_volume=show_volume,
            ylim=ylim,
        )
        plt.show()

    if weekly_summary.empty:
        print("[skip] Sem dados para match semanal de score")
    else:
        plot_score_match_summary(
            weekly_summary,
            match_tolerance=match_tolerance,
            title=f"{title_prefix} — Semanal",
            xlabel="Semana",
            model_col=model_col,
            show_volume=show_volume,
            ylim=ylim,
        )
        plt.show()

    return daily_summary, weekly_summary


# ---------------------------------------------------------------------------
# Extended funnel monitoring (volume panel + merge + suite runner)
# ---------------------------------------------------------------------------

FUNNEL_EXTENDED_RATE_COLUMNS = [
    "elegivel_pct_total",
    "sent_pct_elegivel",
    "sent_pct_total",
    "activated_pct_sent",
    "activated_pct_total",
]

FUNNEL_EXTENDED_RATE_LABELS = {
    "elegivel_pct_total": "Elegível (% do total)",
    "sent_pct_elegivel": "Enviado (% dos elegíveis)",
    "sent_pct_total": "Enviado (% do total)",
    "activated_pct_sent": "Ativado (% dos enviados)",
    "activated_pct_total": "Ativado (% do total)",
}

FUNNEL_EXTENDED_RATE_COLORS = {
    "elegivel_pct_total": "#60A5FA",
    "sent_pct_elegivel": "#F59E0B",
    "sent_pct_total": "#FB923C",
    "activated_pct_sent": "#A78BFA",
    "activated_pct_total": "#34D399",
}


def compute_extended_funnel_rates(
    df: pd.DataFrame,
    time_grain: Literal["day", "week"] = "week",
    date_col: str = "requested_at",
) -> pd.DataFrame:
    """Extended funnel rates aligned with blend monitoring semantics."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = _prepare_funnel_columns(df)
    work, period_order = _build_period_column(work, date_col, time_grain)

    rows = []
    for period, group in work.groupby("period", sort=False):
        total = len(group)
        elegivel = int(group[FUNNEL_COL_ELEGIVEL].eq(1).sum())
        sent = int(group[FUNNEL_COL_SENT].eq(1).sum())
        sent_among_elegivel = int(
            group.loc[group[FUNNEL_COL_ELEGIVEL].eq(1), FUNNEL_COL_SENT].eq(1).sum()
        )
        activated = int(group[FUNNEL_COL_ACTIVATED].eq(1).sum())
        activated_among_sent = int(
            group.loc[group[FUNNEL_COL_SENT].eq(1), FUNNEL_COL_ACTIVATED].eq(1).sum()
        )

        rows.append(
            {
                "period": period,
                "volume": total,
                "elegivel_n": elegivel,
                "sent_n": sent,
                "sent_among_elegivel_n": sent_among_elegivel,
                "activated_n": activated,
                "activated_among_sent_n": activated_among_sent,
                "elegivel_pct_total": (elegivel / total * 100) if total else np.nan,
                "sent_pct_elegivel": (sent_among_elegivel / elegivel * 100) if elegivel else np.nan,
                "sent_pct_total": (sent / total * 100) if total else np.nan,
                "activated_pct_sent": (activated_among_sent / sent * 100) if sent else np.nan,
                "activated_pct_total": (activated / total * 100) if total else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["period"] = pd.Categorical(out["period"], categories=period_order, ordered=True)
    return out.sort_values("period").reset_index(drop=True)


def plot_funnel_rates_with_volume(
    rates_df: pd.DataFrame,
    title: str = "Funil",
    xlabel: str = "Período",
    rate_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_labels: bool = True,
    ylim: Tuple[float, float] = (0, 100),
) -> Tuple[plt.Figure, np.ndarray]:
    """Funnel line chart + volume panel (Blend4 style)."""
    if rates_df is None or rates_df.empty:
        raise ValueError("rates_df is empty — no data to plot.")

    if rate_columns is None:
        rate_columns = FUNNEL_RATE_COLUMNS  # default: 3 main lines

    labels_map = {**FUNNEL_EXTENDED_RATE_LABELS, **FUNNEL_RATE_LABELS}
    colors_map = {**FUNNEL_EXTENDED_RATE_COLORS, **FUNNEL_RATE_COLORS}

    fig, axes = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [0.65, 0.35]},
    )
    ax_rates, ax_vol = axes

    x = np.arange(len(rates_df))
    periods = rates_df["period"].astype(str).tolist()

    for col in rate_columns:
        values = rates_df[col].values
        color = colors_map.get(col, "#64748B")
        ax_rates.plot(
            x, values,
            marker="o", linewidth=2, markersize=6,
            color=color, label=labels_map.get(col, col),
        )
        if show_labels:
            for xi, val in zip(x, values):
                if pd.notna(val):
                    ax_rates.text(
                        xi, val + 1.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8,
                        color=color, fontweight="bold",
                    )

    ax_rates.set_ylabel("Proporção (%)")
    ax_rates.set_title(title)
    ax_rates.set_ylim(*ylim)
    ax_rates.grid(axis="y", alpha=0.25)
    ax_rates.legend(title="Métrica", bbox_to_anchor=(1.02, 1), loc="upper left")

    volume = rates_df["volume"].values
    ax_vol.bar(x, volume, color="#64748B", width=0.8, alpha=0.85)
    for bar, n in zip(ax_vol.bar(x, volume, color="#64748B", width=0.8, alpha=0.85), volume):
        if n > 0:
            ax_vol.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_volume(int(n)),
                ha="center", va="bottom", fontsize=8, color="#334155",
            )

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(periods, rotation=45, ha="right")
    ax_vol.set_xlabel(xlabel)
    ax_vol.set_ylabel("Requisições (n)")
    ax_vol.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    return fig, axes


def merge_funnel_with_scoring(
    df_funil: pd.DataFrame,
    df_scoring: pd.DataFrame,
    on: str = "contract_id",
    scoring_cols: Optional[List[str]] = None,
    how: Literal["inner", "left"] = "left",
) -> pd.DataFrame:
    """
    Join funnel gold data with scoring/production dataframe
    to enable model/rating breakdowns.
    """
    if scoring_cols is None:
        scoring_cols = [
            "message_decisao",
            "message_classificacao",
            "rating_blend4",
            "rating_manual_blend4",
            "rating_json_blend4",
        ]

    keep_cols = [on] + [c for c in scoring_cols if c in df_scoring.columns]
    scoring = df_scoring[keep_cols].drop_duplicates(subset=[on])

    out = df_funil.merge(scoring, on=on, how=how)
    return out


def build_funnel_summary_table(
    rates_df: pd.DataFrame,
    rate_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Display-friendly funnel table (% rounded to 1 decimal)."""
    if rates_df is None or rates_df.empty:
        return pd.DataFrame()

    if rate_columns is None:
        rate_columns = FUNNEL_EXTENDED_RATE_COLUMNS

    cols = ["period", "volume"] + [c for c in rate_columns if c in rates_df.columns]
    table = rates_df[cols].copy()
    for col in rate_columns:
        if col in table.columns:
            table[col] = table[col].round(1)
    return table


def run_funnel_monitoring_suite(
    df: pd.DataFrame,
    df_daily: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_col: str = "message_decisao",
    dimension_col: Optional[str] = None,
    dimension_values: Optional[List[str]] = None,
    extended: bool = False,
    include_binary: bool = True,
    show_volume: bool = True,
) -> None:
    """
    Blend4-style funnel monitoring:
      1) Binary stacks (elegível → sent → activated): daily/weekly, geral + por modelo
      2) Funnel line charts: daily/weekly, geral + por modelo
      3) Optional breakdown by another dimension (e.g. segmentacao)
    """
    if models is None:
        models = MODEL_ORDER

    rate_columns = FUNNEL_EXTENDED_RATE_COLUMNS if extended else FUNNEL_RATE_COLUMNS
    compute_fn = compute_extended_funnel_rates if extended else compute_funnel_rates

    def _plot_rates(data, time_grain, suffix):
        rates_df = compute_fn(data, time_grain=time_grain)
        if rates_df.empty:
            print(f"[skip] Sem dados: {suffix} | {time_grain}")
            return

        period_label = "Dia" if time_grain == "day" else "Semana"
        title = f"Funil — {suffix} ({period_label})"

        if show_volume:
            plot_funnel_rates_with_volume(
                rates_df,
                title=title,
                xlabel=f"{period_label} (requested_at)",
                rate_columns=rate_columns,
            )
        else:
            plot_funnel_rates(
                rates_df,
                title=title,
                xlabel=f"{period_label} (requested_at)",
            )
        plt.show()

        display(build_funnel_summary_table(rates_df, rate_columns=rate_columns))

    # --- Binary monitoring (same visual language as Blend4) ---
    if include_binary:
        for time_grain, base_df in [("day", df_daily), ("week", df)]:
            run_binary_batch(base_df, time_grain=time_grain, scope="general")
            if model_col in base_df.columns:
                run_binary_batch(base_df, time_grain=time_grain, scope="by_model", models=models, model_col=model_col)

    # --- Line charts ---
    for time_grain, base_df, label in [
        ("day", df_daily, "Diário"),
        ("week", df, "Semanal"),
    ]:
        print(f"\n=== Funil {label} — Geral ===")
        _plot_rates(base_df, time_grain, "Geral")

        if model_col in base_df.columns:
            print(f"\n=== Funil {label} — Por Modelo ===")
            for model in models:
                df_model = base_df[base_df[model_col] == model]
                if df_model.empty:
                    print(f"[skip] {model}: sem volume")
                    continue
                print(f"→ {model}")
                _plot_rates(df_model, time_grain, model)

    # --- Optional dimension (segmentacao, rating, etc.) ---
    if dimension_col and dimension_col in df.columns:
        values = dimension_values or sorted(df[dimension_col].dropna().unique().tolist())
        print(f"\n=== Funil — Por {dimension_col} ===")
        for value in values:
            subset = df[df[dimension_col] == value]
            if subset.empty:
                continue
            subset_daily = df_daily[df_daily[dimension_col] == value]
            print(f"→ {dimension_col} = {value}")
            _plot_rates(subset_daily, "day", f"{dimension_col}={value}")
            _plot_rates(subset, "week", f"{dimension_col}={value}")

def prepare_blend_funnel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map df_funil_blend4 columns to standardized funnel int flags."""
    out = df.copy()
    out["is_elegivel"] = np.where(
        out["pre_analysis_result"].isin(["APROVAR", "DERIVAR"]), 1, 0
    )
    for src, dst in [
        ("proposta_iniciada", "is_iniciada"),
        ("proposta_enviada", "is_enviada"),
        ("proposta_aprovada", "is_aprovada"),
        ("proposta_ativada", "is_ativada"),
    ]:
        out[dst] = out[src].fillna(False).astype(int)
    return out

BLEND_FUNNEL_METRICS = [
    "elegivel_pct_total",      # %_elegivel
    "iniciada_pct_elegivel",   # %_iniciada
    "enviada_pct_iniciada",    # %_enviada
    "aprovada_pct_enviada",    # %_aprovada
    "ativada_pct_aprovada",    # %_ativada
    "conversao_pct_total",     # %_conversao
]

def compute_blend_funnel_rates(
    df: pd.DataFrame,
    time_grain: str = "week",
    date_col: str = "requested_at",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = prepare_blend_funnel_columns(df) if "is_elegivel" not in df.columns else df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    if time_grain == "day":
        work["period"] = work[date_col].dt.strftime("%Y-%m-%d")
        period_order = sorted(work["period"].unique())
    else:
        work = prepare_week_columns(work, date_col)
        work["period"] = work["year_week"]
        period_order = _ordered_week_labels(work, week_col="period")

    rows = []
    for period, group in work.groupby("period", sort=False):
        total = len(group)
        elegivel = int(group["is_elegivel"].sum())
        iniciada = int(group["is_iniciada"].sum())
        enviada = int(group["is_enviada"].sum())
        aprovada = int(group["is_aprovada"].sum())
        ativada = int(group["is_ativada"].sum())

        iniciada_elegivel = int(group.loc[group["is_elegivel"].eq(1), "is_iniciada"].sum())
        enviada_iniciada = int(group.loc[group["is_iniciada"].eq(1), "is_enviada"].sum())
        enviada_elegivel = int(group.loc[group["is_elegivel"].eq(1), "is_enviada"].sum())
        aprovada_enviada = int(group.loc[group["is_enviada"].eq(1), "is_aprovada"].sum())
        ativada_aprovada = int(group.loc[group["is_aprovada"].eq(1), "is_ativada"].sum())

        rows.append(
            {
                "period": period,
                "volume": total,
                "elegivel_pct_total": (elegivel / total * 100) if total else np.nan,
                "iniciada_pct_elegivel": (iniciada_elegivel / elegivel * 100) if elegivel else np.nan,
                "enviada_pct_iniciada": (enviada_iniciada / iniciada * 100) if iniciada else np.nan,
                "enviada_pct_elegivel": (enviada_elegivel / elegivel * 100) if elegivel else np.nan,
                "aprovada_pct_enviada": (aprovada_enviada / enviada * 100) if enviada else np.nan,
                "ativada_pct_aprovada": (ativada_aprovada / aprovada * 100) if aprovada else np.nan,
                "conversao_pct_total": (ativada / total * 100) if total else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["period"] = pd.Categorical(out["period"], categories=period_order, ordered=True)
    return out.sort_values("period").reset_index(drop=True)


def plot_blend_funnel_with_volume(
    rates_df: pd.DataFrame,
    title: str,
    xlabel: str,
    rate_columns=None,
    ylim=(0, 100),
):
    if rates_df is None or rates_df.empty:
        print(f"[skip] Sem dados para: {title}")
        return

    if rate_columns is None:
        rate_columns = BLEND_FUNNEL_METRICS

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [0.65, 0.35]},
    )
    ax_rates, ax_vol = axes
    x = np.arange(len(rates_df))
    periods = rates_df["period"].astype(str).tolist()

    for col in rate_columns:
        color = BLEND_FUNNEL_COLORS.get(col, "#64748B")
        label = BLEND_FUNNEL_LABELS.get(col, col)
        values = rates_df[col].values
        ax_rates.plot(x, values, marker="o", linewidth=2, markersize=6, color=color, label=label)
        for xi, val in zip(x, values):
            if pd.notna(val):
                ax_rates.text(
                    xi, val + 1.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    color=color, fontweight="bold",
                )

    ax_rates.set_ylabel("Proporção (%)")
    ax_rates.set_title(title)
    ax_rates.set_ylim(*ylim)
    ax_rates.grid(axis="y", alpha=0.25)
    ax_rates.legend(title="Métrica", bbox_to_anchor=(1.02, 1), loc="upper left")

    volume = rates_df["volume"].values
    bars = ax_vol.bar(x, volume, color="#64748B", width=0.8, alpha=0.85)
    for bar, n in zip(bars, volume):
        if n > 0:
            ax_vol.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                _format_volume(int(n)), ha="center", va="bottom",
                fontsize=8, color="#334155",
            )

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(periods, rotation=45, ha="right")
    ax_vol.set_xlabel(xlabel)
    ax_vol.set_ylabel("Requisições (n)")
    ax_vol.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

FUNNEL_STEP_TITLES = {
    "is_elegivel": "Elegível",
    "is_iniciada": "Proposta iniciada",
    "is_enviada": "Proposta enviada",
    "is_aprovada": "Proposta aprovada",
    "is_ativada": "Proposta ativada",
}

BLEND_FUNNEL_LABELS = {
    "elegivel_pct_total": "Elegível (% do total)",
    "iniciada_pct_elegivel": "Iniciada (% dos elegíveis)",
    "enviada_pct_iniciada": "Enviada (% das iniciadas)",
    "aprovada_pct_enviada": "Aprovada (% das enviadas)",
    "ativada_pct_aprovada": "Ativada (% das aprovadas)",
    "conversao_pct_total": "Conversão (% do total)",
    "enviada_pct_elegivel": "Enviada (% dos elegíveis)",
}

BLEND_FUNNEL_COLORS = {
    "elegivel_pct_total": "#60A5FA",
    "iniciada_pct_elegivel": "#38BDF8",
    "enviada_pct_iniciada": "#F59E0B",
    "aprovada_pct_enviada": "#FB923C",
    "ativada_pct_aprovada": "#A78BFA",
    "conversao_pct_total": "#34D399",
    "enviada_pct_elegivel": "#FBBF24",
}

def plot_funnel_binary_step(df, metric_col, time_grain, title_suffix, min_label_pct=5.0):
    df_bin = prepare_binary_column(df, metric_col)
    pct_df, _, volume = compute_category_mix(df_bin, category_col=metric_col, time_grain=time_grain)
    if pct_df.empty:
        print(f"[skip] Sem dados: {metric_col} | {title_suffix}")
        return

    pct_df = pct_df.reindex(columns=BINARY_ORDER, fill_value=0)
    period_label = "Dia" if time_grain == "day" else "Semana"
    step_name = FUNNEL_STEP_TITLES.get(metric_col, metric_col)

    fig, axes = plot_mix_and_volume(
        pct_df, volume,
        category_order=BINARY_ORDER,
        color_map=BINARY_COLOR_MAP,
        text_color_map=BINARY_TEXT_COLOR_MAP,
        reverse_stack=True,
        reverse_legend=True,
        legend_title=step_name,
        title=f"{step_name} — {title_suffix} ({period_label})",
        xlabel=period_label,
        min_label_pct=min_label_pct,
        figsize=(12, 8),
    )
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, ["Não", "Sim"], title=step_name,
                   bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()


def plot_pre_analysis_comparison(
    df: pd.DataFrame,
    models: list,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    category_col: str = "pre_analysis_result",
    color_map: dict = None,
    figsize=(18, 5),
    ylim=(0, 100),
    suptitle: str = None,
):
    """Three side-by-side charts (APROVAR / DERIVAR / REPROVAR), models compared on each."""
    color_map = color_map or MODEL_COLOR_MAP
    outcomes = ["APROVAR", "DERIVAR", "REPROVAR"]
    period_label = "Dia" if time_grain == "day" else "Semana"

    # Pré-calcula mix por modelo (evita recomputar 3 vezes)
    pct_by_model = {}
    all_periods = []

    for model in models:
        df_model = df[df[model_col] == model]
        pct_df, _, _ = compute_category_mix(
            df_model, category_col=category_col, time_grain=time_grain
        )
        if not pct_df.empty:
            pct_by_model[model] = pct_df
            all_periods.extend(pct_df.index.tolist())

    if not pct_by_model:
        print("[skip] Pré-análise: sem dados")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    if suptitle:
        fig.suptitle(suptitle, y=1.1, fontsize=13)

    for ax, outcome in zip(axes, outcomes):
        has_data = False

        for model in models:
            if model not in pct_by_model:
                continue
            pct_df = pct_by_model[model].reindex(periods).fillna(0)
            if outcome not in pct_df.columns:
                continue

            y = pct_df[outcome].values
            color = color_map.get(model, "#64748B")

            ax.plot(x, y, marker="o", linewidth=2, color=color, label=model)
            for xi, val in zip(x, y):
                if val > 0:
                    ax.text(xi, val + 0.8, f"{val:.1f}%", ha="center", fontsize=7, color=color)
            has_data = True

        ax.set_title(outcome)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

        if not has_data:
            ax.set_visible(False)

    axes[0].set_ylabel("Proporção (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(periods, ha="right")

    axes[-1].set_xlabel(period_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Modelo", loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(models))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # espaço para legenda centralizada
    plt.show()
        

FUNNEL_COMPARE_METRICS = {
    "elegivel_pct_total": "Elegível (% do total)",
    "enviada_pct_elegivel": "Enviada (% dos elegíveis)",
    "ativada_pct_aprovada": "Ativada (% das aprovadas)",
    "conversao_pct_total": "Conversão (% do total)",
}

def plot_funnel_metric_comparison(
    df: pd.DataFrame,
    models: list,
    metrics: dict,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    color_map: dict = None,
    ylim=(0, 100),
    figsize=(14, 9),
    suptitle: str = None,
):
    """Compare funnel metrics between models in a 2x2 grid."""
    color_map = color_map or MODEL_COLOR_MAP
    period_label = "Dia" if time_grain == "day" else "Semana"
    metric_items = list(metrics.items())

    # Pré-calcula rates por modelo
    rates_by_model = {}
    all_periods = []

    for model in models:
        df_model = df[df[model_col] == model]
        rates = compute_blend_funnel_rates(df_model, time_grain=time_grain)
        if rates.empty:
            continue
        rates_by_model[model] = rates.set_index(rates["period"].astype(str))
        all_periods.extend(rates["period"].astype(str).tolist())

    if not rates_by_model:
        print("[skip] Funil: sem dados")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    n_metrics = len(metric_items)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    if suptitle:
        fig.suptitle(suptitle, y=1.1, fontsize=13)

    for ax, (metric_col, metric_label) in zip(axes, metric_items):
        has_data = False

        for model in models:
            if model not in rates_by_model:
                continue

            rates = rates_by_model[model].reindex(periods)
            y = rates[metric_col].values
            color = color_map.get(model, "#64748B")

            ax.plot(x, y, marker="o", linewidth=2, color=color, label=model)
            for xi, val in zip(x, y):
                if pd.notna(val):
                    ax.text(xi, val + 1.0, f"{val:.1f}%", ha="center", fontsize=7, color=color)
            has_data = True

        ax.set_title(metric_label)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

        if not has_data:
            ax.set_visible(False)

    # Eixos vazios se metrics < 4
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

  # ylabel só no subplot da esquerda (índices 0 e 2)
    for i in [0, 2]:
        if i < n_metrics:
            axes[i].set_ylabel("Proporção (%)")

    for ax in axes[:n_metrics]:
        ax.set_xticks(x)
        ax.set_xticklabels(periods, ha="right")

    axes[min(3, n_metrics - 1)].set_xlabel(period_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title="Modelo",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(models),
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def build_blend_comparison_summary_table(
    df: pd.DataFrame,
    models: list,
    model_col: str = "bureau_nm_ajust",
    time_grain: str = "week",
    baseline_model: str = "BLEND3_3",
    challenger_model: str = "BLEND4",
):
    work = prepare_blend_funnel_columns(df)
    work = work[work[model_col].isin(models)].copy()
    work["requested_at"] = pd.to_datetime(work["requested_at"])

    if time_grain == "week":
        work = prepare_week_columns(work, "requested_at")
        dt_col = "year_week"
    else:
        work["day"] = work["requested_at"].dt.strftime("%Y-%m-%d")
        dt_col = "day"

    group = [model_col, dt_col]

    aux = work.groupby(group).size().reset_index(name="qtd").merge(
        (work.groupby(group).size() / work.groupby(dt_col).size()).reset_index(name="mix"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_elegivel"].sum() / work.groupby(group).size()).reset_index(name="%_elegivel"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_enviada"].sum() / work.groupby(group)["is_elegivel"].sum()).reset_index(name="%_enviada_elegivel"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_ativada"].sum() / work.groupby(group)["is_aprovada"].sum()).reset_index(name="%_ativada"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_ativada"].sum() / work.groupby(group).size()).reset_index(name="%_conversao"),
        how="left", on=group,
    )

    metricas = ["mix", "%_elegivel", "%_enviada_elegivel", "%_ativada", "%_conversao"]

    tabela = aux.pivot(
        index=dt_col,
        columns=model_col,
        values=metricas,
    )

  # diff: challenger - baseline (blend4 - blend3)
    tabela[("diff_conversao", "")] = (
        tabela[("%_conversao", challenger_model)] - tabela[("%_conversao", baseline_model)]
    )

    return tabela.sort_index()

def color_negative_diff(row):
    val = row.get(("diff_conversao", ""))
    if pd.notna(val) and val < 0:
        return ["background-color: #cc0000"] * len(row)
    return [""] * len(row)

def plot_funnel_rating_comparison(
    df: pd.DataFrame,
    rating: str,
    models: list,
    metrics: dict,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    rating_col: str = "rating_score_ds",
    color_map: dict = None,
    min_volume_per_model: int = 50,
    ylim=(0, 110),
    show_volume: bool = True,
):
    """Compare funnel metrics between models for a single rating."""
    color_map = color_map or {}
    df_rating = df[df[rating_col] == rating].copy()
    if df_rating.empty:
        print(f"[skip] Rating {rating}: sem dados")
        return

    vol = df_rating.groupby(model_col).size()
    if not all(vol.get(m, 0) >= min_volume_per_model for m in models):
        print(f"[skip] Rating {rating}: volume insuficiente — {vol.to_dict()}")
        return

    period_label = "Dia" if time_grain == "day" else "Semana"
    rates_by_model = {}
    all_periods = []

    for model in models:
        rates = compute_blend_funnel_rates(
            df_rating[df_rating[model_col] == model],
            time_grain=time_grain,
        )
        if rates.empty:
            continue
        rates_by_model[model] = rates.set_index(rates["period"].astype(str))
        all_periods.extend(rates["period"].astype(str).tolist())

    if not rates_by_model:
        print(f"[skip] Rating {rating}: sem rates")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    for metric_col, metric_label in metrics.items():
        if show_volume:
            fig, axes = plt.subplots(
                2, 1, figsize=(12, 7), sharex=True,
                gridspec_kw={"height_ratios": [0.65, 0.35]},
            )
            ax_rates, ax_vol = axes
        else:
            fig, ax_rates = plt.subplots(figsize=(12, 5))
            ax_vol = None

        for model in models:
            if model not in rates_by_model:
                continue
            rates = rates_by_model[model].reindex(periods)
            y = rates[metric_col].values
            color = color_map.get(model, "#64748B")

            ax_rates.plot(x, y, marker="o", linewidth=2, color=color, label=model)
            for xi, val in zip(x, y):
                if pd.notna(val):
                    ax_rates.text(xi, val + 1.0, f"{val:.1f}%", ha="center", fontsize=8, color=color)

            if show_volume and ax_vol is not None:
                vol_vals = rates["volume"].fillna(0).values
                offset = (list(models).index(model) - 0.5) * 0.35
                ax_vol.bar(x + offset, vol_vals, width=0.35, color=color, alpha=0.75, label=model)

        ax_rates.set_title(f"Rating {rating} — {metric_label} ({period_label})")
        ax_rates.set_ylabel("Proporção (%)")
        ax_rates.set_ylim(*ylim)
        ax_rates.grid(axis="y", alpha=0.25)
        ax_rates.legend(title="Modelo")

        if ax_vol is not None:
            ax_vol.set_xticks(x)
            ax_vol.set_xticklabels(periods, rotation=45, ha="right")
            ax_vol.set_ylabel("Requisições (n)")
            ax_vol.legend(title="Modelo")
            ax_vol.grid(axis="y", alpha=0.2)
        else:
            ax_rates.set_xticks(x)
            ax_rates.set_xticklabels(periods, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

RATING_FUNNEL_METRICS = {
    "elegivel_pct_total": "Elegível",
    "enviada_pct_elegivel": "Envio",
    "conversao_pct_total": "Conversão",
}

MODEL_LINESTYLE = {
    "BLEND4": "-",      # sólida
    "BLEND3_3": "--",    # tracejada
}

def plot_funnel_rating_overview_comparison(
    df: pd.DataFrame,
    rating: str,
    models: list,
    metrics: dict = None,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    rating_col: str = "rating_score_ds",
    color_map: dict = None,
    min_volume_per_model: int = 50,
    ylim=(0, 110),
    show_labels: bool = True,
):
    """
    One chart per rating: elegível, envio and conversão for all models on the same axes.
    Color = metric, linestyle = model. No volume subplot.
    """
    metrics = metrics or RATING_FUNNEL_METRICS
    color_map = color_map or MODEL_COLOR_MAP

    df_rating = df[df[rating_col] == rating].copy()
    if df_rating.empty:
        print(f"[skip] Rating {rating}: sem dados")
        return

    vol = df_rating.groupby(model_col).size()
    if not all(vol.get(m, 0) >= min_volume_per_model for m in models):
        print(f"[skip] Rating {rating}: volume insuficiente — {vol.to_dict()}")
        return

    period_label = "Dia" if time_grain == "day" else "Semana"
    rates_by_model = {}
    all_periods = []

    for model in models:
        rates = compute_blend_funnel_rates(
            df_rating[df_rating[model_col] == model],
            time_grain=time_grain,
        )
        if rates.empty:
            continue
        rates_by_model[model] = rates.set_index(rates["period"].astype(str))
        all_periods.extend(rates["period"].astype(str).tolist())

    if not rates_by_model:
        print(f"[skip] Rating {rating}: sem rates")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(12, 5))

    for model in models:
        if model not in rates_by_model:
            continue

        rates = rates_by_model[model].reindex(periods)
        linestyle = MODEL_LINESTYLE.get(model, "-")

        for metric_col, metric_label in metrics.items():
            metric_color = BLEND_FUNNEL_COLORS.get(metric_col, "#64748B")
            y = rates[metric_col].values
            label = f"{model} — {metric_label}"

            ax.plot(
                x, y,
                marker="o",
                linewidth=2,
                color=metric_color,
                linestyle=linestyle,
                label=label,
            )

            if show_labels:
                for xi, val in zip(x, y):
                    if pd.notna(val):
                        ax.text(
                            xi, val + 1.2, f"{val:.1f}%",
                            ha="center", va="bottom",
                            fontsize=7, color=metric_color,
                        )

    ax.set_title(f"Funil — Rating {rating} — BLEND4 vs BLEND3_3 ({period_label})")
    ax.set_ylabel("Proporção (%)")
    ax.set_xlabel(period_label)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modelo — Métrica", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def build_blend_comparison_summary_table(
    df: pd.DataFrame,
    models: list,
    model_col: str = "bureau_nm_ajust",
    time_grain: str = "week",
    baseline_model: str = "BLEND3_3",
    challenger_model: str = "BLEND4",
):
    work = prepare_blend_funnel_columns(df)
    work = work[work[model_col].isin(models)].copy()
    work["requested_at"] = pd.to_datetime(work["requested_at"])

    if time_grain == "week":
        work = prepare_week_columns(work, "requested_at")
        dt_col = "year_week"
    else:
        work["day"] = work["requested_at"].dt.strftime("%Y-%m-%d")
        dt_col = "day"

    group = [model_col, dt_col]

    aux = work.groupby(group).size().reset_index(name="qtd").merge(
        (work.groupby(group).size() / work.groupby(dt_col).size()).reset_index(name="mix"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_elegivel"].sum() / work.groupby(group).size()).reset_index(name="%_elegivel"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_enviada"].sum() / work.groupby(group)["is_elegivel"].sum()).reset_index(name="%_enviada_elegivel"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_ativada"].sum() / work.groupby(group)["is_aprovada"].sum()).reset_index(name="%_ativada"),
        how="left", on=group,
    ).merge(
        (work.groupby(group)["is_ativada"].sum() / work.groupby(group).size()).reset_index(name="%_conversao"),
        how="left", on=group,
    )

    metricas = ["mix", "%_elegivel", "%_enviada_elegivel", "%_ativada", "%_conversao"]

    tabela = aux.pivot(
        index=dt_col,
        columns=model_col,
        values=metricas,
    )

  # diff: challenger - baseline (blend4 - blend3)
    tabela[("diff_conversao", "")] = (
        tabela[("%_conversao", challenger_model)] - tabela[("%_conversao", baseline_model)]
    )

    return tabela.sort_index()

def color_negative_diff(row):
    val = row.get(("diff_conversao", ""))
    if pd.notna(val) and val < 0:
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)


def build_blend_comparison_summary_table_counts(
    df: pd.DataFrame,
    models: list,
    model_col: str = "bureau_nm_ajust",
    time_grain: str = "week",
    baseline_model: str = "BLEND3",
    challenger_model: str = "BLEND4",
):
    work = prepare_blend_funnel_columns(df)
    work = work[work[model_col].isin(models)].copy()
    work["requested_at"] = pd.to_datetime(work["requested_at"])

    if time_grain == "week":
        work = prepare_week_columns(work, "requested_at")
        dt_col = "year_week"
    else:
        work["day"] = work["requested_at"].dt.strftime("%Y-%m-%d")
        dt_col = "day"

    rows = []
    for keys, group in work.groupby([model_col, dt_col], sort=False):
        model, period = keys
        total = len(group)
        elegivel = int(group["is_elegivel"].sum())
        enviada_elegivel = int(group.loc[group["is_elegivel"].eq(1), "is_enviada"].sum())
        ativada_aprovada = int(group.loc[group["is_aprovada"].eq(1), "is_ativada"].sum())
        conversao = int(group["is_ativada"].sum())

        rows.append({
            model_col: model,
            dt_col: period,
            "mix": total,
            "elegivel": elegivel,
            "enviada_elegivel": enviada_elegivel,
            "ativada": ativada_aprovada,
            "conversao": conversao,
        })

    aux = pd.DataFrame(rows)
    metricas = ["mix", "elegivel", "enviada_elegivel", "ativada", "conversao"]

    tabela = aux.pivot(
        index=dt_col,
        columns=model_col,
        values=metricas,
    )

    tabela[("diff_conversao", "")] = (
        tabela[("conversao", challenger_model)] - tabela[("conversao", baseline_model)]
    )

    return tabela.sort_index()

def plot_funnel_rating_overview_side_by_side(
    df: pd.DataFrame,
    rating: str,
    models: list,
    metrics: dict = None,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    rating_col: str = "rating_score_ds",
    color_map: dict = None,
    min_volume_per_model: int = 50,
    ylim=(0, 110),
    show_labels: bool = True,
):
    """
    One row of subplots per rating: one panel per model.
    Each panel shows Elegível, Envio and Conversão with metric colors.
    """
    metrics = metrics or RATING_FUNNEL_METRICS
    color_map = color_map or MODEL_COLOR_MAP

    df_rating = df[df[rating_col] == rating].copy()
    if df_rating.empty:
        print(f"[skip] Rating {rating}: sem dados")
        return

    vol = df_rating.groupby(model_col).size()
    if not all(vol.get(m, 0) >= min_volume_per_model for m in models):
        print(f"[skip] Rating {rating}: volume insuficiente — {vol.to_dict()}")
        return

    period_label = "Dia" if time_grain == "day" else "Semana"
    rates_by_model = {}
    all_periods = []

    for model in models:
        rates = compute_blend_funnel_rates(
            df_rating[df_rating[model_col] == model],
            time_grain=time_grain,
        )
        if rates.empty:
            continue
        rates_by_model[model] = rates.set_index(rates["period"].astype(str))
        all_periods.extend(rates["period"].astype(str).tolist())

    available_models = [m for m in models if m in rates_by_model]
    if not available_models:
        print(f"[skip] Rating {rating}: sem rates")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    fig, axes = plt.subplots(
        1, len(available_models),
        figsize=(7 * len(available_models), 5),
        sharex=True,
        sharey=True,
    )
    if len(available_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, available_models):
        rates = rates_by_model[model].reindex(periods)
        model_color = color_map.get(model, "#64748B")

        for metric_col, metric_label in metrics.items():
            metric_color = BLEND_FUNNEL_COLORS.get(metric_col, "#64748B")
            y = rates[metric_col].values

            ax.plot(
                x, y,
                marker="o",
                linewidth=2,
                color=metric_color,
                label=metric_label,
            )

            if show_labels:
                for xi, val in zip(x, y):
                    if pd.notna(val):
                        ax.text(
                            xi, val + 1.2, f"{val:.1f}%",
                            ha="center", va="bottom",
                            fontsize=7, color=metric_color,
                        )

        ax.set_title(model, color=model_color, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45, ha="right")
        ax.legend(title="Métrica", loc="upper right")

    axes[0].set_ylabel("Proporção (%)")
    axes[-1].set_xlabel(period_label)
    fig.suptitle(
        f"Funil — Rating {rating} — BLEND4 vs BLEND3_3 ({period_label})",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

RATING_ORDER = ["A", "B", "C", "D", "E", "N/I"]
RATING_COLOR_MAP = {
    "A": "#7DD3FC",   # azul claro
    "B": "#5EEAD4",   # verde-água
    "C": "#93C5FD",   # azul
    "D": "#818CF8",   # roxo
    "E": "#334155",   # cinza escuro
    "N/I": "#CBD5E1", # cinza claro
}

def _auto_ylim(values, pad_pct=5.0, floor=0, ceiling=100):
    """Compute y-axis limits from data with padding."""
    vals = pd.Series(values).dropna()
    if vals.empty:
        return (floor, ceiling)

    lo = max(floor, vals.min() - pad_pct)
    hi = min(ceiling, vals.max() + pad_pct)

    if lo >= hi:
        hi = min(ceiling, lo + pad_pct)

    return (lo, hi)


def plot_funnel_metrics_by_rating_side_by_side(
    df: pd.DataFrame,
    models: list,
    ratings: list = None,
    metrics: dict = None,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    rating_col: str = "rating_score_ds",
    rating_color_map: dict = None,
    min_volume_per_model: int = 50,
    ylim=None,
    auto_ylim_pad: float = 5.0,
    show_labels: bool = False,
    figsize=(16, 14),
    suptitle: str = None,
):
    """
    Grid Nx2: rows = Elegível / Envio / Conversão;
    left column = all ratings for models[0], right column = all ratings for models[1].

    ylim:
      - None: auto scale per row (Elegível, Envio and Conversão can differ)
      - tuple, e.g. (0, 110): same fixed scale for all rows
      - dict, e.g. {"elegivel_pct_total": (0, 110), ...}: fixed scale per metric
    """
    metrics = metrics or RATING_FUNNEL_METRICS
    ratings = ratings or ["A", "B", "C", "D", "E"]
    rating_color_map = rating_color_map or RATING_COLOR_MAP

    if len(models) != 2:
        raise ValueError("models must contain exactly 2 items: [blend3_left, blend4_right]")

    period_label = "Dia" if time_grain == "day" else "Semana"
    metric_items = list(metrics.items())

    rates_by_model_rating = {}
    all_periods = []

    for model in models:
        for rating in ratings:
            df_slice = df[(df[model_col] == model) & (df[rating_col] == rating)].copy()
            if df_slice.empty:
                continue

            vol = len(df_slice)
            if vol < min_volume_per_model:
                continue

            rates = compute_blend_funnel_rates(df_slice, time_grain=time_grain)
            if rates.empty:
                continue

            key = (model, rating)
            rates_by_model_rating[key] = rates.set_index(rates["period"].astype(str))
            all_periods.extend(rates["period"].astype(str).tolist())

    if not rates_by_model_rating:
        print("[skip] Sem dados para o grid por rating")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    fig, axes = plt.subplots(
        len(metric_items), 2,
        figsize=figsize,
        sharex=True,
        sharey="row",
    )
    if len(metric_items) == 1:
        axes = np.array([axes])

    model_titles = {
        models[0]: "Blend 3 — BLEND3",
        models[1]: "Blend 4 — BLEND4",
    }

    for row_idx, (metric_col, metric_label) in enumerate(metric_items):
        row_values = []
        for model in models:
            for rating in ratings:
                key = (model, rating)
                if key not in rates_by_model_rating:
                    continue
                row_values.extend(
                    rates_by_model_rating[key]
                    .reindex(periods)[metric_col]
                    .tolist()
                )

        if isinstance(ylim, dict):
            row_ylim = ylim.get(metric_col, _auto_ylim(row_values, pad_pct=auto_ylim_pad))
        elif ylim is None:
            row_ylim = _auto_ylim(row_values, pad_pct=auto_ylim_pad)
        else:
            row_ylim = ylim

        label_offset = (row_ylim[1] - row_ylim[0]) * 0.03

        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            has_data = False

            for rating in ratings:
                key = (model, rating)
                if key not in rates_by_model_rating:
                    continue

                rates = rates_by_model_rating[key].reindex(periods)
                y = rates[metric_col].values
                color = rating_color_map.get(rating, "#64748B")

                ax.plot(
                    x, y,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=f"Rating {rating}",
                )

                if show_labels:
                    for xi, val in zip(x, y):
                        if pd.notna(val):
                            ax.text(
                                xi, val + 0.5 * label_offset, f"{val:.1f}%",
                                ha="center", va="bottom",
                                fontsize=6, color=color,
                            )
                has_data = True

            ax.set_title(f"{metric_label} — {model_titles.get(model, model)}")
            ax.set_ylim(*row_ylim)
            ax.grid(axis="y", alpha=0.25)

            if row_idx == len(metric_items) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(periods, rotation=45, ha="right")
                ax.set_xlabel(period_label)

            if col_idx == 0:
                ax.set_ylabel("Proporção (%)")

            if has_data:
                ax.legend(title="Rating", loc="best", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, y=1.1, fontsize=13)

    plt.tight_layout()
    plt.show()

def plot_pre_analysis_by_rating_side_by_side(
    df: pd.DataFrame,
    models: list,
    ratings: list = None,
    outcomes: list = None,
    time_grain: str = "week",
    model_col: str = "bureau_nm_ajust",
    rating_col: str = "rating_score_ds",
    category_col: str = "pre_analysis_result",
    rating_color_map: dict = None,
    min_volume_per_model: int = 50,
    ylim=None,
    auto_ylim_pad: float = 5.0,
    show_labels: bool = True,
    figsize=(16, 12),
    suptitle: str = None,
):
    """
    Grid 3x2: rows = APROVAR / DERIVAR / REPROVAR;
    left column = all ratings for models[0], right column = all ratings for models[1].

    ylim:
      - None: auto scale per row (APROVAR, DERIVAR and REPROVAR can differ)
      - tuple, e.g. (0, 110): same fixed scale for all rows
      - dict, e.g. {"APROVAR": (0, 110), ...}: fixed scale per outcome
    """
    ratings = ratings or ["A", "B", "C", "D", "E"]
    outcomes = outcomes or ["APROVAR", "DERIVAR", "REPROVAR"]
    rating_color_map = rating_color_map or RATING_COLOR_MAP

    if len(models) != 2:
        raise ValueError("models must contain exactly 2 items: [blend4_left, blend3_right]")

    period_label = "Dia" if time_grain == "day" else "Semana"

    pct_by_model_rating = {}
    all_periods = []

    for model in models:
        for rating in ratings:
            df_slice = df[(df[model_col] == model) & (df[rating_col] == rating)].copy()
            if df_slice.empty or len(df_slice) < min_volume_per_model:
                continue

            pct_df, _, _ = compute_category_mix(
                df_slice, category_col=category_col, time_grain=time_grain
            )
            if pct_df.empty:
                continue

            key = (model, rating)
            pct_by_model_rating[key] = pct_df
            all_periods.extend(pct_df.index.tolist())

    if not pct_by_model_rating:
        print("[skip] Pré-análise: sem dados para o grid por rating")
        return

    periods = sorted(set(all_periods))
    x = np.arange(len(periods))

    fig, axes = plt.subplots(
        len(outcomes), 2,
        figsize=figsize,
        sharex=True,
        sharey="row",
    )
    if len(outcomes) == 1:
        axes = np.array([axes])

    model_titles = {
        models[0]: f"Blend 3 — {models[0]}",
        models[1]: f"Blend 4 — {models[1]}",
    }

    for row_idx, outcome in enumerate(outcomes):
        row_values = []
        for model in models:
            for rating in ratings:
                key = (model, rating)
                if key not in pct_by_model_rating:
                    continue
                pct_df = pct_by_model_rating[key].reindex(periods).fillna(0)
                if outcome not in pct_df.columns:
                    continue
                row_values.extend(pct_df[outcome].tolist())

        if isinstance(ylim, dict):
            row_ylim = ylim.get(outcome, _auto_ylim(row_values, pad_pct=auto_ylim_pad))
        elif ylim is None:
            row_ylim = _auto_ylim(row_values, pad_pct=auto_ylim_pad)
        else:
            row_ylim = ylim

        label_offset = (row_ylim[1] - row_ylim[0]) * 0.03

        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            has_data = False

            for rating in ratings:
                key = (model, rating)
                if key not in pct_by_model_rating:
                    continue

                pct_df = pct_by_model_rating[key].reindex(periods).fillna(0)
                if outcome not in pct_df.columns:
                    continue

                y = pct_df[outcome].values
                color = rating_color_map.get(rating, "#64748B")

                ax.plot(x, y, marker="o", linewidth=2, color=color, label=f"Rating {rating}")

                if show_labels:
                    for xi, val in zip(x, y):
                        if val > 0:
                            ax.text(
                                xi, val + 0.05 * label_offset, f"{val:.1f}%",
                                ha="center", va="bottom", fontsize=6, color=color,
                            )

                has_data = True

            # title_prefix = outcome if col_idx == 0 else ""
            # ax.set_title(f"{title_prefix} — {model_titles.get(model, model)}".strip(" —"))
            ax.set_title(f"{outcome} — {model_titles.get(model, model)}")
            ax.set_ylim(*row_ylim)
            ax.grid(axis="y", alpha=0.25)

            if row_idx == len(outcomes) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(periods, rotation=45, ha="right")
                ax.set_xlabel(period_label)

            if col_idx == 0:
                ax.set_ylabel("Proporção (%)")

            if has_data:
                ax.legend(title="Rating", loc="best", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, y=1.1, fontsize=13)

    plt.tight_layout()
    plt.show()

def build_funnel_by_rating_table(
    df,
    models,
    ratings=None,
    time_grain="week",
    model_col="bureau_nm_ajust",
    rating_col="rating_score_ds",
    min_volume_per_model=50,
):
    ratings = ratings or ["A", "B", "C", "D", "E"]
    metric_cols = ["elegivel_pct_total", "enviada_pct_elegivel", "conversao_pct_total"]
    rows = []

    for model in models:
        for rating in ratings:
            df_slice = df[(df[model_col] == model) & (df[rating_col] == rating)].copy()
            if df_slice.empty or len(df_slice) < min_volume_per_model:
                continue

            rates = compute_blend_funnel_rates(df_slice, time_grain=time_grain)
            if rates.empty:
                continue

            part = rates[["period", "volume"] + metric_cols].copy()
            part["model"] = model
            part["rating"] = rating
            rows.append(part)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for col in metric_cols:
        out[col] = out[col].round(1)
    return out