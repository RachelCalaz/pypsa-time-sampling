import geopandas as gpd
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import pypsa
from concurrent.futures import ProcessPoolExecutor
import os
from shapely.geometry import Point
import xarray as xr
import warnings
warnings.simplefilter(action="ignore") # Comment out for debugging and development
import re
import time
import tsam.timeseriesaggregation as tsam
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from pathlib import Path

from _helpers import (
    remove_leap_day,
    normalize_and_rename_df, 
)

PMIN_SUFFIX = "__pmin"

"""
********************************************************************************
    Time step reduction
********************************************************************************
"""

def assign_reduced_ts(n_attr, reduced_df):
    if n_attr.empty:
        return n_attr
    cols = n_attr.columns.intersection(reduced_df.columns)
    return reduced_df[cols]


def _write_reduced_to_network(n, reduced_df):
    pmin_cols = reduced_df.columns[reduced_df.columns.str.endswith(PMIN_SUFFIX)]
    pmin_df = reduced_df[pmin_cols].rename(columns=lambda c: c[: -len(PMIN_SUFFIX)])
    other_df = reduced_df.drop(columns=pmin_cols)

    n.generators_t.p_max_pu = assign_reduced_ts(n.generators_t.p_max_pu, other_df)
    n.generators_t.p_min_pu = assign_reduced_ts(n.generators_t.p_min_pu, pmin_df)
    n.loads_t.p_set = assign_reduced_ts(n.loads_t.p_set, other_df)
    n.storage_units_t.inflow = assign_reduced_ts(n.storage_units_t.inflow, other_df)


def average_every_nhours(n, offset):
    logging.info(f"Resampling the network to {offset}")
    m = n.copy()#with_time=False)
    snapshots_unstacked = n.snapshots.get_level_values(1)

    snapshot_weightings = n.snapshot_weightings.copy().set_index(snapshots_unstacked).resample(offset).sum()
    snapshot_weightings = remove_leap_day(snapshot_weightings)
    snapshot_weightings=snapshot_weightings[snapshot_weightings.index.year.isin(n.investment_periods)]
    snapshot_weightings.index = pd.MultiIndex.from_arrays([snapshot_weightings.index.year, snapshot_weightings.index])
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                resampled = df.set_index(snapshots_unstacked).resample(offset).mean()
                resampled = remove_leap_day(resampled)
                resampled=resampled[resampled.index.year.isin(n.investment_periods)]
                resampled.index = snapshot_weightings.index
                pnl[k] = resampled
    return m

def single_year_segmentation(n, snapshots, segments, config):

    p_max_pu_norm = n.generators_t.p_max_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    p_min_pu_norm = n.generators_t.p_min_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    load_norm = n.loads_t.p_set.loc[snapshots].abs().max().clip(lower=1e-6)
    inflow_norm = n.storage_units_t.inflow.loc[snapshots].abs().max().clip(lower=1e-6)

    p_max_pu = (n.generators_t.p_max_pu.loc[snapshots] / p_max_pu_norm).fillna(1)
    p_min_pu = (n.generators_t.p_min_pu.loc[snapshots] / p_min_pu_norm).fillna(0).add_suffix(PMIN_SUFFIX)
    load = (n.loads_t.p_set.loc[snapshots] / load_norm).fillna(1)
    inflow = (n.storage_units_t.inflow.loc[snapshots] / inflow_norm).fillna(0)

    raw = pd.concat([p_max_pu, p_min_pu, load, inflow], axis=1, sort=False)

    multi_index = False
    if isinstance(raw.index, pd.MultiIndex):
        multi_index = True
        raw.index = raw.index.droplevel(0)

    y = snapshots.get_level_values(0)[0] if multi_index else snapshots[0].year

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=config.get('solver_name', 'highs'),
    )

    segmented_df = agg.createTypicalPeriods()
    weightings = segmented_df.index.get_level_values("Segment Duration")
    cumsum = np.cumsum(weightings[:-1])

    if np.floor(y/4)-y/4 == 0: # check if leap year and add Feb 29
        cumsum = np.where(cumsum >= 1416, cumsum + 24, cumsum)

    offsets = np.insert(cumsum, 0, 0)
    start_snapshot = snapshots[0][1] if n.multi_invest else snapshots[0]
    new_snapshots = pd.DatetimeIndex([start_snapshot + pd.Timedelta(hours=offset) for offset in offsets])
    if multi_index:
        new_snapshots = pd.MultiIndex.from_arrays(
            [new_snapshots.year.astype(int), new_snapshots],
            names=n.snapshots.names,
        )
    weightings = pd.Series(weightings, index=new_snapshots, name="weightings", dtype="float64")
    segmented_df.index = new_snapshots

    segmented_df[p_max_pu.columns] *= p_max_pu_norm.values
    segmented_df[p_min_pu.columns] *= p_min_pu_norm.values
    segmented_df[load.columns] *= load_norm.values
    segmented_df[inflow.columns] *= inflow_norm.values
     
    logging.info(f"Segmentation complete for period: {y}")

    return segmented_df, weightings

def apply_time_segmentation(n, segments, config):
    logging.info(f"Aggregating time series to {segments} segments.")    
    years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]

    if len(years) == 1:
        segmented_df, weightings = single_year_segmentation(n, n.snapshots, segments, config)
    else:

        with ProcessPoolExecutor(max_workers = min(len(years),config['nprocesses'])) as executor:
            parallel_seg = {
                year: executor.submit(
                    single_year_segmentation,
                    n,
                    n.snapshots[n.snapshots.get_level_values(0) == year],
                    segments,
                    config
                )
                for year in years
            }

        segmented_df = pd.concat(
            [parallel_seg[year].result()[0] for year in parallel_seg], axis=0
        )
        weightings = pd.concat(
            [parallel_seg[year].result()[1] for year in parallel_seg], axis=0
        )

    n.set_snapshots(segmented_df.index)
    n.snapshot_weightings = weightings   

    _write_reduced_to_network(n, segmented_df)

    # Adjusting ramp limits by snapshot weightings for segmentation method only
    weights = n.snapshot_weightings.objective
    gens = n.generators.index[n.generators['ramp_limit_up'].notnull()]
    n.generators_t['ramp_limit_up'] = pd.DataFrame(weights.values[:, None] * n.generators.loc[gens, 'ramp_limit_up'].values,index=n.snapshots, columns=gens).clip(upper=1)
    n.generators_t['ramp_limit_down'] = pd.DataFrame(weights.values[:, None] * n.generators.loc[gens, 'ramp_limit_down'].values,index=n.snapshots, columns=gens).clip(upper=1)

    return n

def single_year_tsam_clustering(n, snapshots, periods, method, config):

    p_max_pu_norm = n.generators_t.p_max_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    p_min_pu_norm = n.generators_t.p_min_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    load_norm = n.loads_t.p_set.loc[snapshots].abs().max().clip(lower=1e-6)
    inflow_norm = n.storage_units_t.inflow.loc[snapshots].abs().max().clip(lower=1e-6)

    p_max_pu = (n.generators_t.p_max_pu.loc[snapshots] / p_max_pu_norm).fillna(1)
    p_min_pu = (n.generators_t.p_min_pu.loc[snapshots] / p_min_pu_norm).fillna(0).add_suffix(PMIN_SUFFIX)
    load = (n.loads_t.p_set.loc[snapshots] / load_norm).fillna(1)
    inflow = (n.storage_units_t.inflow.loc[snapshots] / inflow_norm).fillna(0)

    raw = pd.concat([p_max_pu, p_min_pu, load, inflow], axis=1, sort=False)

    PeakMax = ['peak_load', 'peak_solar', 'peak_wind']
    PeakMin = ['peak_load', 'peak_vre']

    raw['peak_load'] = load.sum(axis=1)

    solar_carriers = ['solar_pv', 'solar_pv_low']
    wind_carriers = ['wind', 'wind_low']
    solar_cl = n.generators.query('carrier in @solar_carriers').index
    wind_cl = n.generators.query('carrier in @wind_carriers').index

    raw['peak_solar'] = p_max_pu[solar_cl].sum(axis=1)
    raw['peak_wind'] = p_max_pu[wind_cl].sum(axis=1)
    raw['peak_vre'] = raw['peak_solar'] + raw['peak_wind']

    multi_index = False
    if isinstance(raw.index, pd.MultiIndex):
        multi_index = True
        raw.index = raw.index.droplevel(0)

    y = snapshots.get_level_values(0)[0] if multi_index else snapshots[0].year

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=24,
        noTypicalPeriods=int(periods),
        clusterMethod=method,
        solver=config.get("solver_name", "highs"),
        extremePeriodMethod="new_cluster_center",
        addPeakMax=PeakMax,
        addPeakMin=PeakMin,
    )

    clustered_df = agg.createTypicalPeriods()
    period_ids = clustered_df.index.get_level_values(0).unique()
    period_weightings = agg.clusterPeriodNoOccur
    clustered_df = clustered_df.drop(columns = ['peak_load', 'peak_vre', 'peak_wind', 'peak_solar']).reset_index(level=0, drop=True)

    weightings = []
    for typical_period in period_ids:
        weight = period_weightings[typical_period]
        weightings.extend([weight] * 24)

    weightings = pd.Series(weightings, name="weightings", dtype="float64")

    start_snapshot = snapshots[0][1] if n.multi_invest else snapshots[0]

    reduced_snapshots = pd.date_range(
        start=start_snapshot,
        periods=len(clustered_df),
        freq="h",
    )

    if multi_index:
        reduced_snapshots = pd.MultiIndex.from_arrays(
            [np.array([y] * len(reduced_snapshots), dtype=int), reduced_snapshots],
            names=snapshots.names,
        )

    clustered_df.index = reduced_snapshots
    weightings.index = reduced_snapshots

    clustered_df[p_max_pu.columns] *= p_max_pu_norm.values
    clustered_df[p_min_pu.columns] *= p_min_pu_norm.values
    clustered_df[load.columns] *= load_norm.values
    clustered_df[inflow.columns] *= inflow_norm.values

    logging.info(f"{method} clustering complete for period: {y}")

    return clustered_df, weightings


def single_year_kshape_clustering(n, snapshots, periods, config):

    if KShape is None or TimeSeriesScalerMeanVariance is None:
        raise ImportError("KSHAPE requires tslearn. Install with: pip install tslearn")

    p_max_pu_norm = n.generators_t.p_max_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    p_min_pu_norm = n.generators_t.p_min_pu.loc[snapshots].abs().max().clip(lower=1e-6)
    load_norm = n.loads_t.p_set.loc[snapshots].abs().max().clip(lower=1e-6)
    inflow_norm = n.storage_units_t.inflow.loc[snapshots].abs().max().clip(lower=1e-6)

    p_max_pu = (n.generators_t.p_max_pu.loc[snapshots] / p_max_pu_norm).fillna(1)
    p_min_pu = (n.generators_t.p_min_pu.loc[snapshots] / p_min_pu_norm).fillna(0).add_suffix(PMIN_SUFFIX)
    load = (n.loads_t.p_set.loc[snapshots] / load_norm).fillna(1)
    inflow = (n.storage_units_t.inflow.loc[snapshots] / inflow_norm).fillna(0)

    raw = pd.concat([p_max_pu, p_min_pu, load, inflow], axis=1, sort=False)

    multi_index = False
    if isinstance(raw.index, pd.MultiIndex):
        multi_index = True
        raw.index = raw.index.droplevel(0)

    y = snapshots.get_level_values(0)[0] if multi_index else snapshots[0].year

    n_days = len(raw) // 24
    values = raw.to_numpy().reshape(n_days, 24, raw.shape[1])

    scaled = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(values)

    model = KShape(
        n_clusters=int(periods),
        random_state=config.get("random_seed", 0),
        max_iter=config.get("kshape_max_iter", 100),
        n_init=config.get("kshape_n_init", 1),
    )

    labels = model.fit_predict(scaled)

    typical_days = []
    weights = []

    for cluster_id in range(int(periods)):
        members = values[labels == cluster_id]

        if len(members) == 0:
            continue

        typical_days.append(members.mean(axis=0))
        weights.append(float(len(members)))

    clustered_df = pd.DataFrame(
        np.vstack(typical_days),
        columns=raw.columns,
    )

    weightings = []
    for w in weights:
        weightings.extend([w] * 24)

    start_snapshot = snapshots[0][1] if n.multi_invest else snapshots[0]

    reduced_snapshots = pd.date_range(
        start=start_snapshot,
        periods=len(clustered_df),
        freq="h",
    )

    if multi_index:
        reduced_snapshots = pd.MultiIndex.from_arrays(
            [np.array([y] * len(reduced_snapshots), dtype=int), reduced_snapshots],
            names=snapshots.names,
        )

    clustered_df.index = reduced_snapshots
    weightings = pd.Series(
        weightings,
        index=reduced_snapshots,
        name="weightings",
        dtype="float64",
    )

    clustered_df[p_max_pu.columns] *= p_max_pu_norm.values
    clustered_df[p_min_pu.columns] *= p_min_pu_norm.values
    clustered_df[load.columns] *= load_norm.values
    clustered_df[inflow.columns] *= inflow_norm.values

    logging.info(f"KSHAPE clustering complete for period: {y}")

    return clustered_df, weightings




def apply_time_kshape_clustering(n, periods, config):
    logging.info(f"Aggregating time series to {periods} typical periods using KSHAPE.")

    years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]

    if len(years) == 1:
        clustered_df, weightings = single_year_kshape_clustering(n, n.snapshots, periods, config)
    else:
        with ProcessPoolExecutor(max_workers=min(len(years), config["nprocesses"])) as executor:
            parallel = {
                year: executor.submit(
                    single_year_kshape_clustering,
                    n,
                    n.snapshots[n.snapshots.get_level_values(0) == year],
                    periods,
                    config,
                )
                for year in years
            }

        clustered_df = pd.concat(
            [parallel[year].result()[0] for year in parallel],
            axis=0,
        )
        weightings = pd.concat(
            [parallel[year].result()[1] for year in parallel],
            axis=0,
        )

    n.set_snapshots(clustered_df.index)
    n.snapshot_weightings = weightings

    _write_reduced_to_network(n, clustered_df)

    return n


def apply_time_tsam_clustering(n, periods, method, config):
    logging.info(f"Aggregating time series to {periods} typical periods using {method}.")
    years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]

    if len(years) == 1:
        clustered_df, weightings = single_year_tsam_clustering(n, n.snapshots, periods, method, config)
    else:
        with ProcessPoolExecutor(max_workers=min(len(years), config["nprocesses"])) as executor:
            parallel = {
                year: executor.submit(
                    single_year_tsam_clustering,
                    n,
                    n.snapshots[n.snapshots.get_level_values(0) == year],
                    periods,
                    method,
                    config,
                )
                for year in years
            }

        clustered_df = pd.concat(
            [parallel[year].result()[0] for year in parallel], axis=0
            )
        weightings = pd.concat(
            [parallel[year].result()[1] for year in parallel], axis=0
            )

    n.set_snapshots(clustered_df.index)
    n.snapshot_weightings = weightings

    _write_reduced_to_network(n, clustered_df)

    return n


def apply_time_sampling(opts, n, tsam_clustering):

    # n average hours
    for o in opts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            t0 = time.perf_counter()
            n = average_every_nhours(n, m[0])
            logging.info(f"Time sampling ({m[0]} averaging) completed in {time.perf_counter() - t0:.1f}s")
            break

    for o in opts:
        m = re.match(r"^(\d+)(SEG|KMEANS|KMEDOIDS|HAC|KSHAPE)$", o, re.IGNORECASE)
        if m is None:
            continue

        n_periods = m.group(1)
        op = m.group(2).lower()
        t0 = time.perf_counter()

        if op == "seg":
            # variable segmentation
            n = apply_time_segmentation(n, n_periods, tsam_clustering)
        if op == "kshape":
            n = apply_time_kshape_clustering(n, n_periods, tsam_clustering)
        if op in ["kmeans", "kmedoids", "hac"]:
            # kmeans, kmedoids and hac with tsam
            method_map = {
            "kmeans": "k_means",
            "kmedoids": "k_medoids",
            "hac": "hierarchical",
            }
            n = apply_time_tsam_clustering(n, n_periods, method_map[op], tsam_clustering)

        elapsed = time.perf_counter() - t0
        mins, secs = divmod(elapsed, 60)
        logging.info(
            f"Time sampling ({n_periods}{op.upper()}) completed in "
            f"{int(mins)}m {secs:.1f}s ({elapsed:.1f}s total)"
        )
        break

    return n
