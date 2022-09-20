"""
Script for PVAnalytics function/pipeline that estimates azimuth and tilt.
"""

# This script acts as a function for running the az-tilt algorithm.
# In the Validation Hub, this would be a function that the user submitted
# for validation

import pandas as pd
import pvlib
from pvanalytics import system
import numpy as np
from pvanalytics.quality import gaps
from pvanalytics.quality import data_shifts as ds
from pvanalytics.quality.outliers import hampel, zscore
import rdtools

tz_conversion_dict = {'America/New_York': 'Etc/GMT+4',
                      'America/Los_Angeles': 'Etc/GMT+7',
                      'America/Chicago': 'Etc/GMT+5',
                      'America/Phoenix': 'Etc/GMT+7',
                      'America/Indiana/Indianapolis': 'Etc/GMT+5',
                      'America/Denver': 'Etc/GMT+6'}


def run_az_tilt_estimation(time_series, latitude, longitude, time_zone):
    # Sort the data just in case
    time_series = time_series.sort_index()
    # Remove any duplicated data from the time series, if present
    time_series = time_series.drop_duplicates()
    time_series = time_series[~time_series.index.duplicated(keep='first')]
    # Convert the data to 15-minute data--center align
    time_series = time_series.shift(0.5, freq='15T').resample('15T').mean()
    time_series.index.freq = "15T"
    # Correct for timezone (localize + remove DST if present)
    time_series.index = time_series.index.tz_localize(
        time_zone, ambiguous=True, nonexistent="shift_forward")
    time_series.index = time_series.index.tz_convert(
        tz_conversion_dict[time_zone])
    time_series = time_series.drop_duplicates()
    # Pull the PSM3 data down that is associated with the system via pvlib
    psm3s = []
    years = list(time_series.index.year.drop_duplicates())
    years = [x for x in years if x < 2021]
    for year in years:
        psm3 = None
        tries = 0
        while (psm3 is None) | (tries < 10):
            psm3, _ = pvlib.iotools.get_psm3(
                latitude, longitude,
                '4z5fRAXbGB3qldVVd3c6WH5CuhtY5mhgC2DyD952',
                'kirsten.perry@nrel.gov', year,
                attributes=['air_temperature',
                            'ghi', 'clearsky_ghi',
                            'clearsky_dni',
                            'clearsky_dhi'],
                map_variables=True,
                interval=30)
            tries = tries + 1
        psm3s.append(psm3)
    psm3 = pd.concat(psm3s)
    # reindex the PSM3 data based on the time series so they're consistent
    psm3 = psm3.reindex(pd.date_range(psm3.index[0],
                                      psm3.index[-1],
                                      freq='15T')
                        ).interpolate()
    psm3 = psm3.reindex(time_series.index)
    # Mask clearsky periods and daytime periods in the PSM3 data.
    is_clear = (psm3.ghi_clear == psm3.ghi)
    is_daytime = (psm3.ghi > 0)
    # Run the associated data-cleaning routines on the time series:
    # 1) Removal of data shifts/capacity shifts in the time series
    # 2) Removal of frozen/stuck data
    # 3) Removal of negative data
    # 4) Removal of data periods with low data 'completeness'
    # 5) Removal of outliers via Hampel + z-score outlier filters
    # 6) Removal of clipped values via clipping filter
    # 7) Filter to daytime clearsky data only, as determined by PSM3
    # Detect any data shifts and remove them
    time_series_daily = time_series.resample('D').sum()
    start_date, end_date = ds.get_longest_shift_segment_dates(
        time_series_daily)
    time_series = time_series[start_date:end_date]
    # Trim based on frozen data values
    stale_data_mask = gaps.stale_values_diff(time_series)
    time_series = time_series.asfreq('15T')
    time_series = time_series[~stale_data_mask]
    time_series = time_series.asfreq('15T')
    # Remove negative data
    time_series = time_series[(time_series >= 0) | (time_series.isna())]
    time_series = time_series.asfreq('15T')
    # Trim based on start-stop dates
    completeness_mask = gaps.trim_incomplete(time_series)
    time_series = time_series[completeness_mask]
    time_series = time_series.asfreq('15T')
    # Remove any outliers via Hampel and z-score filters
    hampel_outlier_mask = hampel(time_series, window=5)
    zscore_outlier_mask = zscore(time_series, zmax=2,
                                 nan_policy='omit')
    time_series = time_series[(~hampel_outlier_mask) &
                              (~zscore_outlier_mask)]
    # Apply one last filter to remove any leftover outliers
    time_series = time_series[abs(time_series - time_series.mean()) <
                              4 * time_series.std()]
    time_series = time_series.asfreq('15T')
    # Apply clipping filter from Rdtools.
    clip_mask = rdtools.filtering.xgboost_clip_filter(time_series)
    time_series = time_series[clip_mask]
    # Reindex the time series after all of the filtering so it
    # has the same index as PSM3
    time_series = time_series.reindex(is_clear.index)
    # Trim based on clearsky values
    time_series_clearsky = time_series[(is_clear) & (is_daytime)]
    time_series_clearsky = time_series_clearsky.dropna()
    psm3_clearsky = psm3.loc[time_series_clearsky.index]
    solpos_clearsky = pvlib.solarposition.get_solarposition(
        time_series_clearsky.index, latitude, longitude)
    # Check the time series size. If it covers less than 6 months,
    # axe it. If more than 6 months, run the az-tilt estimator
    # algorithm on it.
    time_series_length_days = (time_series.index.max() -
                               time_series.index.min()).days
    if time_series_length_days < 183:
        print("Not enough data... not running.")
        best_tilt = np.nan
        best_azimuth = np.nan
    else:
        best_tilt, best_azimuth, r2 = system.infer_orientation_fit_pvwatts(
            time_series_clearsky,
            psm3_clearsky.ghi_clear,
            psm3_clearsky.dhi_clear,
            psm3_clearsky.dni_clear,
            solpos_clearsky.zenith,
            solpos_clearsky.azimuth,
            temperature=psm3_clearsky.temp_air,
        )
    return best_tilt, best_azimuth
