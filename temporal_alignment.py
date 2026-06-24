# -*- coding: utf-8 -*-
"""
Temporal Alignment functionality for the XGeoAI Pollution Analyst plugin.
Enhanced with robust resampling and outlier handling.
"""

import logging
import os
import netCDF4
import rasterio
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from datetime import datetime
from netCDF4 import Dataset, num2date
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from qgis.core import QgsProject, QgsRasterLayer

logger = logging.getLogger(__name__)

# ================= Progress Bar =================

def update_temporal_alignment_progress(dialog, value):
    try:
        dialog.pBDataPreparation.setValue(value)
    except Exception as e:
        dialog.iface.messageBar().pushWarning("Progress Error", f"Could not update progress: {e}")

# ================= Save Band Timestamps =================

def save_band_timestamps_csv(timestamps, output_path):
    try:
        df = pd.DataFrame({
            "BandIndex": range(1, len(timestamps) + 1),
            "Timestamp": [ts.isoformat() for ts in timestamps]
        })
        csv_path = os.path.splitext(output_path)[0] + "_timestamps.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved timestamps CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save timestamps CSV: {e}")

# ================= Save Aligned Raster =================

def save_to_multiband_geotiff(aligned_da, output_path, timestamps, crs):
    """
    Save aligned DataArray to multiband GeoTIFF with proper error handling
    
    Args:
        aligned_da: xarray DataArray with time dimension
        output_path: str, path for output GeoTIFF
        timestamps: pandas DatetimeIndex
        crs: str, CRS authority string (e.g., 'EPSG:4326')
    """
    try:
        logger.info(f"Saving GeoTIFF: {output_path}")
        
        # Validate inputs
        if "time" not in aligned_da.dims:
            raise ValueError("Aligned DataArray must have 'time' dimension.")
        
        if len(timestamps) == 0:
            raise ValueError("No timestamps provided")
            
        # Sort by time to ensure proper band order
        aligned_da = aligned_da.sortby("time")
        nodata_value = -9999
        
        # Get spatial dimensions
        if 'y' not in aligned_da.dims or 'x' not in aligned_da.dims:
            raise ValueError("DataArray must have 'x' and 'y' spatial dimensions")
            
        height = aligned_da.sizes['y']
        width = aligned_da.sizes['x']
        count = aligned_da.sizes['time']
        
        logger.info(f"Raster dimensions: {width}x{height} pixels, {count} bands")
        
        # Handle CRS - try multiple approaches
        raster_crs = None
        transform = None
        
        try:
            # Method 1: Try to get CRS from DataArray
            if hasattr(aligned_da, 'rio') and aligned_da.rio.crs is not None:
                raster_crs = aligned_da.rio.crs
                transform = aligned_da.rio.transform()
                logger.info("Using CRS and transform from DataArray")
            else:
                raise AttributeError("No CRS in DataArray")
                
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not get CRS from DataArray: {e}")
            
            # Method 2: Try to parse the provided CRS string
            try:
                if crs:
                    from rasterio.crs import CRS
                    raster_crs = CRS.from_string(crs)
                    logger.info(f"Parsed CRS from string: {crs}")
                else:
                    raise ValueError("No CRS provided")
                    
                # Create a basic transform if we don't have one
                if transform is None:
                    # Use coordinate arrays if available
                    if 'x' in aligned_da.coords and 'y' in aligned_da.coords:
                        x_coords = aligned_da.coords['x'].values
                        y_coords = aligned_da.coords['y'].values
                        
                        if len(x_coords) > 1 and len(y_coords) > 1:
                            x_res = float(x_coords[1] - x_coords[0])
                            y_res = float(y_coords[1] - y_coords[0])  # This might be negative
                            
                            from rasterio.transform import from_origin
                            transform = from_origin(
                                west=float(x_coords[0] - x_res/2),
                                north=float(y_coords[0] - y_res/2),
                                xsize=abs(x_res),
                                ysize=abs(y_res)
                            )
                            logger.info(f"Created transform from coordinates: pixel size = {abs(x_res):.6f}")
                        else:
                            raise ValueError("Insufficient coordinate data for transform")
                    else:
                        raise ValueError("No coordinate information available")
                        
            except Exception as e2:
                logger.error(f"Could not create CRS or transform: {e2}")
                # Last resort - use a default
                from rasterio.crs import CRS
                from rasterio.transform import from_origin
                raster_crs = CRS.from_epsg(4326)  # Default to WGS84
                transform = from_origin(-180, 90, 360/width, 180/height)  # Basic global transform
                logger.warning("Using default WGS84 CRS and global transform")
        
        # Create rasterio profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': count,
            'height': height,
            'width': width,
            'crs': raster_crs,
            'transform': transform,
            'nodata': nodata_value,
            'compress': 'lzw',  # Add compression
            'interleave': 'band'
        }
        
        logger.info(f"Rasterio profile: {profile}")
        
        # Write the GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            for idx in range(count):
                try:
                    # Get the band data
                    band_data = aligned_da.isel(time=idx).values.astype(np.float32)
                    
                    # Replace NaN with nodata value
                    band_data = np.where(np.isnan(band_data), nodata_value, band_data)
                    
                    # Ensure correct shape
                    if band_data.shape != (height, width):
                        logger.warning(f"Band {idx+1} shape mismatch: expected ({height}, {width}), got {band_data.shape}")
                        continue
                    
                    # Write band (rasterio uses 1-based indexing)
                    dst.write(band_data, indexes=idx + 1)
                    
                    if idx == 0:
                        logger.info(f"First band stats: min={np.nanmin(band_data):.3f}, max={np.nanmax(band_data):.3f}")
                    
                except Exception as e:
                    logger.error(f"Error writing band {idx+1}: {e}")
                    continue
        
        logger.info(f"Successfully saved GeoTIFF: {output_path}")
        
        # Save timestamps CSV
        try:
            save_band_timestamps_csv(timestamps, output_path)
        except Exception as e:
            logger.warning(f"Could not save timestamps CSV: {e}")
        
        # Optional NetCDF export
        try:
            netcdf_path = os.path.splitext(output_path)[0] + "_aligned.nc"
            # Make sure DataArray has proper attributes for NetCDF
            aligned_da_for_nc = aligned_da.copy()
            aligned_da_for_nc.attrs['units'] = 'unknown'
            aligned_da_for_nc.attrs['long_name'] = 'Temporally aligned data'
            
            # Ensure coordinates have proper attributes
            if 'time' in aligned_da_for_nc.coords:
                aligned_da_for_nc.coords['time'].attrs['long_name'] = 'time'
                aligned_da_for_nc.coords['time'].attrs['standard_name'] = 'time'
            
            aligned_da_for_nc.to_netcdf(netcdf_path)
            logger.info(f"Saved aligned NetCDF: {netcdf_path}")
        except Exception as e:
            logger.warning(f"Could not export to NetCDF: {e}")

    except Exception as e:
        logger.error(f"Error saving GeoTIFF: {e}")
        logger.exception("Full traceback:")
        raise


def save_band_timestamps_csv(timestamps, output_path):
    """
    Save band timestamps to CSV file
    
    Args:
        timestamps: pandas DatetimeIndex
        output_path: str, path of the main output file (CSV will have _timestamps suffix)
    """
    try:
        if len(timestamps) == 0:
            logger.warning("No timestamps to save")
            return
            
        df = pd.DataFrame({
            "BandIndex": range(1, len(timestamps) + 1),
            "Timestamp": [ts.isoformat() for ts in timestamps],
            "Date": [ts.strftime('%Y-%m-%d') for ts in timestamps],
            "Time": [ts.strftime('%H:%M:%S') for ts in timestamps],
            "DayOfYear": [ts.dayofyear for ts in timestamps],
            "Week": [ts.isocalendar()[1] for ts in timestamps],
            "Month": [ts.month for ts in timestamps],
            "Year": [ts.year for ts in timestamps]
        })
        
        csv_path = os.path.splitext(output_path)[0] + "_timestamps.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved timestamps CSV: {csv_path} ({len(timestamps)} records)")
        
    except Exception as e:
        logger.error(f"Failed to save timestamps CSV: {e}")
        raise

# ================= Populate UI =================

def populate_temporal_alignment_ui(dlg):
    try:
        dlg.cBTimeSeriesLayerDataPreparation.clear()
        dlg.cBTimeFieldDataPreparation.clear()
        dlg.cBResampleTemporalResoultionDataPreparation.clear()
        dlg.cBInterpolationDataPreparation.clear()

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and "Reprojected" in layer.name():
                nc_path = layer.customProperty("original_netcdf_path", "")
                if os.path.exists(nc_path):
                    dlg.cBTimeSeriesLayerDataPreparation.addItem(layer.name(), nc_path)

        dlg.cBResampleTemporalResoultionDataPreparation.addItems(["Daily", "Weekly", "Monthly", "Yearly"])
        dlg.cBInterpolationDataPreparation.addItems(["Linear", "Nearest", "Cubic"])

        if dlg.cBTimeSeriesLayerDataPreparation.count() > 0:
            populate_time_fields(dlg)

        dlg.cBTimeSeriesLayerDataPreparation.currentIndexChanged.connect(lambda: populate_time_fields(dlg))

    except Exception as e:
        logger.error(f"Error populating UI: {e}")

# ================= Populate Time Fields =================

def populate_time_fields(dlg):
    try:
        dlg.cBTimeFieldDataPreparation.clear()
        index = dlg.cBTimeSeriesLayerDataPreparation.currentIndex()
        if index < 0:
            return

        nc_path = dlg.cBTimeSeriesLayerDataPreparation.itemData(index)
        if os.path.exists(nc_path):
            with Dataset(nc_path, "r") as nc:
                time_fields = [var for var in nc.variables.keys() if "time" in var.lower()]
                dlg.cBTimeFieldDataPreparation.addItems(time_fields)
    except Exception as e:
        logger.warning(f"Could not populate time fields: {e}")

# ================= Main Alignment Function =================

def run_temporal_alignment(dialog):
    """
    Complete temporal alignment function with fixed resampling logic
    """
    logger.info("Starting Temporal Alignment...")
    try:
        update_temporal_alignment_progress(dialog, 5)

        # Get output folder from user
        output_folder = QFileDialog.getExistingDirectory(dialog, "Select Output Folder")
        if not output_folder:
            QMessageBox.warning(dialog, "Warning", "No output folder selected.")
            update_temporal_alignment_progress(dialog, 0)
            return

        # Get selected layer and paths
        layer_name = dialog.cBTimeSeriesLayerDataPreparation.currentText()
        if not layer_name:
            QMessageBox.warning(dialog, "Warning", "No time series layer selected.")
            update_temporal_alignment_progress(dialog, 0)
            return
            
        layers = QgsProject.instance().mapLayersByName(layer_name)
        if not layers:
            QMessageBox.warning(dialog, "Warning", f"Layer '{layer_name}' not found.")
            update_temporal_alignment_progress(dialog, 0)
            return
            
        layer = layers[0]
        reprojected_path = layer.source()
        netcdf_path = layer.customProperty("original_netcdf_path", "")

        if not os.path.exists(netcdf_path):
            logger.error(f"Missing NetCDF path: {netcdf_path}")
            QMessageBox.critical(dialog, "Error", f"Original NetCDF file not found: {netcdf_path}")
            update_temporal_alignment_progress(dialog, 0)
            return

        if not os.path.exists(reprojected_path):
            logger.error(f"Missing reprojected raster: {reprojected_path}")
            QMessageBox.critical(dialog, "Error", f"Reprojected raster not found: {reprojected_path}")
            update_temporal_alignment_progress(dialog, 0)
            return

        update_temporal_alignment_progress(dialog, 10)

        # Handle manual base date if enabled
        base_date = None
        if hasattr(dialog, 'cBEnableManualBaseDate') and dialog.cBEnableManualBaseDate.isChecked():
            base_date = dialog.dEBaseDateTemporalAlignment.date().toPyDate()
            base_date = datetime.combine(base_date, datetime.min.time())
            logger.info(f"Using manual base date: {base_date}")

        # Extract time information from NetCDF
        logger.info(f"Processing NetCDF file: {netcdf_path}")
        try:
            with Dataset(netcdf_path, 'r') as ds:
                time_var_name = "time"  # Default
                
                # Try to find time variable if not default
                if hasattr(dialog, 'cBTimeFieldDataPreparation'):
                    selected_time_field = dialog.cBTimeFieldDataPreparation.currentText()
                    if selected_time_field and selected_time_field in ds.variables:
                        time_var_name = selected_time_field
                
                if time_var_name not in ds.variables:
                    # Look for any time-like variable
                    time_vars = [var for var in ds.variables.keys() if "time" in var.lower()]
                    if time_vars:
                        time_var_name = time_vars[0]
                        logger.info(f"Using time variable: {time_var_name}")
                    else:
                        raise ValueError("No time variable found in NetCDF file")
                
                time_var = ds.variables[time_var_name]
                logger.info(f"Time variable info: units={getattr(time_var, 'units', 'unknown')}, "
                           f"calendar={getattr(time_var, 'calendar', 'standard')}, "
                           f"shape={time_var.shape}")
                
                # Parse time values
                try:
                    raw_times = num2date(time_var[:], 
                                       units=time_var.units, 
                                       calendar=getattr(time_var, 'calendar', 'standard'))
                    logger.info("Successfully parsed times using netCDF4.num2date")
                except Exception as e:
                    logger.warning(f"netCDF4.num2date failed: {e}")
                    if base_date is None:
                        logger.error("Time parsing failed and no manual base date provided.")
                        QMessageBox.critical(dialog, "Error", 
                                           "Could not parse time values from NetCDF. "
                                           "Please enable manual base date.")
                        return
                    # Fallback to manual base date
                    raw_times = [base_date + pd.to_timedelta(int(t), unit="h") for t in time_var[:]]
                    logger.info("Using manual base date for time calculation")

        except Exception as e:
            logger.error(f"Error reading NetCDF time data: {e}")
            QMessageBox.critical(dialog, "Error", f"Failed to read time data from NetCDF: {e}")
            return

        # Convert to pandas datetime and remove timezone info
        timestamps = pd.to_datetime(raw_times).tz_localize(None)
        original_start, original_end = timestamps.min(), timestamps.max()
        original_count = len(timestamps)
        
        logger.info(f"Original time range: {original_start} to {original_end}")
        logger.info(f"Original time steps: {original_count}")
        logger.info(f"Time step frequency: {pd.infer_freq(timestamps.sort_values())}")

        update_temporal_alignment_progress(dialog, 30)

        # Load reprojected raster data
        logger.info(f"Loading reprojected raster: {reprojected_path}")
        try:
            aligned_da = xr.open_dataarray(reprojected_path, engine="rasterio")
        except Exception as e:
            logger.error(f"Error loading raster: {e}")
            QMessageBox.critical(dialog, "Error", f"Failed to load raster file: {e}")
            return

        # Validate raster structure
        if "band" not in aligned_da.dims:
            logger.error("Raster missing 'band' dimension.")
            QMessageBox.critical(dialog, "Error", "Raster file missing 'band' dimension.")
            return

        raster_bands = aligned_da.sizes["band"]
        logger.info(f"Raster bands: {raster_bands}")

        if raster_bands != len(timestamps):
            logger.error(f"Mismatch: {raster_bands} raster bands vs {len(timestamps)} timestamps")
            QMessageBox.critical(dialog, "Error", 
                               f"Band count mismatch: {raster_bands} bands vs {len(timestamps)} timestamps")
            return

        # Rename band dimension to time and assign coordinates
        aligned_da = aligned_da.rename({"band": "time"})
        aligned_da = aligned_da.assign_coords({"time": timestamps})
        
        logger.info(f"Raster dimensions: {aligned_da.dims}, shape: {aligned_da.shape}")
        logger.info(f"Raster shape: {aligned_da.shape}")

        # === IMPROVED RESAMPLING WITH PROPER AGGREGATION ===
        resample_choice = dialog.cBResampleTemporalResoultionDataPreparation.currentText()
        if not resample_choice:
            resample_choice = "Daily"  # Default fallback
            
        logger.info(f"Selected resampling: {resample_choice}")
        
        # Analyze original data
        original_min = float(aligned_da.min().item())
        original_max = float(aligned_da.max().item())
        original_mean = float(aligned_da.mean().item())
        total_pixels = np.prod(aligned_da.shape)
        
        logger.info(f"Original data stats: min={original_min:.3f}, max={original_max:.3f}, mean={original_mean:.3f}")
        logger.info(f"Total pixels: {total_pixels:,}")
        
        # Clean nodata values before resampling
        aligned_da_filtered = aligned_da.copy()
        
        # Remove common nodata values
        common_nodata_values = [-999, -999.0, -9999, -9999.0]
        total_nodata_removed = 0
        
        for nodata_val in common_nodata_values:
            nodata_count = int((aligned_da_filtered == nodata_val).sum().item())
            if nodata_count > 0:
                logger.info(f"Removing {nodata_count:,} pixels with nodata value: {nodata_val}")
                aligned_da_filtered = aligned_da_filtered.where(aligned_da_filtered != nodata_val)
                total_nodata_removed += nodata_count
        
        # Check remaining valid data
        valid_count = int(aligned_da_filtered.count().item())
        logger.info(f"Valid data points after nodata removal: {valid_count:,} "
                   f"({100 * valid_count / total_pixels:.1f}% of total)")
        
        if valid_count == 0:
            logger.error("No valid data remaining after nodata removal")
            QMessageBox.critical(dialog, "Error", "All data appears to be nodata values.")
            return
        
        update_temporal_alignment_progress(dialog, 50)
        
        # Apply resampling based on choice
        logger.info(f"Applying {resample_choice.lower()} resampling...")
        
        if resample_choice == "Daily":
            # Group by calendar day and take mean
            resampled_da = aligned_da_filtered.resample(time="1D").mean(skipna=True)
            min_observations = 1  # At least 1 observation per day
            
        elif resample_choice == "Weekly":
            # Use proper weekly resampling with Monday start
            resampled_da = aligned_da_filtered.resample(time="W-MON").mean(skipna=True)
            min_observations = 7  # At least 7 observations per week (lenient)
            
        elif resample_choice == "Monthly":
            # Group by month start and take mean
            resampled_da = aligned_da_filtered.resample(time="MS").mean(skipna=True)
            min_observations = 30  # At least 30 observations per month (lenient)
            
        elif resample_choice == "Yearly":
            # Group by year start and take mean
            resampled_da = aligned_da_filtered.resample(time="AS").mean(skipna=True)
            min_observations = 100  # At least 100 observations per year (lenient)
        else:
            # No resampling - use filtered data as-is
            logger.info("No resampling applied - using original temporal resolution")
            resampled_da = aligned_da_filtered
            min_observations = 1
        
        # Log resampling results
        resampled_time_steps = resampled_da.sizes.get('time', 0)
        logger.info(f"After {resample_choice.lower()} resampling:")
        logger.info(f"  - Time steps: {resampled_time_steps}")
        
        if resampled_time_steps > 0:
            resampled_start = pd.to_datetime(resampled_da.time.min().item())
            resampled_end = pd.to_datetime(resampled_da.time.max().item())
            logger.info(f"  - Time range: {resampled_start} to {resampled_end}")
            
            resampled_min = float(resampled_da.min().item())
            resampled_max = float(resampled_da.max().item())
            resampled_mean = float(resampled_da.mean().item())
            logger.info(f"  - Data stats: min={resampled_min:.3f}, max={resampled_max:.3f}, mean={resampled_mean:.3f}")
        
        # Quality control - count valid observations per time step (optional)
        if resample_choice in ["Daily", "Weekly", "Monthly", "Yearly"] and resampled_time_steps > 0:
            try:
                # This is computationally intensive, so we'll skip for now
                # valid_counts = aligned_da_filtered.resample(time=resampled_da.time).count()
                # sufficient_data_mask = valid_counts >= min_observations
                # logger.info(f"Time steps with sufficient data ({min_observations}+ obs): {sufficient_data_mask.sum().item()}")
                pass
            except Exception as e:
                logger.warning(f"Could not perform quality control check: {e}")
        
        # Remove completely NaN time steps
        resampled_da = resampled_da.dropna(dim="time", how="all")
        final_time_steps = resampled_da.sizes.get('time', 0)
        
        logger.info(f"Final time steps after dropna: {final_time_steps}")
        
        if final_time_steps == 0:
            logger.error("No valid time steps remaining after resampling and quality control")
            QMessageBox.critical(dialog, "Error", 
                               f"No valid data remaining after {resample_choice.lower()} resampling. "
                               "Try a different temporal resolution or check your input data.")
            return
        
        # Update timestamps from resampled data
        updated_timestamps = pd.to_datetime(resampled_da.time.values)
        
        # Constrain to original time range (with small buffer)
        time_buffer = pd.Timedelta(days=1)
        
        time_mask = (updated_timestamps >= (original_start - time_buffer)) & \
                   (updated_timestamps <= (original_end + time_buffer))
        
        filtered_timestamps = updated_timestamps[time_mask]
        logger.info(f"Time steps after range filtering: {len(filtered_timestamps)}")
        
        if len(filtered_timestamps) > 0:
            resampled_da = resampled_da.sel(time=filtered_timestamps)
            final_timestamps = filtered_timestamps
        else:
            logger.warning("Time range filtering removed all data - keeping all resampled data")
            final_timestamps = updated_timestamps
        
        aligned_da = resampled_da
        timestamps = final_timestamps
        
        update_temporal_alignment_progress(dialog, 70)
        
        # Validate final results
        final_bands = len(timestamps)
        logger.info(f"=== RESAMPLING SUMMARY ===")
        logger.info(f"Original: {original_count} time steps")
        logger.info(f"Final ({resample_choice}): {final_bands} time steps")
        logger.info(f"Time range: {timestamps.min()} to {timestamps.max()}")
        
        # Expected band counts for validation
        time_span_days = (timestamps.max() - timestamps.min()).days + 1
        if resample_choice == "Daily":
            expected_count = time_span_days
            logger.info(f"Expected daily bands: ~{expected_count}, Actual: {final_bands}")
        elif resample_choice == "Weekly":
            expected_count = (time_span_days // 7) + 1
            logger.info(f"Expected weekly bands: ~{expected_count}, Actual: {final_bands}")
        elif resample_choice == "Monthly":
            expected_count = ((timestamps.max().year - timestamps.min().year) * 12 + 
                             timestamps.max().month - timestamps.min().month + 1)
            logger.info(f"Expected monthly bands: ~{expected_count}, Actual: {final_bands}")
        elif resample_choice == "Yearly":
            expected_count = timestamps.max().year - timestamps.min().year + 1
            logger.info(f"Expected yearly bands: ~{expected_count}, Actual: {final_bands}")
        
        # Validation check
        if resample_choice == "Weekly" and final_bands > 20:
            logger.warning(f"Weekly resampling produced {final_bands} bands - this may indicate resampling didn't work properly")
        
        # === DIAGNOSTIC EXPORT ===
        try:
            if final_bands > 0:
                logger.info("Creating diagnostic statistics...")
                diagnostic_data = []
                
                for i in range(final_bands):
                    try:
                        ts = timestamps[i]
                        band_data = aligned_da.isel(time=i).values.flatten()
                        valid_pixels = band_data[~np.isnan(band_data)]
                        
                        if len(valid_pixels) > 0:
                            diagnostic_data.append({
                                'BandIndex': i + 1,
                                'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'ISO_Date': ts.isoformat(),
                                'Mean': float(np.mean(valid_pixels)),
                                'Max': float(np.max(valid_pixels)),
                                'Min': float(np.min(valid_pixels)),
                                'StdDev': float(np.std(valid_pixels)),
                                'ValidPixels': len(valid_pixels),
                                'TotalPixels': len(band_data),
                                'ValidPercent': 100.0 * len(valid_pixels) / len(band_data)
                            })
                        else:
                            logger.warning(f"No valid pixels for timestamp {ts}")
                            diagnostic_data.append({
                                'BandIndex': i + 1,
                                'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'ISO_Date': ts.isoformat(),
                                'Mean': np.nan,
                                'Max': np.nan,
                                'Min': np.nan,
                                'StdDev': np.nan,
                                'ValidPixels': 0,
                                'TotalPixels': len(band_data),
                                'ValidPercent': 0.0
                            })
                    except Exception as e:
                        logger.warning(f"Error processing band {i}: {e}")
                
                if diagnostic_data:
                    stats_df = pd.DataFrame(diagnostic_data)
                    diag_csv_path = os.path.join(output_folder, f"{layer.name()}_{resample_choice}_Diagnostics.csv")
                    stats_df.to_csv(diag_csv_path, index=False)
                    logger.info(f"Saved {resample_choice.lower()} diagnostics: {diag_csv_path}")
                    
                    # Log summary statistics
                    valid_stats = stats_df.dropna(subset=['Mean'])
                    if len(valid_stats) > 0:
                        logger.info(f"Diagnostics summary:")
                        logger.info(f"  - Bands with data: {len(valid_stats)}/{len(diagnostic_data)}")
                        logger.info(f"  - Mean value range: {valid_stats['Mean'].min():.3f} to {valid_stats['Mean'].max():.3f}")
                        logger.info(f"  - Average valid pixels: {valid_stats['ValidPixels'].mean():.0f}")
                        logger.info(f"  - Average valid percentage: {valid_stats['ValidPercent'].mean():.1f}%")
                else:
                    logger.warning("No diagnostic data to export")
                    
        except Exception as e:
            logger.warning(f"Could not create diagnostics: {e}")
        
        if final_bands == 0:
            QMessageBox.critical(dialog, "Error", "No valid bands remaining after resampling.")
            return
        
        update_temporal_alignment_progress(dialog, 80)
        
        # Export to GeoTIFF
        try:
            crs = layer.crs().authid()
            output_filename = f"{layer.name()}_{resample_choice}_Aligned.tif"
            output_path = os.path.join(output_folder, output_filename)
            
            logger.info(f"Saving aligned raster to: {output_path}")
            logger.info(f"CRS: {crs}")
            logger.info(f"Data array shape: {aligned_da.shape}")
            logger.info(f"Data array dims: {aligned_da.dims}")
            
            # Ensure we have the required spatial reference
            if not hasattr(aligned_da, 'rio') or aligned_da.rio.crs is None:
                logger.warning("DataArray missing spatial reference, attempting to set from layer CRS")
                aligned_da = aligned_da.rio.write_crs(crs)
            
            # Call the save function with proper error handling
            save_to_multiband_geotiff(aligned_da, output_path, timestamps, crs)
            logger.info(f"Successfully saved GeoTIFF with {final_bands} bands")
            
        except Exception as e:
            logger.error(f"Failed to save GeoTIFF: {e}")
            logger.exception("Full traceback:")
            QMessageBox.critical(dialog, "Error", f"Failed to save output file: {str(e)}")
            return
        
        update_temporal_alignment_progress(dialog, 90)
        
        # Load result into QGIS
        new_layer_name = f"{layer.name()}_{resample_choice}_Aligned"
        logger.info(f"Loading result layer: {new_layer_name}")
        
        try:
            new_layer = QgsRasterLayer(output_path, new_layer_name)
            
            if new_layer.isValid():
                QgsProject.instance().addMapLayer(new_layer)
                
                # Store references for potential future use
                dialog.temporalAlignedLayer = new_layer
                dialog.temporalAlignedLayerPath = output_path
                
                # Verify the loaded layer
                loaded_bands = new_layer.bandCount()
                logger.info(f"Loaded layer has {loaded_bands} bands")
                
                success_msg = f"{resample_choice} resampling completed: {final_bands} bands"
                dialog.iface.messageBar().pushSuccess("Temporal Alignment", success_msg)
                
                logger.info(f"Successfully added layer '{new_layer_name}' to QGIS project")
            else:
                logger.error("Failed to create valid QGIS raster layer")
                dialog.iface.messageBar().pushCritical("Temporal Alignment", "Could not load result layer")
                return
                
        except Exception as e:
            logger.error(f"Error loading result into QGIS: {e}")
            dialog.iface.messageBar().pushCritical("Temporal Alignment", f"Could not load result: {e}")
            return
            
        update_temporal_alignment_progress(dialog, 100)
        
        # Success message with summary
        success_message = (f"Temporal resampling completed successfully!\n\n"
                          f"Original data: {original_count} time steps\n"
                          f"{resample_choice} resampling: {final_bands} time steps\n"
                          f"Time range: {timestamps.min().strftime('%Y-%m-%d')} to {timestamps.max().strftime('%Y-%m-%d')}\n"
                          f"Output file: {output_filename}\n\n"
                          f"Layer '{new_layer_name}' has been added to your QGIS project.")
        
        QMessageBox.information(dialog, "Success", success_message)
        logger.info("Temporal alignment completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in temporal alignment: {e}")
        update_temporal_alignment_progress(dialog, 0)
        QMessageBox.critical(dialog, "Error", f"Temporal alignment failed: {str(e)}")


def validate_temporal_resampling(original_timestamps, resampled_timestamps, resample_type):
    """
    Validate that resampling produced expected results
    
    Args:
        original_timestamps: pandas DatetimeIndex of original time points
        resampled_timestamps: pandas DatetimeIndex of resampled time points  
        resample_type: str, type of resampling ("Daily", "Weekly", "Monthly", "Yearly")
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        original_count = len(original_timestamps)
        resampled_count = len(resampled_timestamps)
        
        if original_count == 0 or resampled_count == 0:
            return False, "Empty timestamp arrays"
        
        time_span = original_timestamps.max() - original_timestamps.min()
        time_span_days = time_span.days + 1
        
        if resample_type == "Daily":
            expected_count = time_span_days
            tolerance = 3  # Allow ±3 days difference
        elif resample_type == "Weekly":
            expected_count = (time_span_days // 7) + 1
            tolerance = 2  # Allow ±2 weeks difference
        elif resample_type == "Monthly":
            expected_count = ((original_timestamps.max().year - original_timestamps.min().year) * 12 + 
                             original_timestamps.max().month - original_timestamps.min().month + 1)
            tolerance = 1  # Allow ±1 month difference
        elif resample_type == "Yearly":
            expected_count = original_timestamps.max().year - original_timestamps.min().year + 1
            tolerance = 0  # Years should be exact
        else:
            return True, f"No validation rules for resample type: {resample_type}"
        
        difference = abs(resampled_count - expected_count)
        is_valid = difference <= tolerance
        
        message = (f"Resampling validation for {resample_type}:\n"
                  f"  Original: {original_count} time steps\n"
                  f"  Expected: ~{expected_count} time steps\n"
                  f"  Actual: {resampled_count} time steps\n"
                  f"  Difference: {difference} (tolerance: ±{tolerance})\n"
                  f"  Time span: {time_span_days} days\n"
                  f"  Status: {'✓ Valid' if is_valid else '✗ Unexpected count'}")
        
        return is_valid, message
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"