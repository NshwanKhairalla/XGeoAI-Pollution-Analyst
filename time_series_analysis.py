
# time_series_analysis.py
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import base
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore, pearsonr, spearmanr, kendalltau, entropy
from matplotlib import pyplot as plt
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer, QgsFields, QgsField, QgsFeature, QgsGeometry,
    QgsPointXY, QgsVectorFileWriter, QgsCoordinateReferenceSystem
)
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem, QButtonGroup
from PyQt5.QtCore import Qt, QUrl, QVariant
import urllib.parse
import re


class MessageBoxHandler:
    @staticmethod
    def information(parent, message):
        QMessageBox.information(parent, "Info", message)

    @staticmethod
    def critical(parent, message):
        QMessageBox.critical(parent, "Error", message)

class TimeSeriesAnalysis:
    def __init__(self, dialog):
        self.dialog = dialog
        self.dataframe = None
        self.loaded_columns = []
        self.crs = None
        self.output_folder = ""
        self.setup_ui()

    def log(self, message):
        logging.info(message)
        self.dialog.tELog.append(f"INFO: {message}")

    def warn(self, message):
        logging.warning(message)
        self.dialog.tELog.append(f"WARNING: {message}")

    def populate_dataset_combo(self):
        self.dialog.cBDatasetTimeSeries.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsVectorLayer.VectorLayer and layer.dataProvider().name() == 'delimitedtext':
                self.dialog.cBDatasetTimeSeries.addItem(layer.name())

        self.log("Refreshed CSV dataset list in combo box.")

    def populate_temporal_scope_options(self):
        """Populate lWTemporalScope with months January..December (checkable)."""
        self.dialog.lWTemporalScope.clear()
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        for m in months:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.dialog.lWTemporalScope.addItem(item)
        self.log("Populated temporal scope options with months Jan–Dec.")

    def refresh_layers_timeseries(self):
        """Refresh cBDatasetTimeSeries from the QGIS Layers Browser (CSV layers)."""
        try:
            # Remember the user's current selection
            prev_name = self.dialog.cBDatasetTimeSeries.currentText()

            # Collect names of CSV (delimitedtext) vector layers currently in the project
            layer_names = []
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsVectorLayer) and layer.providerType() == "delimitedtext":
                    layer_names.append(layer.name())
            layer_names.sort()

            # Refill the combo (avoid spurious signals during refill)
            self.dialog.cBDatasetTimeSeries.blockSignals(True)
            self.dialog.cBDatasetTimeSeries.clear()
            self.dialog.cBDatasetTimeSeries.addItems(layer_names)
            self.dialog.cBDatasetTimeSeries.blockSignals(False)

            # Restore selection if still present
            if prev_name and prev_name in layer_names:
                idx = self.dialog.cBDatasetTimeSeries.findText(prev_name, Qt.MatchExactly)
                if idx >= 0:
                    self.dialog.cBDatasetTimeSeries.setCurrentIndex(idx)
                    # currentIndexChanged will trigger populate_columns() via setup_ui
            elif layer_names:
                # Select first item to keep the UI consistent
                self.dialog.cBDatasetTimeSeries.setCurrentIndex(0)

            count = len(layer_names)
            MessageBoxHandler.information(self.dialog, f"Datasets refreshed: found {count} CSV layer(s).")
            self.log(f"Refreshed time-series datasets: {count} CSV layer(s) found.")
        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"Failed to refresh datasets: {e}")
            self.log(f"ERROR refreshing time-series datasets: {e}")


    def go_to_next_tab(self):
        index = self.dialog.tabsXgeoAi.indexOf(self.dialog.tSpatialMetricsAggregation)
        if index != -1:
            self.dialog.tabsXgeoAi.setCurrentIndex(index)
            MessageBoxHandler.information(self.dialog, "Switched to Spatial Metrics Aggregation tab.")
            self.log("Switched to Spatial Metrics Aggregation tab.")


    def detect_coordinate_columns(self, df, prefer_uppercase=True):
        """
        Detect X and Y coordinate columns with preference for uppercase.

        Args:
            df: pandas DataFrame
            prefer_uppercase: bool, if True prioritizes X,Y over x,y

        Returns:
            tuple: (x_col, y_col) or (None, None) if not found
        """
        # Define possible column names in order of preference
        if prefer_uppercase:
            x_candidates = ['X', 'XCOORD', 'LONGITUDE', 'LON', 'x', 'xcoord', 'longitude', 'lon']
            y_candidates = ['Y', 'YCOORD', 'LATITUDE', 'LAT', 'y', 'ycoord', 'latitude', 'lat']
        else:
            x_candidates = ['x', 'xcoord', 'longitude', 'lon', 'X', 'XCOORD', 'LONGITUDE', 'LON']
            y_candidates = ['y', 'ycoord', 'latitude', 'lat', 'Y', 'YCOORD', 'LATITUDE', 'LAT']

        # Find first matching column for X
        x_col = None
        for candidate in x_candidates:
            if candidate in df.columns:
                x_col = candidate
                break

        # Find first matching column for Y
        y_col = None
        for candidate in y_candidates:
            if candidate in df.columns:
                y_col = candidate
                break

        return x_col, y_col

    def setup_ui(self):
        self.populate_dataset_combo()

        self.dialog.cBDatasetTimeSeries.currentIndexChanged.connect(self.populate_columns)
        self.dialog.pBLoadDataTimeSeries.clicked.connect(self.load_selected_columns)
        self.dialog.pBExportPlotTimeSeries.clicked.connect(self.export_plot)
        self.dialog.pBAggregateTimeSeries.clicked.connect(self.aggregate_data)
        self.dialog.pBDecomposeTimeSeries.clicked.connect(self.decompose_loess)
        self.dialog.pBRunCorrelationTimeSeries.clicked.connect(self.run_correlation)
        self.dialog.tBChooseExportFolderTimeSeries.clicked.connect(self.choose_export_folder)
        self.dialog.pBExportTimeSeriesResults.clicked.connect(self.export_time_series_data)
        self.dialog.pBComputeSpatialTargets.clicked.connect(self.compute_spatial_targets)
        self.dialog.pBNextTimeSeries.clicked.connect(self.go_to_next_tab)
        self.populate_aggregation_controls()
        self.populate_correlation_methods()
        self.populate_temporal_scope_options()
        self.dialog.cBCorrelationMethodTimeSeries.clear()
        self.dialog.cBCorrelationMethodTimeSeries.addItems(["Pearson", "Spearman", "Kendall"])
        # Create button groups
        self.plot_type_group = QButtonGroup()
        self.plot_type_group.addButton(self.dialog.rBLineChart)
        self.plot_type_group.addButton(self.dialog.rBBoxPlot)

        self.data_type_group = QButtonGroup()
        self.data_type_group.addButton(self.dialog.rBRawData)
        self.data_type_group.addButton(self.dialog.rBSeasonalTrends)
        self.data_type_group.addButton(self.dialog.rBAnomalies)

        self.dialog.pBRefreshLayersTimeseries.clicked.connect(self.refresh_layers_timeseries)


    def populate_columns(self):
        layer_name = self.dialog.cBDatasetTimeSeries.currentText()
        if not layer_name:
            return

        layer = QgsProject.instance().mapLayersByName(layer_name)[0]
        path = layer.dataProvider().dataSourceUri().split('?')[0]
        df = pd.read_csv(path)
        self.crs = layer.crs()
        self.dialog.lWValueFieldSelectorTimeSeries.clear()
        self.dialog.lWTimeSeriesTargets.clear()
        self.dialog.cBTimestampMetricsCalculationsTimeSeries.clear()
        self.dialog.cBPollutantMetricsCalculationsTimeSeries.clear()
        self.dialog.lWLandUseFractionColumn.clear()

        for col in df.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.dialog.lWValueFieldSelectorTimeSeries.addItem(item)
            self.dialog.cBTimestampMetricsCalculationsTimeSeries.addItem(col)
            self.dialog.cBPollutantMetricsCalculationsTimeSeries.addItem(col)
            self.dialog.cBTimestampMetricsCalculationsTimeSeries.addItem(col)

        self.dataframe = df
        self.populate_time_column_combobox()
        numeric_columns = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        self.dialog.cBPollutantTimeSeriesColumn.clear()
        self.dialog.lWLandUseFractionColumn.clear()
        self.dialog.cBPollutantTimeSeriesColumn.addItems(numeric_columns)
        self.dialog.lWLandUseFractionColumn.addItems(numeric_columns)
        MessageBoxHandler.information(self.dialog, f"Loaded columns from {layer_name}")
        self.log(f"Loaded columns from {layer_name}")

    def populate_time_column_combobox(self):
        if self.dataframe is not None:
            self.dialog.cBTimeColumnSelection.clear()
            datetime_columns = [
                col for col in self.dataframe.columns
                if pd.api.types.is_datetime64_any_dtype(self.dataframe[col])
                or any(hint in col.lower() for hint in ['time', 'date', 'timestamp'])
            ]
            self.dialog.cBTimeColumnSelection.addItems(datetime_columns)
            self.log(f"Populated timestamp selector with: {datetime_columns}")

    def load_selected_columns(self):
        self.loaded_columns = []
        for i in range(self.dialog.lWValueFieldSelectorTimeSeries.count()):
            item = self.dialog.lWValueFieldSelectorTimeSeries.item(i)
            if item.checkState() == Qt.Checked:
                self.loaded_columns.append(item.text())
                self.dialog.lWTimeSeriesTargets.addItem(item.text())
        MessageBoxHandler.information(self.dialog, f"Loaded selected columns: {self.loaded_columns}")
        self.log(f"Loaded selected columns: {self.loaded_columns}")

    def export_plot(self):
        time_col = self.dialog.cBTimeColumnSelection.currentText()
        target_items = self.dialog.lWTimeSeriesTargets.selectedItems()
        if not target_items:
            QMessageBox.critical(self.dialog, 'Error', 'Please select at least one target column from the list before exporting the plot.')
            return

        # Determine plot style
        plot_style = None
        if self.dialog.rBBoxPlot.isChecked():
            plot_style = 'box'
        elif self.dialog.rBLineChart.isChecked():
            plot_style = 'line'
        if plot_style is None:
            QMessageBox.critical(self.dialog, 'Error', 'Please select a plot type (Line Chart or Box Plot).')
            return

        # Determine data type
        data_type = None
        if self.dialog.rBSeasonalTrends.isChecked():
            data_type = 'seasonal'
        elif self.dialog.rBAnomalies.isChecked():
            data_type = 'anomaly'
        elif self.dialog.rBRawData.isChecked():
            data_type = 'raw'
        if data_type is None:
            QMessageBox.critical(self.dialog, 'Error', 'Please select a data type (Raw Data, Seasonal Trends, or Anomalies).')
            return

        # Always prompt for an export folder
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Export Folder for Plots")
        if folder:
            self.output_folder = folder
            self.log(f"Output folder selected for plots: {folder}")
        elif not self.output_folder:
            QMessageBox.critical(self.dialog, "Error", "No output folder selected.")
            self.log("ERROR: Plot export aborted due to no output folder selection.")
            return

        # Helper: build a continuous, smoothed time series (duplicate-safe)
        def _continuous_and_smooth(series):
            import numpy as np
            import pandas as pd

            s = pd.to_numeric(series, errors='coerce').dropna().copy()
            if s.empty:
                return s

            # Ensure DatetimeIndex (tz-naive)
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, errors='coerce')
            s = s[~s.index.isna()]
            try:
                if s.index.tz is not None:
                    s.index = s.index.tz_convert(None)
            except Exception:
                try:
                    s.index = s.index.tz_localize(None)
                except Exception:
                    pass

            # Sort and collapse duplicates by mean
            s = s.sort_index()
            if s.index.has_duplicates:
                s = s.groupby(level=0).mean().sort_index()

            # Infer typical step and reindex to continuous range
            step_ns = None
            if len(s.index) >= 2:
                diffs = (s.index.asi8[1:] - s.index.asi8[:-1])
                diffs = diffs[diffs > 0]
                if diffs.size:
                    step_ns = int(np.median(diffs))
            if step_ns and step_ns > 0:
                try:
                    freq = pd.to_timedelta(step_ns, unit='ns')
                    full_index = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq)
                    s = s.reindex(full_index)
                except Exception:
                    pass

            # Interpolate and smooth
            try:
                s = s.interpolate(method='time').ffill().bfill()
            except Exception:
                s = s.ffill().bfill()

            win = max(5, int(len(s) * 0.03))
            if win % 2 == 0:
                win += 1
            try:
                s = s.rolling(window=win, center=True, min_periods=1).mean()
            except Exception:
                pass
            return s

        # Do the plotting
        self.dialog.pBTimeSeriesAnalysis.setValue(10)
        success_count = 0
        fail_count = 0

        for idx_item, target_item in enumerate(target_items):
            target_col = target_item.text()
            try:
                df = self.dataframe[[time_col, target_col]].copy()
                # Parse timestamps robustly
                df[time_col] = self.parse_timestamps(df[time_col])
                df = df.dropna(subset=[time_col])
                df.sort_values(by=time_col, inplace=True)
                df.set_index(time_col, inplace=True)

                # Choose data slice
                if data_type == 'raw':
                    data_to_plot = df[[target_col]]
                elif data_type == 'seasonal':
                    data_to_plot = df[[target_col]].resample('M').mean()
                elif data_type == 'anomaly':
                    from scipy.stats import zscore
                    z = zscore(df[target_col].dropna())
                    df['__anomaly__'] = (np.abs(z) > 2).astype(int)
                    data_to_plot = df[df['__anomaly__'] == 1][[target_col]]
                else:
                    data_to_plot = df[[target_col]]

                # Ensure clean DatetimeIndex
                data_to_plot = data_to_plot.copy()
                data_to_plot.index = pd.to_datetime(data_to_plot.index, errors='coerce')
                data_to_plot = data_to_plot[~data_to_plot.index.isna()]

                # Plot
                plt.figure(figsize=(12, 6))
                ax = plt.gca()

                if plot_style == 'line':
                    series = data_to_plot[target_col]
                    if series.dropna().shape[0] < 2:
                        plt.text(0.5, 0.5, 'Not enough data to plot', ha='center', va='center', transform=ax.transAxes)
                    else:
                        series = _continuous_and_smooth(series)
                        if series is not None and len(series) >= 2:
                            ax.plot(series.index, series.values, linestyle='solid')
                        else:
                            plt.text(0.5, 0.5, 'Not enough data after smoothing', ha='center', va='center', transform=ax.transAxes)
                    plt.title(f"{target_col} - Line Chart ({data_type.title()} Data)")
                    plt.xlabel('Time')
                    plt.ylabel(target_col)

                    # Date axis formatting
                    import matplotlib.dates as mdates
                    try:
                        locator = mdates.AutoDateLocator()
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(formatter)
                    except Exception:
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=30, ha='right')

                elif plot_style == 'box':
                    y = None
                    title_suffix = ''
                    if data_type == 'raw':
                        monthly = data_to_plot.resample('M').mean()
                        y = monthly[target_col].dropna()
                        title_suffix = 'Raw Monthly Data'
                    elif data_type == 'seasonal':
                        y = data_to_plot[target_col].dropna()
                        title_suffix = 'Seasonal Data'
                    elif data_type == 'anomaly':
                        y = data_to_plot[target_col].dropna()
                        title_suffix = 'Anomalies' if not y.empty else 'No Anomalies'
                    if y is None or y.empty:
                        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    else:
                        plt.boxplot(y, labels=['Values'])
                    plt.title(f"{target_col} - Box Plot ({title_suffix})")
                    plt.ylabel(target_col)

                plt.tight_layout()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_target = re.sub(r'[^A-Za-z0-9_]+', '_', target_col)
                out_path = os.path.join(self.output_folder, f"plot_{safe_target}_{data_type}_{plot_style}_{timestamp}.png")
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.log(f"Exported plot for {target_col}: {out_path}")
                success_count += 1

            except Exception as e:
                self.log(f"ERROR exporting plot for {target_col}: {e}")
                fail_count += 1

            # Progress
            self.dialog.pBTimeSeriesAnalysis.setValue(10 + int(80 * (idx_item + 1) / max(1, len(target_items))))

        # Final status
        if fail_count == 0 and success_count > 0:
            MessageBoxHandler.information(self.dialog, "All plots exported successfully.")
        elif success_count > 0 and fail_count > 0:
            MessageBoxHandler.information(self.dialog, f"Export completed with warnings. Success: {success_count}, Failed: {fail_count}")
        else:
            MessageBoxHandler.critical(self.dialog, "Plot export failed for all selected targets.")
        self.dialog.pBTimeSeriesAnalysis.setValue(100)
        self.log(f"Plot export summary → success: {success_count}, failed: {fail_count}")
        def _continuous_and_smooth(series):
            import numpy as np
            import pandas as pd
            # Drop NaNs and ensure DatetimeIndex
            s = pd.to_numeric(series, errors='coerce').dropna().copy()
            if s.empty:
                return series

            # Ensure datetime index
            if not isinstance(s.index, pd.DatetimeIndex):
                try:
                    s.index = pd.to_datetime(s.index, errors='coerce')
                except Exception:
                    return series
            # Remove timezone for safety
            try:
                if s.index.tz is not None:
                    s.index = s.index.tz_convert(None)
            except Exception:
                try:
                    s.index = s.index.tz_localize(None)
                except Exception:
                    pass

            # Sort and collapse duplicate timestamps by mean to make index unique
            s = s.sort_index()
            if s.index.has_duplicates:
                s = s.groupby(level=0).mean()
                s = s.sort_index()

            # Estimate typical step; guard against length<2
            step_ns = None
            if len(s.index) >= 2:
                # Use .asi8 to get int64 ns
                diffs = np.diff(s.index.asi8)
                diffs = diffs[diffs > 0]
                if diffs.size:
                    step_ns = int(np.median(diffs))

            # If we can infer a reasonable step, reindex to a continuous range
            if step_ns is not None and step_ns > 0:
                try:
                    freq = pd.to_timedelta(step_ns, unit='ns')
                    full_index = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq)
                    s = s.reindex(full_index)
                except Exception:
                    # If reindex fails (e.g., still dupes somehow), fall back to current s
                    pass

            # Interpolate missing timestamps linearly in time
            try:
                s = s.interpolate(method='time').ffill().bfill()
            except Exception:
                s = s.ffill().bfill()

            # Smooth with a centered rolling mean; window ~3% of length, min 5 and odd
            win = max(5, int(len(s) * 0.03))
            if win % 2 == 0:
                win += 1
            try:
                s = s.rolling(window=win, center=True, min_periods=1).mean()
            except Exception:
                pass
            return series  # nothing to do

            s = s.sort_index()
            idx = s.index

            # Build a continuous index by estimating the typical step
            step_ns = None
            if len(idx) >= 2:
                diffs = idx.view('int64')  # nanoseconds
                diffs = np.diff(diffs)
                diffs = diffs[diffs > 0]
                if diffs.size:
                    step_ns = int(np.median(diffs))

            if step_ns is not None and step_ns > 0:
                freq = pd.to_timedelta(step_ns, unit='ns')
                full_index = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
                s = s.reindex(full_index)

            # Interpolate missing timestamps linearly in time
            s = s.interpolate(method='time').ffill().bfill()

            # Smooth with a centered rolling mean; window ~3% of length, min 5 and odd
            win = max(5, int(len(s) * 0.03))
            if win % 2 == 0:
                win += 1
            s = s.rolling(window=win, center=True, min_periods=1).mean()
            return s

        
            self.dialog.pBTimeSeriesAnalysis.setValue(10)
            success_count = 0
            fail_count = 0
            for idx_item, target_item in enumerate(target_items):
                target_col = target_item.text()
                try:
                    df = self.dataframe[[time_col, target_col]].copy()
                    # Robust timestamp parsing for the selected visualization time column
                    df[time_col] = self.parse_timestamps(df[time_col])
                    df = df.dropna(subset=[time_col])
                    df.sort_values(by=time_col, inplace=True)
                    df.set_index(time_col, inplace=True)

                    if data_type == 'raw':
                        data_to_plot = df[[target_col]]
                    elif data_type == 'seasonal':
                        # Monthly mean as a basic seasonal view
                        data_to_plot = df[[target_col]].resample('M').mean()
                    elif data_type == 'anomaly':
                        from scipy.stats import zscore
                        z = zscore(df[target_col].dropna())
                        df['anomaly'] = (np.abs(z) > 2).astype(int)
                        data_to_plot = df[df['anomaly'] == 1][[target_col]]
                    else:
                        data_to_plot = df[[target_col]]

                    # Ensure DatetimeIndex
                    data_to_plot = data_to_plot.copy()
                    data_to_plot.index = pd.to_datetime(data_to_plot.index, errors='coerce')
                    data_to_plot = data_to_plot[~data_to_plot.index.isna()]

                    plt.figure(figsize=(12, 6))
                    ax = plt.gca()

                    if plot_style == 'line':
                        # Make continuous & smooth
                        series = data_to_plot[target_col]
                        if series.dropna().shape[0] < 2:
                            plt.text(0.5, 0.5, 'Not enough data to plot', ha='center', va='center', transform=ax.transAxes)
                        else:
                            series = _continuous_and_smooth(series)
                            ax.plot(series.index, series.values, linestyle='solid')
                        plt.title(f"{target_col} - Line Chart ({data_type.title()} Data)")
                        plt.xlabel('Time')
                        plt.ylabel(target_col)

                        # Date axis formatting
                        import matplotlib.dates as mdates
                        try:
                            locator = mdates.AutoDateLocator()
                            formatter = mdates.ConciseDateFormatter(locator)
                            ax.xaxis.set_major_locator(locator)
                            ax.xaxis.set_major_formatter(formatter)
                        except Exception:
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=30, ha='right')

                    elif plot_style == 'box':
                        # Unified, non-duplicated box plot logic
                        y = None
                        title_suffix = ''
                        if data_type == 'raw':
                            monthly_data = data_to_plot.resample('M').mean()
                            y = monthly_data[target_col].dropna()
                            title_suffix = 'Raw Monthly Data'
                        elif data_type == 'seasonal':
                            y = data_to_plot[target_col].dropna()
                            title_suffix = 'Seasonal Data'
                        elif data_type == 'anomaly':
                            y = data_to_plot[target_col].dropna()
                            title_suffix = 'Anomalies' if not y.empty else 'No Anomalies'
                        if y is None or y.empty:
                            plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        else:
                            plt.boxplot(y, labels=['Values'])
                        plt.title(f"{target_col} - Box Plot ({title_suffix})")
                        plt.ylabel(target_col)

                    plt.tight_layout()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_target = re.sub(r'[^A-Za-z0-9_]+', '_', target_col)
                    out_path = os.path.join(self.output_folder, f"plot_{safe_target}_{data_type}_{plot_style}_{timestamp}.png")
                    plt.savefig(out_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log(f"Exported plot for {target_col}: {out_path}")
                    success_count += 1
                except Exception as e:
                    self.log(f"ERROR exporting plot for {target_col}: {e}")
                    fail_count += 1

                self.dialog.pBTimeSeriesAnalysis.setValue(10 + int(80 * (idx_item + 1) / len(target_items)))

            # Final message reflecting success/fail counts
            if fail_count == 0 and success_count > 0:
                MessageBoxHandler.information(self.dialog, "All plots exported successfully.")
            elif success_count > 0 and fail_count > 0:
                MessageBoxHandler.information(self.dialog, f"Export completed with warnings. Success: {success_count}, Failed: {fail_count}")
            else:
                MessageBoxHandler.critical(self.dialog, "Plot export failed for all selected targets.")
            self.dialog.pBTimeSeriesAnalysis.setValue(100)
            self.log(f"Plot export summary → success: {success_count}, failed: {fail_count}")



    def populate_aggregation_controls(self):
        self.dialog.cBAggregationIntervalTimeSeries.clear()
        self.dialog.cBAggregationIntervalTimeSeries.addItems(["Daily", "Weekly", "Monthly"])

        self.dialog.cBAggregationMethodTimeSeries.clear()
        self.dialog.cBAggregationMethodTimeSeries.addItems(["Mean", "Sum", "Median"])

        self.log("Populated aggregation interval and method options.")

    def aggregate_data(self):
        # --- Ask user to select output folder ---
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Folder to Save Aggregated CSV")
        if not folder:
            MessageBoxHandler.critical(self.dialog, "No output folder selected. Aggregation aborted.")
            self.log("ERROR: Aggregation aborted due to no output folder selection.")
            return

        self.output_folder = folder
        self.log(f"Output folder selected: {folder}")


        interval = self.dialog.cBAggregationIntervalTimeSeries.currentText()
        method = self.dialog.cBAggregationMethodTimeSeries.currentText()
        requested_time_col = self.dialog.cBTimeColumnSelection.currentText()
        df = self.dataframe.copy()

        # Match time column case-insensitively
        time_col = next((col for col in df.columns if col.lower() in ['time', 'timestamp', 'date', 'datetime', requested_time_col.lower()]), None)
        if not time_col:
            MessageBoxHandler.critical(self.dialog, f"Time column '{requested_time_col}' not found.")
            self.log(f"ERROR: Time column '{requested_time_col}' not found.")
            return

        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"ERROR converting timestamp: {e}")
            self.log(f"ERROR converting timestamp: {e}")
            return

        # FIXED: Use the new coordinate detection function
        x_col, y_col = self.detect_coordinate_columns(df, prefer_uppercase=True)

        if not x_col or not y_col:
            available_cols = list(df.columns)
            MessageBoxHandler.critical(self.dialog, 
                f"Could not find X/Y coordinate columns.\nAvailable columns: {available_cols}")
            self.log(f"ERROR: Could not find X/Y coordinate columns. Available: {available_cols}")
            return

        self.log(f"Using coordinate columns: X='{x_col}', Y='{y_col}'")

        # Filter out grouping columns from value columns
        exclude_cols = [time_col, x_col, y_col]
        exclude_cols_lower = [c.lower() for c in exclude_cols]
        value_columns = [col for col in self.loaded_columns 
                         if col.lower() not in exclude_cols_lower]

        # Filter to only numeric columns
        numeric_columns = df[value_columns].select_dtypes(include=np.number).columns.tolist()
        if not numeric_columns:
            MessageBoxHandler.critical(self.dialog, "No numeric columns selected for aggregation.")
            self.log("ERROR: No numeric columns selected for aggregation.")
            return

        # Determine time interval rule
        rule = {'Monthly': 'M', 'Weekly': 'W', 'Daily': 'D'}.get(interval, 'D')

        try:
            # Temporarily rename for grouping
            df['__time_group'] = df[time_col]
            df['__time_group'] = pd.to_datetime(df[time_col])

            # Group and aggregate
            group = df.groupby([pd.Grouper(key='__time_group', freq=rule), df[x_col], df[y_col]])

            if method.lower() == 'mean':
                result = group[numeric_columns].mean().reset_index()
            elif method.lower() == 'sum':
                result = group[numeric_columns].sum().reset_index()
            else:
                result = group[numeric_columns].median().reset_index()

            result.drop(columns='__time_group', inplace=True, errors='ignore')

        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"ERROR during aggregation: {e}")
            self.log(f"ERROR during aggregation: {e}")
            return

        # Save result to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = os.path.join(self.output_folder, f"aggregated_{method}_{interval}_{timestamp}.csv")
        try:
            # Ensure coordinate columns are numeric
            result[x_col] = pd.to_numeric(result[x_col], errors='coerce')
            result[y_col] = pd.to_numeric(result[y_col], errors='coerce')

            # Drop rows with missing or invalid coordinates
            result = result.dropna(subset=[x_col, y_col])

            result.to_csv(out_csv, index=False)
            MessageBoxHandler.information(self.dialog, f"Aggregated data saved to: {out_csv}")
            self.log(f"Aggregated data saved to: {out_csv}")
        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"Failed to save CSV: {e}")
            self.log(f"ERROR: Failed to save CSV: {e}")
            return

        # Load into QGIS
        try:
            # Normalize path and build URI
            normalized_path = out_csv.replace('\\', '/')
            encoded_path = urllib.parse.quote(normalized_path, safe=':/')
            uri = f"file:///{encoded_path}?delimiter=,&xField={x_col}&yField={y_col}"
            self.log(f"Attempting to load CSV layer with URI: {uri}")

            # Create and validate layer
            layer = QgsVectorLayer(uri, f"Aggregated_{timestamp}", "delimitedtext")
            if layer.isValid():
                layer.setCrs(self.crs)
                QgsProject.instance().addMapLayer(layer)
                MessageBoxHandler.information(self.dialog, "Aggregated layer loaded into QGIS.")
                self.log("Aggregated layer loaded into QGIS.")
            else:
                error_msg = layer.error().message() or "Unknown error. Layer is not valid."
                MessageBoxHandler.critical(self.dialog, f"Failed to load layer: {error_msg}")
                self.log(f"ERROR: Failed to load aggregated layer: {error_msg}")
        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"Failed to load CSV: {str(e)}")
            self.log(f"ERROR: Failed to load CSV: {str(e)}")


    def decompose_loess(self):
        if not self.dialog.cBEnalblesLOESSSeasonalTrendDecomposition.isChecked():
            MessageBoxHandler.information(self.dialog, "LOESS decomposition not enabled.")
            self.log("LOESS decomposition not enabled.")
            return

        col = self.dialog.lWTimeSeriesTargets.currentItem().text()
        time_col = self.dialog.cBTimeColumnSelection.currentText()
        df = self.dataframe[[time_col, col]].dropna()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        result = seasonal_decompose(df[col], model='additive', period=12)
        result.plot()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(self.output_folder, f"decomposed_{col}_{timestamp}.png")
        plt.savefig(out_path)
        plt.close()
        MessageBoxHandler.information(self.dialog, f"LOESS decomposition saved to {out_path}")
        self.log(f"LOESS decomposition saved to {out_path}")

    def populate_correlation_methods(self):
        self.dialog.cBCorrelationMethodTimeSeries.clear()
        self.dialog.cBCorrelationMethodTimeSeries.addItems(["Pearson", "Spearman", "Kendall"])
        self.log("Populated correlation method options.")

    def run_correlation(self):
        from scipy.stats import pearsonr, spearmanr, kendalltau
        from PyQt5.QtWidgets import QFileDialog

        # Prompt user for output folder
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder to Save Correlation Results")
        if not folder:
            MessageBoxHandler.critical(self.dialog, "No output folder selected.")
            self.log("ERROR: No output folder selected for correlation results.")
            return

        self.output_folder = folder  # Save to internal state

        pollutant = self.dialog.cBPollutantTimeSeriesColumn.currentText()
        # Prevent using coordinate columns as pollutants
        if pollutant.lower() in ["x", "y"]:
            MessageBoxHandler.critical(self.dialog, "Please select a valid pollutant column, not a coordinate column.")
            self.log("ERROR: Selected column is X or Y — invalid for correlation.")
            return

        landuse = [item.text() for item in self.dialog.lWLandUseFractionColumn.selectedItems()]
        method = self.dialog.cBCorrelationMethodTimeSeries.currentText()

        if not pollutant or not landuse:
            MessageBoxHandler.critical(self.dialog, "Select a pollutant and at least one land use field.")
            self.log("ERROR: Pollutant or land use fields not selected.")
            return

        df = self.dataframe[[pollutant] + landuse].dropna()
        results = []

        for lu_col in landuse:
            try:
                if method == "Pearson":
                    r, p = pearsonr(df[pollutant], df[lu_col])
                elif method == "Spearman":
                    r, p = spearmanr(df[pollutant], df[lu_col])
                else:
                    r, p = kendalltau(df[pollutant], df[lu_col])

                self.log(f"{method} correlation between {pollutant} and {lu_col}: r={r:.4f}, p={p:.4e}")
                results.append({
                    "pollutant": pollutant,
                    "landuse": lu_col,
                    "method": method,
                    "r": r,
                    "p": p
                })
            except Exception as e:
                self.log(f"Correlation failed for {lu_col}: {e}")

        if results:
            r0 = results[0]["r"]
            p0 = results[0]["p"]
            self.dataframe["correlation_r"] = r0
            self.dataframe["correlation_p"] = p0
            MessageBoxHandler.information(self.dialog, f"{method} correlation:\n\nr = {r0:.4f}\np = {p0:.4e}")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(self.output_folder, f"correlation_results_multi_{ts}.csv")
            pd.DataFrame(results).to_csv(summary_path, index=False)
            self.log(f"Saved multivariate correlation results to: {summary_path}")
            MessageBoxHandler.information(self.dialog, f"Saved multivariate correlation results:\n{summary_path}")

            # Load into QGIS (as non-spatial table)
            uri = f"file:///{summary_path.replace(os.sep, '/')}?delimiter=,&geomType=none"
            layer = QgsVectorLayer(uri, f"Correlation_Results_{ts}", "delimitedtext")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.log("Correlation results loaded into QGIS.")
            else:
                self.log("WARNING: Failed to load correlation CSV into QGIS.")
                MessageBoxHandler.critical(self.dialog, "Failed to load correlation results into QGIS.")


    def choose_export_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Export Folder")
        if folder:
            self.output_folder = folder
            MessageBoxHandler.information(self.dialog, f"Output folder selected: {folder}")
            self.log(f"Output folder selected: {folder}")

    def export_time_series_data(self):
        if self.dataframe is not None and self.output_folder:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_csv = os.path.join(self.output_folder, f"full_timeseries_{timestamp}.csv")
            self.dataframe.to_csv(out_csv, index=False)
            MessageBoxHandler.information(self.dialog, f"Exported full time series dataset: {out_csv}")
            self.log(f"Exported full time series dataset: {out_csv}")

            # FIXED: Use the new coordinate detection function
            x_col, y_col = self.detect_coordinate_columns(self.dataframe, prefer_uppercase=True)

            if not x_col or not y_col:
                available_cols = list(self.dataframe.columns)
                MessageBoxHandler.critical(self.dialog, 
                    f"Failed to detect X and Y coordinate columns.\nAvailable columns: {available_cols}")
                self.log(f"ERROR: Failed to detect X and Y coordinate columns. Available: {available_cols}")
                return

            self.log(f"Using coordinate columns for export: X='{x_col}', Y='{y_col}'")

            # Prepare file URI
            normalized_path = out_csv.replace("\\", "/")
            encoded_path = urllib.parse.quote(normalized_path, safe=":/")
            uri = f"file:///{encoded_path}?delimiter=,&xField={x_col}&yField={y_col}"

            # Create and load layer
            layer = QgsVectorLayer(uri, f"FullTimeSeries_{timestamp}", "delimitedtext")
            if layer.isValid():
                layer.setCrs(self.crs)
                QgsProject.instance().addMapLayer(layer)
                MessageBoxHandler.information(self.dialog, "Exported CSV layer loaded into QGIS.")
                self.log("Exported CSV layer loaded into QGIS.")
            else:
                error = layer.error().message()
                MessageBoxHandler.critical(self.dialog, f"Failed to load CSV into QGIS: {error}")
                self.log(f"ERROR: Failed to load CSV into QGIS: {error}")

    def compute_multipliers(self, df, att, uid='fid', zindex=5):
        import numpy as np
        import pandas as pd

        # Ensure numeric and handle degenerate cases
        series = pd.to_numeric(df[att], errors='coerce')
        if series.dropna().nunique() < 2:
            # Column is constant or all NaN -> no meaningful multipliers
            df[att+'_Nones'] = 0
            df[att+'_Nzeros'] = 0
            return df

        allvalues = series.values
        medval = np.nanmedian(allvalues)

        allover = df[series >= medval]
        allunder = df[series < medval]

        # Safeguard: if any split is empty, avoid percentile on empty arrays
        def safe_iqr(vals):
            vals = pd.to_numeric(vals, errors='coerce').dropna().values
            if vals.size == 0:
                return np.nan
            q75 = np.percentile(vals, 75)
            q25 = np.percentile(vals, 25)
            return q75 - q25

        iqrup = safe_iqr(allover[att]) if len(allover) else np.nan
        iqrlo = safe_iqr(allunder[att]) if len(allunder) else np.nan

        # Fallbacks: if IQR is 0 or NaN, use std; if still 0/NaN, use small epsilon
        def safe_scale(vals, fallback=np.nan):
            vals = pd.to_numeric(vals, errors='coerce').dropna().values
            if vals.size == 0:
                return fallback
            s = float(np.nanstd(vals))
            return s

        if not np.isfinite(iqrup) or iqrup == 0:
            iqrup = safe_scale(allover[att])
        if not np.isfinite(iqrlo) or iqrlo == 0:
            iqrlo = safe_scale(allunder[att])

        # Final epsilon to avoid division by zero
        if not np.isfinite(iqrup) or iqrup == 0:
            iqrup = 1e-9
        if not np.isfinite(iqrlo) or iqrlo == 0:
            iqrlo = 1e-9

        df[att+'_Nones'] = 0
        df[att+'_Nzeros'] = 0

        for i, row in df.iterrows():
            v = row[att]
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v):
                continue
            if v >= medval:
                try:
                    nrep = round(abs(v - medval) / iqrup * 10) + 1
                    if not np.isfinite(nrep): nrep = 0
                except Exception:
                    nrep = 0
                df.at[i, att+'_Nones'] = int(max(0, nrep))
            else:
                try:
                    nrep = round(abs(v - medval) / iqrlo * 10) + 1
                    if not np.isfinite(nrep): nrep = 0
                except Exception:
                    nrep = 0
                df.at[i, att+'_Nzeros'] = int(max(0, nrep))
        return df


    def _detect_normalization_method(self, df, tp):
        """
        Auto-detect normalization for column `tp`.
        Priority:
        1) Explicit indicator columns (case-insensitive):
            - `<tp>_norm_method`, `norm_method`, `normalization_method`,
            `scaling_method`, `scaler`, `<tp>_scaler`
            Accepted values: minmax, min-max, z, zscore, z-score, standard,
            standardize, raw, none
        2) Companion stats columns:
            - Min–max if both `<tp>_orig_min` and `<tp>_orig_max` exist
            - Z-score if both `<tp>_orig_mean` and `<tp>_orig_std` exist
        3) Heuristics on distribution:
            - Min–max if values ~[0,1]
            - Z-score if mean≈0, std≈1, values within ~[-6,6]
        4) Otherwise → 'raw'
        Returns: (method, params_dict)
        """
        import pandas as pd
        import numpy as np

        # --- FORCE RAW FOR SPECIFIC COLUMNS ---
        # Add any columns that should never be converted
        force_raw_columns = [
            'DN_interp',
            'DN_interp_cleaned_minmax_windowed_cleaned'   # <-- add your column here
        ]
        if tp in force_raw_columns:
            return 'raw', {}

        # 1) Explicit indicator columns
        indicator_cols = [
            f"{tp}_norm_method", "norm_method", "normalization_method",
            "scaling_method", "scaler", f"{tp}_scaler"
        ]
        for c in indicator_cols:
            if c in df.columns and df[c].notna().any():
                val = str(df[c].dropna().iloc[0]).strip().lower()
                if val in ['minmax','min-max']:
                    return 'minmax', {}
                if val in ['z','zscore','z-score','standard','standardize']:
                    return 'zscore', {}
                if val in ['raw','none']:
                    return 'raw', {}

        # 2) Companion stats columns
        if f"{tp}_orig_min" in df.columns and f"{tp}_orig_max" in df.columns:
            return 'minmax', {
                'orig_min': pd.to_numeric(df[f"{tp}_orig_min"], errors='coerce').dropna().iloc[0],
                'orig_max': pd.to_numeric(df[f"{tp}_orig_max"], errors='coerce').dropna().iloc[0]
            }
        if f"{tp}_orig_mean" in df.columns and f"{tp}_orig_std" in df.columns:
            return 'zscore', {
                'orig_mean': pd.to_numeric(df[f"{tp}_orig_mean"], errors='coerce').dropna().iloc[0],
                'orig_std': pd.to_numeric(df[f"{tp}_orig_std"], errors='coerce').dropna().iloc[0]
            }

        # 3) Heuristics
        ser = pd.to_numeric(df[tp], errors='coerce').dropna()
        if not ser.empty:
            mn, mx, mean, std = ser.min(), ser.max(), ser.mean(), ser.std(ddof=0)
            if mn >= -0.05 and mx <= 1.05:
                return 'minmax', {}
            if abs(mean) < 0.25 and 0.5 <= std <= 1.5 and mn >= -6 and mx <= 6:
                return 'zscore', {}

        # 4) Default
        return 'raw', {}


    
    def _convert_integer_threshold_to_column_scale(self, df, tp, threshold_int):
        """
        Convert the integer threshold from lEPollutantThreshold into the scale of df[tp].
        Strategy:
          1) Detect normalization via _detect_normalization_method().
          2) If explicit stats (orig_min/max or orig_mean/std) exist -> use them.
          3) If method == 'minmax' but stats missing:
               - Try to infer a *raw* companion column by stripping common suffixes
                 (e.g., '_minmax', '_zscore', '_z', '_standard', '_normalized', '_norm').
               - If found, use that column's observed min/max as orig_min/max.
               - If not found, look for generic per-dataset stats like '<BASE>min'/'<BASE>max',
                 or common 'DNmin'/'DNmax' (useful for air-pollution DN workflows).
          4) If method == 'zscore' but stats missing:
               - Fall back to the observed mean/std of the candidate raw column (or the column itself).
          5) As an absolute last resort, compare using the raw integer (with a clear WARNING).
        """
        import pandas as pd
        import numpy as np

        # Numeric series from the target column (normalized or not)
        ser = pd.to_numeric(df[tp], errors='coerce')
        method, params = self._detect_normalization_method(df, tp)
        self.log(f"Normalization auto-detect for '{tp}': {method} (params={params})")

        def find_companion_raw_series(df, tp):
            # Try stripping common normalization suffixes to locate a raw companion
            suffixes = ['_minmax','_zscore','_z','_standard','_standardized','_normalized','_norm','_scaled']
            base = tp

            changed = True
            while changed:
                changed = False
                for suf in suffixes:
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        changed = True
                        break
                    if base in df.columns:
                        return pd.to_numeric(df[base], errors="coerce"), base
            # If base exists and is numeric, use it

            if base in df.columns and pd.api.types.is_numeric_dtype(df[base]):
                return pd.to_numeric(df[base], errors='coerce'), base

            # Try a few DN-specific helpers if relevant
            tokens = [t.lower() for t in re.split(r'[^A-Za-z0-9]+', tp) if t]
            dn_like = any(t in ('dn','dn_interp') for t in tokens)
            if dn_like:
                # Prefer most detailed DN columns if present
                for cand in ['DN_interp_windowed', 'DN_interp_cleaned', 'DN_interp']:
                    if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                        return pd.to_numeric(df[cand], errors='coerce'), cand

                # Fallback to DNmin/DNmax as stats only (not series return)
                # We'll handle these in the caller by reading df['DNmin']/df['DNmax']
            return None, base

        # Try to locate a companion "raw" series to infer stats if needed
        raw_series, raw_name = find_companion_raw_series(df, tp)

        # --- Min–max case ----------------------------------------------------
        if method == 'minmax':
            orig_min = params.get('orig_min')
            orig_max = params.get('orig_max')

            # If we don't have explicit stats, try inferring from companion series
            if (orig_min is None or orig_max is None or not (isinstance(orig_max,(int,float)) and isinstance(orig_min,(int,float)) and orig_max > orig_min)):
                # DNmin/DNmax special-case (if they exist and companion not found)
                dnmin = pd.to_numeric(df['DNmin'], errors='coerce').dropna().iloc[0] if 'DNmin' in df.columns and pd.to_numeric(df['DNmin'], errors='coerce').notna().any() else None
                dnmax = pd.to_numeric(df['DNmax'], errors='coerce').dropna().iloc[0] if 'DNmax' in df.columns and pd.to_numeric(df['DNmax'], errors='coerce').notna().any() else None

                if raw_series is not None and raw_series.notna().any():
                    orig_min = float(np.nanmin(raw_series))
                    orig_max = float(np.nanmax(raw_series))
                    self.log(f"Inferred orig_min/max from companion column '{raw_name}': {orig_min} / {orig_max}")
                elif dnmin is not None and dnmax is not None and dnmax > dnmin:
                    orig_min, orig_max = float(dnmin), float(dnmax)
                    self.log(f"Inferred orig_min/max from DNmin/DNmax: {orig_min} / {orig_max}")
                else:
                    # As a *very* last resort, if the data look normalized to [0,1] and
                    # the user entered >1, convert using a guessed [0..max_guess] range
                    # based on common pollutant scales (e.g., 0..100). We'll prefer 100 if
                    # threshold<=100, else 1000.
                    mn, mx = ser.min(skipna=True), ser.max(skipna=True)
                    if mn >= -0.05 and mx <= 1.05 and float(threshold_int) > 1:
                        max_guess = 100.0 if float(threshold_int) <= 100 else 1000.0
                        value = float(threshold_int) / max_guess  # map to 0..1
                        self.warn(
                            f"No min–max stats found for '{tp}'. Heuristically mapping integer "
                            f"{threshold_int} to normalized ~{value:.4f} using [0..{int(max_guess)}] guess."
                        )
                        return value
                    # Otherwise just compare raw (will likely zero out); warn clearly
                    self.warn(
                        f"Min–max detected for '{tp}' but missing orig_min/max and no companion raw column. "
                        f"Comparing using raw integer {threshold_int}. Consider adding columns "
                        f"'{tp}_orig_min'/'{tp}_orig_max' or a raw companion (e.g., '{raw_name}')."
                    )
                    return float(threshold_int)

            # If we reach here, we have orig_min/max
            try:
                value = (float(threshold_int) - float(orig_min)) / (float(orig_max) - float(orig_min))
                return float(value)
            except Exception:
                self.warn("Failed min–max conversion; using raw integer.")
                return float(threshold_int)

        # --- Z-score case ----------------------------------------------------
        if method == 'zscore':
            orig_mean = params.get('orig_mean')
            orig_std  = params.get('orig_std')

            # If missing, try to compute from a companion raw series; else from the column itself
            if orig_mean is None or orig_std in (None, 0):
                source = None
                if raw_series is not None and raw_series.notna().any():
                    source = raw_series
                    self.log(f"Estimating z-score stats from companion '{raw_name}'.")
                else:
                    source = ser
                    self.log(f"Estimating z-score stats from '{tp}' directly (no companion found).")
                vals = pd.to_numeric(source, errors='coerce').dropna().values
                if vals.size >= 2:
                    orig_mean = float(np.nanmean(vals))
                    orig_std  = float(np.nanstd(vals))
                else:
                    self.warn(f"Insufficient data to estimate mean/std for '{tp}'. Using raw integer.")
                    return float(threshold_int)

            if orig_std in (None, 0):
                self.warn(f"Z-score detected but invalid std for '{tp}'. Using raw integer.")
                return float(threshold_int)
            try:
                return float((float(threshold_int) - float(orig_mean)) / float(orig_std))
            except Exception:
                self.warn("Failed z-score conversion; using raw integer.")
                return float(threshold_int)

        # --- Raw / unknown case ----------------------------------------------
        return float(threshold_int)

    def compute_spatial_targets(self):
        from qgis.core import (
            QgsFields, QgsField, QgsFeature, QgsGeometry, QgsPointXY,
            QgsVectorFileWriter, QgsVectorLayer, QgsWkbTypes, QgsProject
        )
        from PyQt5.QtWidgets import QFileDialog, QInputDialog
        from PyQt5.QtCore import QVariant
        import numpy as np
        from scipy.stats import entropy
        from math import log

        if self.dataframe is None or self.dataframe.empty:
            MessageBoxHandler.critical(self.dialog, "No data loaded to compute spatial targets.")
            self.log("ERROR: No data loaded to compute spatial targets.")
            return

        # --- 1) Read month selections from lWTemporalScope ---
        selected_month_names = []
        for i in range(self.dialog.lWTemporalScope.count()):
            item = self.dialog.lWTemporalScope.item(i)
            if item and item.checkState() == Qt.Checked:
                selected_month_names.append(item.text())

        if not selected_month_names:
            MessageBoxHandler.critical(self.dialog, "Please select at least one month in Temporal Scope.")
            self.log("ERROR: No months selected in Temporal Scope.")
            return

        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month_numbers = [month_map[m] for m in selected_month_names if m in month_map]
        self.log(f"Selected months: {selected_month_names} → {selected_month_numbers}")

        # FIXED: Use the new coordinate detection function
        x_col, y_col = self.detect_coordinate_columns(self.dataframe, prefer_uppercase=True)

        if not x_col or not y_col:
            available_cols = list(self.dataframe.columns)
            MessageBoxHandler.critical(self.dialog, 
                f"X/Y columns not found.\nAvailable columns: {available_cols}\n"
                f"Looking for: X, Y, XCOORD, YCOORD, LONGITUDE, LATITUDE")
            self.log(f"ERROR: X/Y columns not found. Available: {available_cols}")
            return

        self.log(f"Using coordinate columns: X='{x_col}', Y='{y_col}'")

        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Folder to Save Spatial Targets")
        if not folder:
            MessageBoxHandler.critical(self.dialog, "Output folder not selected.")
            self.log("ERROR: Output folder not selected.")
            return

        try:
            df = self.dataframe.copy()
            # Use user-selected timestamp column
            timestamp_col = self.dialog.cBTimestampMetricsCalculationsTimeSeries.currentText()
            if timestamp_col not in df.columns:
                MessageBoxHandler.critical(self.dialog, f"Timestamp column '{timestamp_col}' not found.")
                self.log(f"ERROR: Timestamp column '{timestamp_col}' not found.")
                return

            # Log sample values for debugging
            sample_values = df[timestamp_col].head(3).tolist()
            self.log(f"Sample timestamp values: {sample_values}")

            # Parse timestamps with enhanced method
            df[timestamp_col] = self.parse_timestamps(df[timestamp_col])
            if df[timestamp_col].isnull().all():
                MessageBoxHandler.critical(self.dialog, "All timestamps are invalid after parsing")
                self.log("ERROR: All timestamps are invalid after parsing")
                return

            # Drop rows with invalid timestamps
            df_before_filter = df.dropna(subset=[timestamp_col])

            if df_before_filter.empty:
                MessageBoxHandler.critical(self.dialog, "No valid timestamp data after parsing.")
                self.log("ERROR: No valid timestamp data after parsing.")
                return

            # Log timestamp range for debugging
            min_date = df_before_filter[timestamp_col].min().strftime('%Y-%m-%d')
            max_date = df_before_filter[timestamp_col].max().strftime('%Y-%m-%d')
            self.log(f"Timestamp range: {min_date} to {max_date}")

            # --- Filter by selected months ---
            df_filtered = df_before_filter[df_before_filter[timestamp_col].dt.month.isin(selected_month_numbers)]
            self.log(f"Applied month filter → {len(df_filtered)} rows remain")

            if len(df_filtered) == 0:
                MessageBoxHandler.critical(self.dialog, "No data for the selected months. Please adjust your selection.")
                self.log("ERROR: No data after applying month filter.")
                return

            # Validate threshold input (now accepts whole numbers)
            try:
                threshold_int = int(float(self.dialog.lEPollutantThreshold.text()))
            except ValueError:
                MessageBoxHandler.critical(self.dialog, "Invalid threshold value. Please enter a whole number.")
                self.log("ERROR: Invalid threshold value (not an integer).")
                return

            tp = self.dialog.cBPollutantMetricsCalculationsTimeSeries.currentText()
            if tp not in df_filtered.columns:
                MessageBoxHandler.critical(self.dialog, f"Pollutant column '{tp}' not found in data.")
                self.log(f"ERROR: Pollutant column '{tp}' not found.")
                return

            # --- NEW: Popup to select raw column if the selected one seems normalized ---
            normalized_indicators = ['minmax', 'normalized', 'windowed', 'scaled', 'zscore', 'standard']
            if any(ind in tp.lower() for ind in normalized_indicators):
                # Find candidate raw columns: numeric columns that are not obviously normalized
                candidate_cols = [col for col in df_filtered.columns 
                                if pd.api.types.is_numeric_dtype(df_filtered[col]) 
                                and not any(ind in col.lower() for ind in normalized_indicators)]
                # Prefer columns with 'DN_interp' in name (raw pollutant)
                dn_cols = [col for col in candidate_cols if 'dn_interp' in col.lower()]
                if dn_cols:
                    candidate_cols = dn_cols + [c for c in candidate_cols if c not in dn_cols]
                if candidate_cols:
                    selected, ok = QInputDialog.getItem(self.dialog, 
                                                        "Select Raw Pollutant Column", 
                                                        "The selected pollutant column appears to be normalized.\n"
                                                        "Please select a raw (non‑normalized) column to use for threshold calculation:", 
                                                        candidate_cols, 0, False)
                    if ok and selected:
                        tp = selected
                        self.log(f"User switched to raw column: {tp}")
                    else:
                        self.warn("User did not select a raw column. Continuing with normalized column may yield zero above‑threshold.")
                else:
                    self.warn("No raw numeric columns found. Continuing with normalized column.")

            # Convert integer threshold to the column's scale if needed
            threshold_for_compare = self._convert_integer_threshold_to_column_scale(df_filtered, tp, threshold_int)
            self.log(f"Original threshold : {threshold_int}")
            self.log(f"Converted threshold: {threshold_for_compare}")

            self.log(
                f"Pollutant statistics:"
                f" min={df_filtered[tp].min()},"
                f" max={df_filtered[tp].max()},"
                f" mean={df_filtered[tp].mean()}"
            )

            self.log(
                f"Rows above threshold = "
                f"{(pd.to_numeric(df_filtered[tp], errors='coerce') >= threshold_for_compare).sum()}"
            )
            # Ensure fid column exists
            if 'fid' not in df_filtered.columns:
                df_filtered['fid'] = range(len(df_filtered))
                self.log("Created 'fid' column as it was missing.")

            # Validate coordinates before processing
            df_coords_valid = df_filtered[(df_filtered[x_col].notna()) & (df_filtered[y_col].notna()) & 
                            (np.isfinite(df_filtered[x_col])) & (np.isfinite(df_filtered[y_col]))]
            self.log(f"Valid coordinate pairs: {len(df_coords_valid)} out of {len(df_filtered)}")

            if len(df_coords_valid) == 0:
                MessageBoxHandler.critical(self.dialog, "No valid coordinate data found.")
                self.log("ERROR: No valid coordinate data found.")
                return

            df = df_coords_valid.copy()

            # Calculate spatial targets
            df['above_th'] = pd.to_numeric(df[tp], errors='coerce') >= threshold_for_compare

            grouped = df.groupby('fid')
            f_values = grouped['above_th'].mean()

            # Handle cases where no data is above threshold
            above_threshold_data = df.loc[df['above_th']]
            if len(above_threshold_data) == 0:
                self.warn("No data points above threshold. Setting I values to 0.")
                I_values = pd.Series(0, index=f_values.index)
            else:
                I_values = above_threshold_data.groupby('fid')[tp].apply(
                    lambda x: ((pd.to_numeric(x, errors='coerce') - threshold_for_compare)/max(threshold_for_compare, 1e-9)).quantile(0.75) if len(x) > 0 else 0
                )

            f_col = df['fid'].map(f_values).fillna(0)
            I_col = df['fid'].map(I_values).fillna(0)
            df['f'] = f_col
            df['I'] = I_col
            df['Ex'] = df['f'] * df['I']

            # Apply multipliers
            df = self.compute_multipliers(df, 'f')
            df = self.compute_multipliers(df, 'I')
            df = self.compute_multipliers(df, 'Ex')

            # Save to CSV
            current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log(f"Saving spatial targets to CSV with timestamp: {current_timestamp}")
            out_csv = os.path.join(folder, f"spatial_targets_{current_timestamp}.csv")
            df.to_csv(out_csv, index=False)
            self.log(f"Saved spatial targets CSV: {out_csv}")

            # Create GeoPackage with improved error handling
            out_gpkg = os.path.join(folder, f"spatial_targets_{current_timestamp}.gpkg")

            # Prepare fields - exclude fid and geometry columns
            seen = set()
            column_names = []
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['fid', 'geometry']:
                    continue
                if col_lower not in seen:
                    column_names.append(col)
                    seen.add(col_lower)

            fields = QgsFields()

            # Add fid field first
            fields.append(QgsField('fid', QVariant.LongLong))

            # Add other fields
            for col in column_names:
                dtype = df[col].dtype
                if col in [x_col, y_col]:  # Coordinate columns
                    qgis_type = QVariant.Double
                elif np.issubdtype(dtype, np.integer):
                    qgis_type = QVariant.LongLong
                elif np.issubdtype(dtype, np.floating):
                    qgis_type = QVariant.Double
                elif np.issubdtype(dtype, np.bool_):
                    qgis_type = QVariant.Bool
                else:
                    qgis_type = QVariant.String
                fields.append(QgsField(col, qgis_type))

            # Create save options
            save_options = QgsVectorFileWriter.SaveVectorOptions()
            save_options.driverName = "GPKG"
            save_options.fileEncoding = "UTF-8"
            save_options.layerName = f"spatial_targets_{current_timestamp}"

            # Create the writer
            transform_context = QgsProject.instance().transformContext()
            self.log(f"Creating GeoPackage writer for: {out_gpkg}")
            writer = QgsVectorFileWriter.create(
                fileName=out_gpkg,
                fields=fields,
                geometryType=QgsWkbTypes.Point,
                srs=self.crs,
                transformContext=transform_context,
                options=save_options
            )

            if writer.hasError() != QgsVectorFileWriter.NoError:
                MessageBoxHandler.critical(self.dialog, f"Failed to create GeoPackage writer: {writer.errorMessage()}")
                self.log(f"ERROR: Failed to create GeoPackage writer: {writer.errorMessage()}")
                return

            # Add features with better error handling
            features_added = 0
            features_failed = 0

            for idx, row in df.iterrows():
                try:
                    # Validate coordinates
                    x = float(row[x_col])
                    y = float(row[y_col])
                    if np.isnan(x) or np.isnan(y) or not np.isfinite(x) or not np.isfinite(y):
                        features_failed += 1
                        self.log(f"WARNING: Invalid coordinates at index {idx}: x={x}, y={y}")
                        continue

                    # Create feature
                    feat = QgsFeature(fields)
                    feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

                    # Set attributes in the correct order
                    attributes = []

                    # Add fid first
                    fid_value = row.get('fid', idx)
                    attributes.append(int(fid_value) if not pd.isna(fid_value) else idx)

                    # Add other attributes
                    for col in column_names:
                        value = row[col]
                        if pd.isna(value):
                            attributes.append(None)
                        elif isinstance(value, (bool, np.bool_)):
                            attributes.append(bool(value))
                        elif isinstance(value, (int, np.integer)):
                            attributes.append(int(value))
                        elif isinstance(value, (float, np.floating)):
                            if np.isfinite(value):
                                attributes.append(float(value))
                            else:
                                attributes.append(None)
                        else:
                            attributes.append(str(value))

                    feat.setAttributes(attributes)

                    # Add feature to writer
                    if not writer.addFeature(feat):
                        features_failed += 1
                        error_msg = writer.errorMessage()  # Get specific error
                        self.log(f"WARNING: Failed to add feature at index {idx}: {error_msg}")
                    else:
                        features_added += 1

                except Exception as e:
                    features_failed += 1
                    self.log(f"WARNING: Failed to add feature at index {idx}: {e}")

            # Clean up writer
            del writer

            self.log(f"GeoPackage creation completed: {features_added} features added, {features_failed} failed")

            if features_added == 0:
                MessageBoxHandler.critical(self.dialog, "No valid features could be added to the GeoPackage.")
                self.log("ERROR: No valid features could be added to the GeoPackage.")
                return

            # Load layer into QGIS with improved error handling
            try:
                layer_name = f"spatial_targets_{current_timestamp}"
                vlayer = QgsVectorLayer(out_gpkg, layer_name, "ogr")

                if vlayer.isValid():
                    vlayer.setCrs(self.crs)
                    QgsProject.instance().addMapLayer(vlayer)
                    MessageBoxHandler.information(self.dialog, 
                        f"Spatial targets computed successfully!\n"
                        f"- Processed {len(df)} data points\n"
                        f"- Created {features_added} spatial features\n"
                        f"- Failed features: {features_failed}\n"
                        f"- Layer loaded: {layer_name}")
                    self.log(f"Spatial targets computed and layer loaded successfully")
                else:
                    # Get detailed error message
                    error_msg = vlayer.error().message()
                    MessageBoxHandler.critical(self.dialog, 
                        f"Failed to load layer into QGIS: {error_msg}\n"
                        f"However, the GeoPackage was created successfully at: {out_gpkg}")
                    self.log(f"ERROR: Failed to load spatial targets layer: {error_msg}")
                    self.log(f"GeoPackage saved at: {out_gpkg}")

            except Exception as e:
                MessageBoxHandler.critical(self.dialog, 
                    f"Failed to load layer into QGIS: {str(e)}\n"
                    f"However, the GeoPackage was created successfully at: {out_gpkg}")
                self.log(f"ERROR: Failed to load layer into QGIS: {str(e)}")
                self.log(f"GeoPackage saved at: {out_gpkg}")

        except Exception as e:
            MessageBoxHandler.critical(self.dialog, f"Failed to compute spatial targets: {e}")
            self.log(f"ERROR: Failed to compute spatial targets: {e}")
            import traceback
            self.log(f"ERROR traceback: {traceback.format_exc()}")

    
    def parse_timestamps(self, series):
        """Robust timestamp parsing with magnitude-aware heuristics.

        Rules of thumb:
        - First try pandas' parser (works for ISO like '2025-03-01 12:00').
        - If numeric:
            * >1e17 → nanoseconds since epoch
            * >1e14 → microseconds since epoch
            * >1e11 → milliseconds since epoch
            * >1e9  → seconds since epoch
            * 4e4..1e6 → Excel serial date (days since 1899-12-30)
            * 0..1 or very small numbers → likely normalized/time-only → don't coerce to epoch
        - If still not parseable but looks like time-only (HH:MM:SS), anchor to today's date.
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, date, timedelta

        s = pd.Series(series)

        # 1) Try direct parse (handles '2025-...' well)
        p = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
        if p.notna().sum() >= max(1, int(0.8 * len(s))):
            return p

        # 2) Magnitude-aware numeric handling
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().any():
            v = s_num.dropna()
            if len(v):
                vmin = float(v.min())
                vmax = float(v.max())

                # Excel serial days
                if vmin > 40000 and vmax < 1000000:
                    base = datetime(1899, 12, 30)
                    return v.apply(lambda d: base + pd.to_timedelta(d, unit='D')).reindex(s.index)

                # UNIX epoch magnitudes
                if vmin > 1e17:
                    return pd.to_datetime(s_num, unit='ns', errors='coerce')
                if vmin > 1e14:
                    return pd.to_datetime(s_num, unit='us', errors='coerce')
                if vmin > 1e11:
                    return pd.to_datetime(s_num, unit='ms', errors='coerce')
                if vmin > 1e9:
                    return pd.to_datetime(s_num, unit='s', errors='coerce')

                # Very small numbers → likely normalized/time-only: do NOT coerce to epoch
                # Leave as NaT for now and try time-only parsing below.

        # 3) Time-only strings like '12:34:56'
        time_like = s.astype(str).str.fullmatch(r'\s*\d{1,2}:\d{2}(:\d{2}(\.\d{1,6})?)?\s*', na=False)
        if time_like.any():
            today = datetime.now().date()
            def _anchor(t):
                try:
                    tt = pd.to_datetime(t, format='%H:%M:%S', errors='coerce')
                    if pd.isna(tt):
                        tt = pd.to_datetime(t, errors='coerce')
                    if pd.isna(tt):
                        return pd.NaT
                    return datetime.combine(today, tt.time())
                except Exception:
                    return pd.NaT
            return s.apply(_anchor)

        # 4) Fallback: return whatever pandas managed (could be all NaT)
        return p

    def calculate_fractal_dimension(self, df, x_col, y_col):
        """Estimate fractal dimension using box counting approximation."""
        try:
            from sklearn.neighbors import KDTree
            points = df[[x_col, y_col]].dropna().values
            tree = KDTree(points)
            scales = np.logspace(0.5, 2, num=5, base=2)
            counts = []

            for scale in scales:
                neighbors = tree.query_radius(points, r=scale, count_only=True)
                counts.append(np.mean(neighbors))

            logs = np.log(scales)
            log_counts = np.log(counts)
            coeffs = np.polyfit(logs, log_counts, 1)
            fractal_value = coeffs[0]
            return np.full(len(df), round(fractal_value, 3))
        except Exception as e:
            self.log(f"ERROR computing fractal dimension: {e}")
            return np.full(len(df), np.nan)

    def calculate_spatial_autocorrelation(self, df, x_col, y_col):
        """Moran-like spatial autocorrelation approximation."""
        try:
            from scipy.spatial import distance_matrix
            subset = df[[x_col, y_col]].dropna()
            D = distance_matrix(subset.values, subset.values)
            np.fill_diagonal(D, np.nan)
            weights = 1 / D
            W = np.nan_to_num(weights)

            # Use first numeric column for values
            value_cols = ['DN_interp'] if 'DN_interp' in df.columns else df.select_dtypes(include=np.number).columns
            if not len(value_cols):
                return np.full(len(df), np.nan)

            z = (df[value_cols[0]] - df[value_cols[0]].mean()).fillna(0).values
            num = np.dot(z, W.dot(z))
            den = np.dot(z, z)
            I = num / den if den != 0 else 0
            return np.full(len(df), round(I, 3))
        except Exception as e:
            self.log(f"ERROR computing spatial autocorrelation: {e}")
            return np.full(len(df), np.nan)

    def calculate_entropy(self, df):
        """Shannon entropy per row using all numeric attributes."""
        try:
            numeric_df = df[['DN_interp']] if 'DN_interp' in df.columns else df.select_dtypes(include=np.number)
            row_entropy = numeric_df.div(numeric_df.sum(axis=1), axis=0).apply(
                lambda row: entropy(row.dropna()), axis=1
            )
            return row_entropy.fillna(0).round(3)
        except Exception as e:
            self.log(f"ERROR computing entropy: {e}")
            return np.full(len(df), np.nan)