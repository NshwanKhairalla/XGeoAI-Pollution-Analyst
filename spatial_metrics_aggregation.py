import os
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsVectorFileWriter, QgsField,
    QgsFields, QgsFeature, QgsGeometry, QgsPointXY, QgsWkbTypes,
    QgsCoordinateReferenceSystem
)
from PyQt5.QtCore import QVariant, QDateTime
import urllib.parse


class SpatialMetricsAggregation:
    def __init__(self, dialog):
        self.dialog = dialog
        self.current_csv_df = None  # cache for feature selection
        self.setup_ui()

    # -------------------------------------------------------------------------
    # UI wiring
    # -------------------------------------------------------------------------
    def setup_ui(self):
        # Targeting / export selected cols
        self.dialog.cBSelectRefinedCSVLayer.currentIndexChanged.connect(self.populate_fields_list)
        self.dialog.pBDataTargetingSave.clicked.connect(self.export_selected_columns)
        self.dialog.tBChooseFolderDataTargeting.clicked.connect(self.choose_output_folder_targeting)

        # Aggregation
        self.dialog.cBEnableAggregateOverTimeorSpace.stateChanged.connect(self.toggle_aggregation_controls)
        self.dialog.pBAggregateOverTimeorSpace.clicked.connect(self.run_aggregation)
        self.dialog.cBAggregationFunction.addItems(["mean", "median", "sum", "min", "max", "std", "count"])

        # Frequency
        self.dialog.pBFrequencyAnalysisSave.clicked.connect(self.run_frequency_analysis)
        self.dialog.tBChooseFolderFrequencyAnalysis.clicked.connect(self.choose_output_folder_frequency)

        # Intensity
        self.dialog.pBSaveIntensityAnalysis.clicked.connect(self.run_intensity_analysis)
        self.dialog.tBChooseFolderIntensityAnalysis.clicked.connect(self.choose_output_folder_intensity)
        self.dialog.cBIntensityMetricsIntensityAnalysis.addItems([
            "Mean Exceedance",
            "Z-score Based",
            "Median + IQR"
        ])

        # Exposure
        self.dialog.pBSaveExposureAnalysis.clicked.connect(self.run_exposure_analysis)
        self.dialog.tBChooseFolderExposureAnalysis.clicked.connect(self.choose_output_folder_exposure)
        self.dialog.cBMethodofEstimationExposureAnalysis.addItems([
            "Direct Proportional",
            "Weighted by Population",
            "Time Weighted Exposure"
        ])

        # Feature selection
        self.dialog.pBRunFeatureSelection.clicked.connect(self.run_feature_selection)
        self.dialog.pBCSVDataAnalysisNext.clicked.connect(
            lambda: self.dialog.tabsXgeoAi.setCurrentWidget(self.dialog.tSpatialAnalysis)
        )
        self.dialog.cBSelectCSVLayerforExposureAnalysis.currentIndexChanged.connect(self.populate_population_columns)

        # Layer refresh
        self.dialog.pBRefreshLayersSpatialMetricsAggregation.clicked.connect(self.refresh_layers_spatial_metrics)

        # Initial fill
        self.populate_csv_layers()
        self.populate_fields_list()

    # -------------------------------------------------------------------------
    # Layer combos
    # -------------------------------------------------------------------------
    def populate_csv_layers(self):
        self.dialog.cBSelectRefinedCSVLayer.clear()
        self.dialog.cBSelectCSVLayerforFrequencyAnalysis.clear()
        self.dialog.cBSelectCSVLayerforIntensityAnalysis.clear()
        self.dialog.cBSelectCSVLayerforExposureAnalysis.clear()

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and layer.dataProvider().name() in ["delimitedtext", "ogr"]:
                self.dialog.cBSelectRefinedCSVLayer.addItem(layer.name(), layer)
                self.dialog.cBSelectCSVLayerforFrequencyAnalysis.addItem(layer.name(), layer)
                self.dialog.cBSelectCSVLayerforIntensityAnalysis.addItem(layer.name(), layer)
                self.dialog.cBSelectCSVLayerforExposureAnalysis.addItem(layer.name(), layer)

        if self.dialog.cBSelectRefinedCSVLayer.count() > 0:
            self.dialog.cBSelectRefinedCSVLayer.setCurrentIndex(0)
            self.populate_fields_list()

    # -------------------------------------------------------------------------
    # Data source inspection and dataframe creation
    # -------------------------------------------------------------------------
    def get_datasource_info(self, layer):
        """
        Return (file_path, delimiter, provider_name, is_file_based)

        - file_path: actual path on disk if we can resolve it (csv, gpkg, shp, etc.)
        - delimiter: for delimitedtext only, else None
        - provider_name: e.g. 'delimitedtext', 'ogr'
        - is_file_based: True if file_path exists on disk
        """
        try:
            provider = layer.dataProvider().name()
            uri = layer.dataProvider().dataSourceUri()
            self.dialog.tELog.append(f"DEBUG: Raw URI from layer '{layer.name()}': {uri}")
            self.dialog.tELog.append(f"DEBUG: Provider for layer '{layer.name()}': {provider}")

            file_path = None
            delimiter = None
            is_file_based = False

            if provider == "delimitedtext":
                # Example:
                # file:///C:/path/data.csv?encoding=UTF-8&delimiter=;&xField=X&yField=Y&crs=EPSG:4326
                if '?' in uri:
                    path_part, query_part = uri.split('?', 1)
                else:
                    path_part, query_part = uri, ""

                path_part = path_part.replace('file:///', '').replace('file://', '')
                # Windows quirk: "/C:/..." -> "C:/..."
                if path_part.startswith('/') and len(path_part) > 2 and path_part[2] == ':':
                    path_part = path_part[1:]

                path_part = urllib.parse.unquote(path_part)
                path_part = os.path.normpath(path_part)

                file_path = path_part
                is_file_based = os.path.exists(file_path)

                for token in query_part.split('&'):
                    if token.lower().startswith('delimiter='):
                        raw_val = token.split('=', 1)[1]
                        raw_val = urllib.parse.unquote(raw_val)
                        if raw_val in ['\\t', '\t']:
                            delimiter = '\t'
                        else:
                            delimiter = raw_val
                        break

                self.dialog.tELog.append(f"DEBUG: Processed CSV path: {file_path}")
                self.dialog.tELog.append(f"DEBUG: Detected delimiter: '{delimiter}'")

            elif provider == "ogr":
                # Example: C:/path/data.gpkg|layername=my_layer
                path_part = uri.split('|')[0]
                path_part = path_part.replace('file:///', '').replace('file://', '')
                if path_part.startswith('/') and len(path_part) > 2 and path_part[2] == ':':
                    path_part = path_part[1:]
                path_part = urllib.parse.unquote(path_part)
                path_part = os.path.normpath(path_part)

                file_path = path_part
                is_file_based = os.path.exists(file_path)
                delimiter = None

                self.dialog.tELog.append(f"DEBUG: Processed OGR path: {file_path}")

            else:
                # fallback
                path_part = uri.split('|')[0].split('?')[0]
                path_part = path_part.replace('file:///', '').replace('file://', '')
                if path_part.startswith('/') and len(path_part) > 2 and path_part[2] == ':':
                    path_part = path_part[1:]
                path_part = urllib.parse.unquote(path_part)
                path_part = os.path.normpath(path_part)

                if os.path.exists(path_part):
                    file_path = path_part
                    is_file_based = True
                delimiter = None
                self.dialog.tELog.append(f"DEBUG: Fallback path: {file_path}")

            return file_path, delimiter, provider, is_file_based

        except Exception as e:
            raise Exception(f"Failed to inspect data source for layer '{layer.name()}': {e}")

    def get_layer_dataframe(self, layer):
        """
        Create a pandas.DataFrame from any QgsVectorLayer:
        - If layer is a CSV-like layer ('delimitedtext'), read with pandas.read_csv using the correct delimiter.
        - Otherwise (GPKG / SHP / etc. via 'ogr'), iterate features to build an attribute table and attach centroid coords.
        """
        file_path, delimiter, provider, is_file_based = self.get_datasource_info(layer)

        # Case 1: CSV-like source
        if provider == "delimitedtext" and is_file_based and file_path.lower().endswith(('.csv', '.txt', '.tsv')):
            self.dialog.tELog.append(
                f"INFO: Reading CSV layer '{layer.name()}' from {file_path} (sep='{delimiter}') via pandas"
            )
            try:
                df = pd.read_csv(file_path, encoding='utf-8', sep=delimiter if delimiter else ',')
            except UnicodeDecodeError:
                self.dialog.tELog.append("WARNING: UTF-8 decoding failed, retrying latin1")
                df = pd.read_csv(file_path, encoding='latin1', sep=delimiter if delimiter else ',')
            return df

        # Case 2: Generic vector layer (ogr-backed .gpkg, .shp, etc.)
        self.dialog.tELog.append(
            f"INFO: Reading non-delimited layer '{layer.name()}' from provider '{provider}' using features"
        )
        field_names = [f.name() for f in layer.fields()]
        rows = []
        for feat in layer.getFeatures():
            row = {}
            for name in field_names:
                row[name] = feat[name]
            geom = feat.geometry()
            if geom and not geom.isEmpty():
                centroid = geom.centroid().asPoint()
                row['_centroid_x'] = centroid.x()
                row['_centroid_y'] = centroid.y()
            else:
                row['_centroid_x'] = None
                row['_centroid_y'] = None
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    # -------------------------------------------------------------------------
    # CSV/GPKG writer for MGWR-ready output
    # -------------------------------------------------------------------------
    def save_results_as_csv_and_gpkg(
        self,
        base_name,
        df_result,
        crs,
        output_folder,
        x_col,
        y_col
    ):
        """
        Save df_result as:
        - CSV (attributes only, MGWR friendly)
        - GPKG (point geometry from x_col/y_col + same attributes)

        Returns (csv_path, gpkg_path or None)

        Also attempts to load both into QGIS.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_folder, f"{base_name}_{timestamp}.csv")
        gpkg_path = os.path.join(output_folder, f"{base_name}_{timestamp}.gpkg")
        layer_name = f"{base_name}_{timestamp}"

        # 1. Always save CSV
        df_result.to_csv(csv_path, index=False)
        self.dialog.tELog.append(f"INFO: Saved CSV for {base_name}: {csv_path}")

        # 2. Build GPKG only if we have coordinates
        if x_col is None or y_col is None:
            self.dialog.tELog.append(
                f"WARNING: No coordinates provided for {base_name}, skipping GPKG export."
            )
            # still try to load CSV into QGIS
            try:
                csv_norm = os.path.normpath(csv_path).replace("\\", "/")
                if len(csv_norm) > 1 and csv_norm[1] == ':':
                    csv_norm = '/' + csv_norm
                uri_csv = f"file://{csv_norm}?delimiter=,"
                csv_layer = QgsVectorLayer(uri_csv, f"{layer_name}_csv", "delimitedtext")
                if csv_layer.isValid():
                    QgsProject.instance().addMapLayer(csv_layer)
                    self.dialog.tELog.append("INFO: CSV layer loaded into QGIS.")
            except Exception as e:
                self.dialog.tELog.append(f"WARNING: Failed to load CSV into QGIS: {e}")

            return csv_path, None

        if x_col not in df_result.columns or y_col not in df_result.columns:
            self.dialog.tELog.append(
                f"WARNING: Missing {x_col}/{y_col} columns in {base_name} result, skipping GPKG export."
            )
            # still try to load CSV into QGIS
            try:
                csv_norm = os.path.normpath(csv_path).replace("\\", "/")
                if len(csv_norm) > 1 and csv_norm[1] == ':':
                    csv_norm = '/' + csv_norm
                uri_csv = f"file://{csv_norm}?delimiter=,"
                csv_layer = QgsVectorLayer(uri_csv, f"{layer_name}_csv", "delimitedtext")
                if csv_layer.isValid():
                    QgsProject.instance().addMapLayer(csv_layer)
                    self.dialog.tELog.append("INFO: CSV layer loaded into QGIS.")
            except Exception as e:
                self.dialog.tELog.append(f"WARNING: Failed to load CSV into QGIS: {e}")

            return csv_path, None

        # Prepare the schema for the GPKG
        qgs_fields = QgsFields()
        for col in df_result.columns:
            dtype = df_result[col].dtype
            if np.issubdtype(dtype, np.integer):
                qtype = QVariant.Int
            elif np.issubdtype(dtype, np.floating):
                qtype = QVariant.Double
            else:
                qtype = QVariant.String
            qgs_fields.append(QgsField(col, qtype))

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = layer_name
        opts.fileEncoding = "UTF-8"

        # QGIS API compatibility layer:
        # - Newer QGIS: QgsVectorFileWriter.create(...) -> writerObj
        # - Older QGIS: QgsVectorFileWriter.create(...) -> (writerObj, errMsg)
        writer = None
        err_msg = None
        try:
            # Try new style first (single return value)
            writer = QgsVectorFileWriter.create(
                fileName=gpkg_path,
                fields=qgs_fields,
                geometryType=QgsWkbTypes.Point,
                srs=crs,
                transformContext=QgsProject.instance().transformContext(),
                options=opts
            )
        except TypeError:
            # Fallback to old signature that returns (writer, err)
            writer, err_msg = QgsVectorFileWriter.create(
                fileName=gpkg_path,
                fields=qgs_fields,
                geometryType=QgsWkbTypes.Point,
                srs=crs,
                transformContext=QgsProject.instance().transformContext(),
                options=opts
            )

        if writer is None:
            self.dialog.tELog.append(
                f"ERROR: Could not create GPKG for {base_name}: {err_msg}"
            )
            # still load CSV into QGIS
            try:
                csv_norm = os.path.normpath(csv_path).replace("\\", "/")
                if len(csv_norm) > 1 and csv_norm[1] == ':':
                    csv_norm = '/' + csv_norm
                uri_csv = f"file://{csv_norm}?delimiter=,"
                csv_layer = QgsVectorLayer(uri_csv, f"{layer_name}_csv", "delimitedtext")
                if csv_layer.isValid():
                    QgsProject.instance().addMapLayer(csv_layer)
                    self.dialog.tELog.append("INFO: CSV layer loaded into QGIS.")
            except Exception as e:
                self.dialog.tELog.append(f"WARNING: Failed to load CSV into QGIS: {e}")

            return csv_path, None

        # Write features to the GPKG
        for _, row in df_result.iterrows():
            try:
                px = float(row[x_col])
                py = float(row[y_col])
            except (TypeError, ValueError):
                # skip rows with bad/missing coords
                continue

            feat = QgsFeature(qgs_fields)
            geom = QgsGeometry.fromPointXY(QgsPointXY(px, py))
            feat.setGeometry(geom)

            attrs = []
            for col in df_result.columns:
                val = row[col]
                attrs.append(None if pd.isna(val) else val)
            feat.setAttributes(attrs)

            writer.addFeature(feat)

        # finalize writer
        del writer
        self.dialog.tELog.append(f"INFO: Saved GPKG for {base_name}: {gpkg_path}")

        # Load CSV layer into QGIS
        try:
            csv_norm = os.path.normpath(csv_path).replace("\\", "/")
            if len(csv_norm) > 1 and csv_norm[1] == ':':
                csv_norm = '/' + csv_norm
            uri_csv = f"file://{csv_norm}?delimiter=,"
            csv_layer = QgsVectorLayer(uri_csv, f"{layer_name}_csv", "delimitedtext")
            if csv_layer.isValid():
                QgsProject.instance().addMapLayer(csv_layer)
                self.dialog.tELog.append("INFO: CSV layer loaded into QGIS.")
        except Exception as e:
            self.dialog.tELog.append(f"WARNING: Failed to load CSV into QGIS: {e}")

        # Load GPKG layer into QGIS
        try:
            uri_gpkg = f"{gpkg_path}|layername={layer_name}"
            gpkg_layer = QgsVectorLayer(uri_gpkg, layer_name, "ogr")
            if gpkg_layer.isValid():
                QgsProject.instance().addMapLayer(gpkg_layer)
                self.dialog.tELog.append("INFO: GPKG layer loaded into QGIS.")
        except Exception as e:
            self.dialog.tELog.append(f"WARNING: Failed to load GPKG into QGIS: {e}")

        return csv_path, gpkg_path


    # -------------------------------------------------------------------------
    # Populate field list + cache dataframe for feature selection
    # -------------------------------------------------------------------------
    def populate_fields_list(self):
        self.dialog.lWFieldSelection.clear()
        self.dialog.cBTargetVariableFeatureSelection.clear()

        index = self.dialog.cBSelectRefinedCSVLayer.currentIndex()
        if index < 0:
            self.dialog.tELog.append("WARNING: No layer selected in cBSelectRefinedCSVLayer.")
            return

        layer = self.dialog.cBSelectRefinedCSVLayer.itemData(index)
        if not layer:
            self.dialog.tELog.append("WARNING: No layer object found for current selection.")
            return

        # Show all fields in the list widget
        for field in layer.fields():
            self.dialog.lWFieldSelection.addItem(field.name())

        self.dialog.tELog.append(
            f"INFO: Populated field list with {layer.fields().count()} fields from layer '{layer.name()}'"
        )

        # Cache dataframe for later feature selection
        self.current_csv_df = None
        try:
            df = self.get_layer_dataframe(layer)
            self.current_csv_df = df
            self.dialog.tELog.append(
                f"INFO: Cached dataframe for feature selection. Shape: {df.shape}"
            )

            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            self.dialog.cBTargetVariableFeatureSelection.addItems(numeric_columns)
            self.dialog.tELog.append(
                f"INFO: Numeric columns available for target variable: {numeric_columns}"
            )

        except Exception as e:
            self.dialog.tELog.append(
                f"ERROR: Failed to load dataframe for feature selection: {e}"
            )
            self.current_csv_df = None

    # -------------------------------------------------------------------------
    # Folder choosers
    # -------------------------------------------------------------------------
    def choose_output_folder_targeting(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Folder")
        if folder:
            self.dialog.tBChooseFolderDataTargeting.setText(folder)

    def choose_output_folder_frequency(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderFrequencyAnalysis.setText(folder)

    def choose_output_folder_intensity(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderIntensityAnalysis.setText(folder)

    def choose_output_folder_exposure(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderExposureAnalysis.setText(folder)

    # -------------------------------------------------------------------------
    # Export selected columns (data targeting)
    # -------------------------------------------------------------------------
    def export_selected_columns(self):
        folder = self.dialog.tBChooseFolderDataTargeting.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self.dialog, "Warning", "Please select a valid folder.")
            return

        index = self.dialog.cBSelectRefinedCSVLayer.currentIndex()
        layer = self.dialog.cBSelectRefinedCSVLayer.itemData(index)
        if not layer:
            QMessageBox.warning(self.dialog, "Warning", "No CSV layer selected.")
            return

        selected_items = self.dialog.lWFieldSelection.selectedItems()
        selected_columns = [item.text() for item in selected_items]

        # Always include X/Y-like columns if they exist in layer
        all_fields = [f.name() for f in layer.fields()]
        for geo_field in ['X', 'Y', 'Longitude', 'Latitude']:
            if geo_field in all_fields and geo_field not in selected_columns:
                selected_columns.append(geo_field)

        if not selected_columns:
            QMessageBox.warning(self.dialog, "Warning", "No fields selected.")
            return

        # Extract attribute rows from the layer
        features = layer.getFeatures()
        rows = []
        for feat in features:
            row = [feat[field] for field in selected_columns]
            rows.append(row)

        df = pd.DataFrame(rows, columns=selected_columns)

        # Guess X/Y
        x_col = next((c for c in selected_columns if c.lower() in ['x', 'longitude']), None)
        y_col = next((c for c in selected_columns if c.lower() in ['y', 'latitude']), None)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(folder, f"selected_fields_{timestamp}.csv")
        self.dialog.tELog.append(f"INFO: Exporting selected fields to: {csv_path}")
        df.to_csv(csv_path, index=False)

        crs = layer.crs()
        crs_authid = crs.authid() if crs.isValid() else "EPSG:4326"
        crs_authid = crs_authid.replace("|", ":")

        csv_uri_path = os.path.normpath(csv_path).replace("\\", "/")
        if len(csv_uri_path) > 1 and csv_uri_path[1] == ':':
            csv_uri_path = '/' + csv_uri_path

        if x_col and y_col:
            uri = (
                f"file://{csv_uri_path}?encoding=UTF-8&delimiter=,&xField={x_col}&yField={y_col}&crs={crs_authid}"
            )
        else:
            uri = f"file://{csv_uri_path}?encoding=UTF-8&delimiter=,"

        layer_name = f"selected_fields_{timestamp}"
        vlayer = QgsVectorLayer(uri, layer_name, "delimitedtext")

        if vlayer.isValid():
            QgsProject.instance().addMapLayer(vlayer)
            QMessageBox.information(
                self.dialog,
                "Success",
                "Selected columns exported and loaded into QGIS."
            )
            self.dialog.tELog.append(f"INFO: Layer loaded successfully with CRS: {crs_authid}")
        else:
            err_msg = vlayer.error().summary() if vlayer.error() else "Unknown error"
            QMessageBox.critical(
                self.dialog,
                "Error",
                f"Failed to load the exported layer:\n{err_msg}\nURI: {uri}"
            )
            self.dialog.tELog.append(f"ERROR: Failed to load layer: {err_msg}")

    # -------------------------------------------------------------------------
    # Aggregation toggle
    # -------------------------------------------------------------------------
    def toggle_aggregation_controls(self, state):
        enabled = state == 2
        self.dialog.rBAggregateByTime.setEnabled(enabled)
        self.dialog.rBAggregateBySpatialUnit.setEnabled(enabled)
        self.dialog.cBAggregationFunction.setEnabled(enabled)
        self.dialog.pBAggregateOverTimeorSpace.setEnabled(enabled)

    # -------------------------------------------------------------------------
    # Basic helper for dumping a layer to a df with centroid columns (used in aggregation)
    # -------------------------------------------------------------------------
    def layer_to_dataframe(self, layer):
        """
        Lightweight extract of layer attributes + centroid coordinates.
        """
        field_names = [field.name() for field in layer.fields()]
        rows = []
        for feat in layer.getFeatures():
            row = {}
            for name in field_names:
                row[name] = feat[name]
            geom = feat.geometry()
            if geom and not geom.isEmpty():
                centroid = geom.centroid().asPoint()
                row['_centroid_x'] = centroid.x()
                row['_centroid_y'] = centroid.y()
            else:
                row['_centroid_x'] = None
                row['_centroid_y'] = None
            rows.append(row)
        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=field_names + ['_centroid_x', '_centroid_y'])

    # -------------------------------------------------------------------------
    # Aggregation (time or spatial unit). Saves CSV and GPKG already by design.
    # -------------------------------------------------------------------------
    def run_aggregation(self):
        try:
            self.dialog.tELog.append("INFO: Starting aggregation...")
            layer_name = self.dialog.cBSelectRefinedCSVLayer.currentText()

            layer_list = QgsProject.instance().mapLayersByName(layer_name)
            if not layer_list:
                self.log("ERROR: Selected layer not found.")
                return
            layer = layer_list[0]

            selected_items = self.dialog.lWFieldSelection.selectedItems()
            selected_fields = [item.text() for item in selected_items]

            df = self.layer_to_dataframe(layer)

            # Detect coords
            x_col = None
            y_col = None
            for c in df.columns:
                lc = c.lower()
                if lc in ['x', 'x_coord', 'longitude', 'lon'] and x_col is None:
                    x_col = c
                if lc in ['y', 'y_coord', 'latitude', 'lat'] and y_col is None:
                    y_col = c
            if x_col is None and '_centroid_x' in df.columns:
                x_col = '_centroid_x'
            if y_col is None and '_centroid_y' in df.columns:
                y_col = '_centroid_y'

            # Detect time column
            time_col = None
            possible_time_cols = [
                col for col in df.columns
                if any(k in col.lower() for k in ['time', 'date', 'timestamp'])
            ]
            for c in possible_time_cols:
                try:
                    if df[c].apply(lambda v: isinstance(v, QDateTime)).any():
                        df[c] = df[c].apply(
                            lambda v: v.toString("yyyy-MM-dd HH:mm:ss")
                            if isinstance(v, QDateTime) else v
                        )
                    parsed = pd.to_datetime(df[c], errors='coerce')
                    if parsed.notna().sum() / len(parsed) > 0.8:
                        df[c] = parsed
                        time_col = c
                        break
                except Exception:
                    continue

            # Make sure grouping fields are included
            if self.dialog.rBAggregateBySpatialUnit.isChecked():
                if x_col and x_col not in selected_fields:
                    selected_fields.append(x_col)
                if y_col and y_col not in selected_fields:
                    selected_fields.append(y_col)

            if self.dialog.rBAggregateByTime.isChecked():
                if time_col and time_col not in selected_fields:
                    selected_fields.append(time_col)

            self.log(f"DEBUG: Selected fields: {selected_fields}")
            self.log(f"DEBUG: Detected columns - X:{x_col}, Y:{y_col}, Time:{time_col}")

            if not selected_fields:
                self.log("ERROR: No fields selected for aggregation.")
                return

            agg_func = self.dialog.cBAggregationFunction.currentText()
            if agg_func not in ["mean", "median", "sum", "min", "max", "std", "count"]:
                self.log("ERROR: Invalid aggregation function.")
                return

            df_filtered = df[selected_fields].copy()

            numeric_cols = []
            for col in selected_fields:
                if col not in [x_col, y_col, time_col] and pd.api.types.is_numeric_dtype(df_filtered[col]):
                    numeric_cols.append(col)

            if not numeric_cols:
                self.log("ERROR: No numeric columns found for aggregation.")
                return

            self.log(f"DEBUG: Numeric columns for aggregation: {numeric_cols}")

            if self.dialog.rBAggregateByTime.isChecked():
                if not time_col:
                    self.log("ERROR: Time aggregation selected but no valid time column found.")
                    return
                self.log(f"INFO: Aggregating by time column '{time_col}'")
                grouped = df_filtered.groupby(time_col)[numeric_cols].agg(agg_func).reset_index()

            elif self.dialog.rBAggregateBySpatialUnit.isChecked():
                if not x_col or not y_col:
                    self.log("ERROR: Spatial aggregation selected but missing coord columns.")
                    return
                self.log(f"INFO: Aggregating by spatial coordinates '{x_col}', '{y_col}'")
                grouped = df_filtered.groupby([x_col, y_col])[numeric_cols].agg(agg_func).reset_index()

            else:
                self.log("ERROR: No aggregation mode selected.")
                return

            # Ask output folder
            folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
            if not folder:
                self.log("ERROR: No output folder selected.")
                return
            self.dialog.tELog.append(f"INFO: Output folder selected: {folder}")

            # Build final per-point/per-time MGWR-style dataset:
            # - If grouped by space: each row is a location with x,y and aggregated metrics
            # - If grouped by time: each row is a timestamp with aggregated metrics (no point geom)
            out_df = grouped.copy()
            if self.dialog.rBAggregateBySpatialUnit.isChecked():
                # Normalize column names to generic 'x','y' for MGWR and GPKG
                out_df = out_df.rename(columns={x_col: "x", y_col: "y"})

                crs = layer.crs()
                csv_path, gpkg_path = self.save_results_as_csv_and_gpkg(
                    base_name=f"aggregated_spatial_metrics_{agg_func}",
                    df_result=out_df,
                    crs=crs,
                    output_folder=folder,
                    x_col="x",
                    y_col="y"
                )
                self.dialog.tELog.append(
                    f"INFO: Aggregation exported:\nCSV: {csv_path}\nGPKG: {gpkg_path}"
                )
            else:
                # Time aggregation has no spatial geometry -> CSV only
                timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_csv_only = os.path.join(
                    folder,
                    f"aggregated_temporal_metrics_{agg_func}_{timestamp_now}.csv"
                )
                out_df.to_csv(out_csv_only, index=False)
                self.dialog.tELog.append(
                    f"INFO: Temporal aggregation exported CSV: {out_csv_only}"
                )

                try:
                    csv_norm = os.path.normpath(out_csv_only).replace("\\", "/")
                    if len(csv_norm) > 1 and csv_norm[1] == ':':
                        csv_norm = '/' + csv_norm
                    uri_csv = f"file://{csv_norm}?delimiter=,"
                    csv_layer = QgsVectorLayer(
                        uri_csv,
                        f"aggregated_temporal_metrics_{agg_func}_{timestamp_now}_csv",
                        "delimitedtext"
                    )
                    if csv_layer.isValid():
                        QgsProject.instance().addMapLayer(csv_layer)
                        self.log("INFO: Temporal aggregation CSV also loaded into QGIS.")
                except Exception as e:
                    self.log(f"WARNING: Failed to load temporal CSV into QGIS: {e}")

            self.dialog.pBCSVDataAnalysis.setValue(100)
            self.log("INFO: Aggregation complete.")

        except Exception as e:
            import traceback
            self.log(f"ERROR: Aggregation failed: {e}")
            self.log(f"DEBUG: Full traceback: {traceback.format_exc()}")

    # -------------------------------------------------------------------------
    # Frequency analysis
    # -------------------------------------------------------------------------
    def run_frequency_analysis(self):
        try:
            self.dialog.pBCSVDataAnalysis.setValue(0)

            layer_name = self.dialog.cBSelectCSVLayerforFrequencyAnalysis.currentText()
            threshold = self.dialog.sBThresholdLevelFrequencyAnalysis.value()
            folder = self.dialog.tBChooseFolderFrequencyAnalysis.text()

            self.dialog.tELog.append(
                f"INFO: Running frequency analysis on '{layer_name}' with threshold {threshold}."
            )

            if not layer_name or not folder:
                QMessageBox.warning(self.dialog, "Warning", "Please select a layer and output folder.")
                return

            layer_list = QgsProject.instance().mapLayersByName(layer_name)
            if not layer_list:
                QMessageBox.critical(self.dialog, "Error", "Selected CSV layer not found in QGIS.")
                return
            layer = layer_list[0]

            df = self.get_layer_dataframe(layer)
            self.dialog.pBCSVDataAnalysis.setValue(20)

            # detect coordinates
            x_col, y_col = self.detect_xy_columns(df)
            excluded = [x_col, y_col, '_centroid_x', '_centroid_y']

            numeric_cols = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and col not in excluded
            ]

            # We compute frequency (fraction of time above threshold) per variable,
            # then attach those values to every point row so MGWR has per-row features.
            freq_map = {}
            for col in numeric_cols:
                series = df[col]
                count = (series > threshold).sum()
                total = series.notna().sum()
                freq_val = round(count / total, 5) if total > 0 else 0
                freq_map[col] = freq_val

            self.dialog.pBCSVDataAnalysis.setValue(60)

            out_df = pd.DataFrame()
            out_df["x"] = df[x_col] if x_col else df['_centroid_x']
            out_df["y"] = df[y_col] if y_col else df['_centroid_y']

            for var_name, freq_val in freq_map.items():
                out_df[f"freq_{var_name}"] = freq_val

            crs = layer.crs()
            csv_path, gpkg_path = self.save_results_as_csv_and_gpkg(
                base_name="frequency_analysis",
                df_result=out_df,
                crs=crs,
                output_folder=folder,
                x_col="x",
                y_col="y"
            )

            self.dialog.pBCSVDataAnalysis.setValue(100)
            self.dialog.tELog.append(
                f"INFO: Frequency analysis exported:\nCSV: {csv_path}\nGPKG: {gpkg_path}"
            )

        except Exception as e:
            import traceback
            self.dialog.tELog.append(f"ERROR: Failed to run frequency analysis: {e}")
            self.dialog.tELog.append(f"DEBUG: Full traceback: {traceback.format_exc()}")

    # -------------------------------------------------------------------------
    # Intensity analysis
    # -------------------------------------------------------------------------
    def run_intensity_analysis(self):
        try:
            self.dialog.pBCSVDataAnalysis.setValue(0)

            layer_name = self.dialog.cBSelectCSVLayerforIntensityAnalysis.currentText()
            method = self.dialog.cBIntensityMetricsIntensityAnalysis.currentText()
            folder = self.dialog.tBChooseFolderIntensityAnalysis.text()

            self.dialog.tELog.append(
                f"INFO: Running intensity analysis on '{layer_name}' using '{method}'."
            )

            if not layer_name or not folder:
                QMessageBox.warning(self.dialog, "Warning", "Please select a layer and output folder.")
                return

            layer_list = QgsProject.instance().mapLayersByName(layer_name)
            if not layer_list:
                QMessageBox.critical(self.dialog, "Error", "Selected CSV layer not found in QGIS.")
                return
            layer = layer_list[0]

            df = self.get_layer_dataframe(layer)
            self.dialog.pBCSVDataAnalysis.setValue(20)

            x_col, y_col = self.detect_xy_columns(df)
            excluded = [x_col, y_col, '_centroid_x', '_centroid_y']
            numeric_cols = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and col not in excluded
            ]

            intensity_map = {}
            for col in numeric_cols:
                series = df[col].dropna()

                if len(series) == 0:
                    intensity_val = 0
                else:
                    if method == "Mean Exceedance":
                        thr = series.mean()
                        exceedances = series[series > thr]
                        intensity_val = exceedances.mean() if not exceedances.empty else 0

                    elif method == "Median + IQR":
                        med = series.median()
                        iqr = stats.iqr(series)
                        thr = med + iqr
                        exceedances = series[series > thr]
                        intensity_val = exceedances.mean() if not exceedances.empty else 0

                    elif method == "Z-score Based":
                        mu = series.mean()
                        sigma = series.std(ddof=0)
                        if sigma == 0 or np.isnan(sigma):
                            intensity_val = 0
                        else:
                            z = (series - mu) / sigma
                            exceedances = series[z > 2]
                            intensity_val = exceedances.mean() if not exceedances.empty else 0
                    else:
                        intensity_val = 0

                intensity_map[col] = round(intensity_val, 5)

            self.dialog.pBCSVDataAnalysis.setValue(70)

            out_df = pd.DataFrame()
            out_df["x"] = df[x_col] if x_col else df['_centroid_x']
            out_df["y"] = df[y_col] if y_col else df['_centroid_y']

            for var_name, inten_val in intensity_map.items():
                out_df[f"intensity_{var_name}"] = inten_val

            crs = layer.crs()
            csv_path, gpkg_path = self.save_results_as_csv_and_gpkg(
                base_name="intensity_analysis",
                df_result=out_df,
                crs=crs,
                output_folder=folder,
                x_col="x",
                y_col="y"
            )

            self.dialog.pBCSVDataAnalysis.setValue(100)
            self.dialog.tELog.append(
                f"INFO: Intensity analysis exported:\nCSV: {csv_path}\nGPKG: {gpkg_path}"
            )

        except Exception as e:
            import traceback
            self.dialog.tELog.append(f"ERROR: Failed to run intensity analysis: {e}")
            self.dialog.tELog.append(f"DEBUG: Full traceback: {traceback.format_exc()}")

    # -------------------------------------------------------------------------
    # Helper: detect XY columns (including centroid fallback)
    # -------------------------------------------------------------------------
    def detect_xy_columns(self, df):
        x_col, y_col = None, None

        for col in df.columns:
            low = col.lower()
            if low in ['x', 'x_coord', 'longitude', 'lon']:
                x_col = col
            elif low in ['y', 'y_coord', 'latitude', 'lat']:
                y_col = col

        if x_col is None and '_centroid_x' in df.columns:
            x_col = '_centroid_x'
        if y_col is None and '_centroid_y' in df.columns:
            y_col = '_centroid_y'

        return x_col, y_col

    # -------------------------------------------------------------------------
    # Exposure analysis (now also MGWR-ready)
    # -------------------------------------------------------------------------
    def run_exposure_analysis(self):
        try:
            self.dialog.pBCSVDataAnalysis.setValue(0)

            pollutant_layer_name = self.dialog.cBSelectCSVLayerforExposureAnalysis.currentText()

            # if we have a population combo, use it, else reuse pollutant layer
            if hasattr(self.dialog, 'cBPopulationLayerExposureAnalysis'):
                population_layer_name = self.dialog.cBPopulationLayerExposureAnalysis.currentText()
            else:
                population_layer_name = pollutant_layer_name
                self.dialog.tELog.append(
                    "INFO: Using same layer for both pollutant and population data."
                )

            method = self.dialog.cBMethodofEstimationExposureAnalysis.currentText()
            folder = self.dialog.tBChooseFolderExposureAnalysis.text()

            self.dialog.tELog.append(
                f"INFO: Running exposure analysis with method '{method}'."
            )
            self.dialog.tELog.append(
                f"DEBUG: pollutant='{pollutant_layer_name}', population='{population_layer_name}'"
            )

            if not pollutant_layer_name or not folder:
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "Please select pollutant layer and output folder."
                )
                return

            if not os.path.isdir(folder):
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "Please select a valid output folder."
                )
                return

            pollutant_layers = QgsProject.instance().mapLayersByName(pollutant_layer_name)
            if not pollutant_layers:
                available = [l.name() for l in QgsProject.instance().mapLayers().values()]
                self.dialog.tELog.append(
                    f"ERROR: Pollutant layer '{pollutant_layer_name}' not found. Available: {available}"
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    f"Pollutant layer '{pollutant_layer_name}' not found in QGIS."
                )
                return
            pollutant_layer = pollutant_layers[0]

            if population_layer_name == pollutant_layer_name:
                population_layer = pollutant_layer
            else:
                population_layers = QgsProject.instance().mapLayersByName(population_layer_name)
                if not population_layers:
                    available = [l.name() for l in QgsProject.instance().mapLayers().values()]
                    self.dialog.tELog.append(
                        f"ERROR: Population layer '{population_layer_name}' not found. Available: {available}"
                    )
                    QMessageBox.critical(
                        self.dialog,
                        "Error",
                        f"Population layer '{population_layer_name}' not found in QGIS."
                    )
                    return
                population_layer = population_layers[0]

            # Read both into dataframes
            df_pollutant = self.get_layer_dataframe(pollutant_layer)
            self.dialog.tELog.append(
                f"INFO: Pollutant DF loaded. Shape: {df_pollutant.shape}"
            )

            if population_layer == pollutant_layer:
                df_population = df_pollutant.copy()
                self.dialog.tELog.append("INFO: Population DF reused from pollutant DF.")
            else:
                df_population = self.get_layer_dataframe(population_layer)
                self.dialog.tELog.append(
                    f"INFO: Population DF loaded. Shape: {df_population.shape}"
                )

            self.dialog.pBCSVDataAnalysis.setValue(20)

            # detect coords in both
            def detect_xy(df):
                xx, yy = None, None
                for c in df.columns:
                    low = c.lower()
                    if low in ['x', 'x_coord', 'longitude', 'lon']:
                        xx = c
                    elif low in ['y', 'y_coord', 'latitude', 'lat']:
                        yy = c
                if xx is None and '_centroid_x' in df.columns:
                    xx = '_centroid_x'
                if yy is None and '_centroid_y' in df.columns:
                    yy = '_centroid_y'
                return xx, yy

            x_col_p, y_col_p = detect_xy(df_pollutant)
            x_col_pop, y_col_pop = detect_xy(df_population)

            self.dialog.tELog.append(
                f"DEBUG: pollutant coords => X:{x_col_p} Y:{y_col_p}"
            )
            self.dialog.tELog.append(
                f"DEBUG: population coords => X:{x_col_pop} Y:{y_col_pop}"
            )

            if not x_col_p or not y_col_p:
                self.dialog.tELog.append(
                    f"ERROR: Pollutant layer has no usable coordinates. Columns: {list(df_pollutant.columns)}"
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "Pollutant layer has no valid coordinate fields or geometry."
                )
                return

            if not x_col_pop or not y_col_pop:
                self.dialog.tELog.append(
                    f"ERROR: Population layer has no usable coordinates. Columns: {list(df_population.columns)}"
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "Population layer has no valid coordinate fields or geometry."
                )
                return

            # find population magnitude column
            pop_col = None
            if hasattr(self.dialog, 'cBPopulationColumnExposureAnalysis'):
                cand = self.dialog.cBPopulationColumnExposureAnalysis.currentText()
                if cand and (cand in df_population.columns):
                    if pd.api.types.is_numeric_dtype(df_population[cand]):
                        pop_col = cand

            if pop_col is None:
                for col in df_population.columns:
                    low = col.lower()
                    if any(k in low for k in ['pop', 'population', 'residents', 'inhabitants']):
                        if pd.api.types.is_numeric_dtype(df_population[col]):
                            pop_col = col
                            self.dialog.tELog.append(
                                f"INFO: Auto-detected population column '{pop_col}'"
                            )
                            break

            if pop_col is None:
                numeric_cands = df_population.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cands = [
                    c for c in numeric_cands
                    if c not in [x_col_pop, y_col_pop, '_centroid_x', '_centroid_y']
                ]
                if numeric_cands:
                    pop_col = numeric_cands[-1]
                    self.dialog.tELog.append(
                        f"INFO: Using numeric column '{pop_col}' as population proxy"
                    )

            if pop_col is None:
                self.dialog.tELog.append("ERROR: No usable population column found.")
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "No numeric population column found in population layer."
                )
                return

            if not pd.api.types.is_numeric_dtype(df_population[pop_col]):
                self.dialog.tELog.append(
                    f"ERROR: Population column '{pop_col}' is not numeric."
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    f"Population column '{pop_col}' must be numeric."
                )
                return

            # spatial match via KDTree
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                self.dialog.tELog.append("ERROR: scipy.spatial.cKDTree not available.")
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "scipy is required for exposure analysis."
                )
                return

            df_pollutant_valid = df_pollutant.dropna(subset=[x_col_p, y_col_p]).reset_index(drop=True)
            df_population_valid = df_population.dropna(
                subset=[x_col_pop, y_col_pop, pop_col]
            ).reset_index(drop=True)

            if len(df_pollutant_valid) == 0:
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "No valid pollutant points with coordinates."
                )
                return

            if len(df_population_valid) == 0:
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    "No valid population points with coordinates and population values."
                )
                return

            pollutant_coords = df_pollutant_valid[[x_col_p, y_col_p]].to_numpy()
            population_coords = df_population_valid[[x_col_pop, y_col_pop]].to_numpy()

            tree = cKDTree(population_coords)
            _, nearest_idx = tree.query(pollutant_coords, k=1)

            df_pollutant_valid["matched_population"] = df_population_valid.iloc[nearest_idx][pop_col].to_numpy()

            self.dialog.pBCSVDataAnalysis.setValue(50)

            # exposure per pollutant attribute (global summary value per attribute)
            exposure_values = {}
            skip_cols = {x_col_p, y_col_p, "_centroid_x", "_centroid_y", "matched_population"}

            for col in df_pollutant_valid.columns:
                if col in skip_cols:
                    continue
                if not pd.api.types.is_numeric_dtype(df_pollutant_valid[col]):
                    continue

                values = df_pollutant_valid[col].astype(float).dropna()
                pops = df_pollutant_valid.loc[values.index, "matched_population"].astype(float)

                if len(values) == 0:
                    self.dialog.tELog.append(f"WARNING: No numeric data for '{col}', skipping.")
                    continue

                try:
                    if method == "Direct Proportional":
                        exposure_val = values.mean()

                    elif method == "Weighted by Population":
                        wsum = np.sum(pops)
                        if wsum > 0:
                            exposure_val = np.average(values, weights=pops)
                        else:
                            exposure_val = values.mean()

                    elif method == "Time Weighted Exposure":
                        wsum = np.sum(pops)
                        if wsum > 0 and len(values) > 0:
                            exposure_val = (values * pops).sum() / (wsum * len(values))
                        else:
                            exposure_val = values.mean()

                    else:
                        exposure_val = values.mean()

                except Exception as e:
                    self.dialog.tELog.append(
                        f"WARNING: Failed computing exposure for '{col}': {e}"
                    )
                    exposure_val = np.nan

                exposure_values[col] = round(exposure_val, 5) if pd.notna(exposure_val) else None

            if not exposure_values:
                self.dialog.tELog.append(
                    "ERROR: No exposure values computed. Check columns/types."
                )
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "No exposure values were computed. Check data types."
                )
                return

            self.dialog.pBCSVDataAnalysis.setValue(80)

            # build MGWR-style dataframe per point
            out_df = pd.DataFrame()
            out_df["x"] = df_pollutant_valid[x_col_p].values
            out_df["y"] = df_pollutant_valid[y_col_p].values
            out_df["population_nearest"] = df_pollutant_valid["matched_population"].values

            for col_name, val in exposure_values.items():
                out_df[f"exposure_{col_name}"] = val

            crs = pollutant_layer.crs()
            csv_path, gpkg_path = self.save_results_as_csv_and_gpkg(
                base_name="exposure_analysis",
                df_result=out_df,
                crs=crs,
                output_folder=folder,
                x_col="x",
                y_col="y"
            )

            self.dialog.pBCSVDataAnalysis.setValue(100)
            self.dialog.tELog.append(
                f"INFO: Exposure analysis exported:\nCSV: {csv_path}\nGPKG: {gpkg_path}"
            )

        except Exception as e:
            import traceback
            self.dialog.tELog.append(f"ERROR: Exposure analysis failed: {e}")
            self.dialog.tELog.append(f"DEBUG: Full traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self.dialog,
                "Error",
                f"Exposure analysis failed: {str(e)}"
            )
            self.dialog.pBCSVDataAnalysis.setValue(0)

    # -------------------------------------------------------------------------
    # Feature selection (Spearman correlation thresh)
    # -------------------------------------------------------------------------
    def run_feature_selection(self):
        """
        Compute Spearman correlation between every numeric predictor and the chosen target.
        Keep only predictors with |rho| >= threshold.
        Export as CSV and load back into QGIS (table only, no geometry).
        """
        try:
            self.dialog.pBCSVDataAnalysis.setValue(0)

            if not self.dialog.cBEnableFeatureSelectionByCorrelation.isChecked():
                self.dialog.tELog.append("INFO: Feature selection by correlation is disabled.")
                QMessageBox.information(
                    self.dialog,
                    "Info",
                    "Feature selection by correlation is not enabled. Please check the enable checkbox."
                )
                return

            target_column = self.dialog.cBTargetVariableFeatureSelection.currentText()
            threshold = self.dialog.dSBSpearmanThreshold.value()

            if not target_column:
                self.dialog.tELog.append("ERROR: No target variable selected.")
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "Please select a target variable for feature selection."
                )
                return

            folder = QFileDialog.getExistingDirectory(
                self.dialog,
                "Select Output Folder for Feature Selection Results"
            )
            if not folder:
                self.dialog.tELog.append(
                    "WARNING: No output folder selected for feature selection."
                )
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "No output folder selected."
                )
                return

            self.dialog.tELog.append(
                f"INFO: Running feature selection using Spearman |ρ| ≥ {threshold} "
                f"with target '{target_column}'."
            )
            self.dialog.pBCSVDataAnalysis.setValue(10)

            # ensure df is loaded
            if self.current_csv_df is None:
                self.dialog.tELog.append("ERROR: No cached CSV dataframe. Trying reload from layer...")
                index = self.dialog.cBSelectRefinedCSVLayer.currentIndex()
                if index >= 0:
                    layer = self.dialog.cBSelectRefinedCSVLayer.itemData(index)
                    if layer:
                        try:
                            self.current_csv_df = self.get_layer_dataframe(layer)
                            self.dialog.tELog.append(
                                "INFO: Reloaded dataframe from layer for feature selection."
                            )
                        except Exception as e:
                            self.dialog.tELog.append(
                                f"ERROR: Failed to reload dataframe from layer: {e}"
                            )
                            QMessageBox.critical(
                                self.dialog,
                                "Error",
                                "No CSV data available. Please select and load a CSV layer first."
                            )
                            return
                else:
                    QMessageBox.critical(
                        self.dialog,
                        "Error",
                        "No CSV data available. Please select and load a CSV layer first."
                    )
                    return

            df = self.current_csv_df

            if target_column not in df.columns:
                available_cols = list(df.columns)
                self.dialog.tELog.append(
                    f"ERROR: Target column '{target_column}' not in dataframe."
                )
                self.dialog.tELog.append(f"DEBUG: Available columns: {available_cols}")
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    f"Target column '{target_column}' not found in the dataset."
                )
                return

            if not pd.api.types.is_numeric_dtype(df[target_column]):
                self.dialog.tELog.append(
                    f"ERROR: Target '{target_column}' must be numeric."
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    f"Target variable '{target_column}' must be numeric for correlation analysis."
                )
                return

            self.dialog.pBCSVDataAnalysis.setValue(20)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != target_column]

            if not numeric_cols:
                self.dialog.tELog.append(
                    "ERROR: No numeric columns found for correlation analysis."
                )
                QMessageBox.warning(
                    self.dialog,
                    "Warning",
                    "No numeric columns found for correlation analysis."
                )
                return

            self.dialog.tELog.append(
                f"INFO: Found {len(numeric_cols)} numeric columns for correlation."
            )
            self.dialog.pBCSVDataAnalysis.setValue(30)

            results = []
            target_series = df[target_column].dropna()
            if len(target_series) == 0:
                self.dialog.tELog.append(
                    f"ERROR: Target column '{target_column}' has no valid data."
                )
                QMessageBox.critical(
                    self.dialog,
                    "Error",
                    f"Target column '{target_column}' contains no valid data."
                )
                return

            self.dialog.tELog.append(
                f"INFO: Correlating against {len(target_series)} valid target values."
            )

            for i, col in enumerate(numeric_cols):
                try:
                    pairs = df[[target_column, col]].dropna()
                    if len(pairs) < 3:
                        self.dialog.tELog.append(
                            f"WARNING: '{col}' skipped (only {len(pairs)} valid pairs)."
                        )
                        continue

                    corr, p_value = stats.spearmanr(pairs[target_column], pairs[col])

                    if abs(corr) >= threshold:
                        results.append(
                            (col, round(corr, 4), round(p_value, 4), len(pairs))
                        )
                        self.dialog.tELog.append(
                            f"INFO: '{col}' -> ρ={corr:.4f}, p={p_value:.4f}"
                        )
                    else:
                        self.dialog.tELog.append(
                            f"DEBUG: '{col}' below threshold (ρ={corr:.4f})"
                        )

                except Exception as e:
                    self.dialog.tELog.append(
                        f"WARNING: Failed correlation for '{col}': {e}"
                    )

                progress = 30 + (i / max(len(numeric_cols), 1)) * 30
                self.dialog.pBCSVDataAnalysis.setValue(int(progress))

            self.dialog.pBCSVDataAnalysis.setValue(60)

            if not results:
                self.dialog.tELog.append(
                    f"INFO: No features with |ρ| ≥ {threshold} for target '{target_column}'."
                )
                QMessageBox.information(
                    self.dialog,
                    "Info",
                    f"No features found with correlation ≥ {threshold}. "
                    f"Try lowering the threshold."
                )
                self.dialog.pBCSVDataAnalysis.setValue(100)
                return

            result_df = pd.DataFrame(
                results,
                columns=["Attribute", "SpearmanR", "P_Value", "Valid_Pairs"]
            )
            result_df = result_df.sort_values('SpearmanR', key=abs, ascending=False)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = os.path.join(folder, f"feature_selection_{timestamp}.csv")
            result_df.to_csv(out_csv, index=False)

            self.dialog.tELog.append(
                f"INFO: Feature selection complete. {len(results)} features kept."
            )
            self.dialog.tELog.append(f"INFO: Results saved to {out_csv}")

            # load feature selection results into QGIS (table only)
            try:
                csv_norm = os.path.normpath(out_csv).replace("\\", "/")
                if len(csv_norm) > 1 and csv_norm[1] == ':':
                    csv_norm = '/' + csv_norm
                uri = f"file://{csv_norm}?delimiter=,"
                vlayer = QgsVectorLayer(uri, f"FeatureSelection_{timestamp}", "delimitedtext")
                if vlayer.isValid():
                    QgsProject.instance().addMapLayer(vlayer)
                    self.dialog.tELog.append(
                        "INFO: Feature selection results loaded into QGIS."
                    )
                else:
                    err_msg = vlayer.error().summary() if vlayer.error() else "Unknown error"
                    self.dialog.tELog.append(
                        f"WARNING: Could not load feature selection layer into QGIS: {err_msg}"
                    )
            except Exception as e:
                self.dialog.tELog.append(
                    f"WARNING: Could not load feature selection layer into QGIS: {e}"
                )

            self.dialog.pBCSVDataAnalysis.setValue(100)

            QMessageBox.information(
                self.dialog,
                "Success",
                (
                    "Feature selection completed!\n\n"
                    f"Selected {len(results)} features with |ρ| ≥ {threshold}\n"
                    f"Results saved to: {os.path.basename(out_csv)}"
                )
            )

        except Exception as e:
            import traceback
            self.dialog.tELog.append(f"ERROR: Failed to run feature selection: {e}")
            self.dialog.tELog.append(f"DEBUG: Full traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self.dialog,
                "Error",
                f"Feature selection failed: {str(e)}"
            )
            self.dialog.pBCSVDataAnalysis.setValue(0)

    # -------------------------------------------------------------------------
    # Populate population column combo (helper for exposure UI)
    # -------------------------------------------------------------------------
    def populate_population_columns(self):
        # Some UIs have this combobox, some don't
        if not hasattr(self.dialog, 'cBPopulationDatasetExposureAnalysis'):
            return

        self.dialog.cBPopulationDatasetExposureAnalysis.clear()

        index = self.dialog.cBSelectCSVLayerforExposureAnalysis.currentIndex()
        layer = self.dialog.cBSelectCSVLayerforExposureAnalysis.itemData(index)
        if not layer:
            return

        numeric_names = [f.name() for f in layer.fields() if f.isNumeric()]
        self.dialog.cBPopulationDatasetExposureAnalysis.addItems(numeric_names)
        self.dialog.tELog.append(
            f"INFO: Populated population columns from '{layer.name()}' with numeric fields: {numeric_names}"
        )

    # -------------------------------------------------------------------------
    # Refresh layer list in combo
    # -------------------------------------------------------------------------
    def refresh_layers_spatial_metrics(self):
        try:
            combo = self.dialog.cBSelectRefinedCSVLayer
            prev_name = combo.currentText()

            layers = []
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsVectorLayer) and lyr.dataProvider().name() in ["delimitedtext", "ogr"]:
                    layers.append(lyr)
            layers.sort(key=lambda l: l.name().lower())

            combo.blockSignals(True)
            combo.clear()
            for lyr in layers:
                combo.addItem(lyr.name(), lyr)
            combo.blockSignals(False)

            if prev_name:
                idx = combo.findText(prev_name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                elif combo.count() > 0:
                    combo.setCurrentIndex(0)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

            count = combo.count()
            self.dialog.tELog.append(
                f"INFO: Refreshed CSV/GPKG layers: found {count} layer(s)."
            )

            self.populate_fields_list()

        except Exception as e:
            self.dialog.tELog.append(
                f"ERROR: Failed to refresh refined CSV layers: {e}"
            )

    # -------------------------------------------------------------------------
    # Short logger wrapper
    # -------------------------------------------------------------------------
    def log(self, message):
        self.dialog.tELog.append(message)
