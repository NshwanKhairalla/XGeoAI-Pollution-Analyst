import logging
import os
import shutil
import tempfile
import numpy as np
from datetime import datetime
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer
)
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtCore import Qt
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping

class ClipPollutantLayer:
    def __init__(self, dialog):
        """
        Initialize the ClipPollutantLayer class with the main dialog.

        Args:
            dialog: The main dialog of the plugin.
        """
        self.dialog = dialog

    def populate_list_widgets_clip_pollutant(self):
        """
        Populate the QListWidgets with raster and vector layers from the QGIS layers browser.
        Each layer is added with a checkbox (checked by default).
        """
        self.dialog.lWInputPollutantRasterClip.clear()
        self.dialog.lWInputLandUseVectorClip.clear()

        layers = QgsProject.instance().mapLayers().values()

        # Populate raster layers with checkboxes
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                item = QListWidgetItem(layer.name())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, layer)  # Store reference to layer
                self.dialog.lWInputPollutantRasterClip.addItem(item)

        # Populate vector layers with checkboxes
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                item = QListWidgetItem(layer.name())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, layer)  # Store reference to layer
                self.dialog.lWInputLandUseVectorClip.addItem(item)

    def choose_output_folder_clip_pollutant(self):
        """
        Open a dialog to choose the output folder for clipped layers.
        """
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderDataRefinementClipPollutantLayer.setText(folder)

    def clip_pollutant_layers(self):
        """
        Crop and clip raster to the geometry of selected vector mask layers.
        Adds clipped rasters to QGIS and preserves metadata and sidecar files.
        """
        import shutil
        from datetime import datetime
        import geopandas as gpd
        from shapely.geometry import mapping
        import rasterio
        from rasterio.mask import mask
        from qgis.core import QgsProject, QgsRasterLayer
        from PyQt5.QtWidgets import QMessageBox

        selected_rasters = [item.text() for item in self.dialog.lWInputPollutantRasterClip.selectedItems()]
        selected_vectors = [item.text() for item in self.dialog.lWInputLandUseVectorClip.selectedItems()]

        if not selected_rasters or not selected_vectors:
            QMessageBox.warning(self.dialog, "Warning", "Please select at least one raster and one vector layer.")
            return

        output_folder = self.dialog.tBChooseFolderDataRefinementClipPollutantLayer.text()
        if not output_folder or not os.path.exists(output_folder):
            QMessageBox.warning(self.dialog, "Warning", "Please select a valid output folder.")
            return

        if not os.access(output_folder, os.W_OK):
            self.log_message(f"No write permissions for output folder: {output_folder}", level="error")
            return

        temp_dir = os.path.join(output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.log_message(f"Using temporary directory: {temp_dir}", level="info")

        self.dialog.pBDataRefinement.setValue(0)
        total_steps = len(selected_rasters) * len(selected_vectors)
        current_step = 0

        for raster_name in selected_rasters:
            raster_layer = QgsProject.instance().mapLayersByName(raster_name)[0]
            raster_path = raster_layer.source()

            try:
                with rasterio.open(raster_path) as src:
                    raster_crs = src.crs
                    raster_nodata = src.nodata if src.nodata is not None else -9999
                    raster_meta = src.meta.copy()
                self.log_message(f"Raster layer {raster_name} loaded successfully.", level="info")
            except Exception as e:
                self.log_message(f"Error opening raster file {raster_name}: {e}", level="error")
                continue

            for vector_name in selected_vectors:
                vector_layer = QgsProject.instance().mapLayersByName(vector_name)[0]
                vector_path = vector_layer.source()

                try:
                    self.check_vector_layer_validity(vector_path)
                except Exception as e:
                    self.log_message(f"Error with vector layer {vector_name}: {e}", level="error")
                    continue

                vector_crs = vector_layer.crs().authid()
                if raster_crs.to_string() != vector_crs:
                    self.log_message(f"CRS mismatch detected. Reprojecting vector layer: {vector_name}", level="warning")
                    try:
                        reprojected_path = os.path.join(
                            temp_dir, f"{vector_name}_reprojected_{datetime.now().strftime('%Y%m%d%H%M%S')}.shp"
                        )
                        self.reproject_vector(vector_path, reprojected_path, raster_crs)
                        vector_path = reprojected_path
                        vector_layer = QgsVectorLayer(vector_path, f"{vector_name}_reprojected", "ogr")
                        if not vector_layer.isValid():
                            raise Exception("Failed to load reprojected vector.")
                    except Exception as e:
                        self.log_message(f"Error reprojecting vector layer {vector_name}: {e}", level="error")
                        continue

                try:
                    vector_data = gpd.read_file(vector_path)
                    shapes = [mapping(geom) for geom in vector_data.geometry]

                    with rasterio.open(raster_path) as src:
                        clipped_data, clipped_transform = mask(
                            src, shapes, crop=True, nodata=raster_nodata, filled=True
                        )
                        clipped_meta = src.meta.copy()
                        clipped_meta.update({
                            "height": clipped_data.shape[1],
                            "width": clipped_data.shape[2],
                            "transform": clipped_transform,
                            "nodata": raster_nodata,
                            "count": src.count,
                            "driver": "GTiff"
                        })

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    clipped_raster_path = os.path.join(
                        output_folder, f"{raster_name}_clipped_{vector_name}_{timestamp}.tif"
                    )

                    with rasterio.open(clipped_raster_path, "w", **clipped_meta) as dest:
                        dest.write(clipped_data)

                    self.log_message(f"Clipped {raster_name} using {vector_name}. Output saved to {clipped_raster_path}", level="info")

                    # Copy metadata sidecar
                    aux_path = raster_path + ".aux.xml"
                    aux_dst = clipped_raster_path + ".aux.xml"
                    if os.path.exists(aux_path):
                        shutil.copy(aux_path, aux_dst)
                        self.log_message(f"Copied aux.xml metadata sidecar: {aux_dst}", level="info")

                    # Copy timestamps sidecar
                    self.copy_timestamp_csv_for_clipped_raster(raster_path, clipped_raster_path)

                    # Add clipped raster to QGIS
                    clipped_layer = QgsRasterLayer(clipped_raster_path, f"{raster_name}_clipped_{vector_name}_{timestamp}")
                    if clipped_layer.isValid():
                        QgsProject.instance().addMapLayer(clipped_layer)
                        self.log_message(f"Added clipped layer to QGIS: {clipped_layer.name()}", level="info")
                    else:
                        raise Exception("Clipped raster is invalid.")

                    current_step += 1
                    self.dialog.pBDataRefinement.setValue(int((current_step / total_steps) * 100))

                except Exception as e:
                    self.log_message(f"Error processing {raster_name} with {vector_name}: {e}", level="error")

        # Cleanup and UI refresh
        self.populate_list_widgets_clip_pollutant()
        self.dialog.pBDataRefinement.setValue(0)
        self.log_message("Clipping process completed.", level="info")
        QMessageBox.information(self.dialog, "Success", "Clipping process completed.")

    def copy_timestamp_csv_for_clipped_raster(self, original_raster_path, clipped_raster_path):
        """
        Copies the timestamp CSV sidecar alongside clipped raster.
        """
        try:
            original_csv = os.path.splitext(original_raster_path)[0] + "_timestamps.csv"
            clipped_csv = os.path.splitext(clipped_raster_path)[0] + "_timestamps.csv"

            if os.path.exists(original_csv):
                shutil.copyfile(original_csv, clipped_csv)
                self.log_message(f"Copied timestamps CSV: {clipped_csv}", level="info")
            else:
                self.log_message(f"No timestamps CSV found for: {original_csv}", level="warning")
        except Exception as e:
            self.log_message(f"Error copying timestamp CSV: {e}", level="error")


    def check_vector_layer_validity(self, vector_path):
        """
        Check if the vector layer is valid and contains polygons.

        Args:
            vector_path: Path to the vector layer.
        """
        vector_data = gpd.read_file(vector_path)
        if vector_data.empty:
            raise Exception("Vector layer has no features.")

        if not all(vector_data.geometry.type == "Polygon"):
            raise Exception("Vector layer must contain polygons.")

        feature_count = len(vector_data)
        self.log_message(f"Vector layer {vector_path} is valid and contains {feature_count} polygons.", level="info")

    def reproject_vector(self, input_path, output_path, target_crs):
        """
        Reproject a vector layer to the target CRS using GeoPandas.
        """
        vector_data = gpd.read_file(input_path)
        vector_data = vector_data.to_crs(target_crs)
        vector_data.to_file(output_path)
        self.log_message(f"Reprojected vector layer saved to {output_path}", level="info")

    def log_message(self, message, level="info"):
        """
        Log messages to the QTextEdit in the Log tab.

        Args:
            message: The message to log.
            level: The log level (e.g., "info", "error").
        """
        if level == "error":
            self.dialog.tELog.append(f"<font color='red'>ERROR: {message}</font>")
        else:
            self.dialog.tELog.append(f"INFO: {message}")

import logging
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.PyQt.QtWidgets import QMessageBox

def refresh_clip_pollutant_ui(dialog):
    """
    Refresh the UI for Data Spatial Refinement step by repopulating the raster and vector list widgets.

    This function reloads:
    - Input Pollutant Raster: lWInputPollutantRasterClip
    - Input Land Use Vector: lWInputLandUseVectorClip
    using the layers available in the QGIS Layers Browser.
    """
    try:
        logging.info("Refreshing Spatial Refinement layer lists...")

        # Clear existing items
        dialog.lWInputPollutantRasterClip.clear()
        dialog.lWInputLandUseVectorClip.clear()

        layers = QgsProject.instance().mapLayers().values()

        # Populate raster layers
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                dialog.lWInputPollutantRasterClip.addItem(layer.name())
                logging.debug(f"Added raster layer: {layer.name()}")

        # Populate vector layers
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                dialog.lWInputLandUseVectorClip.addItem(layer.name())
                logging.debug(f"Added vector layer: {layer.name()}")

        logging.info("Spatial Refinement layer lists refreshed.")

    except Exception as e:
        logging.error(f"Failed to refresh Spatial Refinement layer lists: {e}")
        QMessageBox.critical(dialog, "Error", f"Failed to refresh Spatial Refinement layer lists:\n{e}")

import shutil
import os

def copy_timestamp_csv_for_clipped_raster(original_raster_path, clipped_raster_path):
    """
    Copies the timestamp sidecar CSV from the original raster to the clipped raster location.
    
    Parameters:
    - original_raster_path: Path to the aligned (pre-clipped) raster .tif
    - clipped_raster_path: Path to the clipped raster .tif
    """
    try:
        # Derive paths
        original_csv = os.path.splitext(original_raster_path)[0] + "_timestamps.csv"
        clipped_csv = os.path.splitext(clipped_raster_path)[0] + "_timestamps.csv"

        if os.path.exists(original_csv):
            shutil.copyfile(original_csv, clipped_csv)
            logging.info(f"Copied timestamps CSV to match clipped raster: {clipped_csv}")
        else:
            logging.warning(f"No timestamps CSV found for original raster: {original_csv}")

    except Exception as e:
        logging.error(f"Error copying timestamp CSV: {e}")

