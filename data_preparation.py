
# -*- coding: utf-8 -*-
"""
Data Preparation functionality for the XGeoAI Pollution Analyst plugin.

Patched to:
- Accept and drive the Data Preparation progress bar (pBDataPreparation).
- Provide export_selected_layers_to_csv() that calls csv_conversion.convert_selected_vectors_to_csv
  while updating the same progress bar and using iface for QGIS message bar.
- Restore and harden go_to_data_spatial_refinement_tab(dialog) for navigation from the dialog.
"""

import logging
import os
import tempfile
import subprocess
from datetime import datetime
from typing import List

from osgeo import gdal
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsVectorFileWriter,
    QgsWkbTypes,
    Qgis,
)
from qgis.PyQt.QtWidgets import (
    QMessageBox, QListWidgetItem, QFileDialog, QApplication, QTabWidget
)
from PyQt5.QtCore import Qt

from .reproject_netcdf_with_rioxarray import reproject_netcdf_with_rioxarray, detect_main_variable

__all__ = ["DataPreparationManager", "go_to_data_spatial_refinement_tab"]


class DataPreparationManager:
    """
    Manages the Data Preparation tab functionality.
    """

    def __init__(self, input_pollutant_list_widget, input_landuse_list_widget,
                 target_crs_pollutant_combobox, target_crs_landuse_combobox,
                 progress_bar=None, iface=None):
        self.input_pollutant_list_widget = input_pollutant_list_widget
        self.input_landuse_list_widget = input_landuse_list_widget
        self.target_crs_pollutant_combobox = target_crs_pollutant_combobox
        self.target_crs_landuse_combobox = target_crs_landuse_combobox
        self.selected_pollutant_layers = []
        self.selected_landuse_layers = []
        self.converted_pollutant_layers = []
        self.converted_landuse_layers = []

        # NEW: wiring for progress and message bar
        self.progress_bar = progress_bar
        self.iface = iface

    def _reset_prep_progress(self, value=0):
        """Safely reset/update the Data Preparation progress bar."""
        try:
            if self.progress_bar:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(value))
                QApplication.processEvents()
        except Exception:
            logging.debug("Progress bar not available during reset/update.")

    def _add_layer_to_list(self, list_widget, layer):
        """
        Adds a layer to a QListWidget with a checkbox.

        :param list_widget: QListWidget to populate.
        :param layer: QgsMapLayer to add.
        """
        try:
            item = QListWidgetItem(layer.name())
            item.setCheckState(Qt.Unchecked)
            # Store the actual layer object for retrieval
            item.setData(Qt.UserRole, layer)
            list_widget.addItem(item)
            logging.debug(f"Added layer '{layer.name()}' to list widget.")
        except Exception as e:
            logging.error(f"Error adding layer {layer.name()} to list: {e}")

    def populate_preparation_layer_lists(self, pollutant_layers, landuse_layers):
        """
        Populates the QListWidgets with the selected layers from the Data Import tab.
        """
        logging.info("Populating input layer lists in Data Preparation tab.")
        try:
            self.input_pollutant_list_widget.clear()
            self.input_landuse_list_widget.clear()

            for layer in pollutant_layers:
                self._add_layer_to_list(self.input_pollutant_list_widget, layer)

            for layer in landuse_layers:
                self._add_layer_to_list(self.input_landuse_list_widget, layer)

            logging.info("Layer lists populated successfully.")
        except Exception as e:
            logging.error(f"Error populating layer lists: {e}")
            QMessageBox.critical(None, "Error", f"Failed to populate layer lists: {e}")

    def populate_crs_comboboxes(self):
        """
        Populates the CRS comboboxes with available CRS choices for Europe.
        """
        logging.info("Populating CRS comboboxes.")
        try:
            europe_crs_list = [
                ("32632", "WGS 84 / UTM zone 32N"),
                ("25832", "ETRS89 / UTM zone 32N"),
                ("3035", "ETRS89 / LAEA Europe"),
                ("4326", "WGS 84"),
                ("3857", "WGS 84 / Pseudo-Mercator"),
                ("4258", "ETRS89"),
            ]

            self.target_crs_pollutant_combobox.clear()
            self.target_crs_landuse_combobox.clear()

            for crs_code, crs_name in europe_crs_list:
                self.target_crs_pollutant_combobox.addItem(f"{crs_name} (EPSG:{crs_code})", crs_code)
                self.target_crs_landuse_combobox.addItem(f"{crs_name} (EPSG:{crs_code})", crs_code)

            logging.info("CRS comboboxes populated successfully.")
        except Exception as e:
            logging.error(f"Error populating CRS comboboxes: {e}")
            QMessageBox.critical(None, "Error", f"Failed to populate CRS comboboxes: {e}")

    def perform_crs_conversion(self):
        """
        Performs CRS conversion on the selected layers to the target CRS using GDAL.
        """
        logging.info("Performing CRS conversion.")
        output_folder = QFileDialog.getExistingDirectory(None, "Select Output Folder for Reprojected Layers")
        if not output_folder:
            QMessageBox.warning(None, "Warning", "No output folder selected. Operation cancelled.")
            return False

        self.reprojection_output_folder = output_folder
        try:
            # Get selected layers
            self.selected_pollutant_layers = self._get_selected_layers(self.input_pollutant_list_widget)
            self.selected_landuse_layers = self._get_selected_layers(self.input_landuse_list_widget)

            if not self.selected_pollutant_layers and not self.selected_landuse_layers:
                QMessageBox.warning(None, "Warning", "No layers selected for conversion.")
                return False

            pollutant_target_crs_code = self.target_crs_pollutant_combobox.currentData()
            landuse_target_crs_code = self.target_crs_landuse_combobox.currentData()

            if not pollutant_target_crs_code or not landuse_target_crs_code:
                QMessageBox.warning(None, "Warning", "Please select a target CRS for both pollutant and land use layers.")
                return False

            self.converted_pollutant_layers = []
            self.converted_landuse_layers = []

            # Convert pollutant layers
            for layer in self.selected_pollutant_layers:
                converted_layer = self._convert_raster_crs(layer, pollutant_target_crs_code)
                if converted_layer:
                    self.converted_pollutant_layers.append(converted_layer)

            # Convert land use layers
            for layer in self.selected_landuse_layers:
                converted_layer = self._convert_vector_crs(layer, landuse_target_crs_code)
                if converted_layer:
                    self.converted_landuse_layers.append(converted_layer)

            if not self.converted_pollutant_layers and not self.converted_landuse_layers:
                logging.error("No layers were successfully converted.")
                QMessageBox.critical(None, "Error", "No layers were successfully converted.")
                return False

            logging.info("CRS conversion completed successfully.")
            QMessageBox.information(None, "Success", "CRS conversion completed successfully.")
            return True
        except Exception as e:
            logging.error(f"Error during CRS conversion: {e}")
            QMessageBox.critical(None, "Error", f"Failed to perform CRS conversion: {e}")
            return False

    def _get_selected_layers(self, list_widget):
        """
        Retrieves the selected layers from a QListWidget.

        :param list_widget: QListWidget containing layers.
        :return: List of selected QgsMapLayer objects.
        """
        selected_layers = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.checkState() == Qt.Checked:
                layer = item.data(Qt.UserRole)
                if layer:
                    selected_layers.append(layer)
        return selected_layers

    def _convert_raster_crs(self, layer, target_crs_code):
        """
        Reprojects a NetCDF raster layer to the given CRS using rioxarray and exports as GeoTIFF.
        This ensures compatibility with QGIS and preserves spatial accuracy.
        """
        import logging
        from qgis.core import QgsRasterLayer, QgsCoordinateReferenceSystem
        from .reproject_netcdf_with_rioxarray import reproject_netcdf_with_rioxarray

        logger = logging.getLogger(__name__)
        logger.info(f"Reprojecting NetCDF raster layer '{layer.name()}' to EPSG:{target_crs_code} using GeoTIFF export.")

        try:
            # Determine source file
            original_netcdf_path = layer.customProperty("original_netcdf_path", "") or layer.source()
            if not os.path.exists(original_netcdf_path):
                raise FileNotFoundError(f"Original NetCDF file not found: {original_netcdf_path}")

            logger.info(f"Opening NetCDF file for reprojection: {original_netcdf_path}")

            # Generate a SAFE filename
            output_filename = self._safe_filename(layer.name()) + "_Reprojected.tif"
            full_output_path = os.path.join(self.reprojection_output_folder, output_filename)

            # Now call reproject with correct output_path
            output_path, detected_variable = reproject_netcdf_with_rioxarray(
                original_netcdf_path,
                target_crs=f"EPSG:{target_crs_code}",
                output_path=full_output_path
            )
            if not output_path or not os.path.exists(output_path):
                raise ValueError(f"Reprojection failed or output path does not exist: {output_path}")

            # Load as a raster layer
            new_layer_name = f"{layer.name()} (Reprojected)"
            reprojected_layer = QgsRasterLayer(output_path, new_layer_name, "gdal")
            if not reprojected_layer.isValid():
                raise ValueError(f"Reprojected layer '{new_layer_name}' is invalid.")

            # Set CRS and metadata
            reprojected_layer.setCrs(QgsCoordinateReferenceSystem(f"EPSG:{target_crs_code}"))
            reprojected_layer.setCustomProperty("original_netcdf_path", original_netcdf_path)
            reprojected_layer.setCustomProperty("data_variable_name", detected_variable)
            reprojected_layer.setCustomProperty("source_geotiff_path", output_path)

            logger.info(f"Successfully reprojected NetCDF '{layer.name()}' to '{output_path}' with variable '{detected_variable}'")
            return reprojected_layer

        except Exception as e:
            logger.error(f"Error reprojecting raster layer: {e}", exc_info=True)
            return None

    def _convert_vector_crs(self, vector_layer, target_crs_code):
        """
        Converts a vector layer's CRS using QgsCoordinateTransform.
        """
        logging.info(f"Reprojecting vector layer {vector_layer.name()} to EPSG:{target_crs_code}.")
        try:
            source_crs = vector_layer.crs()
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{target_crs_code}")

            if not source_crs.isValid() or not target_crs.isValid():
                raise ValueError(f"Invalid CRS for vector layer {vector_layer.name()}.")

            # Safe file name for vector output
            safe_layer_name = vector_layer.name().replace(" ", "_").replace("(", "").replace(")", "")
            output_path = os.path.join(self.reprojection_output_folder, f"{safe_layer_name}_reprojected.shp")

            # Get the geometry type of the source layer
            geometry_type = vector_layer.wkbType()

            # Create a new vector layer with the target CRS
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "ESRI Shapefile"
            options.fileEncoding = "UTF-8"
            transform_context = QgsCoordinateTransformContext()
            transform = QgsCoordinateTransform(source_crs, target_crs, transform_context)

            writer = QgsVectorFileWriter.create(
                fileName=output_path,
                fields=vector_layer.fields(),
                geometryType=geometry_type,  # Use the correct geometry type
                srs=target_crs,
                transformContext=transform_context,
                options=options,
            )

            if writer.hasError() != QgsVectorFileWriter.NoError:
                raise IOError(f"Failed to create output file: {output_path}")

            # Reproject features and write to the new layer
            for feature in vector_layer.getFeatures():
                new_feature = QgsFeature(feature)
                new_geometry = feature.geometry()
                if new_geometry:
                    new_geometry.transform(transform)
                    new_feature.setGeometry(new_geometry)
                writer.addFeature(new_feature)

            del writer  # Ensure the writer is closed

            reprojected_layer = QgsVectorLayer(output_path, f"{vector_layer.name()} (Reprojected)", "ogr")
            if not reprojected_layer.isValid():
                raise ValueError(f"Reprojected layer {vector_layer.name()} is invalid.")

            logging.info(f"Vector layer {vector_layer.name()} reprojected to EPSG:{target_crs_code}.")
            return reprojected_layer
        except Exception as e:
            logging.error(f"Error reprojecting vector layer {vector_layer.name()}: {e}")
            return None

    def load_converted_layers(self):
        """
        Loads the converted layers into QGIS.
        """
        logging.info("Loading converted layers into QGIS browser and memory.")
        try:
            project = QgsProject.instance()

            for layer in self.converted_pollutant_layers + self.converted_landuse_layers:
                if layer and layer.isValid():
                    project.addMapLayer(layer)
                    logging.debug(f"Loaded layer '{layer.name()}' into QGIS Layers Browser.")
                else:
                    logging.error(f"Layer '{layer.name() if layer else 'Unknown'}' is invalid and was not added.")

            logging.info("Converted layers loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading converted layers: {e}")
            QMessageBox.critical(None, "Error", f"Failed to load converted layers: {e}")

    def _safe_filename(self, name):
        """
        Makes a safe filename by replacing unsafe characters.
        """
        import re
        name = re.sub(r'[^\w\-_.]', '_', name)  # Replace spaces and weird chars with underscores
        return name

    # -----------------------------
    # NEW: CSV export wiring
    # -----------------------------
    def export_selected_layers_to_csv(self, output_folder: str):
        """
        Run CSV conversion on the currently checked pollutant + land use layers,
        driving the Data Preparation progress bar during the whole process.

        Returns
        -------
        str or None
            Path to the grouped CSV if successful, otherwise None.
        """
        # Lazy import to avoid heavy import on plugin load
        try:
            from . import csv_conversion as csvconv
        except Exception as e:
            logging.error(f"Cannot import csv_conversion module: {e}")
            QMessageBox.critical(None, "Error", f"CSV conversion module not found:\n{e}")
            return None

        # Collect selected layers
        self.selected_pollutant_layers = self._get_selected_layers(self.input_pollutant_list_widget)
        self.selected_landuse_layers = self._get_selected_layers(self.input_landuse_list_widget)

        if not self.selected_pollutant_layers and not self.selected_landuse_layers:
            QMessageBox.warning(None, "Warning", "No layers selected for CSV export.")
            return None

        if not output_folder:
            QMessageBox.warning(None, "Warning", "No output folder selected.")
            return None

        os.makedirs(output_folder, exist_ok=True)

        # Start / reset the Data Preparation progress bar
        self._reset_prep_progress(0)
        try:
            if self.iface is not None:
                self.iface.messageBar().pushMessage("Data Preparation", "Starting CSV export…", level=Qgis.Info)
        except Exception:
            pass

        # Choose the converter entrypoint (function or staticmethod)
        converter = None
        if hasattr(csvconv, "CSVConversion") and hasattr(csvconv.CSVConversion, "convert_selected_vectors_to_csv"):
            converter = csvconv.CSVConversion.convert_selected_vectors_to_csv
        elif hasattr(csvconv, "convert_selected_vectors_to_csv"):
            converter = csvconv.convert_selected_vectors_to_csv

        if converter is None:
            logging.error("No suitable converter found in csv_conversion.py")
            QMessageBox.critical(None, "Error", "CSV conversion entrypoint not found in csv_conversion.py")
            self._reset_prep_progress(100)
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = converter(
                pollutant_layers=self.selected_pollutant_layers,
                landuse_layers=self.selected_landuse_layers,
                output_folder=output_folder,
                progress_bar=self.progress_bar,   # key wiring to pBDataPreparation
                iface=self.iface,
                timestamp=timestamp
            )
            self._reset_prep_progress(100)

            try:
                if self.iface is not None:
                    if result_path and os.path.exists(result_path):
                        self.iface.messageBar().pushMessage("Data Preparation", "CSV export completed.", level=Qgis.Info)
                    else:
                        self.iface.messageBar().pushMessage("Data Preparation", "CSV export finished (no path returned).", level=Qgis.Warning)
            except Exception:
                pass

            return result_path

        except Exception as e:
            logging.exception("CSV export failed")
            self._reset_prep_progress(100)
            try:
                if self.iface is not None:
                    self.iface.messageBar().pushMessage("Data Preparation", f"CSV export failed: {e}", level=Qgis.Critical)
            except Exception:
                pass
            QMessageBox.critical(None, "Error", f"CSV export failed:\n{e}")
            return None


def go_to_data_spatial_refinement_tab(dialog) -> bool:
    """
    Switch the main tab widget to the 'Data Spatial Refinement' tab if present.

    Tries multiple strategies:
      - Known tab widget attributes: tWMain, tabWidget, twMain, tabs
      - Any QTabWidget child of the dialog
      - Match by tab text/objectName containing 'spatial' and 'refine'/'refinement'
      - Fallback to attributes tDataSpatialRefinement / tDataSpatialRefinementTab

    Returns True if a tab was activated, otherwise False.
    """
    logging.info("Switching to the Data Spatial Refinement tab.")
    try:
        # 1) gather candidate tab widgets
        candidates = []
        for name in ("tWMain", "tabWidget", "twMain", "tabs"):
            if hasattr(dialog, name):
                w = getattr(dialog, name)
                if isinstance(w, QTabWidget):
                    candidates.append(w)
        for child in dialog.findChildren(QTabWidget):
            if child not in candidates:
                candidates.append(child)

        if not candidates:
            logging.warning("No QTabWidget found on dialog while navigating to Spatial Refinement tab.")
            return False

        def matches_spatial_refine(widget, tab_text: str) -> bool:
            try:
                obj = widget.objectName() or ""
            except Exception:
                obj = ""
            txt = (tab_text or "").lower()
            obj = obj.lower()
            return (("spatial" in txt and ("refine" in txt or "refinement" in txt))
                    or ("spatial" in obj and "refin" in obj))

        # 2) try to match by tab text/objectName
        for tw in candidates:
            for idx in range(tw.count()):
                w = tw.widget(idx)
                if matches_spatial_refine(w, tw.tabText(idx)):
                    tw.setCurrentIndex(idx)
                    logging.debug("Successfully switched to the Data Spatial Refinement tab (by text/objectName).")
                    return True

        # 3) fallback by direct page attributes
        for attr in ("tDataSpatialRefinement", "tDataSpatialRefinementTab"):
            if hasattr(dialog, attr):
                page = getattr(dialog, attr)
                for tw in candidates:
                    i = tw.indexOf(page)
                    if i != -1:
                        tw.setCurrentIndex(i)
                        logging.debug("Successfully switched to the Data Spatial Refinement tab (by page attr).")
                        return True

        logging.warning("Could not locate 'Data Spatial Refinement' tab.")
        return False

    except Exception as e:
        logging.error(f"Error switching to the Data Spatial Refinement tab: {e}")
        QMessageBox.critical(dialog if dialog else None, "Error",
                             f"Failed to switch to Data Spatial Refinement tab: {e}")
        return False
