# -*- coding: utf-8 -*-
"""
Data Import functionality for the XGeoAI Pollution Analyst plugin.
"""

import logging
from qgis.core import QgsProject, QgsMapLayer
from PyQt5.QtWidgets import QMessageBox, QListWidgetItem
from PyQt5.QtCore import Qt


class DataImportManager:
    """
    Manages the Data Import tab functionality.
    """

    def __init__(self, pollutant_list_widget, landuse_list_widget):
        """
        Initializes the manager with the necessary UI elements.

        :param pollutant_list_widget: QListWidget for pollutant data type.
        :param landuse_list_widget: QListWidget for land use data.
        """
        self.pollutant_list_widget = pollutant_list_widget
        self.landuse_list_widget = landuse_list_widget
        self.selected_pollutant_layers = []
        self.selected_landuse_layers = []

    def populate_layer_lists(self):
        """
        Populates the QListWidgets with raster and vector layers from the QGIS project.
        """
        logging.info("Populating pollutant and land use layer lists.")
        try:
            self.pollutant_list_widget.clear()
            self.landuse_list_widget.clear()

            layers = QgsProject.instance().mapLayers().values()

            for layer in layers:
                if layer.type() == QgsMapLayer.RasterLayer:
                    self._add_layer_to_list(self.pollutant_list_widget, layer)
                elif layer.type() == QgsMapLayer.VectorLayer:
                    self._add_layer_to_list(self.landuse_list_widget, layer)

            logging.info("Layer lists populated successfully.")
        except Exception as e:
            logging.error(f"Error populating layer lists: {e}")
            QMessageBox.critical(None, "Error", f"Failed to populate layer lists: {e}")

    def _add_layer_to_list(self, list_widget, layer):
        """
        Adds a layer to a QListWidget with a checkbox.

        :param list_widget: QListWidget to populate.
        :param layer: QgsMapLayer to add.
        """
        try:
            item = QListWidgetItem(layer.name())
            item.setCheckState(Qt.Unchecked)
            list_widget.addItem(item)
            logging.debug(f"Added layer '{layer.name()}' to the list widget.")
        except Exception as e:
            logging.error(f"Error adding layer '{layer.name()}' to the list widget: {e}")

    def load_selected_layers(self):
        """
        Loads the user-selected layers into the plugin memory.

        :return: Tuple of selected pollutant and land use layers.
        """
        logging.info("Loading selected layers.")
        try:
            # Load selected pollutant layers
            self.selected_pollutant_layers = []
            for i in range(self.pollutant_list_widget.count()):
                item = self.pollutant_list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    layers = QgsProject.instance().mapLayersByName(item.text())
                    if layers:
                        self.selected_pollutant_layers.append(layers[0])
                        logging.debug(f"Selected pollutant layer: {item.text()}")
                    else:
                        logging.warning(f"Pollutant layer '{item.text()}' not found in the project.")

            # Load selected land use layers
            self.selected_landuse_layers = []
            for i in range(self.landuse_list_widget.count()):
                item = self.landuse_list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    layers = QgsProject.instance().mapLayersByName(item.text())
                    if layers:
                        self.selected_landuse_layers.append(layers[0])
                        logging.debug(f"Selected land use layer: {item.text()}")
                    else:
                        logging.warning(f"Land use layer '{item.text()}' not found in the project.")

            # Ensure layers are selected
            if not self.selected_pollutant_layers or not self.selected_landuse_layers:
                QMessageBox.warning(None, "Warning", "Please select at least one layer from each list.")
                logging.warning("User did not select layers from both lists.")
                return [], []

            logging.info("Selected layers loaded successfully.")
            return self.selected_pollutant_layers, self.selected_landuse_layers

        except Exception as e:
            logging.error(f"Error loading selected layers: {e}")
            QMessageBox.critical(None, "Error", f"Failed to load selected layers: {e}")
            return [], []
