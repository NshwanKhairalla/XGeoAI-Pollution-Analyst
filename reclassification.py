import os
import logging
from datetime import datetime
import pandas as pd
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext
)
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QComboBox
from PyQt5.QtCore import Qt

class Reclassification:
    def __init__(self, dialog):
        self.dialog = dialog
        self.standard_classes = [
            'Agricultural',
            'Urban',
            'Water',
            'Natural',
            'Industrial',
            'Forest',
            'Built-up',
            'Bare Soil',
            'Wetlands',
            'Grassland',
            'Wooded',
            'Mining/Extraction',
            'Green Urban Areas',
            'Production Facilities',
            'Communication Networks',
            'Abandoned/Waste Land',
            'Streets',
            'Rice Fields',
            'Corn Fields',
            'Cereal Crops',
            'Water Bodies',
            'Non-Agricultural Green Areas',
            'Urban/Industrial',
            'Meadows',
            'Orchards',
            'Dump Sites',
            'Sports Areas',
            'Arable Land',
            
            # SIARL-specific additions
            'Leguminous Crops',
            'Industrial Crops',
            'Temporary Grasslands',
            'Pastures',
            'Other Annual Crops',
            'Other Permanent Crops',
            'Mixed Crops',
            'Non-Cultivated Areas'
        ]


        self.setup_ui()

    def setup_ui(self):
        self.dialog.cBEnableReclassification.stateChanged.connect(self.toggle_reclassification_ui)
        self.dialog.pBAutoPopulateClassesReclassification.clicked.connect(self.auto_populate_classes)
        self.dialog.cBSelectLandUseLayerReclassification.currentIndexChanged.connect(self.populate_class_fields)
        self.dialog.pBApplyReclassification.clicked.connect(self.apply_reclassification)
        self.dialog.tBChooseFolderDataCleaningReclassificationCSVSave.clicked.connect(self.choose_output_folder)
        self.dialog.pBSaveReclassificationResult.clicked.connect(self.save_reclassification_result)
        self.dialog.pBDataCleaningNext.clicked.connect(self.go_to_time_series_analysis)
        self.populate_csv_layers()
        self.populate_standard_classes()
        self.dialog.pBRefreshLayersofReclassification.clicked.connect(self.refresh_land_use_layer_list)

    def toggle_reclassification_ui(self):
        enabled = self.dialog.cBEnableReclassification.isChecked()
        logging.info(f"Reclassification module {'enabled' if enabled else 'disabled'}.")

        self.dialog.cBSelectLandUseLayerReclassification.setEnabled(enabled)
        self.dialog.cBSelectClassFieldReclassification.setEnabled(enabled)
        self.dialog.cBStandarizedClassDropdown.setEnabled(enabled)
        self.dialog.pBAutoPopulateClassesReclassification.setEnabled(enabled)
        self.dialog.pBApplyReclassification.setEnabled(enabled)
        self.dialog.tBChooseFolderDataCleaningReclassificationCSVSave.setEnabled(enabled)
        self.dialog.pBSaveReclassificationResult.setEnabled(enabled)
        self.dialog.tWReclassificationTable.setEnabled(enabled)
        self.dialog.pBDataCleaning.setEnabled(enabled)

    def populate_csv_layers(self):
        self.dialog.cBSelectLandUseLayerReclassification.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if layer.type() == QgsVectorLayer.VectorLayer and layer.dataProvider().name() == 'delimitedtext':
                self.dialog.cBSelectLandUseLayerReclassification.addItem(layer.name())
        logging.info("Populated CSV layers in land use selection combo box.")

    def populate_class_fields(self):
        layer_name = self.dialog.cBSelectLandUseLayerReclassification.currentText()
        if not layer_name:
            return

        layer = QgsProject.instance().mapLayersByName(layer_name)[0]
        self.dialog.cBSelectClassFieldReclassification.clear()
        fields = layer.fields()
        for field in fields:
            self.dialog.cBSelectClassFieldReclassification.addItem(field.name())
        logging.info(f"Populated fields for layer {layer_name}.")

    def populate_standard_classes(self):
        self.dialog.cBStandarizedClassDropdown.clear()
        self.dialog.cBStandarizedClassDropdown.addItems(self.standard_classes)

    def auto_populate_classes(self):
          self.dialog.pBDataCleaning.setValue(0)
          layer_name = self.dialog.cBSelectLandUseLayerReclassification.currentText()
          field_name = self.dialog.cBSelectClassFieldReclassification.currentText()
          logging.info(f"Auto-populate triggered for layer: {layer_name}, field: {field_name}")

          if not layer_name or not field_name:
               QMessageBox.warning(self.dialog, "Warning", "Please select a land use layer and a class field.")
               return

          layer = QgsProject.instance().mapLayersByName(layer_name)[0]
          logging.info(f"Layer found: {layer.name()} with {layer.featureCount()} features.")

          unique_values = set()
          total_features = layer.featureCount()

          for i, feature in enumerate(layer.getFeatures()):
               try:
                    val = feature[field_name]
                    logging.debug(f"Feature {i}: {field_name} = {val}")
                    unique_values.add(str(val))
               except Exception as e:
                    logging.error(f"Error reading feature {i}: {e}")

               if i % 10 == 0 and total_features > 0:
                    progress = int(i / total_features * 100)
                    self.dialog.pBDataCleaning.setValue(progress)

          logging.info(f"Unique values found: {unique_values}")

          table = self.dialog.tWReclassificationTable
          table.setColumnCount(3)
          table.setHorizontalHeaderLabels(['Original Class', 'New Class', 'Actions'])
          table.setRowCount(0)
          for row_idx, val in enumerate(unique_values):
               table.insertRow(row_idx)
               table.setItem(row_idx, 0, QTableWidgetItem(val))
               combo = QComboBox()
               combo.addItems(self.standard_classes)
               table.setCellWidget(row_idx, 1, combo)
               action_item = QTableWidgetItem("-")
               action_item.setFlags(action_item.flags() & ~Qt.ItemIsEditable)
               table.setItem(row_idx, 2, action_item)

          self.dialog.pBDataCleaning.setValue(100)
          logging.info(f"Auto-populated {len(unique_values)} unique classes from field {field_name}.")


    def apply_reclassification(self):
          self.dialog.pBDataCleaning.setValue(0)
          layer_name = self.dialog.cBSelectLandUseLayerReclassification.currentText()
          field_name = self.dialog.cBSelectClassFieldReclassification.currentText()
          if not layer_name or not field_name:
               QMessageBox.warning(self.dialog, "Warning", "Please select a land use layer and a class field.")
               return

          layer = QgsProject.instance().mapLayersByName(layer_name)[0]
          crs = layer.crs()

          # Fix: get only the CSV path before any query parameters
          path = layer.dataProvider().dataSourceUri().split('?')[0]
          logging.info(f"Reading CSV from: {path}")
          df = pd.read_csv(path)

          if 'X' not in df.columns or 'Y' not in df.columns:
               QMessageBox.critical(self.dialog, "Error", "The selected CSV must contain X and Y columns.")
               return

          table = self.dialog.tWReclassificationTable
          mapping = {}
          for row in range(table.rowCount()):
               item = table.item(row, 0)
               combo = table.cellWidget(row, 1)
               if item is None or combo is None:
                    logging.warning(f"Row {row}: Skipping because of missing data.")
                    continue
               original = item.text()
               new_class = combo.currentText()
               mapping[original] = new_class

          timestamp = datetime.now().strftime('%Y%m%d_%H%M')
          new_field = f'LU_RECLASS_{timestamp}'

          total_rows = len(df)
          new_column = []
          for idx, val in enumerate(df[field_name]):
               mapped_value = mapping.get(str(val), val)
               new_column.append(mapped_value)
               if idx % 100 == 0:
                    progress = int(idx / total_rows * 100)
                    self.dialog.pBDataCleaning.setValue(progress)

          df[new_field] = new_column
          
          # Add global land use fraction columns based on area
          if 'area' not in df.columns:
               QMessageBox.critical(self.dialog, "Error", "The dataset must contain an 'area' column for fraction calculation.")
               return

          total_area = df['area'].sum()
          class_area_totals = df.groupby(new_field)['area'].sum()
          for lu_class, class_area in class_area_totals.items():
               frac_col = f"frac_{lu_class.replace(' ', '_')}"
               frac_value = class_area / total_area if total_area > 0 else 0
               df[frac_col] = frac_value
          logging.info("Added global land use fraction columns based on area.")
          
          # Add per-row land use area fraction columns
          class_area_totals = df.groupby(new_field)['area'].sum()
          unique_classes = class_area_totals.index.tolist()

          for lu_class in unique_classes:
               col_name = f"row_frac_{lu_class.replace(' ', '_')}"
               total_class_area = class_area_totals[lu_class]
               df[col_name] = df.apply(
                   lambda row: row["area"] / total_class_area if row[new_field] == lu_class and total_class_area > 0 else 0,
                   axis=1
               )
          logging.info("Added per-row land use area fraction columns (row_frac_*) based on class area totals.")

          self.df_reclassified = df
          self.crs_reclassified = crs

          self.dialog.pBDataCleaning.setValue(100)
          logging.info(f"Applied reclassification and created new column: {new_field}.")
          QMessageBox.information(self.dialog, "Success", f"Reclassification applied. New column: {new_field}")


    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderDataCleaningReclassificationCSVSave.setText(folder)
            logging.info(f"Selected output folder: {folder}")

    def save_reclassification_result(self):
        self.dialog.pBDataCleaning.setValue(0)
        output_folder = self.dialog.tBChooseFolderDataCleaningReclassificationCSVSave.text()
        if not hasattr(self, 'df_reclassified'):
            QMessageBox.warning(self.dialog, "Warning", "Please apply reclassification before saving.")
            return
        if not output_folder or not os.path.exists(output_folder):
            QMessageBox.warning(self.dialog, "Warning", "Please choose a valid output folder.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_name = f'reclassified_{timestamp}.csv'
        output_path = os.path.join(output_folder, output_name)

        self.df_reclassified.to_csv(output_path, index=False)
        logging.info(f"Saved reclassified CSV to {output_path}.")

        uri = f"file:///{output_path}?delimiter=,&xField=X&yField=Y"
        layer = QgsVectorLayer(uri, f"Reclassified_{timestamp}", "delimitedtext")
        if layer.isValid():
            layer.setCrs(self.crs_reclassified)
            QgsProject.instance().addMapLayer(layer)
            logging.info("Loaded reclassified CSV as QGIS point layer.")
            QMessageBox.information(self.dialog, "Success", f"Saved and loaded reclassified CSV: {output_name}")
        else:
            QMessageBox.warning(self.dialog, "Warning", "Saved CSV but failed to load into QGIS.")

        self.dialog.pBDataCleaning.setValue(100)


    def go_to_time_series_analysis(self):
        logging.info("Switching to Time Series Analysis tab.")
        index = self.dialog.tabsXgeoAi.indexOf(self.dialog.tTimeSeriesAnalysis)
        if index != -1:
            self.dialog.tabsXgeoAi.setCurrentIndex(index)
            logging.info("Successfully switched to Time Series Analysis tab.")
        else:
            logging.error("Time Series Analysis tab not found.")
            QMessageBox.warning(self.dialog, "Error", "Time Series Analysis tab not found.")


    def refresh_land_use_layer_list(self):
        self.dialog.cBSelectLandUseLayerReclassification.clear()
        layers = QgsProject.instance().mapLayers().values()

        count = 0
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                self.dialog.cBSelectLandUseLayerReclassification.addItem(layer.name())
                count += 1

        logging.info(f"Refreshed land use layers list with {count} vector layer(s) from the QGIS layer browser.")
        QMessageBox.information(self.dialog, "Layers Refreshed", f"{count} vector layers loaded from the QGIS project.")