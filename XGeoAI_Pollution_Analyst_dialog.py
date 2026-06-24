# -*- coding: utf-8 -*-
"""
/***************************************************************************
 XGeoAIPollutionAnalystDialog
                                 A QGIS plugin
 Explainable, Geospatial, Artificial Intelligence-Driven Analysis for Pollution Data.
 ***************************************************************************/"""

import os
import logging

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QFileDialog
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QDir
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsProcessingFeedback, QgsProcessingException, Qgis
from qgis import processing
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

from .help_and_instructions import HelpAndInstructions
from .log import LogManager, QTextEditHandler
from .data_import import DataImportManager
from .data_preparation import DataPreparationManager, go_to_data_spatial_refinement_tab
from .clip_pollutant_layer import ClipPollutantLayer, refresh_clip_pollutant_ui
from .vectorization import Vectorization
from .temporal_alignment import populate_temporal_alignment_ui, run_temporal_alignment, populate_time_fields, update_temporal_alignment_progress
from .grid_resampling import GridResampling
from .vector_alignment import VectorAlignment
from .csv_conversion import CSVConversion
from .data_quality_cleaning import DataQualityCleaning
from .reclassification import Reclassification
from .time_series_analysis import TimeSeriesAnalysis
from .spatial_metrics_aggregation import SpatialMetricsAggregation
from .spatial_analysis import EnhancedSpatialAnalysis
from .predictive_modeling import PredictiveModelingController
from .interpret_results import InterpretResults



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'XGeoAI_Pollution_Analyst_dialog_base.ui'))


class XGeoAIPollutionAnalystDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        """Constructor."""
        super(XGeoAIPollutionAnalystDialog, self).__init__(parent)
        self.iface = iface  # Store the iface object
        self.setupUi(self)

        # Ensure progress bars start in a known state
        try:
            if hasattr(self, 'pBDataPreparation') and self.pBDataPreparation:
                self.pBDataPreparation.setRange(0, 100)
                self.pBDataPreparation.setValue(0)
            if hasattr(self, 'pBDataRefinement') and self.pBDataRefinement:
                self.pBDataRefinement.setRange(0, 100)
                self.pBDataRefinement.setValue(0)
        except Exception:
            pass

        # Set help and instructions text
        HelpAndInstructions.set_help_text(self.tEHelpInstructions)

        # Initialize logging
        if hasattr(self, 'tELog') and self.tELog is not None:
            self.log_handler = QTextEditHandler(self.tELog)
            self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(self.log_handler)
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.error("Log widget (tELog) is not initialized.")

        # Log manager actions
        self.pBSaveLog.clicked.connect(lambda: LogManager.save_log(self.tELog))
        self.pBCopyLog.clicked.connect(lambda: LogManager.copy_log(self.tELog))
        self.pBClearLog.clicked.connect(lambda: LogManager.clear_log(self.tELog))

        # Initialize Data Import Manager
        self.data_import_manager = DataImportManager(self.lWPollutantDataType, self.lWLandUseData)
        self.data_import_manager.populate_layer_lists()

        # Initialize Data Preparation Manager
        self.data_preparation_manager = DataPreparationManager(
            self.lWInputPolluatantLayer,
            self.lWInputLandUseLayer,
            self.cBTargetCRSPollutantLayer,
            self.cBTargetCRSLandUseLayer,
            progress_bar=self.pBDataPreparation,                      
            iface=self.iface 
        )

        # Populate CRS comboboxes
        self.data_preparation_manager.populate_crs_comboboxes()

        # Connect buttons
        self.pBBegintheanalysis.clicked.connect(self.go_to_data_import)
        self.pBDataImportNext.clicked.connect(self.go_to_data_preparation)
        self.pBDataPreparationCRSConversion.clicked.connect(self.handle_data_preparation_CRS_Conversion)

        # Initialize the ClipPollutantLayer class
        self.clip_pollutant_layer = ClipPollutantLayer(self)

        # Automatically populate the list widgets with layers from the QGIS layer browser
        self.populate_list_widgets_clip_pollutant()

        # Enable multi-selection for the list widgets
        self.lWInputPollutantRasterClip.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lWInputLandUseVectorClip.setSelectionMode(QAbstractItemView.MultiSelection)

        # Connect buttons to functions
        self.tBChooseFolderDataRefinementClipPollutantLayer.clicked.connect(self.clip_pollutant_layer.choose_output_folder_clip_pollutant)
        self.pBClipPollutantLayer.clicked.connect(self.clip_pollutant_layer.clip_pollutant_layers)
        self.pBRefreshClip.clicked.connect(lambda: refresh_clip_pollutant_ui(self))

        # Initialize logging
        self.tELog.setReadOnly(True)

        # Initialize the Vectorization class with iface
        self.vectorization = Vectorization(
            iface=self.iface,
            lWInputPollutantRasterClip=self.lWInputPollutantRasterClip,
            tBChooseFolderDataRefinementVectorPollutantLayer=self.tBChooseFolderDataRefinementVectorPollutantLayer,
            pBConvertPollutantLayertoVector=self.pBConvertPollutantLayertoVector,
            pBDataRefinement=self.pBDataRefinement,
            log=self.tELog
        )

        # Connect Temporal Alignment logic
        self.cBEnableTemporalAlignment.stateChanged.connect(
            lambda: populate_temporal_alignment_ui(self) if self.cBEnableTemporalAlignment.isChecked() else None
        )
        self.pBSyncTimestampsDataPreparation.clicked.connect(
            lambda: run_temporal_alignment(self)
        )

        self.cBTimeSeriesLayerDataPreparation.currentIndexChanged.connect(
            lambda: populate_time_fields(self)
        )

        # Connect manual base date checkbox to date picker
        self.cBEnableManualBaseDate.stateChanged.connect(
            lambda: self.dEBaseDateTemporalAlignment.setEnabled(self.cBEnableManualBaseDate.isChecked())
        )
        self.dEBaseDateTemporalAlignment.setEnabled(False)

        self.pBNextDataPreparation.clicked.connect(lambda: go_to_data_spatial_refinement_tab(self))

        self.plugin_memory = {
            "grid_resampled_layers": {}
        }

        self.grid_resampling = GridResampling(self)

        self.vector_alignment = VectorAlignment(self)

        # Initialize CSV conversion
        self.csv_conversion = CSVConversion(self)

        # Connect Choose Folder button
        self.tBChooseFolderDataRefinementCSVSave.clicked.connect(self.csv_conversion.choose_output_folder)

        # Connect Convert to CSV button
        self.pBConvertDataCSV.clicked.connect(self.csv_conversion.convert_data_to_csv)

        self.pBDataRefinementNext.clicked.connect(self.go_to_data_quality_cleaning)

        self.data_quality_cleaning = DataQualityCleaning(self)
        self.data_quality_cleaning.initialize()

        # Connect enabling of data cleaning section
        self.cBEnableDataCleaning.toggled.connect(self.data_quality_cleaning.enable_data_cleaning_section)

        # Connect selecting a CSV from layer browser
        self.cBImportCSVFileDataCleaning.currentIndexChanged.connect(self.data_quality_cleaning.load_selected_csv_layer)

        # Connect Save Clean Data Folder button
        self.tBSaveCleanData.clicked.connect(self.data_quality_cleaning.choose_output_folder)

        # Connect the button to clean AND save data
        self.pBSaveCleanDataCSV.clicked.connect(self.data_quality_cleaning.clean_and_save_data)

        self.reclassification_module = Reclassification(self)
        self.time_series_analysis = TimeSeriesAnalysis(self)
        self.tabsXgeoAi.currentChanged.connect(self.on_tab_change)
        self.spatial_metrics_aggregation = SpatialMetricsAggregation(self)
        self.spatial_analysis = EnhancedSpatialAnalysis(self, self.iface)
        self.predictive_ctrl = PredictiveModelingController(self, self.iface)
        self.interpret_ctrl = InterpretResults(self, self.iface)

    def on_tab_change(self, index):
        tab = self.tabsXgeoAi.widget(index)
        if tab == self.tTimeSeriesAnalysis:
            if hasattr(self, 'time_series_analysis'):
                self.time_series_analysis.populate_dataset_combo()

    def closeEvent(self, event):
        if hasattr(self, 'log_handler') and self.log_handler is not None:
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler = None
        super().closeEvent(event)

    def go_to_data_import(self):
        logging.info("Switching to the Data Import tab.")
        try:
            self.tabsXgeoAi.setCurrentWidget(self.tDataImport)
            logging.debug("Successfully switched to the Data Import tab.")
        except Exception as e:
            logging.error(f"Error switching to the Data Import tab: {e}")
            QMessageBox.critical(self, "Error", f"Failed to switch to Data Import tab: {e}")

    def go_to_data_preparation(self):
        logging.info("Switching to the Data Preparation tab.")
        try:
            pollutant_layers, landuse_layers = self.data_import_manager.load_selected_layers()
            if not pollutant_layers or not landuse_layers:
                logging.warning("Layer selection incomplete. Cannot proceed.")
                QMessageBox.warning(self, "Warning", "Please select at least one layer from both lists.")
                return

            self.data_preparation_manager.populate_preparation_layer_lists(pollutant_layers, landuse_layers)
            self.tabsXgeoAi.setCurrentWidget(self.tDataPreparation)
            logging.info("Successfully switched to the Data Preparation tab.")
        except Exception as e:
            logging.error(f"Error switching to the Data Preparation tab: {e}")
            QMessageBox.critical(self, "Error", f"Failed to switch to Data Preparation tab: {e}")

    def handle_data_preparation_CRS_Conversion(self):
        logging.info("Handling Data Preparation CRS Conversion button.")
        try:
            if self.data_preparation_manager.perform_crs_conversion():
                self.data_preparation_manager.load_converted_layers()
                logging.info("CRS conversion completed and layers loaded successfully.")
        except Exception as e:
            logging.error(f"Error handling Data CRS conversion button: {e}")
            QMessageBox.critical(self, "Error", f"Failed to perform CRS conversion and load layers: {e}")

    def populate_list_widgets_clip_pollutant(self):
        self.lWInputPollutantRasterClip.clear()
        self.lWInputLandUseVectorClip.clear()

        layers = QgsProject.instance().mapLayers().values()

        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.lWInputPollutantRasterClip.addItem(layer.name())

        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                self.lWInputLandUseVectorClip.addItem(layer.name())


    def setup_connections(self):
        self.tBChooseFolderDataRefinementCSVSave.clicked.connect(self.csv_conversion.choose_output_folder)
        self.pBConvertDataCS.clicked.connect(self.csv_conversion.convert_data_to_csv)

    def go_to_data_quality_cleaning(self):
        try:
            self.tabsXgeoAi.setCurrentWidget(self.tDataQualityCleaning)
            logging.info("Switched to the Data Quality Cleaning tab.")
            self.iface.messageBar().pushMessage("Info", "Moved to Data Quality Cleaning tab.", level=Qgis.Info)
        except Exception as e:
            logging.exception("Failed to switch to the Data Quality Cleaning tab.")
            self.iface.messageBar().pushMessage("Error", f"Could not switch tabs: {str(e)}", level=Qgis.Critical)