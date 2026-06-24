import os
import logging
import pandas as pd
from datetime import datetime
from qgis.core import QgsProject, QgsVectorLayer, Qgis
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

class CSVConversion:
    def __init__(self, dialog):
        self.dialog = dialog
        self.selected_csv_output_folder = None

    def choose_output_folder(self):
        logging.info("Opening folder dialog to choose CSV output folder...")
        folder = QFileDialog.getExistingDirectory(None, "Choose Output Folder")
        if folder:
            folder_name = os.path.basename(folder)
            self.dialog.tBChooseFolderDataRefinementCSVSave.setText(folder_name)
            self.selected_csv_output_folder = folder
            logging.info(f"Output folder selected: {folder}")
        else:
            logging.warning("No folder selected.")

    def get_layer_by_name(self, name):
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == name:
                return layer
        return None

    def convert_data_to_csv(self):
        logging.info("Starting CSV conversion process...")
        output_folder = self.selected_csv_output_folder

        if not output_folder:
            logging.warning("No output folder selected.")
            self.dialog.iface.messageBar().pushMessage("Warning", "Please select an output folder.", level=Qgis.Warning)
            return

        pollutant_layers = []
        landuse_layers = []

        selected_pollutant = self.dialog.lWInputPollutantVectorAlign.selectedItems()
        selected_landuse = self.dialog.lWInputLandUseVectorAlign.selectedItems()

        logging.info(f"Pollutant layers selected: {[item.text() for item in selected_pollutant]}")
        logging.info(f"Land use layers selected: {[item.text() for item in selected_landuse]}")

        for item in selected_pollutant:
            layer = self.get_layer_by_name(item.text())
            if layer:
                pollutant_layers.append(layer)

        for item in selected_landuse:
            layer = self.get_layer_by_name(item.text())
            if layer:
                landuse_layers.append(layer)

        if not pollutant_layers and not landuse_layers:
            self.dialog.iface.messageBar().pushMessage("Warning", "No layers selected for export.", level=Qgis.Warning)
            return

        # Now call the static helper
        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = self.convert_selected_vectors_to_csv(
            pollutant_layers,
            landuse_layers,
            output_folder,
            self.dialog.pBDataRefinement,
            self.dialog.iface,
            timestamp_now
        )

        if csv_path:
            self.load_csv_as_layer(csv_path, timestamp_now)


    @staticmethod
    def convert_selected_vectors_to_csv(pollutant_layers, landuse_layers, output_folder, progress_bar, iface, timestamp=None):
        """
        Export selected pollutant + land use vector layers to two CSVs:
        - combined_export_grouped_{timestamp}.csv  (spatial: X,Y columns)
        - combined_table_{timestamp}.csv           (table copy)
        Progress bar is updated smoothly; UI messages are throttled.
        """
        from PyQt5.QtWidgets import QApplication

        try:
            # Defensive init for the progress bar
            try:
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
            except Exception:
                pass

            combined_data = []

            # 1) Count total features up front
            all_layers = [lyr for lyr in (pollutant_layers + landuse_layers) if lyr and lyr.isValid()]
            total_features = sum(int(lyr.featureCount() or 0) for lyr in all_layers)

            if total_features == 0:
                iface.messageBar().pushMessage("Warning", "No features found in the selected layers.", level=Qgis.Warning)
                return None

            processed_features = 0
            # Target about 50 progress updates across the whole run
            update_every = max(1, total_features // 50)
            notify_every = 200  # throttle messageBar notifications

            def process_layer(layer):
                nonlocal processed_features
                features = layer.getFeatures()
                attr_names = [field.name() for field in layer.fields()]

                for feat in features:
                    record = feat.attributes()
                    record_dict = {}

                    # Copy attributes with special handling for "timestamp"
                    for field_name, value in zip(attr_names, record):
                        if field_name.strip().lower() == "timestamp":
                            try:
                                if hasattr(value, "toString"):
                                    record_dict[field_name] = value.toString(Qt.ISODate)
                                elif isinstance(value, str):
                                    record_dict[field_name] = value.strip()
                                else:
                                    record_dict[field_name] = str(value)
                            except Exception as timestamp_err:
                                logging.warning(f"Could not parse timestamp for feature {feat.id()}: {timestamp_err}")
                                record_dict[field_name] = ""
                        else:
                            record_dict[field_name] = value

                    # Geometry → centroid → point (robust to non-point geoms)
                    try:
                        geom = feat.geometry()
                        if geom is None or geom.isEmpty() or not geom.isGeosValid():
                            raise ValueError("Invalid or empty geometry.")

                        # For points, asPoint() is fine; for others, use centroid
                        try:
                            pt = geom.asPoint()
                            if not pt or (pt.x() == 0 and pt.y() == 0 and not geom.isMultipart()):
                                # fallback to centroid if asPoint looked bogus
                                pt = geom.centroid().asPoint()
                        except Exception:
                            pt = geom.centroid().asPoint()

                        record_dict['X'] = round(pt.x(), 6)
                        record_dict['Y'] = round(pt.y(), 6)
                    except Exception as ge:
                        logging.warning(f"Skipping feature {feat.id()} due to geometry error: {ge}")
                        continue  # skip bad feature

                    combined_data.append(record_dict)

                    processed_features += 1
                    # Smooth progress updates
                    if processed_features % update_every == 0 or processed_features == total_features:
                        try:
                            progress = int((processed_features / total_features) * 100)
                            progress_bar.setValue(progress)
                        except Exception:
                            pass
                        QApplication.processEvents()

                    # Throttled info ping
                    if processed_features % notify_every == 0:
                        iface.messageBar().pushMessage(
                            "Processing",
                            f"Processed {processed_features} of {total_features} features…",
                            level=Qgis.Info,
                        )

            # 2) Process layers
            for layer in pollutant_layers:
                process_layer(layer)
            for layer in landuse_layers:
                process_layer(layer)

            # 3) Build dataframe and group by X,Y
            df = pd.DataFrame(combined_data)
            if df.empty:
                iface.messageBar().pushMessage("Warning", "No valid features to export after filtering.", level=Qgis.Warning)
                return None

            df_grouped = df.groupby(["X", "Y"], as_index=False).first()

            os.makedirs(output_folder, exist_ok=True)
            filename_spatial = f"combined_export_grouped_{timestamp}.csv" if timestamp else "combined_export_grouped.csv"
            filename_table  = f"combined_table_{timestamp}.csv"            if timestamp else "combined_table.csv"

            csv_path_spatial = os.path.join(output_folder, filename_spatial)
            csv_path_table   = os.path.join(output_folder, filename_table)

            # 4) Save outputs
            df_grouped.to_csv(csv_path_spatial, index=False)
            df_grouped.to_csv(csv_path_table, index=False)

            logging.info(f"Saved spatial CSV: {csv_path_spatial}")
            logging.info(f"Saved table CSV: {csv_path_table}")

            # 5) Final success message (once)
            iface.messageBar().pushMessage("Success", f"CSV export completed: {filename_spatial}", level=Qgis.Info)
            logging.info(f"CSV export completed: {filename_spatial}")
            logging.info(f"CSV export completed: {filename_table}")
            QMessageBox.information(None, "CSV Export Completed", f"Spatial CSV: {filename_spatial}\nTable CSV: {filename_table}")

            return csv_path_spatial

        except Exception as e:
            logging.exception("Error during CSV export")
            iface.messageBar().pushMessage("Error", f"Export failed: {str(e)}", level=Qgis.Critical)
            QMessageBox.critical(None, "Grouped CSV Export Error", str(e))
            return None

        finally:
            # Always drive the bar to 100% so the user sees completion
            try:
                progress_bar.setValue(100)
            except Exception:
                pass


    def load_csv_as_layer(self, csv_path, timestamp):
        try:
            safe_path = os.path.normpath(csv_path).replace("\\", "/")

            # Find CRS
            crs = None
            pollutant_layers = [self.get_layer_by_name(item.text()) for item in self.dialog.lWInputPollutantVectorAlign.selectedItems()]
            landuse_layers = [self.get_layer_by_name(item.text()) for item in self.dialog.lWInputLandUseVectorAlign.selectedItems()]

            if pollutant_layers:
                crs = pollutant_layers[0].crs().authid()
            elif landuse_layers:
                crs = landuse_layers[0].crs().authid()
            else:
                crs = "EPSG:4326"

            # --- Load CSV as SPATIAL LAYER ---
            uri_spatial = f"file:///{safe_path}?delimiter=,&xField=X&yField=Y&crs={crs}&detectTypes=yes&geomType=point"
            layer_spatial = QgsVectorLayer(uri_spatial, f"Combined_CSV_Export_{timestamp}", "delimitedtext")

            if layer_spatial.isValid():
                QgsProject.instance().addMapLayer(layer_spatial)
                self.dialog.iface.messageBar().pushMessage("Success", "CSV spatial layer added to QGIS.", level=Qgis.Info)
                logging.info(f"CSV spatial layer loaded successfully: {layer_spatial.name()}")
            else:
                self.dialog.iface.messageBar().pushMessage("Error", "Failed to load CSV as spatial layer.", level=Qgis.Warning)
                logging.error("CSV spatial layer is not valid.")

            # --- Load CSV also as NON-SPATIAL TABLE ---
            csv_path_table = os.path.join(os.path.dirname(safe_path), f"combined_table_{timestamp}.csv")
            csv_path_table = os.path.normpath(csv_path_table).replace("\\", "/")

            uri_table = f"file:///{csv_path_table}?delimiter=,&detectTypes=yes"
            layer_table = QgsVectorLayer(uri_table, f"Combined_CSV_Table_{timestamp}", "delimitedtext")

            if layer_table.isValid():
                QgsProject.instance().addMapLayer(layer_table)
                self.dialog.iface.messageBar().pushMessage("Success", "CSV table added to QGIS.", level=Qgis.Info)
                logging.info(f"CSV table loaded successfully: {layer_table.name()}")
            else:
                self.dialog.iface.messageBar().pushMessage("Error", "Failed to load CSV as table.", level=Qgis.Warning)
                logging.error("CSV table layer is not valid.")


        except Exception as e:
            logging.exception("Failed to load CSV as layer and table")
            QMessageBox.critical(None, "CSV Load Error", str(e))
            self.dialog.iface.messageBar().pushMessage("Error", f"Failed to load CSV: {str(e)}", level=Qgis.Critical)
            logging.error(f"Error loading CSV: {str(e)}")