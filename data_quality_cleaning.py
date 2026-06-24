import pandas as pd
import numpy as np
import os
from datetime import datetime
from qgis.core import QgsProject, QgsVectorLayer, QgsFields, QgsField, QgsFeature, QgsGeometry, QgsPointXY, QgsWkbTypes
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QVariant, QStringListModel, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from urllib.parse import urlparse, unquote


import logging

class DataQualityCleaning:
    def __init__(self, dialog):
        self.dialog = dialog
        self.iface = dialog.iface
        self.loaded_layers = {}
        self.selected_layer = None
        self.dataframe = None
        self.output_folder = None
        self.dialog.cBImportCSVFileDataCleaning.currentIndexChanged.connect(self.on_csv_layer_selected)

    def initialize(self):
        if not self.dialog.cBEnableDataCleaning.isChecked():
            return
        self.populate_csv_layers()
        self.populate_comboboxes()


    def populate_csv_layers(self):
        self.dialog.cBImportCSVFileDataCleaning.blockSignals(True)  # Prevent premature trigger
        self.dialog.cBImportCSVFileDataCleaning.clear()
        self.loaded_layers.clear()

        for layer_id, layer in QgsProject.instance().mapLayers().items():
            if (
                layer.type() == QgsVectorLayer.VectorLayer and
                layer.providerType().lower() == "delimitedtext" and
                layer.isValid() and
                layer.geometryType() != QgsWkbTypes.NoGeometry
            ):
                self.dialog.cBImportCSVFileDataCleaning.addItem(layer.name(), layer_id)
                self.loaded_layers[layer_id] = layer  # Store by layer ID

        self.dialog.cBImportCSVFileDataCleaning.blockSignals(False)
        logging.info("Populated spatial CSV layers using layer IDs")


    def populate_comboboxes(self):
        self.dialog.cBMissingValueHandling.clear()
        self.dialog.cBMissingValueHandling.addItems([
            "No Action",
            "Drop rows with missing values",
            "Impute with mean",
            "Impute with median",
            "Impute with zero"
        ])

        self.dialog.cBOutlierDetection.clear()
        self.dialog.cBOutlierDetection.addItems([
            "No Action",
            "Detect and flag",
            "Winsorize (clip to ±3σ)",
            "Remove outlier rows"
        ])

        self.dialog.cBDuplicateDataHandling.clear()
        self.dialog.cBDuplicateDataHandling.addItems([
            "No Action",
            "Drop duplicate rows"
        ])

        self.dialog.cBDataTypeConversion.clear()
        self.dialog.cBDataTypeConversion.addItems([
            "No Action",
            "Convert to numeric (where possible)",
            "Convert timestamp to datetime"
        ])

        self.dialog.cBDataNormalization.clear()
        self.dialog.cBDataNormalization.addItems([
            "No Action",
            "Min-Max Scaling",
            "Z-Score Normalization"
        ])

        self.dialog.cBCollinearityThreshold.clear()
        self.dialog.cBCollinearityThreshold.addItems([
            "No Action",
            "0.7",
            "0.8",
            "0.85",
            "0.9"
        ])

    def enable_data_cleaning_section(self):
        enabled = self.dialog.cBEnableDataCleaning.isChecked()
        self.log(f"Data cleaning enabled: {enabled}")

        if enabled:
          self.populate_csv_layers()
          self.populate_comboboxes()
        else:
          self.dialog.cBImportCSVFileDataCleaning.clear()
          self.dialog.cBMissingValueHandling.clear()
          self.dialog.cBOutlierDetection.clear()
          self.dialog.cBDuplicateDataHandling.clear()
          self.dialog.cBDataTypeConversion.clear()
          self.dialog.cBDataNormalization.clear()
          self.dialog.cBCollinearityThreshold.clear()


    def load_selected_csv_layer(self):
        """Loads the selected CSV layer and prepares it for data cleaning operations."""
        try:
            # Get selected layer ID from combo box user data
            layer_id = self.dialog.cBImportCSVFileDataCleaning.currentData()
            
            if not layer_id:
                logging.error("No valid layer selected in combo box")
                QMessageBox.warning(self.dialog, "Error", "No valid layer selected!")
                return

            # Retrieve layer from registry using ID
            layer = QgsProject.instance().mapLayer(layer_id)
            if not layer or not layer.isValid():
                logging.error(f"Layer with ID {layer_id} not found or invalid")
                QMessageBox.critical(self.dialog, "Error", "Selected layer is invalid or missing!")
                return

            # Parse URI components
            data_source = layer.dataProvider().dataSourceUri()
            uri_parts = data_source.split('?')
            file_path = uri_parts[0]
            params = uri_parts[1] if len(uri_parts) > 1 else ""

            # Handle URI encoding and Windows path formatting
            parsed_uri = urlparse(file_path)
            if parsed_uri.scheme == 'file':
                file_path = unquote(parsed_uri.path)
                if os.name == 'nt' and file_path.startswith('/'):
                    file_path = file_path[1:].replace('/', '\\')

            # Validate file existence
            if not os.path.exists(file_path):
                logging.error(f"CSV file not found: {file_path}")
                QMessageBox.critical(self.dialog, "Error", f"File not found:\n{file_path}")
                return

            # Read CSV with error handling
            try:
                df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                logging.info(f"Successfully read CSV with {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logging.error(f"CSV read failed: {str(e)}")
                QMessageBox.critical(self.dialog, "Read Error", f"Failed to read CSV:\n{str(e)}")
                return

            # Case-insensitive X/Y column detection
            x_col = next((col for col in df.columns if col.strip().lower() == 'x'), None)
            y_col = next((col for col in df.columns if col.strip().lower() == 'y'), None)

            if not x_col or not y_col:
                logging.error("Missing spatial columns (X/Y not found)")
                QMessageBox.warning(self.dialog, "Data Error", "CSV must contain X and Y coordinate columns!")
                return

            # Numeric validation for coordinates
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            valid_rows = df.dropna(subset=[x_col, y_col])
            
            if len(valid_rows) == 0:
                logging.error("All rows contain invalid coordinate values")
                QMessageBox.warning(self.dialog, "Data Error", "All rows have invalid X/Y values!")
                return
                
            # Update dataframe and spatial references
            self.dataframe = valid_rows.reset_index(drop=True)
            self.x_col = x_col
            self.y_col = y_col
            logging.info(f"Valid spatial data: {len(self.dataframe)} rows")

            # Populate column list view
            model = QStandardItemModel()
            for column in self.dataframe.columns:
                item = QStandardItem(column)
                item.setCheckable(True)
                item.setSelectable(True)
                item.setEditable(False)
                # Highlight coordinate columns
                if column in (x_col, y_col):
                    item.setBackground(Qt.gray)
                    item.setEnabled(False)  # Disable selection for spatial columns
                model.appendRow(item)

            self.dialog.lVTargetColumnsDataCleaning.setModel(model)
            self.dialog.lVTargetColumnsDataCleaning.setSelectionMode(3)  # Multi-selection
            self.dialog.lVTargetColumnsDataCleaning.repaint()
            
            logging.info(f"List view populated with {model.rowCount()} columns")
            QMessageBox.information(self.dialog, "Success", f"Loaded {len(self.dataframe)} valid rows with {model.rowCount()} columns")

        except Exception as e:
            logging.error(f"Critical error in load_selected_csv_layer: {str(e)}", exc_info=True)
            QMessageBox.critical(self.dialog, "Unexpected Error", f"Operation failed:\n{str(e)}")


    def log(self, message):
        logging.info(message)

    def perform_cleaning(self):
        if self.dataframe is None:
            self.log("No data loaded.")
            return

        self.dialog.pBDataCleaning.setValue(10)

        # --- Detect X/Y columns case-insensitively ---
        x_col = next((col for col in self.dataframe.columns if col.lower() == 'x'), None)
        y_col = next((col for col in self.dataframe.columns if col.lower() == 'y'), None)
        spatial_cols = [x_col, y_col]
        id_cols = ['fid']  # Preserve fid
        protected_cols = spatial_cols + id_cols

        # --- Get selected target columns explicitly ---
        selected_indexes = self.dialog.lVTargetColumnsDataCleaning.selectedIndexes()
        target_columns = [index.data() for index in selected_indexes if index.data() not in protected_cols]

        if not target_columns:
            self.log("No target columns selected from list view. Cleaning operations will be skipped.")
            QMessageBox.warning(self.dialog, "No Columns Selected", "Please select target columns from the list view.")
            return

        
        # Create cleaned versions of each target column
        cleaned_map = {}  
        for col in target_columns:
            new_col = f"{col}_cleaned"
            self.dataframe[new_col] = self.dataframe[col]
            cleaned_map[col] = new_col

        # --- Missing Value Handling ---
        missing_option = self.dialog.cBMissingValueHandling.currentText()
        if missing_option != "No Action":
            cols_to_check = list(set(target_columns + spatial_cols))
            if missing_option == "Drop rows with missing values":
                self.dataframe.dropna(subset=cols_to_check, inplace=True)
                self.log(f"Dropped rows with missing values in {cols_to_check}")
            elif target_columns:
                if missing_option == "Impute with mean":
                    for col in target_columns:
                        self.dataframe[cleaned_map[col]] = self.dataframe[cleaned_map[col]].fillna(self.dataframe[cleaned_map[col]].mean())
                elif missing_option == "Impute with median":
                    for col in target_columns:
                        self.dataframe[cleaned_map[col]] = self.dataframe[cleaned_map[col]].fillna(self.dataframe[cleaned_map[col]].median())
                elif missing_option == "Impute with zero":
                    for col in target_columns:
                        self.dataframe[cleaned_map[col]] = self.dataframe[cleaned_map[col]].fillna(0)
                self.log(f"Imputed missing values using '{missing_option}' for {len(target_columns)} columns")

        self.dialog.pBDataCleaning.setValue(30)

        # --- Outlier Handling ---
        outlier_option = self.dialog.cBOutlierDetection.currentText()
        if outlier_option != "No Action":
            numeric_cols = [col for col in target_columns if pd.api.types.is_numeric_dtype(self.dataframe[col])]
            if numeric_cols:
                z_scores = pd.DataFrame({col: (self.dataframe[cleaned_map[col]] - self.dataframe[cleaned_map[col]].mean()) / self.dataframe[cleaned_map[col]].std() for col in numeric_cols})
                if outlier_option == "Detect and flag":
                    for col in numeric_cols:
                        self.dataframe[f"{col}_zscore"] = z_scores[col]
                    self.dataframe['outlier_flag'] = (np.abs(z_scores) > 3).any(axis=1)
                elif outlier_option == "Winsorize (clip to ±3σ)":
                    for col in numeric_cols:
                        mean = self.dataframe[col].mean()
                        std = self.dataframe[col].std()
                        self.dataframe[cleaned_map[col]] = np.clip(self.dataframe[cleaned_map[col]], mean - 3*std, mean + 3*std)
                elif outlier_option == "Remove outlier rows":
                    mask = (np.abs(z_scores) <= 3).all(axis=1)
                self.dataframe = self.dataframe[mask]
                self.log(f"Applied outlier handling to {len(numeric_cols)} columns")

        self.dialog.pBDataCleaning.setValue(50)

        # --- Duplicate Removal ---
        if self.dialog.cBDuplicateDataHandling.currentText() == "Drop duplicate rows":
            self.dataframe.drop_duplicates(inplace=True)
            self.log("Dropped duplicate rows")

        self.dialog.pBDataCleaning.setValue(60)

        # --- Data Type Conversion ---
        dtype_option = self.dialog.cBDataTypeConversion.currentText()
        if dtype_option != "No Action":
            if dtype_option == "Convert to numeric (where possible)":
                for col in target_columns:
                    self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='ignore')
            elif dtype_option == "Convert timestamp to datetime":
                for col in target_columns:
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], errors='coerce')
            self.log(f"Applied data type conversion: {dtype_option}")

        self.dialog.pBDataCleaning.setValue(70)

        # --- Normalization ---
        normalization_option = self.dialog.cBDataNormalization.currentText()
        if normalization_option != "No Action":
            numeric_cols = [col for col in target_columns if pd.api.types.is_numeric_dtype(self.dataframe[col])]
            if numeric_cols:
                if normalization_option == "Min-Max Scaling":
                    for col in numeric_cols:
                        self.dataframe[f"{col}_minmax"] = (self.dataframe[cleaned_map[col]] - self.dataframe[cleaned_map[col]].min()) / (self.dataframe[cleaned_map[col]].max() - self.dataframe[cleaned_map[col]].min())
                elif normalization_option == "Z-Score Normalization":
                    self.dataframe[numeric_cols] = (self.dataframe[numeric_cols] - self.dataframe[numeric_cols].mean()) / \
                                                self.dataframe[numeric_cols].std()
                self.log(f"Applied {normalization_option} to {len(numeric_cols)} columns")

        
        # --- BLOCK Spatial Windowing (Central + Surrounding Average) ---
        if self.dialog.cBEnableSpatialWindowingWholeBlock.isChecked():
            if x_col and y_col:
                try:
                    from scipy.spatial import cKDTree

                    points = self.dataframe[[x_col, y_col]].values
                    tree = cKDTree(points)
                    _, indices = tree.query(points, k=9)  # 1 center + 8 neighbors

                    min_neighbors = 6
                    valid_targets = [col for col in target_columns if pd.api.types.is_numeric_dtype(self.dataframe[col])]

                    for col in valid_targets:
                        original_col = self.dataframe[col].copy()
                        new_values = []
                        block_avg_count = 0
                        fallback_count = 0

                        for i, neighbors in enumerate(indices):
                            neighbor_values = original_col.iloc[neighbors].dropna()
                            if len(neighbor_values) >= min_neighbors:
                                block_mean = neighbor_values.mean()
                                new_values.append(block_mean)
                                block_avg_count += 1
                            else:
                                new_values.append(original_col.iloc[i])
                                fallback_count += 1

                        self.dataframe[f"{col}_windowed"] = new_values
                        self.log(f"[BLOCK Windowing] Column '{col}': BlockAvg={block_avg_count}, Fallback={fallback_count}, Total={len(self.dataframe)}")

                    self.log(f"Applied BLOCK spatial windowing to {len(valid_targets)} columns")

                except Exception as e:
                    self.log(f"BLOCK spatial windowing error: {str(e)}")
            else:
                self.log("BLOCK spatial windowing skipped: X/Y columns not found in data")
# --- Spatial Windowing ---
        if self.dialog.cBEnableSpatialWindowing.isChecked():
            if x_col and y_col:
                try:
                    from scipy.spatial import cKDTree

                    # Build KDTree from coordinate points
                    points = self.dataframe[[x_col, y_col]].values
                    tree = cKDTree(points)
                     
                    
                    # Alternative: Use k=9 but with better boundary handling
                    _, indices = tree.query(points, k=9)  # 1 center + 8 neighbors

                    min_neighbors = 6  # Increased minimum for better stability
                    valid_targets = [col for col in target_columns if pd.api.types.is_numeric_dtype(self.dataframe[col])]

                    for col in valid_targets:
                        original_col = self.dataframe[col].copy()
                        new_values = []
                        smoothed_count = 0
                        fallback_count = 0

                        for i, neighbors in enumerate(indices):
                            neighbor_indices = neighbors[1:]  # exclude self
                            
                            # CRITICAL FIX: Only smooth if we have enough neighbors
                            if len(neighbor_indices) >= min_neighbors:
                                # Get neighbor values
                                neighbor_values = original_col.iloc[neighbor_indices].dropna()
                                
                                # Additional check: ensure we have enough valid neighbor values
                                if len(neighbor_values) >= min_neighbors:
                                    # Use weighted average instead of median for smoother results
                                    # Or use median but with distance weighting
                                    median_val = neighbor_values.median()
                                    
                                    # Apply conservative smoothing (blend with original)
                                    alpha = 0.7  # Smoothing factor (0.7 = 70% neighbor influence)
                                    smoothed_val = alpha * median_val + (1 - alpha) * original_col.iloc[i]
                                    new_values.append(smoothed_val)
                                    smoothed_count += 1
                                else:
                                    # Fallback: keep original value
                                    new_values.append(original_col.iloc[i])
                                    fallback_count += 1
                            else:
                                # Fallback: keep original value for boundary points
                                new_values.append(original_col.iloc[i])
                                fallback_count += 1

                        # Assign smoothed values to column
                        self.dataframe[f"{col}_windowed"] = new_values
                        self.log(f"[Spatial Windowing] Column '{col}': Smoothed={smoothed_count}, Fallback={fallback_count}, Total={len(self.dataframe)}")

                    self.log(f"Applied spatial windowing to {len(valid_targets)} columns using min_neighbors={min_neighbors}")

                except Exception as e:
                    self.log(f"Spatial windowing error: {str(e)}")
            else:
                self.log("Spatial windowing skipped: X/Y columns not found in data")


        # --- Collinearity Filtering ---
        collinearity_option = self.dialog.cBCollinearityThreshold.currentText()
        if collinearity_option != "No Action":
            try:
                threshold = float(collinearity_option)
                numeric_cols = [col for col in target_columns if pd.api.types.is_numeric_dtype(self.dataframe[col])]
                if numeric_cols:
                    corr_matrix = self.dataframe[numeric_cols].corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
                    if to_drop:
                        self.dataframe.drop(columns=to_drop, inplace=True)
                        self.log(f"Removed {len(to_drop)} collinear columns above threshold {threshold}")
            except Exception as e:
                self.log(f"Collinearity filtering error: {str(e)}")
        else:
            self.log("Collinearity filtering skipped (No Action selected)")

        self.dialog.pBDataCleaning.setValue(90)
        self.log("Cleaning process completed")


    def choose_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if self.output_folder:
            self.log(f"Selected output folder: {self.output_folder}")
        else:
            self.log("No output folder selected.")


    def save_cleaned_data(self):
        if self.dataframe is None or self.output_folder is None:
            self.log("No data to save or output folder missing")
            return

        # ========================================================================
        # 1. Detect spatial columns in cleaned data
        # ========================================================================
        x_col = next((col for col in self.dataframe.columns if col.strip().lower() == 'x'), None)
        y_col = next((col for col in self.dataframe.columns if col.strip().lower() == 'y'), None)

        if not x_col or not y_col:
            self.log("X/Y columns missing in cleaned data")
            QMessageBox.warning(self.dialog, "Error", "Cleaned data lacks X/Y columns!")
            return

        # ========================================================================
        # 2. Get CRS from original layer with strict validation
        # ========================================================================
        crs_authid = "EPSG:4326"
        crs_valid = False

        if self.selected_layer and self.selected_layer.isValid():
            layer_crs = self.selected_layer.crs()
            if layer_crs.isValid():
                crs_authid = layer_crs.authid()
                crs_valid = True
                self.log(f"Using original layer CRS: {crs_authid}")
            else:
                self.log("Original layer CRS is invalid")
        else:
            self.log("No valid layer reference found")

        if not crs_valid:
            QMessageBox.warning(
                self.dialog,
                "CRS Warning",
                "Using default EPSG:4326.\n"
                "Ensure original layer has a valid CRS set in QGIS!"
            )

        # ========================================================================
        # 3. Final coordinate validation
        # ========================================================================
        try:
            self.dataframe[x_col] = pd.to_numeric(self.dataframe[x_col], errors='coerce')
            self.dataframe[y_col] = pd.to_numeric(self.dataframe[y_col], errors='coerce')
            valid_data = self.dataframe.dropna(subset=[x_col, y_col])
            if valid_data.empty:
                QMessageBox.critical(self.dialog, "Error", "All coordinates invalid after cleaning!")
                return
        except Exception as e:
            self.log(f"Coordinate validation failed: {str(e)}")
            return

        # ========================================================================
        # 4. Determine cleaned vs untouched columns
        # ========================================================================
        protected_cols = ['fid', x_col, y_col]
        cleaned_columns = [col for col in self.dataframe.columns if col not in protected_cols]
        untouched_columns = [col for col in protected_cols if col in self.dataframe.columns]

        self.log(f"Cleaned columns to export: {cleaned_columns}")
        self.log(f"Untouched (preserved) columns: {untouched_columns}")

        # ========================================================================
        # 5. Save CSV and create layer
        # ========================================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_folder, f"cleaned_data_{timestamp}.csv")

        try:
            valid_data.to_csv(output_path, index=False, encoding='utf-8')
            self.log(f"CSV successfully saved to: {output_path}")
        except Exception as e:
            QMessageBox.critical(self.dialog, "Save Error", f"CSV save failed:\n{str(e)}")
            return

        # ========================================================================
        # 6. Create layer URI with original CRS
        # ========================================================================
        uri = (
            f"file:///{output_path}?type=csv&"
            f"delimiter=,&xField={x_col}&yField={y_col}&"
            f"crs={crs_authid}&"
            "decimalSeparator=."
        ).replace("\\", "/")  # Ensure valid URI on Windows

        vlayer = QgsVectorLayer(uri, f"Cleaned Data {timestamp}", "delimitedtext")

        if vlayer.isValid():
            QgsProject.instance().addMapLayer(vlayer)
            self.log(f"Successfully loaded layer with CRS {crs_authid}")
            QMessageBox.information(
                self.dialog, "Success",
                f"Saved and loaded layer with:\n"
                f"• CRS: {crs_authid}\n"
                f"• {len(valid_data)} features\n"
                f"• X: {x_col}, Y: {y_col}"
            )
        else:
            error_msg = (
                f"Layer creation failed. Reasons:\n"
                f"1. CRS mismatch (original: {crs_authid})\n"
                f"2. Invalid coordinates in {x_col}/{y_col}\n"
                f"3. File encoding issues\n"
                f"Error: {vlayer.error().summary()}"
            )
            QMessageBox.critical(self.dialog, "Layer Error", error_msg)

        self.log(f"Final URI parameters: {uri}")
        if self.selected_layer:
            self.log(f"Original layer CRS validity: {self.selected_layer.crs().isValid()}")
            self.log(f"Original layer CRS details: {self.selected_layer.crs().description()}")
       
    def clean_and_save_data(self):
          self.perform_cleaning()
          self.save_cleaned_data()
          self.log("Data cleaning and saving completed.")
          QMessageBox.information(self.dialog, "Data Cleaning", "Data cleaning and saving completed successfully.")
          
    def on_csv_layer_selected(self, index):
        """Updates the selected layer reference when combo box changes."""
        layer_id = self.dialog.cBImportCSVFileDataCleaning.currentData()
        if layer_id:
            self.selected_layer = QgsProject.instance().mapLayer(layer_id)  # Critical: Update reference
            self.load_selected_csv_layer()
        

