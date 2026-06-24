import os
from datetime import datetime
from pathvalidate import sanitize_filename
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsVectorFileWriter, QgsFeature,
    QgsGeometry, QgsWkbTypes, QgsCoordinateTransformContext,
    QgsField, QgsSpatialIndex, QgsFeatureRequest, QgsApplication
)
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import QVariant, QDateTime, Qt, QCoreApplication

class VectorAlignment:
    def __init__(self, dialog):
        self.dialog = dialog
        self.setup_ui()
        # Feature cache for improved performance
        self.feature_cache = {}
        # Timestamp parsing cache
        self.timestamp_cache = {}

    def setup_ui(self):
        self.dialog.tBChooseFolderDataRefinementCSVSaveVectorAlign.clicked.connect(self.choose_output_folder)
        self.dialog.pBRefreshAlgin.clicked.connect(self.populate_vector_lists)
        self.dialog.pBVectorAlign.clicked.connect(self.run_vector_alignment)
        
        # Add timestamp selection combobox if not already present
        if not hasattr(self.dialog, 'cbTimestampSelectionMethod'):
            from PyQt5.QtWidgets import QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget
            
            # Find the parent container for the vector alignment UI elements
            # Try to find a parent widget/layout for inserting the combobox
            try:
                # Option 1: Look for tab widget
                if hasattr(self.dialog, 'tabWidget') and self.dialog.tabWidget is not None:
                    for i in range(self.dialog.tabWidget.count()):
                        if "vector" in self.dialog.tabWidget.tabText(i).lower():
                            container = self.dialog.tabWidget.widget(i)
                            break
                    else:
                        container = None
                
                # Option 2: Look for a groupbox with vector in the name
                if not container:
                    for child in self.dialog.findChildren(QWidget):
                        if "vector" in child.objectName().lower() and hasattr(child, 'layout'):
                            container = child
                            break
                    
                # Option 3: Default to the dialog itself
                if not container:
                    container = self.dialog
                
                # Create the combobox and label
                self.dialog.cbTimestampSelectionMethod = QComboBox(self.dialog)
                self.dialog.cbTimestampSelectionMethod.addItems([
                    "DN Weighted", 
                    "DN Weighted with Diversity",
                    "Weighted Average", 
                    "Max DN", 
                    "Random Weighted", 
                    "Area Weighted", 
                    "Feature Centroid", 
                    "Highest Weight"
                ])
                self.dialog.cbTimestampSelectionMethod.setCurrentText("DN Weighted with Diversity")
                
                # Create a container for our controls
                hbox = QHBoxLayout()
                label = QLabel("Timestamp Selection Method:", self.dialog)
                hbox.addWidget(label)
                hbox.addWidget(self.dialog.cbTimestampSelectionMethod)
                
                # Try to find the right place to put our controls
                # Option 1: Try to insert below the output folder selection
                parent_container = self.dialog.tBChooseFolderDataRefinementCSVSaveVectorAlign.parentWidget()
                if parent_container and hasattr(parent_container, 'layout') and parent_container.layout():
                    # Insert into the parent's layout if it exists
                    parent_container.layout().addLayout(hbox)
                else:
                    # Option 2: Try to find a vertical layout near the button
                    for child in self.dialog.findChildren(QVBoxLayout):
                        if self.dialog.pBVectorAlign in [child.itemAt(i).widget() for i in range(child.count()) 
                                                          if child.itemAt(i) and child.itemAt(i).widget()]:
                            child.insertLayout(child.count()-1, hbox)
                            break
                    else:
                        # Option 3: Create a standalone widget and place it
                        timestamp_widget = QWidget(self.dialog)
                        timestamp_widget.setLayout(hbox)
                        # Place it near the pBVectorAlign button
                        button_pos = self.dialog.pBVectorAlign.pos()
                        timestamp_widget.move(button_pos.x(), button_pos.y() - 40)
                        timestamp_widget.show()
                
            except Exception as e:
                # If we hit any issues adding the UI, log it but continue
                self.dialog.tELog.append(f"WARNING: Could not add timestamp selection UI: {str(e)}")
                # Create default value that we'll use in the code
                self.dialog.cbTimestampSelectionMethod = type('obj', (object,), {
                    'currentText': lambda: "DN Weighted with Diversity"
                })
        
        self.populate_vector_lists()

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderDataRefinementCSVSaveVectorAlign.setText(folder)

    def populate_vector_lists(self):
        self.dialog.lWInputPollutantVectorAlign.clear()
        self.dialog.lWInputLandUseVectorAlign.clear()

        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer) and layer.isValid():
                if layer.geometryType() in [QgsWkbTypes.PolygonGeometry, QgsWkbTypes.LineGeometry, QgsWkbTypes.PointGeometry]:
                    self.dialog.lWInputPollutantVectorAlign.addItem(layer.name())
                    self.dialog.lWInputLandUseVectorAlign.addItem(layer.name())

        self.dialog.tELog.append("INFO: Vector layer lists updated.")

    def parse_timestamp(self, raw_time, feature_id=None):
        """Helper function to parse timestamp values from various formats with caching"""
        # Use cache if feature_id is provided
        if feature_id is not None and feature_id in self.timestamp_cache:
            return self.timestamp_cache[feature_id]
            
        if raw_time is None:
            return None
            
        feat_timestamp = None
        
        # Handle different storage types
        if isinstance(raw_time, QDateTime):
            feat_timestamp = raw_time
        elif isinstance(raw_time, QVariant) and raw_time.userType() == QVariant.DateTime:
            feat_timestamp = raw_time.toDateTime()
        elif isinstance(raw_time, str):
            cleaned_time = raw_time.strip().replace('Z', '+00:00')
            # Try ISO format first
            feat_timestamp = QDateTime.fromString(cleaned_time, Qt.ISODate)
            
            # Fallback to Python parsing if QDateTime fails
            if not feat_timestamp.isValid():
                try:
                    dt_str = cleaned_time.split('+')[0].replace('T', ' ')
                    py_dt = datetime.fromisoformat(dt_str)
                    feat_timestamp = QDateTime(
                        py_dt.year, py_dt.month, py_dt.day,
                        py_dt.hour, py_dt.minute, py_dt.second
                    )
                except Exception:
                    pass
        
        # Store in cache if valid
        if feature_id is not None and feat_timestamp and feat_timestamp.isValid():
            self.timestamp_cache[feature_id] = feat_timestamp
                    
        return feat_timestamp if feat_timestamp and feat_timestamp.isValid() else None

    def diagnose_timestamp_issues(self, poll_layer, timestamp_field_name, feature_count=5):
        """Diagnostic function to identify timestamp parsing issues"""
        self.dialog.tELog.append("\nDIAGNOSTICS: Timestamp Field Analysis")
        self.dialog.tELog.append(f"Field name being used: '{timestamp_field_name}'")
        
        # Check first few features
        self.dialog.tELog.append(f"Examining first {feature_count} features:")
        for i, feat in enumerate(poll_layer.getFeatures()):
            if i >= feature_count:
                break
                
            raw_value = feat[timestamp_field_name]
            parsed = self.parse_timestamp(raw_value)
            
            self.dialog.tELog.append(f"Feature {feat.id()}: Raw='{raw_value}' ({type(raw_value).__name__})")
            self.dialog.tELog.append(f"  Parsed: {parsed.toString(Qt.ISODate) if parsed else 'FAILED'}")
        
        # Count valid vs invalid
        valid_count = 0
        invalid_count = 0
        invalid_samples = []
        
        for feat in poll_layer.getFeatures():
            raw_value = feat[timestamp_field_name]
            parsed = self.parse_timestamp(raw_value)
            
            if parsed:
                valid_count += 1
            else:
                invalid_count += 1
                if len(invalid_samples) < 3 and raw_value is not None:
                    invalid_samples.append(f"'{raw_value}' ({type(raw_value).__name__})")
        
        self.dialog.tELog.append(f"Valid timestamps: {valid_count}, Invalid: {invalid_count}")
        if invalid_samples:
            self.dialog.tELog.append(f"Invalid examples: {', '.join(invalid_samples)}")
        
        self.dialog.tELog.append("END DIAGNOSTICS\n")

    def run_vector_alignment(self):
        dialog = self.dialog
        dialog.tELog.append("INFO: Starting vector interpolation process...")

        if not dialog.cBEnableVectorAlignment.isChecked():
            dialog.tELog.append("INFO: Vector interpolation is disabled.")
            return
            
        # Get timestamp selection method
        timestamp_selection = dialog.cbTimestampSelectionMethod.currentText()
        dialog.tELog.append(f"INFO: Using timestamp selection method: {timestamp_selection}")

        output_folder = dialog.tBChooseFolderDataRefinementCSVSaveVectorAlign.text()
        if not os.path.exists(output_folder) or not os.access(output_folder, os.W_OK):
            QMessageBox.critical(dialog, "Error", f"Output folder invalid or not writable: {output_folder}")
            return

        try:
            poll_name = dialog.lWInputPollutantVectorAlign.selectedItems()[0].text()
            lu_name = dialog.lWInputLandUseVectorAlign.selectedItems()[0].text()
            poll_layer = QgsProject.instance().mapLayersByName(poll_name)[0]
            lu_layer = QgsProject.instance().mapLayersByName(lu_name)[0]
            dialog.tELog.append(f"INFO: Using layers: {poll_name} and {lu_name}")
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Layer retrieval failed: {str(e)}")
            return

        if poll_layer.geometryType() != QgsWkbTypes.PolygonGeometry or lu_layer.geometryType() != QgsWkbTypes.PolygonGeometry:
            QMessageBox.critical(dialog, "Error", "Both layers must be polygon type.")
            return

        # Validate Timestamp field
        timestamp_field = next((f for f in poll_layer.fields() if f.name().lower() == "timestamp"), None)
        if not timestamp_field:
            QMessageBox.critical(dialog, "Error", "Pollutant layer missing 'Timestamp' field")
            return
        timestamp_field_name = timestamp_field.name()
        
        # Run diagnostics on timestamp field
        self.diagnose_timestamp_issues(poll_layer, timestamp_field_name)

        # Create output layer with land use attributes + new fields
        crs = lu_layer.crs()
        out_layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", f"Interpolated_{lu_name}", "memory")
        out_provider = out_layer.dataProvider()
        
        # Preserve original land use attributes and add new fields
        out_provider.addAttributes(
            lu_layer.fields().toList() + [
                QgsField("DN_interp", QVariant.Double),
                QgsField("DNmin", QVariant.Double),
                QgsField("DNmax", QVariant.Double),
                QgsField("src_count", QVariant.Int),
                QgsField("NumDates", QVariant.Int),
                QgsField("area_ratio", QVariant.Double),
                QgsField("Timestamp", QVariant.DateTime),
                QgsField("x", QVariant.Double),
                QgsField("y", QVariant.Double)
            ]
        )
        out_layer.updateFields()

        # Get field names
        dn_field = next(f.name() for f in poll_layer.fields() if f.name().lower() == "dn")

        # Create spatial index for pollution layer
        dialog.tELog.append("INFO: Building spatial index for pollutant layer...")
        poll_index = QgsSpatialIndex(poll_layer.getFeatures())
        
        # Processing setup
        dialog.pBDataRefinement.setMaximum(lu_layer.featureCount())
        dialog.pBDataRefinement.setValue(0)
        
        # Stats tracking
        total_skipped = 0
        total_warnings = 0
        batch_size = 50
        features_to_add = []

        # Track unique timestamps
        used_timestamps = {}

        # Process land use features
        for lu_idx, lu_feat in enumerate(lu_layer.getFeatures()):
            # Process UI events periodically
            if lu_idx % 10 == 0:
                QCoreApplication.processEvents()

            
            lu_geom = lu_feat.geometry()
            if not lu_geom.isGeosValid():
                total_skipped += 1
                continue

            lu_area = lu_geom.area()
            
            # FIXED: Initialize variables properly
            weighted_dn_sum = 0.0  # Sum of (DN * weight)
            total_weight = 0.0     # Sum of weights
            min_dn = float('inf')
            max_dn = float('-inf')
            contributing_features = 0
            timestamp_data = []
            unique_dates = set()

            # Use spatial index to get candidate features
            intersecting_ids = poll_index.intersects(lu_geom.boundingBox())
            if not intersecting_ids:
                continue

            # Request only the needed features
            request = QgsFeatureRequest().setFilterFids(intersecting_ids)
            
            # FIXED: Collect all intersection data first, then calculate
            intersection_data = []  # Store (weight, dn_value, timestamp, area)
            
            for poll_feat in poll_layer.getFeatures(request):
                try:
                    poll_geom = poll_feat.geometry()
                    if not poll_geom.isGeosValid():
                        continue

                    # Quick bounding box check
                    if not poll_geom.boundingBox().intersects(lu_geom.boundingBox()):
                        continue

                    inter_geom = lu_geom.intersection(poll_geom)
                    if inter_geom.isEmpty():
                        continue

                    inter_area = inter_geom.area()
                    if inter_area <= 0:
                        continue
                    
                    # FIXED: Calculate weight as intersection area / total land use area
                    # This ensures weights are proportional and don't exceed 1.0 total
                    weight = inter_area / lu_area if lu_area > 0 else 0
                    
                    dn_value = poll_feat[dn_field]
                    if dn_value is None:
                        continue
                    
                    dn_float = float(dn_value)
                    
                    # Parse timestamp
                    raw_time = poll_feat[timestamp_field_name]
                    feat_timestamp = self.parse_timestamp(raw_time, poll_feat.id())
                    
                    if feat_timestamp:
                        # Store intersection data
                        intersection_data.append((weight, dn_float, feat_timestamp, inter_area))
                        
                        # Track unique dates
                        date_key = feat_timestamp.date().toString(Qt.ISODate)
                        unique_dates.add(date_key)
                        
                        contributing_features += 1
                        
                        if lu_idx < 3:  # Debug logging
                            dialog.tELog.append(
                                f"DEBUG: LU {lu_idx} | Weight: {weight:.4f} | Area: {inter_area:.2f} | "
                                f"DN: {dn_value} | Timestamp: {feat_timestamp.toString(Qt.ISODate)}"
                            )
                    else:
                        total_warnings += 1
                        if lu_idx < 3:
                            dialog.tELog.append(f"WARN: Invalid timestamp: {raw_time}")

                except Exception as e:
                    total_warnings += 1
                    if lu_idx < 10:
                        dialog.tELog.append(f"ERROR: {str(e)}")
                    continue

            # Skip features with no valid intersections
            if not intersection_data:
                if lu_idx < 20:
                    dialog.tELog.append(f"WARN: No valid intersections for LU feature {lu_idx}")
                continue

            # FIXED: Calculate weighted average DN properly
            for weight, dn_value, timestamp, area in intersection_data:
                weighted_dn_sum += dn_value * weight
                total_weight += weight
                min_dn = min(min_dn, dn_value)
                max_dn = max(max_dn, dn_value)
                timestamp_data.append((weight, area, timestamp, dn_value))

            # FIXED: Calculate final interpolated DN as weighted average
            if total_weight > 0:
                dn_interp = weighted_dn_sum / total_weight
            else:
                # Fallback to simple average if weights are invalid
                dn_interp = sum(dn for _, dn, _, _ in intersection_data) / len(intersection_data)
            
            # Reset min/max if no values found
            if min_dn == float('inf'):
                min_dn = 0.0
            if max_dn == float('-inf'):
                max_dn = 0.0

            # FIXED: Log the calculation for debugging
            if lu_idx < 5:
                dialog.tELog.append(
                    f"DEBUG: LU {lu_idx} | Weighted DN Sum: {weighted_dn_sum:.4f} | "
                    f"Total Weight: {total_weight:.4f} | Final DN: {dn_interp:.4f}"
                )

            # Temporal selection logic (timestamp selection remains the same)
            best_timestamp = None
            
            if timestamp_selection == "DN Weighted with Diversity":
                timestamp_data.sort(key=lambda x: -x[3])  # Sort by DN value
                top_candidates = timestamp_data[:min(3, len(timestamp_data))]
                
                least_used_count = float('inf')
                for _, _, timestamp, _ in top_candidates:
                    ts_key = timestamp.toString(Qt.ISODate)
                    usage_count = used_timestamps.get(ts_key, 0)
                    if usage_count < least_used_count:
                        least_used_count = usage_count
                        best_timestamp = timestamp
                
                if best_timestamp is None:
                    best_timestamp = top_candidates[0][2]
                    
                ts_key = best_timestamp.toString(Qt.ISODate)
                used_timestamps[ts_key] = used_timestamps.get(ts_key, 0) + 1
                
            elif timestamp_selection == "Weighted Average":
                total_weighted_time = 0
                total_weights = 0
                
                for weight, _, timestamp, _ in timestamp_data:
                    msecs = timestamp.toMSecsSinceEpoch()
                    total_weighted_time += msecs * weight
                    total_weights += weight
                
                if total_weights > 0:
                    avg_msecs = int(total_weighted_time / total_weights)
                    best_timestamp = QDateTime.fromMSecsSinceEpoch(avg_msecs)
            
            elif timestamp_selection == "Max DN":
                timestamp_data.sort(key=lambda x: -x[3])
                best_timestamp = timestamp_data[0][2]
                
            else:  # Default: Highest Weight
                timestamp_data.sort(key=lambda x: -x[0])
                best_timestamp = timestamp_data[0][2]

            # FIXED: Ensure area_ratio doesn't exceed 1.0
            # Total weight should represent coverage, capped at 100%
            area_ratio = min(total_weight, 1.0)

            # Create output feature
            new_feat = QgsFeature(out_layer.fields())
            new_feat.setGeometry(lu_geom)
            # Extract centroid for X and Y
            centroid = lu_geom.centroid().asPoint()
            x_coord = round(centroid.x(), 2)
            y_coord = round(centroid.y(), 2)

            new_feat.setAttributes(
                lu_feat.attributes() + [
                    round(dn_interp, 4),      # FIXED: Now properly weighted average
                    round(min_dn, 4),
                    round(max_dn, 4),
                    contributing_features,
                    len(unique_dates),
                    round(area_ratio, 4),     # FIXED: Properly capped at 1.0
                    best_timestamp,
                    x_coord,
                    y_coord
                ]
            )
            features_to_add.append(new_feat)
            
            # Add features in batches
            if len(features_to_add) >= batch_size or lu_idx == lu_layer.featureCount() - 1:
                out_provider.addFeatures(features_to_add)
                features_to_add = []
            
            # Update progress
            dialog.pBDataRefinement.setValue(lu_idx + 1)

        # Validation check
        if out_layer.featureCount() == 0:
            QMessageBox.critical(dialog, "Error", "No valid features created - check input data.")
            return

        # Finalize output layer
        try:
            base_name = sanitize_filename(f"{poll_name}_interp_{lu_name}", platform="auto")
            timestamp_now = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_filename = f"{base_name[:50]}_{timestamp_now}.gpkg"
            output_path = os.path.normpath(os.path.join(output_folder, safe_filename))
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Filename generation failed: {str(e)}")
            return

        # Remove existing file if present
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Failed to remove existing file: {str(e)}")
                return


        # Save using current QGIS API
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "GPKG"
        save_options.layerName = "interpolated_data"
        save_options.fileEncoding = "UTF-8"
        save_options.transformContext = QgsCoordinateTransformContext()

       
        # Ensure the output layer is valid before saving

        error_code, error_message = QgsVectorFileWriter.writeAsVectorFormat(
            out_layer,
            output_path,
            save_options
        )

        
        if error_code != QgsVectorFileWriter.NoError:
            QMessageBox.critical(dialog, "Error", f"Failed to save: {error_message}")
            return

        # Load result layer
        final_layer = QgsVectorLayer(output_path, f"Interpolated_{lu_name}", "ogr")
        if final_layer.isValid():
            QgsProject.instance().addMapLayer(final_layer)
            dialog.plugin_memory.setdefault("interpolated_vectors", {})[final_layer.name()] = final_layer
            dialog.tELog.append("INFO: Output layer loaded successfully.")
        else:
            QMessageBox.critical(dialog, "Error", "Final output layer is invalid.")
            return

        
        # Final reporting
        dialog.tELog.append(f"INFO: Processed {lu_layer.featureCount()} features.")
        dialog.tELog.append(f"INFO: Created {out_layer.featureCount()} output features.")
        dialog.tELog.append(f"INFO: Skipped {total_skipped} invalid geometries.")
        dialog.tELog.append(f"INFO: Encountered {total_warnings} processing warnings.")
        dialog.pBDataRefinement.setValue(dialog.pBDataRefinement.maximum())
        
        # Stats about timestamp diversity
        num_timestamps = len(used_timestamps)
        dialog.tELog.append(f"INFO: Used {num_timestamps} different timestamps in output")
        
        # Display timestamp usage histogram
        if used_timestamps:
            dialog.tELog.append("INFO: Timestamp usage distribution:")
            sorted_usage = sorted([(ts, count) for ts, count in used_timestamps.items()], 
                                 key=lambda x: -x[1])
            for i, (ts, count) in enumerate(sorted_usage[:5]):  # Show top 5
                dialog.tELog.append(f"  {ts}: {count} features")
        
        QMessageBox.information(dialog, "Success", f"Interpolation complete.\nSaved to: {output_path}")
        dialog.tELog.append(f"INFO: Output saved to {output_path}")
        dialog.tELog.append("INFO: Vector interpolation process completed.")
        
        # Clear caches
        self.timestamp_cache = {}
        self.feature_cache = {}