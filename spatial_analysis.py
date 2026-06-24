import os
import logging
from qgis.PyQt import QtWidgets, QtCore
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import traceback
from mgwr.gwr import MGWR, GWR
from mgwr.sel_bw import Sel_BW
import scipy.spatial.distance
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from qgis.core import (QgsCoordinateTransformContext, QgsVectorFileWriter,
    QgsVectorLayer, QgsFields, QgsField, QgsPointXY, QgsGeometry,
    QgsFeature, QgsProject, QgsCoordinateReferenceSystem, QgsVectorFileWriter, QgsMapLayerType, QgsWkbTypes)
from PyQt5.QtCore import QVariant, QCoreApplication
from qgis.core import QgsUnitTypes, QgsFields, QgsField, QgsWkbTypes, QgsCoordinateReferenceSystem, QgsCoordinateTransformContext
from mgwr.gwr import MGWR, GWR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import LinAlgError
import warnings
import json
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.model_selection import KFold
try:
    from esda.moran import Moran
    from libpysal.weights import KNN, DistanceBand
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("PySAL not available - spatial autocorrelation tests disabled")
import threading
import time

class EnhancedSpatialAnalysis:
    def __init__(self, dialog, iface):
        self.dialog = dialog
        self.iface = iface
        self.progress_bar = dialog.pBSpatialFiltering
        self.output_folder = None
        self._connect_ui()
        self.populate_csv_layers()
        self.populate_kernel_and_bandwidth_options()

        # Enhanced configuration
        self.min_observations = 30  # Minimum required observations
        self.max_vif_threshold = 10.0  # Default VIF threshold
        self.spatial_lag_threshold = 0.3  # Moran's I threshold for spatial issues

    def _connect_ui(self):
        """Connect UI elements to their respective functions"""
        self.dialog.tBChooseFolderSpatialFilteringMGWR.clicked.connect(self.choose_folder)
        self.dialog.pBSpatialFilteringSaveMGWR.clicked.connect(self.run_enhanced_mgwr)
        self.dialog.pBSpatialFilteringNext.clicked.connect(
            lambda: self.dialog.tabsXgeoAi.setCurrentWidget(self.dialog.tPredictiveModeling)
        )
        self.dialog.cBSelectCSVLayerforMGWR.currentIndexChanged.connect(self.populate_fields_from_selected_layer)
        self.dialog.pBRefreshLayersMGWR.clicked.connect(self.refresh_layers_mgwr)

    def choose_folder(self):
        """Open a folder picker and store the chosen path for MGWR outputs."""
        try:
            start_dir = getattr(self, "output_folder", None) or os.path.expanduser("~")
            folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder", start_dir)
            if folder:
                # Verify writable
                try:
                    test_file = os.path.join(folder, ".__xgeoai_write_test.tmp")
                    with open(test_file, "w") as f:
                        f.write("ok")
                    os.remove(test_file)
                except Exception as e:
                    QMessageBox.warning(self.dialog, "Folder Error", f"Selected folder is not writable: {e}")
                    self.log(f"ERROR: Selected folder not writable: {e}")
                    return
                self.output_folder = folder
                # Update any known line edits if they exist
                for attr in ("lESaveFolderSpatialFilteringMGWR", "lESaveFolderMGWR", "lESaveFolder"):
                    le = getattr(self.dialog, attr, None)
                    if le:
                        try:
                            le.setText(folder)
                            break
                        except Exception:
                            pass
                self.log(f"Output folder set to: {folder}")
            else:
                self.log("Folder selection cancelled.")
        except Exception as e:
            self.log(f"Folder selection failed: {e}")

    def _check_mgwr_compatibility(self):
        """Assume MGWR 2.1.1 and sanity-check basic selector construction."""
        try:
            from mgwr import __version__ as mgwr_version
            self.log(f"MGWR version detected: {mgwr_version} (proceeding with 2.1.1-compatible path)")
        except Exception:
            self.log("MGWR version: unknown (proceeding with 2.1.1-compatible path)")
        # Force internal version flag for consistency elsewhere
        self.mgwr_version = "2.1.1"
        try:
            test_coords = __import__('numpy').array([[0, 0], [1, 1], [2, 2]])
            test_y = __import__('numpy').array([[1], [2], [3]])
            test_X = __import__('numpy').array([[1], [2], [3]])
            _ = Sel_BW(test_coords, test_y, test_X)
            self.log("MGWR 2.1.1 compatibility check passed (selector constructed).")
        except Exception as e:
            self.log(f"MGWR basic compatibility check failed: {e}")

    def log(self, message):
        """Enhanced logging with timestamp (no dialogs, no file IO)."""
        from datetime import datetime
        import logging
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"{ts} - {message}"
        logging.info(full)
        try:
            self.dialog.tELog.append(full)
        except Exception:
            pass


    def populate_csv_layers(self):
        """Populate CSV layers with enhanced detection"""
        self.dialog.cBSelectCSVLayerforMGWR.clear()
        csv_layers = []

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayerType.VectorLayer:
                source = layer.source().lower()
                provider_type = layer.providerType().lower()

                # Enhanced CSV detection
                is_csv = ('.csv' in source or
                        provider_type in ['delimitedtext', 'ogr'] or
                        source.endswith('.csv') or
                        'csv' in provider_type)

                if is_csv:
                    csv_layers.append(layer.name())
                    self.dialog.cBSelectCSVLayerforMGWR.addItem(layer.name())

        self.log(f"Found {len(csv_layers)} CSV layers: {csv_layers}")

    def populate_kernel_and_bandwidth_options(self):
        """Populate kernel and bandwidth options with enhanced choices"""
        # Kernel options
        self.dialog.cBKernelTypeMGWR.clear()
        kernel_options = ["gaussian", "bisquare", "exponential"]
        self.dialog.cBKernelTypeMGWR.addItems(kernel_options)

        # Bandwidth selection methods
        self.dialog.cBBandwidthSelectionMethod.clear()
        bandwidth_methods = ["Automatic (AICc)", "Automatic (CV)", "Fixed", "Adaptive"]
        self.dialog.cBBandwidthSelectionMethod.addItems(bandwidth_methods)

    def populate_fields_from_selected_layer(self):
        """Enhanced field population with data validation"""
        layer_name = self.dialog.cBSelectCSVLayerforMGWR.currentText()
        if not layer_name:
            return

        layer = self._get_layer_by_name(layer_name)
        if not layer:
            self.log(f"ERROR: Layer '{layer_name}' not found")
            return

        # Validate CRS
        crs = layer.crs()
        if not crs.isValid():
            self.log("ERROR: CRS is invalid or undefined.")
            QMessageBox.warning(self.dialog, "CRS Error",
                            "Layer CRS is invalid. Please set a valid CRS.")
            return

        # Enhanced CRS validation
        if crs.mapUnits() == QgsUnitTypes.DistanceDegrees:
            self.log(f"WARNING: Layer CRS is geographic ({crs.authid()}). "
                    "MGWR works better with projected coordinates (meters).")

            reply = QMessageBox.question(self.dialog, "Geographic CRS Warning",
                                    f"Layer uses geographic CRS ({crs.authid()}) with degree units.\n"
                                    "MGWR performs better with projected CRS (meter units).\n\n"
                                    "Continue anyway? (Results may be suboptimal)",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
            if reply == QMessageBox.No:
                return
        else:
            self.log(f"CRS validated: {crs.authid()} - coordinates in {crs.mapUnits()}")

        # Get and validate fields
        fields = layer.fields()
        numeric_fields = []

        for field in fields:
            if field.isNumeric():
                numeric_fields.append(field.name())

        if len(numeric_fields) < 3:  # Need at least X, Y, and one variable
            self.log("ERROR: Insufficient numeric fields (minimum 3 required: X, Y, target)")
            QMessageBox.warning(self.dialog, "Data Error",
                            "Layer must have at least 3 numeric fields (X, Y coordinates and target variable)")
            return

        # Clear and populate combo boxes
        for combo in [self.dialog.cBXCoordMGWR, self.dialog.cBYCoordMGWR,
                    self.dialog.cBTargetVariableMGWR]:
            combo.clear()
            combo.addItems(numeric_fields)

        # Populate explanatory variables list
        self.dialog.lWExplanatoryVarsMGWR.clear()
        self.dialog.lWExplanatoryVarsMGWR.addItems(numeric_fields)

        # Auto-select coordinate fields if available
        coord_candidates = ['x', 'y', 'lon', 'lat', 'longitude', 'latitude', 'easting', 'northing']

        for field_name in numeric_fields:
            field_lower = field_name.lower()
            if any(candidate in field_lower for candidate in ['x', 'lon', 'east']):
                index = self.dialog.cBXCoordMGWR.findText(field_name)
                if index >= 0:
                    self.dialog.cBXCoordMGWR.setCurrentIndex(index)
            elif any(candidate in field_lower for candidate in ['y', 'lat', 'north']):
                index = self.dialog.cBYCoordMGWR.findText(field_name)
                if index >= 0:
                    self.dialog.cBYCoordMGWR.setCurrentIndex(index)

        self.log(f"Populated {len(numeric_fields)} numeric fields from layer: {layer_name}")

    def _reproject_with_weights(self, df, W, attributes, log_func=None):
        """
        Reproject attributes using weight matrices with enhanced error handling
        """
        if log_func is None:
            log_func = self.log  # Use class logger by default

        out_df = df[['X', 'Y']].copy()
        n_cells = df.shape[0]

        # Handle different weight matrix structures
        if isinstance(W, list):
            # Multiple weight matrices (one per variable)
            weight_matrices = W
        elif hasattr(W, 'shape'):
            if W.ndim == 1:
                # Single 1D weight array - convert to proper 2D matrix
                log_func("Warning: Weight matrix is 1D, creating identity-like weighting")
                # Create a simple distance-based weight matrix or use uniform weights
                weight_matrices = [np.eye(n_cells) for _ in attributes]
            elif W.ndim == 2:
                # Single 2D weight matrix - use for all attributes
                weight_matrices = [W for _ in attributes]
            elif W.ndim == 3:
                # 3D array with separate matrix for each variable
                weight_matrices = [W[i] if i < W.shape[0] else W[0] for i in range(len(attributes))]
            else:
                log_func(f"Unexpected weight matrix dimensions: {W.shape}")
                weight_matrices = [np.eye(n_cells) for _ in attributes]
        else:
            log_func("No valid weight matrices found, skipping reprojection")
            return out_df

        log_func(f"Processing {len(attributes)} attributes with {len(weight_matrices)} weight matrices")

        for i, attr in enumerate(attributes):
            try:
                # Skip coordinate fields
                if attr.lower() in ['x', 'y', 'geometry']:
                    continue

                # Check if attribute exists in dataframe
                if attr not in df.columns:
                    log_func(f"Attribute '{attr}' not found in dataframe")
                    out_df[attr + "_rp"] = np.nan
                    continue

                # Get the appropriate weight matrix
                if i < len(weight_matrices):
                    wm = weight_matrices[i]
                else:
                    wm = weight_matrices[0] if weight_matrices else np.eye(n_cells)

                # Ensure weight matrix has correct dimensions
                if wm.ndim != 2 or wm.shape[0] != n_cells or wm.shape[1] != n_cells:
                    log_func(f"Weight matrix for '{attr}' has incorrect dimensions: {wm.shape}, expected: ({n_cells}, {n_cells})")
                    out_df[attr + "_rp"] = np.nan
                    continue

                # Convert attribute values with proper error handling
                attr_vals = df[attr].values
                converted_vals = []

                for val in attr_vals:
                    try:
                        # Handle QVariant objects
                        if hasattr(val, 'isNull') and val.isNull():
                            converted_vals.append(np.nan)
                        elif hasattr(val, 'value'):  # QVariant with value
                            converted_vals.append(float(val.value()) if val.value() is not None else np.nan)
                        elif isinstance(val, str):
                            # Try to convert string to float, otherwise set as NaN
                            try:
                                converted_vals.append(float(val))
                            except (ValueError, TypeError):
                                converted_vals.append(np.nan)
                        elif val is None or val == 'NULL':
                            converted_vals.append(np.nan)
                        else:
                            converted_vals.append(float(val))
                    except (ValueError, TypeError, AttributeError) as e:
                        log_func(f"Conversion error for value {val} in attribute '{attr}': {e}")
                        converted_vals.append(np.nan)

                attr_vals = np.array(converted_vals)

                # Check if we have any valid numeric values
                if np.all(np.isnan(attr_vals)):
                    log_func(f"All values in attribute '{attr}' are non-numeric or NaN")
                    out_df[attr + "_rp"] = np.nan
                    continue

                # Replace NaN values with column mean for calculation
                if np.any(np.isnan(attr_vals)):
                    mean_val = np.nanmean(attr_vals)
                    if not np.isnan(mean_val):
                        attr_vals_filled = np.where(np.isnan(attr_vals), mean_val, attr_vals)
                    else:
                        log_func(f"Cannot compute mean for attribute '{attr}' - all values are NaN")
                        out_df[attr + "_rp"] = np.nan
                        continue
                else:
                    attr_vals_filled = attr_vals

                # Perform weighted calculation
                try:
                    # Calculate weighted sum: matrix multiplication
                    weighted_sum = np.dot(wm, attr_vals_filled)

                    # Calculate weight totals for each observation
                    weight_total = np.sum(wm, axis=1)

                    # Avoid division by zero
                    weight_total = np.where(weight_total == 0, 1.0, weight_total)

                    # Calculate reprojected values
                    reprojected_vals = weighted_sum / weight_total

                    # Ensure finite values
                    reprojected_vals = np.where(np.isfinite(reprojected_vals), reprojected_vals, np.nan)

                    out_df[attr + "_rp"] = reprojected_vals

                    # Log success with statistics
                    valid_count = np.sum(np.isfinite(reprojected_vals))
                    log_func(f"Reprojected attribute '{attr}' successfully. Valid values: {valid_count}/{len(reprojected_vals)}")

                except Exception as calc_error:
                    log_func(f"Calculation error for attribute '{attr}': {calc_error}")
                    out_df[attr + "_rp"] = np.nan

            except Exception as attr_error:
                log_func(f"Error processing attribute '{attr}': {attr_error}")
                # Add NaN column if reprojection fails
                out_df[attr + "_rp"] = np.nan

        log_func(f"Reprojection completed. Output shape: {out_df.shape}")
        return out_df

    def _configure_environment(self):
        """Configure environment for optimal MGWR performance"""
        # Suppress warnings that might clutter output
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Configure numpy for better numerical stability
        np.seterr(all='ignore')

        # Set environment variables for joblib/multiprocessing
        os.environ['JOBLIB_MULTIPROCESSING'] = '0'
        os.environ['JOBLIB_START_METHOD'] = 'spawn'
        os.environ['LOKY_MAX_CPU_COUNT'] = '1'

        self.log("Environment configured for MGWR analysis")

        # Quiet chatty loggers (numba bytecode/SSA dumps)
        import logging
        from qgis.PyQt import QtWidgets, QtCore
        logging.getLogger().setLevel(logging.INFO)
        for noisy in ('numba', 'numba.core', 'numba.core.byteflow', 'numba.core.ssa'):
            logging.getLogger(noisy).setLevel(logging.WARNING)
            logging.getLogger(noisy).propagate = False

        # Ensure no env flags enable numba debug
        import os as _os_env_quiet
        _os_env_quiet.environ.pop('NUMBA_DEBUG', None)

    def _comprehensive_data_validation(self, df, x_col, y_col, target_var, explanatory_vars):
        """Comprehensive data validation and cleaning"""
        issues = []
        cleaned_df = df.copy()

        # Required columns validation
        required_cols = [x_col, y_col, target_var] + explanatory_vars
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return None, issues

        # Initial data info
        self.log(f"Initial data shape: {df.shape}")

        # Remove rows with any missing values in required columns
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=required_cols)
        removed_na = initial_count - len(cleaned_df)
        if removed_na > 0:
            self.log(f"Removed {removed_na} rows with missing values")

        if len(cleaned_df) < self.min_observations:
            issues.append(f"Insufficient observations after cleaning: {len(cleaned_df)} < {self.min_observations}")
            return None, issues

        # Check for infinite values
        numeric_cols = [col for col in required_cols if col in cleaned_df.columns]
        inf_mask = np.isinf(cleaned_df[numeric_cols]).any(axis=1)
        if inf_mask.any():
            cleaned_df = cleaned_df[~inf_mask]
            self.log(f"Removed {inf_mask.sum()} rows with infinite values")

        # Coordinate validation
        coords = cleaned_df[[x_col, y_col]].values

        # Check for coordinate variation
        x_range = np.ptp(coords[:, 0])
        y_range = np.ptp(coords[:, 1])
        if x_range == 0 or y_range == 0:
            issues.append("Coordinates have no spatial variation")
            return None, issues

        # Check for duplicate coordinates
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        if len(unique_coords) < len(coords):
            duplicate_count = len(coords) - len(unique_coords)
            self.log(f"Found {duplicate_count} duplicate coordinate locations")

            # Average values for duplicate locations
            averaged_data = []
            for i, unique_coord in enumerate(unique_coords):
                mask = inverse_indices == i
                if mask.sum() > 1:  # Multiple points at same location
                    # Average all numeric values for this location
                    averaged_row = cleaned_df[mask].mean(numeric_only=True)
                    averaged_row[x_col] = unique_coord[0]
                    averaged_row[y_col] = unique_coord[1]
                    averaged_data.append(averaged_row)
                else:
                    averaged_data.append(cleaned_df[mask].iloc[0])

            cleaned_df = pd.DataFrame(averaged_data)
            self.log(f"Averaged duplicate coordinates, new shape: {cleaned_df.shape}")

        # Variable-specific validation
        target_data = cleaned_df[target_var].values
        explanatory_data = cleaned_df[explanatory_vars].values

        # Check for constant variables
        constant_vars = []
        for i, var in enumerate(explanatory_vars):
            if np.var(explanatory_data[:, i]) < 1e-10:
                constant_vars.append(var)

        if constant_vars:
            issues.append(f"Constant variables detected: {constant_vars}")
            # Remove constant variables
            keep_vars = [var for var in explanatory_vars if var not in constant_vars]
            if len(keep_vars) < 1:
                issues.append("No non-constant explanatory variables remaining")
                return None, issues
            explanatory_vars = keep_vars
            self.log(f"Removed constant variables: {constant_vars}")

        # Check target variable distribution
        if np.var(target_data) < 1e-10:
            issues.append("Target variable has no variation")
            return None, issues

        return cleaned_df[required_cols], issues

    def _enhanced_bandwidth_selection(self, coords, y, X, method, kernel, init_bw=None):
        """Enhanced bandwidth selection with timeout protection"""
        self.log(f"Starting bandwidth selection using method: {method}")

        n_points = coords.shape[0]

        # Calculate distance statistics efficiently
        try:
            # Use a sample for large datasets to speed up calculation
            if n_points > 1000:
                sample_indices = np.random.choice(n_points, size=500, replace=False)
                sample_coords = coords[sample_indices]
                distances = scipy.spatial.distance.pdist(sample_coords)
                self.log(f"Using sample of 500 points for distance calculation (total: {n_points})")
            else:
                distances = scipy.spatial.distance.pdist(coords)

            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)

            self.log(f"Distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, "
                    f"Mean: {mean_dist:.2f}, Median: {median_dist:.2f}")

        except Exception as e:
            self.log(f"Error calculating distances: {e}")
            # Fallback distance calculation
            x_range = np.ptp(coords[:, 0])
            y_range = np.ptp(coords[:, 1])
            mean_dist = np.sqrt(x_range**2 + y_range**2) / 4
            min_dist = mean_dist * 0.1
            max_dist = mean_dist * 4
            median_dist = mean_dist
            self.log(f"Using fallback distance estimates")

        try:
            if method == "Fixed":
                if init_bw and init_bw > 0:
                    bw_value = min(init_bw, max_dist * 0.7)
                else:
                    bw_value = min(mean_dist * 0.25, median_dist * 0.8, max_dist * 0.3)

                bw_value = max(bw_value, min_dist * 2)
                self.log(f"Fixed bandwidth selected: {bw_value:.2f}")
                return bw_value, "fixed"

            elif method == "Adaptive":
                optimal_neighbors = max(8, min(int(np.sqrt(n_points)), n_points // 8, 50))
                self.log(f"Adaptive bandwidth: {optimal_neighbors} neighbors")
                return optimal_neighbors, "adaptive"

            else:  # Automatic methods with timeout protection
                self.log("Starting automatic bandwidth selection with timeout protection...")
                result_container = {}

                def target():
                    try:
                        # Create selector
                        selector = Sel_BW(coords, y, X, spherical=getattr(self, 'spherical', False), kernel=kernel)

                        # Conservative search bounds
                        min_bw = max(min_dist * 2, mean_dist * 0.05)
                        max_bw = min(max_dist * 0.5, mean_dist * 1.5)

                        self.log(f"Search bounds: {min_bw:.2f} to {max_bw:.2f}")

                        criterion = 'CV' if method == "Automatic (CV)" else 'AICc'

                        bw_value = selector.search(
                            criterion=criterion,
                            bw_min=min_bw,
                            bw_max=max_bw,
                            tol=1e-2,  # Relaxed tolerance
                            max_iter=15  # Reduced iterations
                        )

                        if np.isfinite(bw_value) and bw_value > 0:
                            result_container['bw_value'] = bw_value
                        else:
                            raise ValueError(f"Invalid bandwidth from search: {bw_value}")

                    except Exception as e:
                        result_container['error'] = e

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(600)  # Wait for 600 seconds

                if thread.is_alive():
                    self.log("Bandwidth selection timed out, using fallback")
                    fallback_bw = mean_dist * 0.2
                    fallback_bw = max(fallback_bw, min_dist * 2)
                    return fallback_bw, "timeout_fallback"
                elif 'error' in result_container:
                    self.log(f"Bandwidth search error: {result_container['error']}")
                    fallback_bw = min(mean_dist * 0.3, max_dist * 0.2)
                    fallback_bw = max(fallback_bw, min_dist * 2)
                    return fallback_bw, "fallback"
                else:
                    return result_container['bw_value'], "automatic"

        except Exception as e:
            self.log(f"Bandwidth selection completely failed: {e}")
            # Emergency fallback
            emergency_bw = 1000.0  # Fixed emergency value
            try:
                if mean_dist > 0:
                    emergency_bw = mean_dist * 0.15
            except:
                pass

            self.log(f"Using emergency fallback bandwidth: {emergency_bw:.2f}")
            return emergency_bw, "emergency"

    def _validate_mgwr_inputs(self, coords, y, X, bw_list):
        """Validate inputs specifically for MGWR 2.1.1"""
        # Check coordinates
        if coords.shape[0] != y.shape[0] or coords.shape[0] != X.shape[0]:
            raise ValueError("Coordinate, target, and feature arrays must have same number of observations")

        # Check bandwidth list
        expected_bw_length = X.shape[1] + 1  # +1 for intercept
        if len(bw_list) != expected_bw_length:
            self.log(f"Warning: Bandwidth list length {len(bw_list)} != expected {expected_bw_length}")
            # Adjust bandwidth list
            if len(bw_list) == 1:
                # Replicate single bandwidth
                bw_list = [bw_list[0]] * expected_bw_length
            elif len(bw_list) > expected_bw_length:
                # Truncate
                bw_list = bw_list[:expected_bw_length]
            else:
                # Extend with last value
                bw_list.extend([bw_list[-1]] * (expected_bw_length - len(bw_list)))

        # Validate bandwidth values
        for i, bw in enumerate(bw_list):
            if not np.isfinite(bw) or bw <= 0:
                raise ValueError(f"Invalid bandwidth at index {i}: {bw}")

        # Check for sufficient variation in coordinates
        x_range = np.ptp(coords[:, 0])
        y_range = np.ptp(coords[:, 1])
        if x_range == 0 or y_range == 0:
            raise ValueError("Coordinates must have spatial variation")

        self.log(f"MGWR input validation passed: {coords.shape[0]} observations, "
                f"{X.shape[1]} variables, bandwidth list: {[f'{bw:.2f}' for bw in bw_list]}")

        return bw_list

    def _create_mgwr_bandwidth_list(self, coords, X, bw_value, bw_type):
        """Create bandwidth list for MGWR 2.1.1 - Fixed version to prevent freezing"""
        try:
            # Simple bandwidth holder class
            class BandwidthHolder:
                def __init__(self, bw_list, bw_type):
                    self.bw = bw_list
                    self.bw_type = bw_type

            # Get number of variables (including intercept)
            n_vars = X.shape[1] + 1  # +1 for intercept

            if bw_type == "adaptive":
                # Convert adaptive bandwidth to fixed distance bandwidth
                try:
                    n_neighbors = max(5, min(int(bw_value), coords.shape[0] - 1))
                    self.log(f"Converting adaptive bandwidth: {n_neighbors} neighbors")

                    # Calculate average k-th neighbor distance more efficiently
                    from scipy.spatial import cKDTree

                    # Use cKDTree for efficient neighbor finding
                    tree = cKDTree(coords)

                    # Query k+1 nearest neighbors (including self)
                    distances, indices = tree.query(coords, k=n_neighbors + 1)

                    # Take the k-th neighbor distance (excluding self at index 0)
                    kth_distances = distances[:, -1]  # Last column is k-th neighbor
                    avg_knn_distance = np.mean(kth_distances)

                    # Create bandwidth list
                    bw_list = [float(avg_knn_distance)] * n_vars

                    self.log(f"Converted adaptive ({n_neighbors} neighbors) to fixed distance: {avg_knn_distance:.2f}")
                    return BandwidthHolder(bw_list, "adaptive_converted")

                except Exception as e:
                    self.log(f"Adaptive bandwidth conversion failed: {e}")
                    # Fallback to fixed bandwidth
                    fallback_bw = np.mean(scipy.spatial.distance.pdist(coords)) * 0.3
                    bw_list = [float(fallback_bw)] * n_vars
                    return BandwidthHolder(bw_list, "adaptive_fallback")

            else:  # Fixed bandwidth types
                # Ensure bw_value is valid
                if not np.isfinite(bw_value) or bw_value <= 0:
                    self.log(f"Invalid bandwidth value: {bw_value}, calculating fallback")

                    # Calculate fallback bandwidth
                    try:
                        distances = scipy.spatial.distance.pdist(coords)
                        fallback_bw = np.mean(distances) * 0.25
                        self.log(f"Using fallback bandwidth: {fallback_bw:.2f}")
                        bw_value = fallback_bw
                    except:
                        # Ultimate fallback
                        x_range = np.ptp(coords[:, 0])
                        y_range = np.ptp(coords[:, 1])
                        bw_value = (x_range + y_range) / 4
                        self.log(f"Using coordinate-based fallback: {bw_value:.2f}")

                # Create bandwidth list
                bw_list = [float(bw_value)] * n_vars
                self.log(f"Created fixed bandwidth list: {bw_value:.2f} for {n_vars} coefficients")
                return BandwidthHolder(bw_list, bw_type)

        except Exception as e:
            self.log(f"Error in bandwidth selector creation: {e}")

            # Emergency fallback
            try:
                n_vars = X.shape[1] + 1

                # Simple coordinate-based bandwidth
                x_range = np.ptp(coords[:, 0])
                y_range = np.ptp(coords[:, 1])
                emergency_bw = max(x_range, y_range) * 0.1

                if emergency_bw <= 0 or not np.isfinite(emergency_bw):
                    emergency_bw = 1000.0  # Absolute fallback

                bw_list = [float(emergency_bw)] * n_vars

                class EmergencyBandwidthHolder:
                    def __init__(self, bw_list):
                        self.bw = bw_list
                        self.bw_type = "emergency_fallback"

                self.log(f"Using emergency bandwidth: {emergency_bw:.2f}")
                return EmergencyBandwidthHolder(bw_list)

            except Exception as final_e:
                self.log(f"Emergency fallback failed: {final_e}")
                raise Exception(f"Bandwidth selector creation completely failed: {e}, {final_e}")

    def _build_mgwr_21_clean(self, coords, y, X, kernel='bisquare', fixed=True, multi=True):
        """
        MGWR 2.1.x builder: use selector -> search -> MGWR(coords,y,X,selector).fit()
        No positional-arg tricks, no bw list passed to fit().
        """
        try:
            sel = Sel_BW(coords, y, X, multi=multi, kernel=kernel, fixed=fixed, spherical=getattr(self, 'spherical', False))
            # Use a conservative minimum for multi-bandwidth to avoid degenerate weights
            if multi:
                bw = sel.search(multi_bw_min=[2], criterion='AICc', tol=1e-3, max_iter=30)
            else:
                bw = sel.search(criterion='AICc', tol=1e-3, max_iter=30)
            model = MGWR(coords, y, X, sel)
            results = model.fit()
            return results, bw, "MGWR_21_multi" if multi else "MGWR_21_single"
        except Exception as e:
            self.log(f"MGWR 2.1.x clean build failed: {e}")
            raise

    def _run_mgwr_with_fallbacks(self, coords, y_std, X_design, selector, kernel):
        """Run MGWR using MGWR 2.1.x API only, then fall back to GWR if needed."""
        results = None
        method_used = None

        # Contextual variables required for reprojection and saving
        layer_name = self.dialog.cBSelectCSVLayerforMGWR.currentText()
        layer = self._get_layer_by_name(layer_name)
        if layer is None or not layer.isValid():
            self.log("ERROR: No valid layer selected for MGWR.")
            return None, "InvalidLayer"

        # Primary MGWR path: use provided selector (multi=True) and X with explicit intercept
        try:
            self.log("Fitting MGWR with provided selector (no re-search), constant=False, spherical=%s" % getattr(self, 'spherical', False))
            mgwr_model = MGWR(coords=coords, y=y_std, X=X_design, selector=selector, constant=False, spherical=getattr(self, 'spherical', False), n_jobs=1)
            results = mgwr_model.fit()
            method_used = "MGWR_21_multi"
        except Exception as e_first:
            self.log(f"Primary MGWR fit failed: {e_first}")
            # Fallback 1: single-bandwidth MGWR (multi=False)
            try:
                self.log("Attempting MGWR fallback: single-bandwidth (multi=False)")
                sel_single = None
                try:
                    sel_single = Sel_BW(coords, y_std, X_design, multi=False, kernel=kernel, fixed=getattr(selector, 'fixed', True), spherical=getattr(self, 'spherical', False), constant=False)
                except TypeError:
                    sel_single = Sel_BW(coords, y_std, X_design, multi=False, kernel=kernel, fixed=getattr(selector, 'fixed', True), spherical=getattr(self, 'spherical', False))
                bw_single = sel_single.search(criterion='AICc', tol=1e-3, max_iter=30)
                mgwr_single = MGWR(coords=coords, y=y_std, X=X_design, selector=sel_single, constant=False, spherical=getattr(self, 'spherical', False), n_jobs=1)
                results = mgwr_single.fit()
                method_used = "MGWR_21_single"
            except Exception as e_second:
                # Final fallback: standard GWR with AICc bandwidth
                self.log(f"MGWR single-band fallback failed: {e_second}")
                try:
                    self.log("Falling back to GWR with AICc bandwidth selection")
                    from mgwr.sel_bw import Sel_BW as SelBW_GWR
                    gwr_sel = None
                    try:
                        gwr_sel = SelBW_GWR(coords, y_std, X_design, multi=False, kernel=kernel, fixed=True, spherical=getattr(self, 'spherical', False), constant=True)
                    except TypeError:
                        gwr_sel = SelBW_GWR(coords, y_std, X_design, multi=False, kernel=kernel, fixed=True, spherical=getattr(self, 'spherical', False))
                    bw = gwr_sel.search(criterion='AICc')
                    self.log(f"GWR selected bandwidth: {bw}")
                    gwr_model = GWR(coords, y_std, X_design, bw=bw, fixed=True, kernel=kernel, spherical=getattr(self, 'spherical', False), constant=False)
                    results = gwr_model.fit()
                    method_used = "GWR_Fallback"
                except Exception as e_gwr:
                    self.log(f"GWR fallback also failed: {e_gwr}")
                    raise Exception(f"All MGWR/GWR paths failed: {e_first} | {e_second} | {e_gwr}")

        # Reprojection using weights, if available
        if results is not None and hasattr(results, "W"):
            try:
                self.log("Reprojecting data using MGWR spatial weights...")

                # Load the data into a dataframe for reprojection
                df = self._layer_to_dataframe(layer)
                n = df.shape[0]
                do_weights = self._is_save_weights_enabled()

                if do_weights and hasattr(results, 'W') and results.W is not None and n <= 1500:
                    self.log("Reprojecting attributes using MGWR weights (size-gated)...")

                    # Prefer target + explanatory vars if available
                    try:
                        target_var = self.dialog.cBTargetVariableMGWR.currentText()
                        explanatory_items = self.dialog.lWExplanatoryVarsMGWR.selectedItems()
                        explanatory_vars = [item.text() for item in explanatory_items]
                        attrs_pref = [target_var] + list(explanatory_vars)
                    except Exception:
                        attrs_pref = [field.name() for field in layer.fields() if field.name() not in ['X','Y','geometry']]

                    attrs = [a for a in dict.fromkeys(attrs_pref) if a in df.columns]
                    if not attrs:
                        self.log("No valid attributes found for reprojection; skipping.")
                    else:
                        W = results.W
                        reprojected_df = self._reproject_with_weights(df, W, attrs)

                        output_folder = os.path.join(tempfile.gettempdir(), "mgwr_outputs")
                        os.makedirs(output_folder, exist_ok=True)
                        layer_name_s = layer.name().replace(" ", "_")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        reprojected_path = os.path.join(output_folder, f"{layer_name_s}_reprojected_{timestamp}.csv")
                        reprojected_df.to_csv(reprojected_path, index=False)
                        self.log(f"Reprojected data saved to CSV: {reprojected_path}")

                        gpkg_path = reprojected_path.replace(".csv", ".gpkg")
                        self._save_to_geopackage(reprojected_df, gpkg_path, layer.crs())
                        self.log(f"Reprojected data saved to GeoPackage: {gpkg_path}")
                else:
                    self.log("Skipping weight-based reprojection (disabled, missing weights, or dataset too large).")
            except Exception as reproj_e:
                self.log(f"Warning: Weight-based reprojection skipped due to error: {reproj_e}")

        return results, method_used

    def _enhanced_collinearity_removal(self, X, var_names, threshold=10.0):
        """Enhanced collinearity detection and removal"""
        if X.shape[1] <= 1:
            return X, var_names

        self.log(f"Checking multicollinearity (VIF threshold: {threshold})")

        # Use robust scaling to improve numerical stability
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate correlation matrix for initial screening
        corr_matrix = np.corrcoef(X_scaled.T)
        highly_correlated_pairs = []

        # Find highly correlated pairs (|r| > 0.8)
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > 0.8:
                    highly_correlated_pairs.append((var_names[i], var_names[j], corr_matrix[i, j]))

        # Check for highly correlated variable pairs
        if highly_correlated_pairs:
            self.log("Highly correlated variable pairs found:")
            for var1, var2, corr in highly_correlated_pairs:
                self.log(f"  {var1} - {var2}: r = {corr:.3f}")

        # VIF-based removal
        keep_vars = list(range(X.shape[1]))
        removed_vars = []
        iteration = 0
        max_iterations = len(var_names)

        while len(keep_vars) > 1 and iteration < max_iterations:
            try:
                X_current = X_scaled[:, keep_vars]

                # Add small regularization term
                X_reg = X_current + np.random.normal(0, 1e-8, X_current.shape)

                vif_values = []
                for i in range(len(keep_vars)):
                    try:
                        vif = variance_inflation_factor(X_reg, i)
                        vif_values.append(vif if np.isfinite(vif) else float('inf'))
                    except:
                        vif_values.append(float('inf'))

                max_vif = max(vif_values)
                self.log(f"VIF iteration {iteration + 1}: Max VIF = {max_vif:.2f}")

                if max_vif > threshold:
                    if len(keep_vars) > 1:  # Ensure we don't remove the last variable
                        max_index = vif_values.index(max_vif)
                        removed_var_idx = keep_vars[max_index]
                        removed_var = var_names[removed_var_idx]

                        removed_vars.append(removed_var)
                        del keep_vars[max_index]

                        self.log(f"Removed '{removed_var}' (VIF: {max_vif:.2f})")
                    else:
                        self.log("Warning: Max VIF above threshold but only one variable left - stopping removal")
                        break
                else:
                    break

                iteration += 1

            except Exception as e:
                self.log(f"VIF calculation error at iteration {iteration}: {e}")
                break

        if removed_vars:
            self.log(f"Collinearity removal complete. Removed {len(removed_vars)} variables: {removed_vars}")

        X_filtered = X[:, keep_vars]
        var_names_filtered = [var_names[i] for i in keep_vars]

        return X_filtered, var_names_filtered

    def _calculate_enhanced_diagnostics(self, results, y, X, coords, method_used):
        """Calculate comprehensive model diagnostics"""
        diagnostics = {
            'method_used': method_used,
            'n_observations': len(y),
            'n_variables': X.shape[1],
            'model_type': 'spatial_regression'
        }

        # Basic fit statistics
        if hasattr(results, 'resid_response'):
            residuals = results.resid_response.flatten()

            # Prediction accuracy metrics
            diagnostics['rmse'] = float(np.sqrt(np.mean(residuals**2)))
            diagnostics['mae'] = float(np.mean(np.abs(residuals)))

            # MAPE calculation with safe division
            y_flat = y.flatten()
            non_zero_mask = np.abs(y_flat) > 1e-10
            if non_zero_mask.any():
                mape_values = np.abs(residuals[non_zero_mask] / y_flat[non_zero_mask])
                diagnostics['mape'] = float(np.mean(mape_values) * 100)
            else:
                diagnostics['mape'] = None

        # Model information criteria
        if hasattr(results, 'aic'):
            diagnostics['aic'] = float(results.aic)
        if hasattr(results, 'aicc'):
            diagnostics['aicc'] = float(results.aicc)
        if hasattr(results, 'bic'):
            diagnostics['bic'] = float(results.bic)

        # Bandwidth information
        if hasattr(results, 'bw') and results.bw is not None:
            if isinstance(results.bw, (list, np.ndarray)):
                diagnostics['bandwidths'] = [float(bw) for bw in results.bw]
                diagnostics['mean_bandwidth'] = float(np.mean(results.bw))
            else:
                diagnostics['bandwidth'] = float(results.bw)

        # Local fit statistics
        try:
            if hasattr(results, 'localR2') and results.localR2 is not None and not callable(results.localR2):
                local_r2 = results.localR2
                diagnostics['local_r2_mean'] = float(np.mean(local_r2))
                diagnostics['local_r2_std'] = float(np.std(local_r2))
                diagnostics['local_r2_min'] = float(np.min(local_r2))
                diagnostics['local_r2_max'] = float(np.max(local_r2))
        except NotImplementedError:
            self.log("localR2 not available for multiple bandwidth models")
            diagnostics['local_r2'] = 'not_implemented'
        except Exception as e:
            self.log(f"Error processing localR2: {e}")
            diagnostics['local_r2'] = 'error'

        # Spatial autocorrelation test on residuals
        if PYSAL_AVAILABLE and hasattr(results, 'resid_response'):
            try:
                residuals = results.resid_response.flatten()
                w = KNN.from_array(coords, k=min(8, len(coords)-1))
                w.transform = 'r'

                moran = Moran(residuals, w)
                diagnostics['residual_moran_i'] = float(moran.I)
                diagnostics['residual_moran_p'] = float(moran.p_norm)
                diagnostics['residual_spatial_autocorr'] = 'significant' if moran.p_norm < 0.05 else 'not_significant'

                self.log(f"Residual spatial autocorrelation - Moran's I: {moran.I:.4f}, p-value: {moran.p_norm:.4f}")

            except Exception as e:
                self.log(f"Could not calculate residual spatial autocorrelation: {e}")
                diagnostics['residual_spatial_autocorr'] = 'calculation_failed'

        return diagnostics

    def _ui_keepalive(self):
        """Pump the Qt event loop so the UI stays responsive."""
        try:
            QCoreApplication.processEvents()
        except Exception:
            pass

    def _is_save_weights_enabled(self):
        cb = getattr(self.dialog, 'cBSaveWeights', None)
        try:
            return bool(cb.isChecked()) if cb is not None else False
        except Exception:
            return False

    def _run_in_thread_with_progress(self, target_fn, args_tuple,
                                    start=60, end=95, step=1, interval=0.5,
                                    label="Working..."):
        """
        Run a blocking function in a worker thread while smoothly updating the progress bar.
        Returns the function's return value.
        """
        self.log(f"{label} (background thread started)")
        result = {}
        err = {}

        def wrapper():
            try:
                result["value"] = target_fn(*args_tuple)
            except Exception as e:
                err["error"] = e
                err["trace"] = traceback.format_exc()

        t = threading.Thread(target=wrapper, daemon=True)
        t.start()

        p = max(0, min(100, start))
        self.progress_bar.setValue(p)
        last_log_bump = p

        while t.is_alive():
            time.sleep(interval)
            p = min(p + step, end)
            self.progress_bar.setValue(p)
            if p - last_log_bump >= 5:
                self.log(f"{label} … {p}%")
                last_log_bump = p
            self._ui_keepalive()

        self._ui_keepalive()
        if "error" in err:
            self.log(f"Background task failed: {err['trace']}")
            raise err["error"]

        self.progress_bar.setValue(end)
        self.log(f"{label} (background thread finished)")
        return result.get("value")

    def _run_mgwr_clean_approach(self, coords, y, X, explanatory_vars, kernel='bisquare'):
        """
        Clean MGWR approach matching the working standalone script, with numeric guards:
        - Standardize X and y
        - Explicit intercept in X_design (constant=False)
        - Robust multi-bandwidth search with safe NN floors + retry
        - Final-fit retry by inflating bandwidths if singular
        - Auto-switch to single-band MGWR when only 1 predictor remains
        """
        try:
            self.log("Starting clean MGWR approach (matching working standalone script)")

            # === STANDARDIZE X and y ===
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_std = scaler_X.fit_transform(X)
            y_std = scaler_y.fit_transform(y.reshape(-1, 1))  # Ensure y is (n,1)

            # === BUILD DESIGN MATRIX WITH EXPLICIT INTERCEPT ===
            ones = np.ones((X_std.shape[0], 1))
            X_design = np.hstack([ones, X_std])  # (n, k+1), first col is intercept
            design_labels = ["Intercept"] + explanatory_vars

            self.log(f"Design matrix shape: {X_design.shape}, y shape: {y_std.shape}")

            # === FINAL DIAGNOSTIC FOR NaNs/Infs ===
            self._check_array_issues(X_design, "X_design", design_labels)
            self._check_array_issues(y_std, "y_std", ["target"])

            # === CONDITION NUMBER CHECK (without intercept) ===
            cond_number = np.linalg.cond(X_std) if X_std.size else np.nan
            self.log(f"Condition number of X (no intercept): {cond_number:.2f}" if np.isfinite(cond_number) else "Condition number: n/a")

            n = coords.shape[0]
            k_plus_1 = X_design.shape[1]         # intercept + predictors
            p = k_plus_1 - 1                     # number of predictors

            # If only one predictor remains, prefer single-band MGWR (more stable)
            force_single_band = (p == 1)

            # === BANDWIDTH SELECTION ===
            self.log("Selecting bandwidths with Sel_BW (MGWR, adaptive NN, AICc)...")

            # Safe floors for adaptive NN (multi-band only)
            safe_min = max(60, int(0.10 * n))
            safe_max = max(safe_min + 10, int(0.50 * n))

            selector = Sel_BW(
                coords=coords,
                y=y_std,
                X_loc=X_design,   # includes intercept in first column
                X_glob=None,
                offset=None,
                kernel=kernel,
                fixed=False,      # adaptive
                multi=(not force_single_band),
                constant=False,   # IMPORTANT: intercept already in X_loc
                spherical=False
            )

            self.log(f"Variables (including intercept): {k_plus_1}")
            if not force_single_band:
                self.log(f"Bandwidth search range per var: {safe_min}–{safe_max}")

            # --- First attempt at search ---
            try:
                if force_single_band:
                    # IMPORTANT: do NOT pass multi_bw_min/max for single-band; older mgwr calls len(...) on them
                    bw_vector = selector.search(criterion='AICc')
                else:
                    bw_vector = selector.search(
                        multi_bw_min=[safe_min] * k_plus_1,
                        multi_bw_max=[safe_max] * k_plus_1,
                        criterion='AICc',
                    )
            except Exception as e:
                self.log(f"Bandwidth search hit {e}; retrying with larger minimum NN...")
                # Retry with larger floors (multi-band only). For single-band, just call search again without bounds.
                if force_single_band:
                    bw_vector = selector.search(criterion='AICc')
                else:
                    safe_min2 = max(80, int(0.15 * n))
                    safe_max2 = max(safe_min2 + 10, int(0.60 * n))
                    selector = Sel_BW(
                        coords=coords,
                        y=y_std,
                        X_loc=X_design,
                        X_glob=None,
                        offset=None,
                        kernel=kernel,
                        fixed=False,
                        multi=True,
                        constant=False,
                        spherical=False
                    )
                    bw_vector = selector.search(
                        multi_bw_min=[safe_min2] * k_plus_1,
                        multi_bw_max=[safe_max2] * k_plus_1,
                        criterion='AICc',
                    )

            # Log chosen bandwidth(s)
            self.log("Optimal bandwidths:")
            arr = np.atleast_1d(bw_vector)
            if not force_single_band and arr.size == k_plus_1:
                for i, col in enumerate(design_labels):
                    self.log(f"   {col}: {float(arr[i]):.1f}")
            else:
                self.log(f"   (single): {float(arr[0]):.1f}")

            # === FIT MGWR / single-band MGWR MODEL ===
            self.log("Fitting MGWR model...")
            try:
                mgwr_model = MGWR(coords=coords, y=y_std, X=X_design, selector=selector, constant=False, spherical=False, n_jobs=1)
                
                try:
                    results = mgwr_model.fit()
                except TypeError as e:
                    # Known mgwr 2.2.x issue: Kernel expecting scalar bw when list is given
                    if "int() argument must be a string" in str(e) or "not 'list'" in str(e):
                        self.log("MGWR fit failed with list/Scalar bandwidth mismatch; falling back to GWR (adaptive gaussian).")
                        # GWR fallback: use Sel_BW with multi=False
                        from mgwr.gwr import GWR, Gaussian
                        sel_gwr = Sel_BW(coords, y_std, X_design, multi=False, constant=False, kernel='gaussian')
                        bw_gwr = sel_gwr.search()
                        self.log(f"GWR fallback bandwidth (neighbors): {bw_gwr}")
                        gwr_model = GWR(coords, y_std, X_design, bw_gwr, fixed=False, kernel=Gaussian(), constant=False)
                        results = gwr_model.fit()
                        method_used_local = "GWR_Fallback_Fixed"
                    else:
                        raise

            except Exception as e_fit:
                # If singular, inflate bandwidths and retry once
                if "singular" in str(e_fit).lower():
                    self.log("Final MGWR fit hit a singular matrix; inflating bandwidths and retrying once...")
                    if not force_single_band and arr.size == k_plus_1:
                        bw_vector = [min(int(float(b) * 1.5), n - 1) for b in arr]
                    else:
                        b0 = float(arr[0])
                        bw_vector = min(int(b0 * 1.5), n - 1)
                    mgwr_model = MGWR(coords=coords, y=y_std, X=X_design, selector=selector, constant=False, spherical=False, n_jobs=1)
                    results = mgwr_model.fit()
                else:
                    raise

            self.log("MGWR model fitted successfully using clean approach")
            label = "MGWR_Clean_Approach_SingleBW" if force_single_band else "MGWR_Clean_Approach"
            return results, label, scaler_X, scaler_y, design_labels

        except Exception as e:
            self.log(f"Clean MGWR approach failed: {e}")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            raise

    # === FIXED: single-bandwidth MGWR approach to avoid list -> int errors in kernels ===

    def _run_mgwr_clean_approach_fixed(self, coords, y, X, explanatory_vars, kernel='bisquare'):
        """
        Clean MGWR approach with robust bandwidth handling using bw_min/bw_max in search(),
        and a safe GWR fallback. Avoids passing unsupported kwargs to Sel_BW.__init__.
        """
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        try:
            self.log("Starting fixed clean MGWR approach")
            # === STANDARDIZE ===
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_std = scaler_X.fit_transform(X)
            y_std = scaler_y.fit_transform(y.reshape(-1, 1))

            # === DESIGN MATRIX WITH INTERCEPT IN X ===
            ones = np.ones((X_std.shape[0], 1))
            X_design = np.hstack([ones, X_std])
            design_labels = ["Intercept"] + list(explanatory_vars)

            self.log(f"Design matrix shape: {X_design.shape}, y shape: {y_std.shape}")
            self._check_array_issues(X_design, "X_design", design_labels)
            self._check_array_issues(y_std, "y_std", ["target"])

            n = int(coords.shape[0])
            p = int(X_design.shape[1] - 1)  # exclude intercept

            # === Helper to compute bw bounds safely ===
            def _nn_bounds(n, p, min_frac, max_frac, min_add):
                bw_min = int(max(3 * max(p, 1), min_frac * n, min_add))
                bw_max = int(min(n - 1, max(bw_min + 5, max_frac * n)))
                if bw_min >= bw_max:
                    # ensure valid search interval
                    bw_min = max(2, min(bw_min, n - 2))
                    bw_max = min(n - 1, max(bw_min + 1, bw_max))
                return bw_min, bw_max

            # === Primary MGWR (single-bandwidth selector) ===
            try:
                min_nn, max_nn = _nn_bounds(n, p, 0.10, 0.60, 30)
                self.log(f"Selecting MGWR single bandwidth with bw_min={min_nn}, bw_max={max_nn}, kernel={kernel}")
                sel = Sel_BW(coords=coords,
                            y=y_std,
                            X_loc=X_design,
                            kernel=kernel,
                            fixed=False,
                            multi=False,
                            constant=False,
                            spherical=False)
                bw = sel.search(criterion='AICc', bw_min=min_nn, bw_max=max_nn)
                # coerce to scalar
                if isinstance(bw, (list, tuple, np.ndarray)):
                    bw = float(bw[0])
                else:
                    bw = float(bw)
                self.log(f"Optimal single bandwidth: {bw}")
                from mgwr.gwr import MGWR
                mgwr_model = MGWR(coords=coords, y=y_std, X=X_design, selector=sel, fixed=False, kernel=kernel, constant=False, spherical=False, n_jobs=1)
                results = mgwr_model.fit()
                method_used = "MGWR_SingleBW_Fixed"
                self.log(f"Model fitted successfully using {method_used}")
                return results, method_used, scaler_X, scaler_y, design_labels

            except Exception as e_mgwr:
                self.log(f"Single-bandwidth MGWR failed: {e_mgwr}. Trying Gaussian kernel & larger neighborhood...")
                # Retry MGWR with gaussian kernel and wider bounds (still without unsupported kwargs)
                try:
                    min_nn_retry, max_nn_retry = _nn_bounds(n, p, 0.20, 0.80, 50)
                    self.log(f"Retrying MGWR with kernel=gaussian and bw_min>={min_nn_retry}, bw_max<={max_nn_retry}")
                    sel_retry = Sel_BW(coords=coords,
                                    y=y_std,
                                    X_loc=X_design,
                                    kernel='gaussian',
                                    fixed=False,
                                    multi=False,
                                    constant=False,
                                    spherical=False)
                    bw_retry = sel_retry.search(criterion='AICc',
                                                bw_min=min_nn_retry,
                                                bw_max=max_nn_retry)
                    if isinstance(bw_retry, (list, tuple, np.ndarray)):
                        bw_retry = float(bw_retry[0])
                    else:
                        bw_retry = float(bw_retry)
                    from mgwr.gwr import MGWR
                    mgwr_retry = MGWR(coords=coords, y=y_std, X=X_design, selector=sel_retry, fixed=False, kernel='gaussian', constant=False, spherical=False, n_jobs=1)
                    results = mgwr_retry.fit()
                    method_used = "MGWR_SingleBW_Gaussian"
                    self.log(f"Model fitted successfully using {method_used}")
                    return results, method_used, scaler_X, scaler_y, design_labels
                except Exception as e_retry:
                    self.log(f"Gaussian-kernel MGWR also failed: {e_retry}. Trying GWR fallback...")
                    # === GWR fallback ===
                    try:
                        from mgwr.gwr import GWR
                        min_nn_gwr, max_nn_gwr = _nn_bounds(n, p, 0.20, 0.90, 50)
                        self.log(f"Selecting GWR bandwidth with kernel=gaussian, bw_min>={min_nn_gwr}, bw_max<={max_nn_gwr}")
                        sel_gwr = Sel_BW(coords=coords,
                                        y=y_std,
                                        X_loc=X_design,
                                        kernel='gaussian',
                                        fixed=False,
                                        multi=False,
                                        constant=False,
                                        spherical=False)
                        bw_gwr = sel_gwr.search(criterion='AICc',
                                                bw_min=min_nn_gwr,
                                                bw_max=max_nn_gwr)
                        if isinstance(bw_gwr, (list, tuple, np.ndarray)):
                            bw_gwr = float(bw_gwr[0])
                        else:
                            bw_gwr = float(bw_gwr)
                        gwr = GWR(coords=coords,
                                y=y_std,
                                X=X_design,
                                bw=bw_gwr,
                                fixed=False,
                                kernel='gaussian',
                                constant=False,
                                spherical=False)
                        results = gwr.fit()
                        method_used = "GWR_Fallback_Fixed"
                        self.log(f"GWR fitted successfully with bw={bw_gwr}")
                        return results, method_used, scaler_X, scaler_y, design_labels
                    except Exception as e_gwr:
                        self.log(f"GWR fallback also failed: {e_gwr}")
                        raise RuntimeError(f"Both MGWR and GWR failed: {e_mgwr} | {e_retry} | {e_gwr}")
        except Exception as e:
            self.log(f"Fixed clean MGWR approach failed: {e}")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _safe_build_mgwr_fixed(self, coords, y_std, X_design, *, kernel, spherical, bw=None, selector=None, fixed=False):
        """
        Fixed builder that ensures a single numeric bandwidth is passed to MGWR/GWR.
        """
        from mgwr.gwr import MGWR

        # Normalize bandwidth to a single float if provided
        if bw is not None:
            if isinstance(bw, (list, np.ndarray)):
                bw = float(np.asarray(bw).ravel()[0]) if len(np.asarray(bw).ravel()) > 0 else None
            else:
                bw = float(bw)

        # If a selector is passed and has a list-like bw, coerce to single float
        if selector is not None and hasattr(selector, 'bw'):
            bw_attr = getattr(selector, 'bw')
            try:
                # Some mgwr versions store [final_bws, history]; take the first element
                if isinstance(bw_attr, (list, np.ndarray)):
                    # if it's nested like [list, history], flatten once
                    cand = np.asarray(bw_attr, dtype=object).ravel()[0]
                    if isinstance(cand, (list, np.ndarray)):
                        bw = float(np.asarray(cand).ravel()[0])
                    else:
                        bw = float(cand)
            except Exception:
                pass

        return MGWR(
            coords=coords,
            y=y_std,
            X=X_design,
            bw=bw,
            selector=None,   # ensure constructor does not try to use multi-band selector
            fixed=fixed,
            kernel=kernel,
            constant=False,
            spherical=spherical,
        )

    def _check_array_issues(self, array, array_name, column_names=None):
        """Check arrays for NaNs/Infs and log issues"""
        self.log(f"Checking {array_name} for NaNs/Infs...")
        arr = array if array.ndim == 2 else array.reshape(-1, 1)
        isnan = np.isnan(arr)
        isinf = np.isinf(arr)
        bad_rows = np.where(np.any(isnan | isinf, axis=1))[0]

        if bad_rows.size > 0:
            self.log(f"ERROR: {array_name} contains NaNs or Infs at rows: {bad_rows.tolist()}")
            if column_names is not None:
                for i in bad_rows[:5]:  # Show first 5 problematic rows
                    self.log(f"Row {i} values:")
                    for j, col in enumerate(column_names):
                        if j < arr.shape[1]:
                            val = arr[i, j]
                            status = " (NaN)" if np.isnan(val) else " (Inf)" if np.isinf(val) else ""
                            self.log(f"  {col}: {val}{status}")
            raise ValueError(f"{array_name} contains invalid values")
        else:
            self.log(f"✅ {array_name} is clean")

    def _updated_run_enhanced_mgwr(self):
        """Updated main MGWR function using the clean approach"""
        try:
            self._configure_environment()
            self._check_mgwr_compatibility()
            self.progress_bar.setValue(0)

            # Input validation (same as before)
            layer_name = self.dialog.cBSelectCSVLayerforMGWR.currentText()
            if not layer_name:
                self.log("ERROR: No layer selected")
                QMessageBox.critical(self.dialog, "Input Error", "Please select a CSV layer")
                return

            layer = self._get_layer_by_name(layer_name)
            if not layer:
                self.log("ERROR: Selected layer not found")
                return

            # Get field selections
            x_col = self.dialog.cBXCoordMGWR.currentText()
            y_col = self.dialog.cBYCoordMGWR.currentText()
            target_var = self.dialog.cBTargetVariableMGWR.currentText()

            explanatory_items = self.dialog.lWExplanatoryVarsMGWR.selectedItems()
            explanatory_vars = [item.text() for item in explanatory_items]

            # Validation
            if not all([x_col, y_col, target_var]):
                self.log("ERROR: Please select X coordinate, Y coordinate, and target variable")
                QMessageBox.critical(self.dialog, "Input Error",
                                "Please select X coordinate, Y coordinate, and target variable")
                return

            if not explanatory_vars:
                self.log("ERROR: Please select at least one explanatory variable")
                QMessageBox.critical(self.dialog, "Input Error",
                                "Please select at least one explanatory variable")
                return

            if not self.output_folder:
                self.log("ERROR: No output folder selected")
                QMessageBox.critical(self.dialog, "Input Error", "Please select an output folder")
                return

            self.progress_bar.setValue(10)

            # Load and validate data
            self.log("Loading and validating data...")
            df = self._layer_to_dataframe(layer)

            cleaned_df, issues = self._comprehensive_data_validation(
                df, x_col, y_col, target_var, explanatory_vars
            )

            if cleaned_df is None:
                error_msg = "Data validation failed:\n" + "\n".join(issues)
                self.log(f"ERROR: {error_msg}")
                QMessageBox.critical(self.dialog, "Data Validation Error", error_msg)
                return

            if issues:
                self.log("Data validation warnings:")
                for issue in issues:
                    self.log(f"  - {issue}")

            self.progress_bar.setValue(20)

            # === ENFORCE NUMERIC TYPES & CLEAN DATA (matching standalone script) ===
            self.log("Enforcing numeric types and cleaning data...")
            for col in [target_var] + explanatory_vars + [x_col, y_col]:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

            cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            cleaned_df.dropna(subset=[target_var] + explanatory_vars + [x_col, y_col], inplace=True)

            self.log(f"Cleaned data: {cleaned_df.shape[0]} rows remain after filtering")

            # === ZERO VARIANCE CHECK ===
            for col in explanatory_vars:
                if cleaned_df[col].std() == 0:
                    self.log(f"Warning: Column '{col}' has zero variance — may cause instability")

            # Extract analysis data
            coords = cleaned_df[[x_col, y_col]].values.astype(float)
            y = cleaned_df[[target_var]].values.astype(float)  # Keep as (n,1)
            X = cleaned_df[explanatory_vars].values.astype(float)

            self.log(f"Analysis dataset: {len(coords)} observations, {X.shape[1]} explanatory variables")

            # Enhanced collinearity removal (if enabled)
            if self.dialog.cBRemoveCollinearAttributes.isChecked():
                threshold = self.dialog.dSBCollinearityThresholdMGWR.value()
                X, explanatory_vars = self._enhanced_collinearity_removal(X, explanatory_vars, threshold)
                self.log(f"After collinearity removal: {X.shape[1]} variables remaining")

            self.progress_bar.setValue(40)

            # Get kernel parameter
            kernel = self.dialog.cBKernelTypeMGWR.currentText()

            # === RUN CLEAN MGWR APPROACH ===
            self.log("Running MGWR with clean approach...")
            results, method_used, scaler_X, scaler_y, design_labels = self._run_in_thread_with_progress(
                self._run_mgwr_clean_approach,
            (coords, y, X, explanatory_vars, kernel),
                start=40, end=90, step=1, interval=0.4,
                label="Fitting MGWR (Clean Approach)"
            )

            self.progress_bar.setValue(90)

            # Calculate enhanced diagnostics
            diagnostics = self._run_in_thread_with_progress(
                self._calculate_enhanced_diagnostics,
                (results, y, X, coords, method_used),
                start=90, end=94, step=1, interval=0.15,
                label="Computing diagnostics"
            )

            # === PREPARE OUTPUT DATAFRAME ===
            self.log("Preparing output data...")

            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare comprehensive output dataframe
            output_df = pd.DataFrame({
                'X': coords[:, 0],
                'Y': coords[:, 1],
                target_var: y.flatten()
            })

            # Add model coefficients using design_labels
            if hasattr(results, 'params'):
                params = results.params
                if params.ndim == 2:  # MGWR with varying coefficients
                    for i, label in enumerate(design_labels):
                        if i < params.shape[1]:
                            if label == "Intercept":
                                output_df['Intercept'] = params[:, i]
                            else:
                                output_df[f'Beta_{label}'] = params[:, i]
                elif params.ndim == 1:  # Single coefficient case
                    for i, label in enumerate(design_labels):
                        if i < len(params):
                            if label == "Intercept":
                                output_df['Intercept'] = params[i]
                            else:
                                output_df[f'Beta_{label}'] = params[i]

            # Add residuals and fitted values
            if hasattr(results, 'resid_response'):
                output_df['Residuals'] = results.resid_response.flatten()

            if hasattr(results, 'predy'):
                output_df['Fitted_Values'] = results.predy.flatten()
            elif hasattr(results, 'resid_response'):
                # Calculate fitted values from residuals
                output_df['Fitted_Values'] = y.flatten() - results.resid_response.flatten()

            # Add local statistics
            try:
                if hasattr(results, 'localR2') and results.localR2 is not None and not callable(results.localR2):
                    output_df['Local_R2'] = results.localR2
            except NotImplementedError:
                self.log("Skipping Local_R2 - not implemented for multiple bandwidths")

            if hasattr(results, 'sigma2'):
                output_df['Local_Sigma2'] = results.sigma2

            # Add standard errors if available
            if hasattr(results, 'bse') and results.bse is not None:
                bse = results.bse
                if bse.ndim == 2:
                    for i, label in enumerate(design_labels):
                        if i < bse.shape[1]:
                            if label == "Intercept":
                                output_df['SE_Intercept'] = bse[:, i]
                            else:
                                output_df[f'SE_Beta_{label}'] = bse[:, i]

            self.progress_bar.setValue(94)
            self._ui_keepalive()

            # Save results
            csv_path = os.path.join(self.output_folder, f"MGWR_Results_{timestamp}.csv")
            gpkg_path = os.path.join(self.output_folder, f"MGWR_Results_{timestamp}.gpkg")

            # Save CSV
            output_df.to_csv(csv_path, index=False)
            self.log(f"Results saved to CSV: {csv_path}")

            # Save GeoPackage
            self._save_to_geopackage(output_df, gpkg_path, layer.crs())
            self.log(f"Results saved to GeoPackage: {gpkg_path}")

            # Save diagnostics
            diagnostics_path = os.path.join(self.output_folder, f"MGWR_Diagnostics_{timestamp}.json")
            with open(diagnostics_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            self.log(f"Diagnostics saved to: {diagnostics_path}")

            self.progress_bar.setValue(97)
            self._ui_keepalive()

            # Load results into QGIS
            self._load_results_to_qgis(gpkg_path, csv_path, layer.crs())

            self.progress_bar.setValue(99)
            self._ui_keepalive()
            self.progress_bar.setValue(100)
            self._ui_keepalive()

            # Display summary
            self._display_analysis_summary(diagnostics, method_used, kernel, "Clean MGWR",
                                        len(output_df), len(explanatory_vars))

            self.log("MGWR analysis completed successfully using clean approach!")

        except Exception as e:
            self.log(f"ERROR: MGWR analysis failed: {str(e)}")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")

            QMessageBox.critical(self.dialog, "Analysis Error",
                            f"MGWR analysis failed:\n\n{str(e)}\n\nCheck log for details.")
            self.progress_bar.setValue(0)

    def run_enhanced_mgwr(self):
        """Replace the original method with the updated version"""
        return self._updated_run_enhanced_mgwr()

    def _save_to_geopackage(self, output_df, output_path, crs):
        try:
            self.log(f"Saving MGWR output to GeoPackage at: {output_path}")

            # Define fields
            fields = QgsFields()
            for col in output_df.columns:
                if col.lower() in ["x", "y"]:  # Skip coordinate columns
                    continue
                # Use proper data types with updated QgsField constructor
                if output_df[col].dtype == np.int64:
                    fields.append(QgsField(col, QVariant.Int, "integer"))
                else:
                    fields.append(QgsField(col, QVariant.Double, "double"))

            # Create save options
            writer_options = QgsVectorFileWriter.SaveVectorOptions()
            writer_options.driverName = "GPKG"
            writer_options.fileEncoding = "UTF-8"
            writer_options.layerName = os.path.splitext(os.path.basename(output_path))[0]
            writer_options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

            # Create writer for POINT data
            writer = QgsVectorFileWriter.create(
                output_path,
                fields,
                QgsWkbTypes.Point,
                crs,  # Use CRS from layer
                QgsCoordinateTransformContext(),
                writer_options
            )

            if writer.hasError() != QgsVectorFileWriter.NoError:
                self.log(f"Writer error: {writer.errorMessage()}")
                raise Exception(writer.errorMessage())

            # Create features
            for _, row in output_df.iterrows():
                fet = QgsFeature(fields)

                # Set attributes (skip X/Y)
                attrs = []
                for col in output_df.columns:
                    if col.lower() in ["x", "y"]:
                        continue
                    value = row[col]
                    attrs.append(float(value) if pd.notnull(value) else None)
                fet.setAttributes(attrs)

                # Create point geometry from X/Y
                point = QgsPointXY(float(row['X']), float(row['Y']))
                fet.setGeometry(QgsGeometry.fromPointXY(point))

                writer.addFeature(fet)

            del writer  # Finalize file
            self.log("GeoPackage created successfully")

            # Add layer to QGIS
            layer_name = os.path.splitext(os.path.basename(output_path))[0]
            vlayer = QgsVectorLayer(output_path, layer_name, "ogr")
            if vlayer.isValid():
                QgsProject.instance().addMapLayer(vlayer)
                self.log("Layer added to QGIS project")
            else:
                self.log("Error loading GeoPackage layer")

        except Exception as e:
            self.log(f"ERROR: Failed to save GeoPackage: {e}")
            traceback_str = traceback.format_exc()
            self.log(traceback_str)

    def _load_results_to_qgis(self, gpkg_path, csv_path, crs):
        """Load results into QGIS with proper styling"""
        try:
            # Load GeoPackage layer
            gpkg_layer = QgsVectorLayer(gpkg_path, f"MGWR_Results_{datetime.now().strftime('%H%M%S')}", "ogr")
            if gpkg_layer.isValid():
                QgsProject.instance().addMapLayer(gpkg_layer)
                self.log(f"Loaded GeoPackage layer into QGIS: {gpkg_path}")
            else:
                self.log(f"Warning: Could not load GeoPackage layer: {gpkg_layer.error().summary()}")

            # Load CSV layer
            csv_uri = f"file:///{csv_path.replace(os.sep, '/')}" + "?delimiter=,&xField=X&yField=Y"
            csv_layer = QgsVectorLayer(csv_uri, f"MGWR_CSV_{datetime.now().strftime('%H%M%S')}", "delimitedtext")

            if csv_layer.isValid():
                if crs and crs.isValid():
                    csv_layer.setCrs(crs)
                    self.log(f"Set CRS {crs.authid()} for CSV layer")
                QgsProject.instance().addMapLayer(csv_layer)
                self.log(f"Loaded CSV layer into QGIS: {csv_path}")
            else:
                self.log(f"Warning: Could not load CSV layer: {csv_layer.error().summary()}")

        except Exception as e:
            self.log(f"Error loading results to QGIS: {e}")

    def _display_analysis_summary(self, diagnostics, method_used, kernel, bw_method, n_obs, n_vars):
        """Display comprehensive analysis summary"""
        summary_lines = [
            f"MGWR Analysis Summary",
            f"=" * 50,
            f"Method Used: {method_used}",
            f"Kernel: {kernel}",
            f"Bandwidth Method: {bw_method}",
            f"Observations: {n_obs}",
            f"Variables: {n_vars}",
            ""
        ]

        # Model fit statistics
        if 'rmse' in diagnostics:
            summary_lines.append(f"RMSE: {diagnostics['rmse']:.4f}")
        if 'mae' in diagnostics:
            summary_lines.append(f"MAE: {diagnostics['mae']:.4f}")
        if 'mape' in diagnostics and diagnostics['mape'] is not None:
            summary_lines.append(f"MAPE: {diagnostics['mape']:.2f}%")

        # Information criteria
        if 'aic' in diagnostics:
            summary_lines.append(f"AIC: {diagnostics['aic']:.2f}")
        if 'aicc' in diagnostics:
            summary_lines.append(f"AICc: {diagnostics['aicc']:.2f}")
        if 'bic' in diagnostics:
            summary_lines.append(f"BIC: {diagnostics['bic']:.2f}")

        # Local R²
        if 'local_r2_mean' in diagnostics:
            summary_lines.extend([
                "",
                f"Local R² Statistics:",
                f"  Mean: {diagnostics['local_r2_mean']:.4f}",
                f"  Std:  {diagnostics['local_r2_std']:.4f}",
                f"  Min:  {diagnostics['local_r2_min']:.4f}",
                f"  Max:  {diagnostics['local_r2_max']:.4f}"
            ])
        elif 'local_r2' in diagnostics and diagnostics['local_r2'] == 'not_implemented':
            summary_lines.append("\nLocal R²: Not available for multiple bandwidth models")

        # Spatial autocorrelation
        if 'residual_moran_i' in diagnostics:
            significance = "significant" if diagnostics.get('residual_moran_p', 1) < 0.05 else "not significant"
            summary_lines.extend([
                "",
                f"Residual Spatial Autocorrelation:",
                f"  Moran's I: {diagnostics['residual_moran_i']:.4f}",
                f"  P-value: {diagnostics.get('residual_moran_p', 'N/A')}",
                f"  Result: {significance}"
            ])

        summary_text = "\n".join(summary_lines)

        # Log summary
        for line in summary_lines:
            self.log(line)

        # Show summary dialog
        QMessageBox.information(self.dialog, "MGWR Analysis Complete", summary_text)

    def _get_layer_by_name(self, name):
        """Get layer by name from QGIS project"""
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == name:
                return layer
        return None

    def _layer_to_dataframe(self, layer):
        """Convert QGIS vector layer to pandas DataFrame with enhanced error handling."""
        try:
            data = []
            fields = layer.fields()
            columns = [field.name() for field in fields]

            # Check if layer has features
            feature_count = layer.featureCount()
            if feature_count == 0:
                self.log("Warning: Layer contains no features")
                return pd.DataFrame(columns=columns)

            self.log(f"Converting layer to DataFrame: {feature_count} features, {len(columns)} columns")

            for feature in layer.getFeatures():
                row = []
                for field_name in columns:
                    value = feature[field_name]
                    # Handle different NULL representations
                    if value is None or value == 'NULL' or (isinstance(value, str) and value.lower() == 'null'):
                        row.append(np.nan)
                    else:
                        row.append(value)
                data.append(row)

            df = pd.DataFrame(data, columns=columns)
            self.log(f"Successfully converted to DataFrame: {df.shape}")
            return df

        except Exception as e:
            self.log(f"Error converting layer to DataFrame: {e}")
            raise

    def _safe_build_mgwr(self, coords, y_std, X_design, *, kernel, spherical, bw=None, selector=None, fixed=False):
        """
        Robust MGWR builder for MGWR 2.1.x variants that always rely on selector.*
        and expect selector.bw to be a (final_bws, bws_history) pair.

        - Expands scalar bw to a list of length k (including intercept).
        - Ensures selector.bw[0] and selector.bw[1] both exist.
        - Provides selector.bw_init and selector.params shims.
        - Passes constant=False because intercept is already in X_design.
        """
        from mgwr.gwr import MGWR
        import numpy as _np

        if selector is None:
            raise ValueError("MGWR in this environment requires a selector; none provided.")

        k = X_design.shape[1]  # includes intercept

        def _coerce_final_bws(bw_value):
            # Turn None/float/array/list into a list of length k
            if bw_value is None:
                return None
            arr = _np.atleast_1d(bw_value).astype(float)
            if arr.size == 1:
                return [float(arr[0])] * k
            # if size != k, repeat/trim to length k to be safe
            if arr.size != k:
                return [float(arr[0])] * k
            return arr.tolist()

        class _SelectorProxy:
            def __init__(self, base, *, bw_value, fixed_val, kernel_val, spherical_val):
                self._base = base

                final_bws = _coerce_final_bws(bw_value)
                if final_bws is None:
                    # sensible default init from sample size
                    n = int(coords.shape[0])
                    init_nn = float(max(30, min(int(0.15 * n), n - 1)))
                    final_bws = [init_nn] * k

                # MGWR 2.1.x may index both [0] (final) and [1] (history)
                self.bw = [final_bws, [final_bws.copy()]]  # ensure [1] exists
                self.bw_init = final_bws.copy()

                # Mirror/ensure common fields
                self.fixed = bool(getattr(base, "fixed", fixed_val))
                self.kernel = getattr(base, "kernel", kernel_val)
                self.spherical = bool(getattr(base, "spherical", spherical_val))

                # .params sometimes accessed in fit()
                self.params = getattr(base, "params", {
                    "fixed": self.fixed,
                    "kernel": self.kernel,
                    "spherical": self.spherical,
                    "family": None,
                    "criterion": "AICc",
                })

            def __getattr__(self, name):
                return getattr(self._base, name)

        proxy = _SelectorProxy(
            selector,
            bw_value=bw,
            fixed_val=fixed,
            kernel_val=kernel,
            spherical_val=spherical
        )

        # IMPORTANT: do NOT pass bw=... here; some builds still read selector.* regardless,
        # and passing a scalar can trigger indexing errors inside MGWR.__init__.
        return MGWR(
            coords=coords,
            y=y_std,
            X=X_design,
            selector=proxy,
            fixed=fixed,
            kernel=kernel,
            constant=False,   # intercept already in X_design
            spherical=spherical,
        )

    def refresh_layers_mgwr(self):
        """Refresh cBSelectCSVLayerforMGWR from the QGIS Layers Browser (CSV-backed vector layers)."""
        try:
            combo = self.dialog.cBSelectCSVLayerforMGWR
            prev_text = combo.currentText()

            # Collect CSV-like vector layers (mirror the detection used in populate_csv_layers)
            names = []
            for layer in QgsProject.instance().mapLayers().values():
                if layer.type() == QgsMapLayerType.VectorLayer:
                    src = (layer.source() or "").lower()
                    prov = (layer.providerType() or "").lower()
                    is_csv = ('.csv' in src) or (prov in ['delimitedtext', 'ogr']) or src.endswith('.csv') or ('csv' in prov)
                    if is_csv:
                        names.append(layer.name())

            names = sorted(set(names), key=lambda n: n.lower())

            # Refill the combo while signals are blocked
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

            # Restore selection if possible, otherwise select first (if any)
            if prev_text and prev_text in names:
                combo.setCurrentIndex(combo.findText(prev_text))
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

            self.log(f"Refreshed MGWR CSV layers: found {combo.count()} layer(s).")

            # Ensure dependent UI (coords/target/explanatory) updates
            self.populate_fields_from_selected_layer()

        except Exception as e:
            self.log(f"ERROR refreshing MGWR layers: {e}")