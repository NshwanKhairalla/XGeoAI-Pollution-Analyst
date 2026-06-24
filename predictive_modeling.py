# predictive_modeling.py

from typing import List, Optional, Tuple
import os
import math
import traceback
import logging
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem, QButtonGroup
from qgis.PyQt.QtCore import QObject, QVariant

# --- Qt type compatibility helpers (Qt5/Qt6) ---
try:
    from qgis.PyQt.QtCore import QMetaType  # Qt6 style
except Exception:  # Qt5 fallback
    QMetaType = None
from qgis.PyQt.QtCore import QVariant

def _qt_type_double():
    try:
        if QMetaType is not None and hasattr(QMetaType, 'Double'):
            return int(QMetaType.Double)
    except Exception:
        pass
    return int(QVariant.Double)

def _qt_type_int():
    try:
        if QMetaType is not None and hasattr(QMetaType, 'Int'):
            return int(QMetaType.Int)
    except Exception:
        pass
    return int(QVariant.Int)
def _qt_type_string():
    try:
        if QMetaType is not None and hasattr(QMetaType, 'QString'):
            return int(QMetaType.QString)
    except Exception:
        pass
    return int(QVariant.String)
# --- end helpers ---


from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext,
    QgsTask,
    QgsProcessingFeedback,
    QgsApplication,
    QgsCoordinateReferenceSystem
)

from PyQt5.QtGui import QIntValidator, QDoubleValidator

# ML imports
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")
except Exception as e:
    _HAS_XGB = False
    logging.error(f"Error importing XGBoost: {e}")


class PredictiveModelingController(QObject):
    @staticmethod
    def _sanitize_layer_name(name: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in ('_',) else '_' for ch in name)
        # GeoPackage layer name max practical length ~ 255; keep it conservative
        return (safe or 'layer')[:120]

    """
    Controller for Step 9 – Predictive Modeling.

    Expects the dialog to expose the following widgets (exact names per user spec):
      cBInputDataPredictiveModeling    : QComboBox
      cBTargetVariableModeling         : QComboBox
      dSBThresholdF, dSBThresholdI,
      dSBThresholdEX                   : QDoubleSpinBox
      cBEnableBinaryClassification     : QCheckBox
      lWFeatureVariablesModeling       : QListWidget (multi-select of candidate features)
      cBEnableMGWRFeatures             : QCheckBox
      rBRF, rBXGBoost                  : QRadioButton
      lERFNumberofTrees                : QLineEdit
      lERFMaximumDepthofTrees          : QLineEdit
      rBEnableCrossValidation          : QRadioButton (10-fold)
      rBTrainTestSplit                 : QRadioButton (70/30)
      lEXGLearningRate                 : QLineEdit
      lEXGNumberofBoostingRounds       : QLineEdit
      pBRunModeling                    : QPushButton
      tBChooseFolderPredictiveModelingRFXG : QToolButton
      pBPredictiveModelingSaveRFXG     : QPushButton
      pBPredictiveModelingNext         : QPushButton
      pBPredictiveModeling             : QProgressBar
      tInterpretResults                : (QWidget tab) accessible via dialog.tabs or similar

    Optional:
      self.iface for adding layers and logging to QGIS message bar.
      dialog.tELog (QPlainTextEdit / QTextEdit) for textual logs (if present).
    """

    def __init__(self, dialog, iface=None, logger: Optional[logging.Logger] = None):
        super().__init__(dialog)
        self.dlg = dialog
        self.iface = iface
        self.logger = logger or logging.getLogger("PredictiveModeling")
        if not self.logger.handlers:
            # Simple console handler fallback
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # --- Worker-thread note buffer (flushed on_finished) ---
        self._worker_notes = []

        # Runtime state
        self._output_folder: Optional[str] = None
        self._last_results: Optional[pd.DataFrame] = None  # predictions joined to attributes
        self._last_metrics: Optional[dict] = None
        self._last_model = None
        self._last_model_is_xgb = False
        self._last_model_is_classification = False
        self._last_layer_id: Optional[str] = None

        

        # Keep a strong reference to running task (prevents GC, ensures on_finished runs)
        self._active_task = None# Prepare validators for numeric line edits
        self._wire_validators()

        # Connect UI signals
        self._connect_signals()

        # Initial UI state
        self._refresh_input_layers()
        self._update_target_threshold_controls()
        self._toggle_model_params()
        self._update_progress(0)
        try:
            self.dlg.pBPredictiveModelingSaveRFXG.setEnabled(False)
        except Exception:
            pass

        self._log("Predictive Modeling controller initialized.")

        # Button groups for radio buttons
        self._model_group = QButtonGroup(self.dlg)
        self._model_group.addButton(self.dlg.rBRF)
        self._model_group.addButton(self.dlg.rBXGBoost)

        self._eval_group = QButtonGroup(self.dlg)
        self._eval_group.addButton(self.dlg.rBEnableCrossValidation)
        self._eval_group.addButton(self.dlg.rBTrainTestSplit)

    # -------------------------
    # Wiring & UI helpers
    # -------------------------

    def _wire_validators(self):
        try:
            self.dlg.lERFNumberofTrees.setValidator(QIntValidator(1, 100000, self.dlg))
            self.dlg.lERFMaximumDepthofTrees.setValidator(QIntValidator(1, 1000, self.dlg))
            self.dlg.lEXGLearningRate.setValidator(QDoubleValidator(0.000001, 10.0, 6, self.dlg))
            self.dlg.lEXGNumberofBoostingRounds.setValidator(QIntValidator(1, 100000, self.dlg))
        except Exception:
            pass  # If any missing, just skip

    def _connect_signals(self):
        # Source layer selection → refresh fields list + target options
        self.dlg.cBInputDataPredictiveModeling.currentIndexChanged.connect(self._on_input_layer_changed)

        # Target variable selection → threshold controls
        self.dlg.cBTargetVariableModeling.currentIndexChanged.connect(self._update_target_threshold_controls)

        # Binary classification toggle
        self.dlg.cBEnableBinaryClassification.toggled.connect(self._update_target_threshold_controls)

        # RF / XGBoost toggles
        self.dlg.rBRF.toggled.connect(self._toggle_model_params)
        self.dlg.rBXGBoost.toggled.connect(self._toggle_model_params)

        # CV / Train-Test toggles
        self.dlg.rBEnableCrossValidation.toggled.connect(lambda _: self._log("Using 10-fold cross-validation"))
        self.dlg.rBTrainTestSplit.toggled.connect(lambda _: self._log("Using 70/30 train-test split"))

        # Choose folder
        self.dlg.tBChooseFolderPredictiveModelingRFXG.clicked.connect(self._choose_output_folder)

        # Run / Save / Next
        self.dlg.pBRunModeling.clicked.connect(self._on_run_clicked)
        self.dlg.pBPredictiveModelingSaveRFXG.clicked.connect(self._on_save_clicked)
        self.dlg.pBPredictiveModelingNext.clicked.connect(self._go_next_tab)

        self.dlg.bPRefreshLayersPredictiveModeling.clicked.connect(self.refresh_layers_predictive_modeling)


    def _refresh_input_layers(self):
        """Populate cBInputDataPredictiveModeling with vector layers from the project."""
        combo = self.dlg.cBInputDataPredictiveModeling
        combo.blockSignals(True)
        combo.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
                combo.addItem(lyr.name(), lyr.id())
        combo.blockSignals(False)
        self._on_input_layer_changed()

    def _on_input_layer_changed(self):
        layer = self._current_layer()
        if not layer:
            self._log("No valid input layer selected.", level="warning")
            return

        # Populate target combo & feature list
        fields = [f.name() for f in layer.fields()]
        self.dlg.cBTargetVariableModeling.blockSignals(True)
        self.dlg.cBTargetVariableModeling.clear()
        for name in fields:
            self.dlg.cBTargetVariableModeling.addItem(name)
        self.dlg.cBTargetVariableModeling.blockSignals(False)

        # Populate feature candidates list
        lw = self.dlg.lWFeatureVariablesModeling
        lw.clear()
        for name in fields:
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Unchecked)
            lw.addItem(item)

        # Suggest MGWR-derived fields if present
        if self.dlg.cBEnableMGWRFeatures.isChecked():
            self._auto_select_mgwr_features()

        self._log(f"Loaded {len(fields)} fields from '{layer.name()}'.")
        self._update_target_threshold_controls()

    def _auto_select_mgwr_features(self):
        lw = self.dlg.lWFeatureVariablesModeling
        count = lw.count()
        selected = 0
        for i in range(count):
            it = lw.item(i)
            name = it.text()
            if name.lower().startswith("mgwr_") or name.lower().endswith("_localr2"):
                it.setCheckState(Qt.Checked)
                selected += 1
        if selected:
            self._log(f"Auto-selected {selected} MGWR-related features.")

    def _update_target_threshold_controls(self):
        """Enable the correct threshold spin based on selected target and binary toggle."""
        binary = self.dlg.cBEnableBinaryClassification.isChecked()
        target = self.dlg.cBTargetVariableModeling.currentText().strip()

        # Default: disable all
        for w in (self.dlg.dSBThresholdF, self.dlg.dSBThresholdI, self.dlg.dSBThresholdEX):
            w.setEnabled(False)
            w.setToolTip("")

        if not binary:
            return

        # Enable the one that matches typical labels 'F', 'I', 'Ex'
        if target.lower() == 'f':
            self.dlg.dSBThresholdF.setEnabled(True)
            self.dlg.dSBThresholdF.setToolTip("Binary threshold for F (frequency).")
        elif target.lower() in ('i', 'intensity'):
            self.dlg.dSBThresholdI.setEnabled(True)
            self.dlg.dSBThresholdI.setToolTip("Binary threshold for I (intensity).")
        elif target.lower() in ('ex', 'exposure'):
            self.dlg.dSBThresholdEX.setEnabled(True)
            self.dlg.dSBThresholdEX.setToolTip("Binary threshold for Ex (exposure).")
        else:
            # If not classic F/I/Ex, still allow F spin as generic threshold
            self.dlg.dSBThresholdF.setEnabled(True)
            self.dlg.dSBThresholdF.setToolTip("Binary threshold for selected target.")

    def _toggle_model_params(self):
        """Enable/disable model-specific hyperparameters and evaluation radios."""
        is_rf = self.dlg.rBRF.isChecked()
        is_xg = self.dlg.rBXGBoost.isChecked()
        any_model = is_rf or is_xg

        # RF params
        self.dlg.lERFNumberofTrees.setEnabled(is_rf)
        self.dlg.lERFMaximumDepthofTrees.setEnabled(is_rf)

        # XGB params
        self.dlg.lEXGLearningRate.setEnabled(is_xg)
        self.dlg.lEXGNumberofBoostingRounds.setEnabled(is_xg)
        if is_xg and not _HAS_XGB:
            self._warn("XGBoost is not available in this environment. Please install `xgboost` or choose Random Forest.")

        # Evaluation method radios should be enabled once a model is chosen (RF or XGBoost)
        self.dlg.rBEnableCrossValidation.setEnabled(any_model)
        self.dlg.rBTrainTestSplit.setEnabled(any_model)

        # Optional: if nothing is selected yet, clear both evaluation choices (disabled state)
        if not any_model:
            try:
                self._eval_group.setExclusive(False)
                self.dlg.rBEnableCrossValidation.setChecked(False)
                self.dlg.rBTrainTestSplit.setChecked(False)
                self._eval_group.setExclusive(True)
            except Exception:
                pass
            
        if any_model and not (self.dlg.rBEnableCrossValidation.isChecked() or self.dlg.rBTrainTestSplit.isChecked()):
            self.dlg.rBEnableCrossValidation.setChecked(True)

    def _current_layer(self) -> Optional[QgsVectorLayer]:
        data = self.dlg.cBInputDataPredictiveModeling.currentData()
        if data:
            lyr = QgsProject.instance().mapLayer(data)
            if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
                return lyr
        return None

    def _selected_features(self) -> List[str]:
        names = []
        lw = self.dlg.lWFeatureVariablesModeling
        for i in range(lw.count()):
            it = lw.item(i)
            if it.checkState() == Qt.Checked:
                names.append(it.text())
        return names

    # -------------------------
    # Actions
    # -------------------------

    def _choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dlg, "Choose output folder")
        if folder:
            self._output_folder = folder
            self._log(f"Output folder set to: {folder}")
        else:
            self._log("No folder selected.", level="warning")

    def _on_run_clicked(self):
        try:
            self.dlg.pBRunModeling.setEnabled(False)
            self.dlg.pBPredictiveModeling.setStyleSheet("")
        except Exception:
            pass
        try:
            self._run_model_tasked()
        except Exception as e:
            self._error(f"Model run failed: {e}")
            self._debug_tb()

    
    def _run_model_tasked(self):
        self._init_progress_tracker()  
        self._update_progress_tracker(5)  # Initial progress

        layer = self._current_layer()
        if not layer:
            self._warn("Please select a valid input layer.")
            return

        target = self.dlg.cBTargetVariableModeling.currentText().strip()
        if not target:
            self._warn("Please select a target variable.")
            return

        features = self._selected_features()
        if target in features:
            features = [f for f in features if f != target]

        if not features:
            self._warn("Please select at least one feature variable.")
            return

        use_rf = self.dlg.rBRF.isChecked()
        use_xgb = self.dlg.rBXGBoost.isChecked()
        if not (use_rf or use_xgb):
            self._warn("Please choose a model: Random Forest or XGBoost.")
            return

        if use_xgb and not _HAS_XGB:
            self._worker_log("[Worker] XGBoost selected but not installed.")
            return {"error": "XGBoost not installed"}

        is_classification = self.dlg.cBEnableBinaryClassification.isChecked()

        # Hyperparameters
        n_trees = int(self.dlg.lERFNumberofTrees.text()) if self.dlg.lERFNumberofTrees.text() else 300
        max_depth = int(self.dlg.lERFMaximumDepthofTrees.text()) if self.dlg.lERFMaximumDepthofTrees.text() else 10
        cv_mode = self.dlg.rBEnableCrossValidation.isChecked()  # else train-test split

        xg_lr = float(self.dlg.lEXGLearningRate.text()) if self.dlg.lEXGLearningRate.text() else 0.1
        xg_rounds = int(self.dlg.lEXGNumberofBoostingRounds.text()) if self.dlg.lEXGNumberofBoostingRounds.text() else 200

        # Threshold (classification)
        thr = None
        if is_classification:
            if target.lower() == 'f':
                thr = self.dlg.dSBThresholdF.value()
            elif target.lower() in ('i', 'intensity'):
                thr = self.dlg.dSBThresholdI.value()
            elif target.lower() in ('ex', 'exposure'):
                thr = self.dlg.dSBThresholdEX.value()
            else:
                thr = self.dlg.dSBThresholdF.value()

        # === Extract data on main thread (no QGIS access in task) ===
        try:
            self._log("Extracting attributes to DataFrame...")
            df = self._layer_to_dataframe(layer, [target] + features)
            if df.empty:
                self._warn("Input data frame is empty after extraction.")
                return
            used_cols = [target] + features
            df_clean = df.dropna(subset=used_cols).copy()
            if df_clean.empty:
                self._warn("All rows contain missing values in target/features.")
                return
            self._log(f"Data cleaned: {len(df_clean)} rows remain. Building X/y arrays...")
            X = df_clean[features].to_numpy(dtype="float32", copy=True)
            y = df_clean[target].to_numpy(copy=True)
            self._log(f"Array shapes: X={X.shape}, y={y.shape}")
            if is_classification:
                y = (y >= thr).astype(int)
        except Exception as e:
            self._error(f"Failed to extract data before task: {e}")
            self._debug_tb()
            return

        self._log(f"Running model on '{layer.name()}' - Target: {target} | Features: {len(features)} | Binary: {is_classification}")
        self._update_progress(5)

        # Disable Run while task is active
        try:
            self.dlg.pBRunModeling.setEnabled(False)
        except Exception:
            pass

        def work_task(task, *args, **kwargs):
            try:
                # New: Early boot logging with try-except
                try:
                    task.setProgress(12)  # Task entered worker thread
                    self._worker_log("[Worker] Entered worker thread.")
                except Exception as _boot_err:
                    error_msg = f"Worker boot failure: {_boot_err}\n{traceback.format_exc()}"
                    try:
                        self._worker_log(f"[Worker] ERROR during boot: {error_msg}")
                    except Exception:
                        pass
                    return {"error": error_msg}

                if task.isCanceled():
                    return {"df": None, "metrics": {}, "features": features, "is_classification": is_classification, "error": "Canceled"}
                
                task.setProgress(15)
                self._worker_log("[Worker] Initializing metrics/preds containers.")
                metrics = {}
                preds = None
                prob1 = None

                def _safe_fit_predict(_model, Xtr, ytr, Xte=None):
                    try:
                        # Fit
                        _model.fit(Xtr, ytr)
                        # Predict
                        if Xte is None:
                            return _model.predict(Xtr)
                        return _model.predict(Xte)
                    except Exception as _e:
                        raise RuntimeError(f"Model training/prediction failed: {_e}")
                
                if use_xgb and not _HAS_XGB:
                    self._worker_log("[Worker] XGBoost selected but not installed.")
                    return {"error": "XGBoost not installed"}
                # Build model
                task.setProgress(20)
                self._worker_log("[Worker] Building model object (RF/XGBoost)...")
                try:
                    self._worker_log(f"[Worker] Flags: use_rf={use_rf}, use_xgb={use_xgb}, is_classification={is_classification}")
                except Exception:
                    pass
                if use_rf:
                    if is_classification:
                        model = RandomForestClassifier(
                            n_estimators=n_trees,
                            max_depth=max_depth if max_depth else None,
                            n_jobs=1,
                            random_state=42
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_trees,
                            max_depth=max_depth if max_depth else None,
                            n_jobs=1,
                            random_state=42
                        )
                else:
                    if is_classification:
                        model = xgb.XGBClassifier(
                            n_estimators=xg_rounds,
                            learning_rate=xg_lr,
                            max_depth=max_depth if max_depth else 6,
                            n_jobs=1,
                            subsample=1.0,
                            colsample_bytree=1.0,
                            random_state=42,
                            tree_method="hist"
                        )
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=xg_rounds,
                            learning_rate=xg_lr,
                            max_depth=max_depth if max_depth else 6,
                            n_jobs=1,
                            subsample=1.0,
                            colsample_bytree=1.0,
                            random_state=42,
                            tree_method="hist"
                        )

                task.setProgress(30)
                try:
                    self._worker_log(f"[Worker] Model constructed: {type(model).__name__}")
                except Exception:
                    pass

                # Store the trained model reference for later persistence
                try:
                    self._last_model = model
                except Exception:
                    pass

                
                # Explicit model summary before training
                try:
                    mname = type(model).__name__
                    self._worker_log(f"Model: {mname} | is_classification={is_classification} | params={{'max_depth': getattr(model, 'max_depth', None), 'n_estimators': getattr(model, 'n_estimators', None)}}")
                    try:
                        self._last_model_is_classification = bool(is_classification)
                    except Exception:
                        pass; self._note(f"Model: {mname}")
                except Exception:
                    pass
                if cv_mode:
                    # 10-fold CV (Stratified for classification)
                    task.setProgress(40)
                    if is_classification:
                        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                        splitter = kf.split(X, y)
                    else:
                        kf = KFold(n_splits=10, shuffle=True, random_state=42)
                        splitter = kf.split(X)

                    scores = []
                    total_folds = 10
                    for i, idx in enumerate(splitter, 1):
                        tr, te = idx
                        
                        # Update progress - fold start
                        task.setProgress(40 + int(10 * (i-1)/total_folds))
                        task.setProgress(40 + int(40 * (i-1) / total_folds))
                        self._worker_log(f"[Worker] Fold {i}/{total_folds}: starting fit...")
                        _ = _safe_fit_predict(model, X[tr], y[tr])
                        self._worker_log(f"[Worker] Fold {i}/{total_folds}: fit complete.")
                        
                        # Update progress - fold training done
                        task.setProgress(40 + int(10 * (i-1)/total_folds) + 5)
                        p = _safe_fit_predict(model, X[tr], y[tr], X[te])
                        self._worker_log(f"[Worker] Fold {i}/{total_folds}: predict complete.")
                        
                        # Update progress - fold prediction done
                        task.setProgress(40 + int(10 * (i-1)/total_folds) + 7)
                        if is_classification:
                            s = f1_score(y[te], p, zero_division=0)
                        else:
                            s = r2_score(y[te], p)
                        scores.append(s)
                        self._worker_log(f"[Worker] Fold {i}/{total_folds}: score={s:.4f}")
                        
                        # Update progress - fold complete
                        task.setProgress(40 + int(40 * i / total_folds))
                        
                    metrics['cv_score_mean'] = float(np.mean(scores))
                    metrics['cv_score_std'] = float(np.std(scores))
                    self._worker_log(f"[Worker] CV scores: mean={metrics['cv_score_mean']:.4f}, std={metrics['cv_score_std']:.4f}")

                    # Final fit on all data
                    task.setProgress(85)
                    _ = _safe_fit_predict(model, X, y)
                    
                    # Generate predictions
                    task.setProgress(88)
                    preds = _safe_fit_predict(model, X, y)
                    self._worker_log(f"[Worker] Full-data predictions generated: n={len(preds)}")
                    
                    # Generate probabilities if classification
                    task.setProgress(90)
                    if is_classification and hasattr(model, "predict_proba"):
                        prob1 = model.predict_proba(X)[:, 1]
                        self._worker_log(f"[Worker] Probabilities generated: n={len(prob1)}")
                        
                else:
                    # 70/30 split
                    task.setProgress(50)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y if is_classification else None)
                    self._worker_log(f"[Worker] Train/Test shapes: Xtr={Xtr.shape}, Xte={Xte.shape}, ytr={ytr.shape}, yte={yte.shape}")
                    
                    task.setProgress(55)
                    _ = _safe_fit_predict(model, Xtr, ytr)
                    
                    task.setProgress(65)
                    pte = _safe_fit_predict(model, Xtr, ytr, Xte)
                    
                    task.setProgress(70)
                    if is_classification:
                        metrics['accuracy'] = float(accuracy_score(yte, pte))
                        metrics['f1'] = float(f1_score(yte, pte, zero_division=0))
                    else:
                        metrics['r2'] = float(r2_score(yte, pte))
                        self._worker_log(f"[Worker] R2={metrics['r2']:.4f}")
                        metrics['rmse'] = float(math.sqrt(mean_squared_error(yte, pte)))
                        self._worker_log(f"[Worker] RMSE={metrics['rmse']:.4f}")
                        metrics['mae'] = float(mean_absolute_error(yte, pte))
                        self._worker_log(f"[Worker] MAE={metrics['mae']:.4f}")
                        
                    # Full data predictions
                    task.setProgress(75)
                    preds = _safe_fit_predict(model, X, y)
                    self._worker_log(f"[Worker] Full-data predictions generated: n={len(preds)}")
                    
                    task.setProgress(80)
                    if is_classification and hasattr(model, "predict_proba"):
                        prob1 = model.predict_proba(X)[:, 1]
                        self._worker_log(f"[Worker] Probabilities generated: n={len(prob1)}")

                # Build output dataframe
                task.setProgress(92)
                out_df = df_clean.copy()
                self._worker_log(f"[Worker] Building output DataFrame with predictions. Rows={len(out_df)}")
                out_df['prediction'] = preds
                if prob1 is not None:
                    out_df['prob_1'] = prob1

                # Calculate feature importances
                task.setProgress(95)
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    metrics['feature_importances'] = {feat: float(w) for feat, w in zip(features, fi)}
                    self._worker_log("[Worker] Feature importances computed.")

                task.setProgress(100)
                self._worker_log("[Worker] Task completed successfully.")
                return {"df": out_df, "metrics": metrics, "features": features, "is_classification": is_classification}
            except Exception as e:
                import traceback
                error_msg = f"Task error: {str(e)}\n{traceback.format_exc()}"
                try:
                    self._worker_log(f"[Worker] ERROR: {error_msg}")
                except Exception:
                    pass
                try:
                    task.reportError(error_msg)
                except Exception:
                    pass
                return {"df": None, "metrics": {}, "features": features, "is_classification": is_classification, "error": error_msg}
                

        def on_finished(exception, result):
            # Clear active task reference at finish (success or error)
            try:
                self._active_task = None
            except Exception:
                pass
            # Clear previous results immediately
            self._last_results = None
            self._last_metrics = None
            
            # Always re-enable UI
            try:
                self.dlg.pBRunModeling.setEnabled(True)
                self.dlg.pBPredictiveModeling.setStyleSheet("")
            except Exception:
                pass

            try:
                if exception:
                    self._error(f"Model task raised: {exception}")
                    self._update_progress(0)
                    return

                if not isinstance(result, dict):
                    self._warn("Model finished but produced no structured result.")
                    self._update_progress(0)
                    return

                # Flush any worker notes to UI FIRST
                if getattr(self, "_worker_notes", None):
                    for _n in self._worker_notes:
                        self._log(_n)
                    self._worker_notes.clear()

                # Check for error in result
                if "error" in result:
                    self._error(f"Model finished with error: {result['error']}")
                    return

                out_df = result.get("df")
                if out_df is None:
                    self._error("Model finished without data.")
                    return

                # Success path - these are the critical lines that enable the save button
                self._last_results = out_df
                self._last_metrics = result.get("metrics", {})
                n_rows = len(out_df) if hasattr(out_df, "__len__") else 0
                
                self._log(f"Model finished successfully. Stored results with {n_rows} rows. Keys: metrics={list(self._last_metrics.keys())}")
                
                # ENABLE THE SAVE BUTTON
                try:
                    self.dlg.pBPredictiveModelingSaveRFXG.setEnabled(True)
                    self._log("Save button enabled.")
                except Exception as e:
                    self._log(f"Error enabling save button: {e}", level="error")
                
                self._update_progress(100)
            finally:
                try:
                    self.dlg.pBRunModeling.setEnabled(True)
                except Exception:
                    pass

        def on_error(exc):
            try:
                self._active_task = None
            except Exception:
                pass
            try:
                # Still provide a dict for consistency
                self.model_results = {"df": None, "metrics": {}, "error": str(exc)}
                self._error(f"Model failed: {exc}")
                self._update_progress(0)
            finally:
                try:
                    self.dlg.pBRunModeling.setEnabled(True)
                except Exception:
                    pass

        # Create and run the task
        self._active_task = QgsTask.fromFunction(
            "Predictive Modeling",
            work_task,
            on_finished=on_finished,
            on_error=on_error,
            flags=QgsTask.CanCancel
        )

        # Connect progress BEFORE adding to manager
        self._active_task.progressChanged.connect(self._on_task_progress)

        # Add to task manager ONCE
        QgsApplication.taskManager().addTask(self._active_task)

        self._last_model = None
        self._log("Started modeling task…")
        self._update_progress(10)

    def _persist_model_file(self, csv_path: str):
        """Persist the trained model next to the CSV. Uses joblib for sklearn, JSON for XGBoost; falls back to pickle."""
        try:
            import os
            import pickle
            base, _ = os.path.splitext(csv_path)
            mdl = getattr(self, '_last_model', None)
            if mdl is None:
                self._log('No trained model object in memory; skipping model persistence.', level='warning')
                return None
            # Prefer XGBoost native JSON if available
            try:
                if hasattr(mdl, 'get_booster'):
                    out_path = base + '.xgb.json'
                    booster = mdl.get_booster()
                    booster.save_model(out_path)
                    self._log(f'Saved trained XGBoost model: {out_path}')
                    return out_path, 'xgb.json'
            except Exception as e:
                self._warn(f'Could not save XGBoost JSON; falling back to pickle. Error: {e}')
            # Try joblib for sklearn/random forest
            try:
                from joblib import dump
                out_path = base + '.joblib'
                dump(mdl, out_path)
                self._log(f'Saved trained model (joblib): {out_path}')
                return out_path, 'joblib'
            except Exception:
                pass
            # Final fallback: pickle
            out_path = base + '.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump(mdl, f)
            self._log(f'Saved trained model (pickle): {out_path}')
            return out_path, 'pkl'
        except Exception as e:
            self._error(f'Failed to persist trained model: {e}')
            self._debug_tb()
            return None

    
    
    def _register_model_artifact(self, gpkg_path: str, layer_name_hint: str, model_path: str, model_fmt: str, alg: str, is_clf: bool, metrics: dict, features: list, csv_path: str):
        """
        Write a non-spatial 'model metadata' table into the same GeoPackage.
        Uses QgsVectorFileWriter with NoGeometry and QVariant types.
        Returns True on success, False otherwise (non-fatal).
        """
        try:
            from qgis.core import (
                QgsFields, QgsField, QgsFeature, QgsVectorFileWriter,
                QgsCoordinateTransformContext, QgsWkbTypes, QgsVectorLayer, QgsProject
            )
            from PyQt5.QtCore import QVariant
        except Exception as e:
            self._warn(f"QGIS core not available to write model table: {e}")
            return False

        # Build fields
        fields = QgsFields()
        fields.append(QgsField('model_path', QVariant.String))
        fields.append(QgsField('format', QVariant.String))
        fields.append(QgsField('algorithm', QVariant.String))
        fields.append(QgsField('is_classification', QVariant.Int))
        fields.append(QgsField('n_features', QVariant.Int))
        fields.append(QgsField('features', QVariant.String))
        fields.append(QgsField('metrics', QVariant.String))
        fields.append(QgsField('csv_path', QVariant.String))
        fields.append(QgsField('created_at', QVariant.String))

        # One feature of attributes
        feat = QgsFeature()
        feat.setFields(fields)
        feat.setAttributes([
            model_path or '',
            model_fmt or '',
            alg or '',
            1 if is_clf else 0,
            int(len(features or [])),
            json.dumps(features or []),
            json.dumps(metrics or {}),
            csv_path or '',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = 'GPKG'

        # Prefer CreateOrOverwriteLayer to avoid clobbering file
        try:
            if Path(gpkg_path).exists():
                options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
            else:
                options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
        except Exception:
            pass

        # Sanitize layer name and add a micro-timestamp suffix as last-resort
        sanitized_hint = self._sanitize_layer_name(layer_name_hint)
        base_name = self._sanitize_layer_name(f"Model_{sanitized_hint}")[:55]
        options.layerName = base_name

        ctx = QgsCoordinateTransformContext()
        empty_crs = QgsCoordinateReferenceSystem()

        def _try_write(layer_name: str):
            options.layerName = layer_name
            writer = QgsVectorFileWriter.create(
                gpkg_path, fields, QgsWkbTypes.NoGeometry, empty_crs, ctx, options
            )
            # QGIS returns an object even on failure; check for NoError
            try:
                err = writer.hasError()
            except Exception:
                err = 1
            ok = (err == QgsVectorFileWriter.NoError)
            if ok:
                if not writer.addFeature(feat):
                    return False
            # close handle
            del writer
            return ok

        unique_name = None  # Generate a unique name if needed

        # Try base name, then a shorter name, then a unique name
        if not _try_write(base_name):
            if not _try_write("Model_Metadata"):
                unique_name = f"Model_{datetime.now().strftime('%H%M%S')}"
                if not _try_write(unique_name):
                    self._warn("Could not write model metadata table to GeoPackage (non-fatal). The trained model was saved next to the CSV.")
                    return False

        # Try to load it back into QGIS
        try:
            for lname in [n for n in (base_name, "Model_Metadata", unique_name) if n]:
                v = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
                if v and v.isValid():
                    QgsProject.instance().addMapLayer(v)
                    break
        except Exception as e:
            self._warn(f"Model metadata table written but could not be auto-loaded ({e}); add it manually if needed.")

        return True


    def _on_save_clicked(self):
        if self._last_results is None:
            self._warn("No results to save. Please run the model first, wait for \"Model finished successfully...\" in the log, then click Save.")
            return
        if not self._output_folder:
            self._choose_output_folder()
            if not self._output_folder:
                return

        layer = self._current_layer()
        if not layer:
            self._warn("Input layer not available.")
            return

        try:
            safe_layer_name = layer.name().replace(" ", "_")

            # Detect model type for naming
            if hasattr(self.dlg, 'rBXGBoost') and self.dlg.rBXGBoost.isChecked():
                model_tag = "XGBoost"
            else:
                model_tag = "RF"

            base_name = f"Predictive_Modeling_{safe_layer_name}_{model_tag}"
            csv_path = os.path.join(self._output_folder, base_name + ".csv")
            gpkg_path = os.path.join(self._output_folder, base_name + ".gpkg")

            # Save CSV
            self._last_results.to_csv(csv_path, index=False, encoding="utf-8")
            self._log(f"Saved predictions CSV: {csv_path}")

            # Persist trained model next to the CSV
            try:
                _mdl_info = self._persist_model_file(csv_path)
            except Exception:
                _mdl_info = None
                self._warn('Model persist failed; continuing with vector export.')

            
            # --- Register model metadata FIRST (reduces GPKG lock/contention) ---
            try:
                self._log("Registering model metadata table in GeoPackage...")

                # Decide algorithm name
                alg_name = "RandomForest"
                try:
                    if hasattr(self.dlg, 'rBXGBoost') and self.dlg.rBXGBoost.isChecked():
                        alg_name = "XGBoost"
                    elif hasattr(self.dlg, 'cBModelAlgorithm'):
                        txt = self.dlg.cBModelAlgorithm.currentText()
                        if isinstance(txt, str) and "xgboost" in txt.lower():
                            alg_name = "XGBoost"
                except Exception:
                    pass

                # Parse model persistence info (tuple or None)
                mdl_path, mdl_fmt = (None, None)
                if isinstance(_mdl_info, tuple) and len(_mdl_info) == 2:
                    mdl_path, mdl_fmt = _mdl_info

                # Classification flag & metrics/features
                is_clf = bool(getattr(self, '_last_model_is_classification', False))
                metrics = self._last_metrics or {}
                # Store column names as the feature list (simple & robust)
                feats = list(self._last_results.columns)

                reg_status = self._register_model_artifact(
                    gpkg_path=gpkg_path,
                    layer_name_hint=base_name,
                    model_path=mdl_path or '',
                    model_fmt=mdl_fmt or '',
                    alg=alg_name,
                    is_clf=is_clf,
                    metrics=metrics,
                    features=feats,
                    csv_path=csv_path
                )
                if reg_status:
                    self._log("Model metadata table saved to GeoPackage.")
                else:
                    self._warn("Failed to register model table in GeoPackage; continuing.")
            except Exception as e:
                self._warn(f"Exception during model table write: {e}; continuing.")
                # --- End registration block ---
            except Exception as e:
                self._warn(f"Model registration failed: {e}")

            # Save GeoPackage by cloning original layer and appending prediction fields
            self._save_results_to_gpkg(layer, self._last_results, gpkg_path)
            self._log(f"Saved predictions GeoPackage: {gpkg_path}")

            # Also register a non-spatial model metadata table in the same GPKG
            try:
                mdl_path, mdl_fmt = _mdl_info if isinstance(_mdl_info, tuple) else (None, None)
                alg_name = type(self._last_model).__name__ if getattr(self, '_last_model', None) is not None else 'Unknown'
                is_clf = bool(getattr(self, '_last_model_is_classification', False))
                feats = list(self._last_results.columns)
                metrics = self._last_metrics or {}
                self._register_model_artifact(gpkg_path, safe_layer_name, mdl_path or '', mdl_fmt or '', alg_name, is_clf, metrics, feats, csv_path)
            except Exception:
                self._warn('Failed to register model table in GeoPackage; continuing.')

            # Add to QGIS
            added = QgsVectorLayer(gpkg_path + "|layername=" + self._sanitize_layer_name(layer.name()), base_name, "ogr")
            if added and added.isValid():
                QgsProject.instance().addMapLayer(added)
                self._log("Output layer added to QGIS Layers panel.")
            else:
                self._warn("Could not load the saved GeoPackage into QGIS. You can add it manually.")

            QMessageBox.information(self.dlg, "Saved", "Results saved and added to QGIS.")
        except Exception as e:
            self._error(f"Saving failed: {e}")
            self._debug_tb()

    def _save_results_to_gpkg(self, template_layer: QgsVectorLayer, out_df: pd.DataFrame, gpkg_path: str):
        """Create a GPKG copy of template_layer with new fields prediction/prob_1 joined by FID row order."""
        import os
        preds = out_df['prediction'].tolist()
        probs = out_df['prob_1'].tolist() if 'prob_1' in out_df.columns else None

        # Build NA mask aligned with template layer order (best effort)
        used_cols = [c for c in out_df.columns if c in template_layer.fields().names()]
        mask_keep = self._build_non_na_mask(template_layer, used_cols)

        # New fields (prediction, prob_1)
        new_fields = QgsFields(template_layer.fields())
        pf = QgsField('prediction', _qt_type_double()); pf.setLength(20); pf.setPrecision(8)
        new_fields.append(pf)
        has_prob = probs is not None
        if has_prob:
            pr = QgsField('prob_1', _qt_type_double()); pr.setLength(20); pr.setPrecision(8)
            new_fields.append(pr)

        # GeoPackage writer options
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = 'GPKG'
        # Prefer a sanitized layer name (spaces and symbols can be quirky on some stacks)
        layer_name = self._sanitize_layer_name(template_layer.name())
        options.layerName = layer_name
        # If file exists, overwrite the layer (or file if empty)
        try:
            from qgis.core import QgsVectorFileWriter as _W
            if os.path.exists(gpkg_path):
                options.actionOnExistingFile = _W.CreateOrOverwriteLayer
            else:
                options.actionOnExistingFile = _W.CreateOrOverwriteFile
        except Exception:
            pass

        ctx = QgsCoordinateTransformContext()

        # Attempt to create the writer; retry once with a shorter name if it fails
        writer = QgsVectorFileWriter.create(
            gpkg_path,
            new_fields,
            template_layer.wkbType(),
            template_layer.sourceCrs(),
            ctx,
            options
        )
        try:
            err = writer.hasError()
        except Exception:
            err = 1

        if err != QgsVectorFileWriter.NoError:
            # Retry with a very safe fallback layer name
            options.layerName = 'predictions'
            writer = QgsVectorFileWriter.create(
                gpkg_path,
                new_fields,
                template_layer.wkbType(),
                template_layer.sourceCrs(),
                ctx,
                options
            )
            try:
                err = writer.hasError()
            except Exception:
                err = 1
            if err != QgsVectorFileWriter.NoError:
                raise RuntimeError('Failed to write output layer with predictions.')

        # Iterate and write
        kept_index = -1
        for idx, f in enumerate(template_layer.getFeatures()):
            # Check if the original row is kept (no NA across used_cols)
            if mask_keep is None or mask_keep[idx]:
                kept_index += 1
                pred_val = float(preds[kept_index]) if kept_index < len(preds) and preds[kept_index] is not None else None
                prob_val = float(probs[kept_index]) if (has_prob and kept_index < len(probs) and probs[kept_index] is not None) else None
            else:
                pred_val = None
                prob_val = None

            new_f = QgsFeature()
            new_f.setGeometry(f.geometry())
            attrs = list(f.attributes())
            attrs.append(pred_val)
            if has_prob:
                attrs.append(prob_val)
            new_f.setAttributes(attrs)
            writer.addFeature(new_f)

        del writer  # ensure file closed


    def _build_non_na_mask(self, layer: QgsVectorLayer, used_cols: List[str]) -> Optional[List[bool]]:
        """Return a boolean mask (per feature order) indicating rows without NA across used_cols."""
        try:
            if not used_cols:
                return None
            names = layer.fields().names()
            cols = [c for c in used_cols if c in names]
            if not cols:
                return None
            mask = []
            for feat in layer.getFeatures():
                good = True
                for c in cols:
                    v = feat[c]
                    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "na", "null")):
                        good = False
                        break
                mask.append(good)
            return mask
        except Exception:
            return None

    def _go_next_tab(self):
          """Switch to the Interpret Results tab."""
          try:
               tabw = getattr(self.dlg, "tabsXgeoAi", None) or getattr(self.dlg, "tabWidget", None)
               if not tabw:
                    self._log("No QTabWidget named 'tabsXgeoAi' or 'tabWidget' found.", level="warning")
                    return

               # Try by objectName on pages
               for i in range(tabw.count()):
                    w = tabw.widget(i)
                    if w and w.objectName() == "tInterpretResults":
                         tabw.setCurrentIndex(i)
                         self._log("Moved to the next tab: tInterpretResults.")
                         return

               # Fallback: if dialog exposes tInterpretResults directly
               if hasattr(self.dlg, "tInterpretResults"):
                    idx = tabw.indexOf(self.dlg.tInterpretResults)
                    if idx >= 0:
                         tabw.setCurrentIndex(idx)
                         self._log("Moved to the next tab: tInterpretResults.")
                         return

               self._log("tInterpretResults tab not found in the tab widget.", level="warning")
          except Exception:
               self._log("Could not automatically switch to tInterpretResults tab.", level="warning")


    # -------------------------
    # Data extraction
    # -------------------------

    def _layer_to_dataframe(self, layer: QgsVectorLayer, columns: List[str]) -> pd.DataFrame:
        """Convert layer attributes to DataFrame with the given columns (if present)."""
        names = layer.fields().names()
        cols = [c for c in columns if c in names]
        data = []
        for feat in layer.getFeatures():
            row = []
            for c in cols:
                v = feat[c]
                if v is None or (isinstance(v, str) and v.strip().lower() in ("", "na", "null")):
                    row.append(np.nan)
                else:
                    row.append(v)
            data.append(row)
        df = pd.DataFrame(data, columns=cols)
        return df

    def _worker_log(self, msg: str):
        """Thread-safe: log to Python logger only (no GUI access)."""
        try:
            self.logger.info(msg)
        except Exception:
            pass

    def _note(self, msg: str):
        """Accumulate notes from worker to be flushed to UI when finished."""
        try:
            self._worker_notes.append(str(msg))
        except Exception:
            pass

    # -------------------------
    # Logging / Progress / UI msgs
    # -------------------------

    def _update_progress(self, val: int):
        try:
            self.dlg.pBPredictiveModeling.setValue(int(max(0, min(100, val))))
        except Exception:
            pass

    def _log(self, msg: str, level: str = "info"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"

        # To QGIS message bar (plain message to avoid double timestamps there)
        if self.iface:
            if level == "warning":
                self.iface.messageBar().pushWarning("Predictive Modeling", msg)
            elif level == "critical":
                self.iface.messageBar().pushCritical("Predictive Modeling", msg)
            else:
                try:
                    # Some QGIS builds may not have pushSuccess; fallback to pushMessage
                    self.iface.messageBar().pushSuccess("Predictive Modeling", msg)
                except Exception:
                    self.iface.messageBar().pushMessage("Predictive Modeling", msg)

        # To internal logger (timestamped)
        getattr(self.logger, level if hasattr(self.logger, level) else "info")(full_msg)

        # To dialog text log (timestamped) + auto-scroll
        if hasattr(self.dlg, "tELog") and self.dlg.tELog:
            try:
                self.dlg.tELog.appendPlainText(full_msg)
                sb = self.dlg.tELog.verticalScrollBar()
                if sb is not None:
                    sb.setValue(sb.maximum())
            except Exception:
                pass

    def _warn(self, msg: str):
        self._log(msg, level="warning")

    def _error(self, msg: str):
        self._log(msg, level="critical")
        try:
            QMessageBox.critical(self.dlg, "Predictive Modeling", msg)
        except Exception:
            pass

    def _debug_tb(self):
        tb = traceback.format_exc()
        self.logger.error(tb)
        if hasattr(self.dlg, "tELog") and self.dlg.tELog:
            try:
                self.dlg.tELog.appendPlainText(tb)
                sb = self.dlg.tELog.verticalScrollBar()
                if sb is not None:
                    sb.setValue(sb.maximum())
            except Exception:
                pass

    def _on_task_progress(self, p: float):
        """Handle progress updates from task"""
        progress_int = int(p)
        self._update_progress_tracker(progress_int)
        
        # Additional detailed logging if needed
        if progress_int % 10 == 0 and progress_int != self.last_logged_stage:
            self._log(f"BACKGROUND PROGRESS: {progress_int}% - {self.current_stage}")

    def _init_progress_tracker(self):
        """Initialize progress tracking variables"""
        self.progress_milestones = {
            0: "Starting...",
            10: "Data extracted; preparing task…",
            15: "Building model...",
            20: "Model initialized...",
            30: "Model configuration complete",
            40: "Starting evaluation...",
            50: "Evaluation in progress...",
            60: "Processing folds/splits...",
            70: "Generating predictions...",
            80: "Computing metrics...",
            85: "Assembling outputs…",
            90: "Finalizing results...",
            95: "Cleaning up...",
            100: "Task completed"
        }
        self.current_stage = ""
        self.last_logged_stage = -1

    def _update_progress_tracker(self, progress: int):
        """Update progress tracker with detailed logging"""
        # Update progress bar
        self._update_progress(progress)
        
        # Log stage changes
        if progress in self.progress_milestones:
            if progress == 100:
                try:
                    if self._last_results is not None:
                        self.dlg.pBPredictiveModelingSaveRFXG.setEnabled(True)
                except Exception:
                    pass
            stage_message = self.progress_milestones[progress]
            if progress != self.last_logged_stage:
                self._log(f"PROGRESS: {stage_message}")
                self.last_logged_stage = progress
            self.current_stage = stage_message
        elif progress > self.last_logged_stage + 4:  # Log every 5% progress
            self._log(f"PROGRESS: {self.current_stage} - {progress}% complete")
            self.last_logged_stage = progress

    def refresh_layers_predictive_modeling(self):
        """Refresh cBInputDataPredictiveModeling from the QGIS Layers panel."""
        try:
            combo = self.dlg.cBInputDataPredictiveModeling
        except Exception:
            self._warn("Combo box 'cBInputDataPredictiveModeling' not found on the dialog.")
            return

        # Remember current selection by layer ID (how we store combo data)
        prev_id = combo.currentData()
        prev_text = combo.currentText()

        # Collect valid vector layers from the project
        layers = []
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
                layers.append((lyr.name(), lyr.id()))
        layers.sort(key=lambda t: t[0].lower())

        # Refill combo without firing handlers
        combo.blockSignals(True)
        combo.clear()
        for name, layer_id in layers:
            combo.addItem(name, layer_id)
        combo.blockSignals(False)

        # Restore previous selection by ID; fallback to name; else first item
        restored = False
        if prev_id is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == prev_id:
                    combo.setCurrentIndex(i)
                    restored = True
                    break
        if not restored and prev_text:
            idx = combo.findText(prev_text, Qt.MatchExactly)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                restored = True
        if not restored and combo.count() > 0:
            combo.setCurrentIndex(0)

        # Update dependent widgets (fields list, target options, thresholds)
        self._on_input_layer_changed()

        self._log(f"Refreshed predictive modeling input layers: found {combo.count()} layer(s).")
