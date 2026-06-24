
# interpret_results.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import json
import uuid
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from qgis.PyQt.QtCore import QObject, Qt, QThread, pyqtSignal
from qgis.core import (
    QgsApplication,
    QgsMessageLog,
    Qgis,
    QgsVectorLayer,
    QgsProject,
)
from qgis.utils import iface as qgis_iface

# Third-party ML explainability
import shap
from lime.lime_tabular import LimeTabularExplainer

# Optional: XGBoost JSON Booster loading
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


# --------------------------- Utilities ---------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def qgis_log(msg: str, level: str = "info", tag: str = "InterpretResults") -> None:
    lvl = {"info": Qgis.Info, "warning": Qgis.Warning, "error": Qgis.Critical}.get(level, Qgis.Info)
    QgsMessageLog.logMessage(msg, tag, lvl)


def _norm(s: str) -> str:
    """Normalize column keys for fuzzy matching: lowercase, remove spaces/underscores/dots."""
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace(".", "")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column in df for any of the candidate names (fuzzy)."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # try partials: if a candidate is contained within a column key
    for cand in candidates:
        key = _norm(cand)
        for col_key, orig in norm_map.items():
            if key and key in col_key:
                return orig
    return None



# --------------------------- Data Model ---------------------------

@dataclass
class ModelEntry:
    name: str
    source: str  # "memory" or "disk"
    obj: Any = None
    path: Optional[str] = None

# --------------------------- Worker Base ---------------------------

class CancellableWorker(QThread):
    progressed = pyqtSignal(int)          # 0..100
    message = pyqtSignal(str, str)        # (msg, level)
    finished_ok = pyqtSignal(dict)        # results
    failed = pyqtSignal(str)              # error text

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def _emit_progress(self, v: int):
        self.progressed.emit(int(max(0, min(100, v))))

    def _emit_msg(self, msg: str, level: str = "info"):
        self.message.emit(msg, level)

    def _guard(self):
        if self._cancel:
            raise RuntimeError("Cancelled by user.")


# --------------------------- SHAP/LIME Worker ---------------------------

class ShapLimeWorker(CancellableWorker):
    def __init__(
        self,
        mode: str,  # "shap" or "lime"
        model_obj: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        config: Dict[str, Any],
        out_dir: str
    ):
        super().__init__()
        self.mode = mode
        self.model = model_obj
        self.X = X
        self.y = y
        self.config = config
        self.out_dir = out_dir

    def run(self):
        try:
            self._emit_msg(f"Starting {self.mode.upper()} analysis ...")
            ensure_dir(self.out_dir)
            self._emit_progress(5)
            self._guard()

            if self.mode == "shap":
                results = self._run_shap()
            else:
                results = self._run_lime()

            self._emit_progress(100)
            self._emit_msg(f"{self.mode.upper()} analysis completed.")
            self.finished_ok.emit(results)
        except Exception as e:
            err = f"{self.mode.upper()} failed: {e}\n{traceback.format_exc()}"
            self._emit_msg(err, "error")
            self.failed.emit(err)

    # ---- SHAP ----
    def _run_shap(self) -> Dict[str, Any]:
        # Visual options
        vis_type = self.config.get("shap_visual", "summary_beeswarm")
        local_count = int(self.config.get("local_count", 10))
        dependence_feature = self.config.get("dependence_feature")  # optional
        
        # Detect model predict function
        pred_fn = None
        if hasattr(self.model, "predict_proba"):
            pred_fn = self.model.predict_proba
        elif hasattr(self.model, "predict"):
            pred_fn = self.model.predict

        # Tree vs model-agnostic
        is_tree = False
        try:
            import sklearn
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
            from xgboost import XGBRegressor, XGBClassifier
            tree_like = (
                RandomForestRegressor, RandomForestClassifier,
                GradientBoostingRegressor, GradientBoostingClassifier,
                XGBRegressor, XGBClassifier
            )
            is_tree = isinstance(self.model, tree_like)
        except Exception:
            pass

        # XGBoost Booster JSON
        booster = None
        if isinstance(self.model, xgb.Booster) if xgb else False:
            booster = self.model
            is_tree = True

        self._emit_progress(15)
        self._guard()

        # Build explainer
        explainer = None
        if is_tree:
            try:
                self._emit_msg("Using TreeExplainer.")
                explainer = shap.TreeExplainer(self.model if booster is None else booster)
            except Exception as e:
                self._emit_msg(f"TreeExplainer failed ({e}); falling back to KernelExplainer.", "warning")
        if explainer is None:
            self._emit_msg("Using KernelExplainer (model-agnostic).")
            # Use a small background sample for KernelExplainer
            bg = shap.kmeans(self.X, min(50, len(self.X)))
            # Build a robust predict function (handles Booster separately)
            def _predict_kernel(data_nd):
                import numpy as _np
                import pandas as _pd
                if booster is not None:
                    dmat = xgb.DMatrix(_np.array(data_nd))
                    return _np.array(booster.predict(dmat))
                elif hasattr(self.model, "predict_proba"):
                    return _np.array(self.model.predict_proba(_pd.DataFrame(data_nd, columns=self.X.columns)))
                else:
                    return _np.array(self.model.predict(_pd.DataFrame(data_nd, columns=self.X.columns)))
            explainer = shap.KernelExplainer(_predict_kernel, bg)

        self._emit_progress(25)
        self._guard()

        # Compute SHAP values (global + local)
        # Note: for large X, consider sampling for speed
        sample_n = min(len(self.X), int(self.config.get("sample_n", 200)))
        Xs = self.X.sample(sample_n, random_state=42) if len(self.X) > sample_n else self.X.copy()

        self._emit_msg(f"Computing SHAP values on {len(Xs)} samples ...")
        try:
            # Newer shap.Explainer objects are callable (return Explanation)
            if hasattr(explainer, "__call__"):
                shap_values = explainer(self.X if len(self.X) <= sample_n else Xs, check_additivity=False)
            else:
                # Old-style KernelExplainer path
                shap_values = explainer.shap_values(Xs, nsamples=self.config.get("kernel_nsamples", 100))
        except Exception as e:
            # As a last resort, try KernelExplainer compute path
            self._emit_msg(f"Primary SHAP compute failed ({e}); retrying with KernelExplainer.", "warning")
            import numpy as _np
            import pandas as _pd
            bg = shap.kmeans(self.X, min(50, len(self.X)))
            def _predict_kernel(data_nd):
                if booster is not None:
                    dmat = xgb.DMatrix(_np.array(data_nd))
                    return _np.array(booster.predict(dmat))
                elif hasattr(self.model, "predict_proba"):
                    return _np.array(self.model.predict_proba(_pd.DataFrame(data_nd, columns=self.X.columns)))
                else:
                    return _np.array(self.model.predict(_pd.DataFrame(data_nd, columns=self.X.columns)))
            kexp = shap.KernelExplainer(_predict_kernel, bg)
            shap_values = kexp.shap_values(Xs, nsamples=self.config.get("kernel_nsamples", 100))

        self._emit_progress(55)
        self._guard()

        # Save global importances (mean |SHAP|)
        self._emit_msg("Saving SHAP global importances ...")
        if hasattr(shap_values, "values"):
            sv = np.abs(shap_values.values).mean(axis=0)
        elif isinstance(shap_values, list):
            # Multi-class: average across classes
            sv = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values).mean(axis=0)

        imp = pd.DataFrame({"feature": self.X.columns, "mean_abs_shap": sv}).sort_values("mean_abs_shap", ascending=False)
        imp_csv = os.path.join(self.out_dir, f"shap_importances_{now_ts()}.csv")
        imp.to_csv(imp_csv, index=False)

        # Global plots
        pngs = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def save_current(fig_name: str):
                p = os.path.join(self.out_dir, f"{fig_name}_{now_ts()}.png")
                plt.tight_layout()
                plt.savefig(p, dpi=160)
                plt.close()
                pngs.append(p)

            self._emit_msg(f"Creating visualization: {vis_type}")
            if vis_type in ("summary_beeswarm", "bar_importance", "heatmap"):
                if vis_type == "summary_beeswarm":
                    shap.summary_plot(shap_values, Xs if len(self.X) > sample_n else self.X, show=False)
                    save_current("shap_summary_beeswarm")
                if vis_type == "bar_importance":
                    shap.summary_plot(shap_values, Xs if len(self.X) > sample_n else self.X, plot_type="bar", show=False)
                    save_current("shap_summary_bar")
                if vis_type == "heatmap":
                    shap.plots.heatmap(shap_values, show=False)
                    save_current("shap_heatmap")

            if vis_type == "dependence":
                feat = dependence_feature or imp["feature"].iloc[0]
                shap.dependence_plot(feat, shap_values, Xs if len(self.X) > sample_n else self.X, show=False)
                save_current(f"shap_dependence_{safe_filename(feat)}")

            if vis_type in ("force_single", "force_multi", "decision_plot"):
                expl_base = getattr(shap_values, "base_values", None)
                if vis_type == "force_single":
                    idx = 0
                    sv_row = shap_values[idx]
                    fig = shap.force_plot(expl_base[idx] if isinstance(expl_base, np.ndarray) else expl_base, sv_row.values if hasattr(sv_row, "values") else sv_row, (Xs if len(self.X) > sample_n else self.X).iloc[idx], matplotlib=True, show=False)
                    plt.gcf()
                    save_current("shap_force_single")
                if vis_type == "force_multi":
                    # Save first N local forces
                    N = min(local_count, len(Xs))
                    for i in range(N):
                        sv_row = shap_values[i]
                        shap.force_plot(expl_base[i] if isinstance(expl_base, np.ndarray) else expl_base, sv_row.values if hasattr(sv_row, "values") else sv_row, (Xs if len(self.X) > sample_n else self.X).iloc[i], matplotlib=True, show=False)
                        plt.gcf()
                        save_current(f"shap_force_{i}")
                if vis_type == "decision_plot":
                    shap.decision_plot(expl_base if isinstance(expl_base, (list, np.ndarray)) else np.array([expl_base] * len(Xs)), shap_values.values if hasattr(shap_values, "values") else shap_values, (Xs if len(self.X) > sample_n else self.X).columns, show=False)
                    save_current("shap_decision_plot")

        except Exception as e:
            self._emit_msg(f"Plotting error (SHAP): {e}", "warning")

        self._emit_progress(85)
        self._guard()

        return {
            "mode": "shap",
            "importances_csv": imp_csv,
            "images": pngs,
        }

    # ---- LIME ----
    def _run_lime(self) -> Dict[str, Any]:
        vis_type = self.config.get("lime_visual", "local_explanation_bar")
        subset_mode = self.config.get("lime_subset", "All features")  # All features | Top-K | Custom
        top_k = int(self.config.get("lime_top_k", 10))
        custom_features: Optional[List[str]] = self.config.get("lime_custom_features")

        # Choose features subset
        features = list(self.X.columns)
        if subset_mode == "Top-K by model importances":
            feats = self._top_k_by_importance(self.model, self.X, top_k)
            features = feats
        elif subset_mode == "Custom list" and custom_features:
            features = [f for f in custom_features if f in self.X.columns]

        X_use = self.X[features].copy()
        class_names = [str(c) for c in np.unique(self.y)] if self.y is not None else None

        self._emit_progress(20)
        self._guard()

        # Predict function
        if hasattr(self.model, "predict_proba"):
            predict_fn = self.model.predict_proba
            mode = "classification"
        else:
            predict_fn = self.model.predict
            mode = "regression"

        # Build explainer
        self._emit_msg(f"Building LIME explainer with {len(features)} features.")
        explainer = LimeTabularExplainer(
            training_data=X_use.values,
            feature_names=features,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True
        )

        self._emit_progress(40)
        self._guard()

        # Explain a sample of instances
        N = min(int(self.config.get("lime_local_count", 10)), len(X_use))
        sample_idx = np.random.RandomState(42).choice(len(X_use), size=N, replace=False)

        html_paths, csv_rows, pngs = [], [], []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for i, ridx in enumerate(sample_idx, start=1):
                self._guard()
                row = X_use.iloc[ridx].values
                exp = explainer.explain_instance(
                    data_row=row,
                    predict_fn=lambda x: predict_fn(pd.DataFrame(x, columns=features)),
                    num_features=min(10, len(features))
                )

                # HTML/text export
                if vis_type in ("HTML", "textual_explanation"):
                    html = exp.as_html()
                    html_path = os.path.join(self.out_dir, f"lime_explanation_{i}_{now_ts()}.html")
                    with io.open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    html_paths.append(html_path)

                # Bar plot export
                if vis_type in ("local_explanation_bar",):
                    fig = exp.as_pyplot_figure()
                    plt.tight_layout()
                    p = os.path.join(self.out_dir, f"lime_bar_{i}_{now_ts()}.png")
                    plt.savefig(p, dpi=160)
                    plt.close()
                    pngs.append(p)

                # CSV row of weights
                weights = exp.as_list()
                for feat, w in weights:
                    csv_rows.append({"row_index": int(ridx), "feature": feat, "weight": float(w)})

                self._emit_progress(40 + int(50 * i / max(1, N)))

        except Exception as e:
            self._emit_msg(f"LIME plotting/export error: {e}", "warning")

        # Save CSV of local explanation weights
        dfw = pd.DataFrame(csv_rows)
        csv_path = os.path.join(self.out_dir, f"lime_local_weights_{now_ts()}.csv")
        dfw.to_csv(csv_path, index=False)

        self._emit_progress(95)
        self._guard()

        return {
            "mode": "lime",
            "csv_weights": csv_path,
            "html": html_paths,
            "images": pngs
        }

    def _top_k_by_importance(self, model, X: pd.DataFrame, k: int) -> List[str]:
        # Try model feature_importances_ or SHAP fallback
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
            order = np.argsort(imp)[::-1]
            return [X.columns[i] for i in order[:k]]
        try:
            exp = shap.TreeExplainer(model)
            sv = exp.shap_values(X.sample(min(len(X), 200), random_state=42))
            if isinstance(sv, list):
                vals = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            else:
                vals = np.abs(sv).mean(axis=0)
            order = np.argsort(vals)[::-1]
            return [X.columns[i] for i in order[:k]]
        except Exception:
            return list(X.columns[:k])


# --------------------------- RSU Worker ---------------------------

class RSUWorker(CancellableWorker):
    """
     RSU Rank-Sum Analysis Worker
    - Expects a CSV with f, I, Ex metrics per variable (or raw to compute them if provided).
    - Ranks each variable within each metric (ascending/descending depending on definition),
      sums ranks to get RSU score, sorts, saves CSV, and a bar chart (top 20).
    """
    # Common aliases for auto-detection
    _ALIASES = {
        "variable": ["variable", "var", "feature", "name", "column", "layer", "field"],
        "f": ["f", "freq", "frequency", "fvalue", "f_value", "f-score", "fscore"],
        "I": ["I", "i", "intensity", "inten", "ivalue", "i_value"],
        "Ex": ["Ex", "ex", "exposure", "exp", "exval", "ex_value"],
    }

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any], out_dir: str):
        super().__init__()
        self.df = df
        self.params = params
        self.out_dir = out_dir

    def _resolve_columns(self) -> Tuple[Optional[str], str, str, str]:
        """
        Resolve var,f,I,Ex columns from params or via alias-based auto-detection (case-insensitive).
        'variable' is OPTIONAL; f/I/Ex are REQUIRED.
        Returns (var_col|None, f_col, i_col, ex_col).
        """
        var_col = self.params.get("var_col")
        f_col = self.params.get("f_col") or "f"
        i_col = self.params.get("I_col") or "I"
        ex_col = self.params.get("Ex_col") or "Ex"

        cols = list(self.df.columns)

        def exact_or_ci(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            if name in cols:
                return name
            matches = [c for c in cols if c.lower() == str(name).lower()]
            return matches[0] if matches else None

        resolved_var = exact_or_ci(var_col)
        resolved_f = exact_or_ci(f_col) or _find_col(self.df, self._ALIASES["f"])
        resolved_i = exact_or_ci(i_col) or _find_col(self.df, self._ALIASES["I"])
        resolved_ex = exact_or_ci(ex_col) or _find_col(self.df, self._ALIASES["Ex"])

        # Try variable via aliases if not resolved
        if not resolved_var:
            cand = _find_col(self.df, self._ALIASES["variable"])
            if cand:
                resolved_var = cand

        # Validate mandatory metrics
        missing_metrics = [name for name, val in [("f", resolved_f), ("I", resolved_i), ("Ex", resolved_ex)] if not val]
        if missing_metrics:
            available = ", ".join(cols)
            raise ValueError(
                f"Required metric columns missing after auto-detection: {', '.join(missing_metrics)}. "
                f"Looked for aliases {self._ALIASES}. Available columns: {available}"
            )
        return resolved_var, resolved_f, resolved_i, resolved_ex

    def run(self):
        try:
            self._emit_msg("Starting RSU Rank-Sum Analysis ...")
            ensure_dir(self.out_dir)
            self._emit_progress(10)
            self._guard()

            # Resolve columns (variable optional)
            var_col, f_col, i_col, ex_col = self._resolve_columns()
            self._emit_msg(f"RSU columns -> variable='{var_col or '<synthetic>'}', f='{f_col}', I='{i_col}', Ex='{ex_col}'")

            df_work = self.df.copy()

            # Optional GROUPBY fallback to manufacture variables
            groupby_col = self.params.get("groupby")
            if not var_col and groupby_col and groupby_col in df_work.columns:
                self._emit_msg(f"No 'variable' column found. Grouping by '{groupby_col}' and averaging f/I/Ex per group.")
                agg = df_work.groupby(groupby_col, dropna=False)[[f_col, i_col, ex_col]].mean(numeric_only=True).reset_index()
                agg = agg.rename(columns={groupby_col: "variable"})
                work = agg
            elif not var_col:
                # Aggregate the entire table to a single row; label with csv layer or provided label
                label = self.params.get("variable_label") or self.params.get("csv_layer_name") or "RSU"
                self._emit_msg(f"No 'variable' column found. Collapsing to a single variable='{label}' using mean(f/I/Ex).")
                means = df_work[[f_col, i_col, ex_col]].apply(pd.to_numeric, errors="coerce").mean()
                work = pd.DataFrame([{"variable": str(label), f_col: means[f_col], i_col: means[i_col], ex_col: means[ex_col]}])
            else:
                work = df_work[[var_col, f_col, i_col, ex_col]].rename(columns={var_col: "variable"})

            # Coerce metrics to numeric
            for metric in [f_col, i_col, ex_col]:
                if not pd.api.types.is_numeric_dtype(work[metric]):
                    self._emit_msg(f"Coercing non-numeric values in '{metric}' to numeric (invalid -> NaN).", "warning")
                    work[metric] = pd.to_numeric(work[metric], errors="coerce")

            before = len(work)
            work = work.dropna(subset=[f_col, i_col, ex_col])
            dropped = before - len(work)
            if dropped > 0:
                self._emit_msg(f"Dropped {dropped} rows with NaNs in f/I/Ex.", "warning")

            if work.empty:
                raise ValueError("After cleaning, there are no rows with valid f/I/Ex values to rank.")

            # Ranking directions: default higher worse (desc), override via params
            rank_desc = self.params.get("rank_desc", {f_col: True, i_col: True, ex_col: True})

            for metric in [f_col, i_col, ex_col]:
                self._guard()
                desc = bool(rank_desc.get(metric, True))
                work[f"rank_{metric}"] = work[metric].rank(ascending=not desc, method="min")

            work["RSU_sum"] = work[[f"rank_{f_col}", f"rank_{i_col}", f"rank_{ex_col}"]].sum(axis=1)
            work = work.sort_values("RSU_sum", ascending=True).reset_index(drop=True)

            self._emit_progress(70)
            self._guard()

            # Save CSV
            csv_out = os.path.join(self.out_dir, f"rsu_rank_sum_{now_ts()}.csv")
            work.to_csv(csv_out, index=False)

            # Bar chart (Top 20)
            png_path = None
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                topn = min(20, len(work))
                if work['variable'].nunique() == 1:
                    self._emit_msg('RSU: only one unique variable found after combining; plot will show a single bar (ranking not informative).', 'warning')

                fig = plt.figure()
                plt.bar(work["variable"].iloc[:topn], work["RSU_sum"].iloc[:topn])
                plt.xticks(rotation=60, ha="right")
                plt.ylabel("RSU (sum of ranks)")
                plt.title("RSU Rank-Sum (Top 20)")
                plt.tight_layout()
                png_path = os.path.join(self.out_dir, f"rsu_top20_{now_ts()}.png")
                plt.savefig(png_path, dpi=160)
                plt.close()
            except Exception as e:
                self._emit_msg(f"RSU plotting error: {e}", "warning")

            self._emit_progress(100)
            self._emit_msg("RSU analysis completed.")
            self.finished_ok.emit({"csv": csv_out, "png": png_path})
        
        except Exception as e:
            try:
                cols = ", ".join(list(self.df.columns))
            except Exception:
                cols = "<unavailable>"
            err = f"RSU failed: {e} Available columns: {cols}"
            self._emit_msg(err, "error")
            self.failed.emit(err)

        # --------------------------- Main Dialog Logic ---------------------------

class InterpretResults(QObject):
    """
    High-level controller hooking dialog widgets to SHAP/LIME/RSU workers.
    Assumes the dialog exposes the widgets named in the user spec.
    """

    def __init__(self, dlg, iface=None):
        super().__init__(dlg)
        self.dlg = dlg
        self.iface = iface or qgis_iface
        self._workers: List[CancellableWorker] = []
        self._models: Dict[str, ModelEntry] = {}  # key by visible text
        self._last_output_dir: Optional[str] = None

        # Connect signals
        self._connect()

        # Initial population
        self.refresh_models()
        self.refresh_csv_layers()

    # ---------- Public API ----------

    def add_model_path(self, path: str):
        """Manually register a single model file and refresh combos."""
        try:
            if not path or not os.path.isfile(path):
                self._log(f"Model file not found: {path}", "error")
                return
            base = os.path.basename(path)
            label = os.path.splitext(base)[0]
            self._models[label] = ModelEntry(name=label, source="disk", obj=None, path=path)
            # Update combos
            labels = sorted(self._models.keys())
            self._set_items(self.dlg.cBTrainedModelInterpretation, labels)
            # Also add to SHAP/LIME selection lists if missing
            try:
                existing_shap = [self.dlg.cBSHAPModelSelection.itemText(i) for i in range(self.dlg.cBSHAPModelSelection.count())]
                if path not in existing_shap:
                    self.dlg.cBSHAPModelSelection.addItem(path)
                existing_lime = [self.dlg.cBLIMEModelSelection.itemText(i) for i in range(self.dlg.cBLIMEModelSelection.count())]
                if path not in existing_lime:
                    self.dlg.cBLIMEModelSelection.addItem(path)
            except Exception:
                pass
            self._log(f"Registered model: {label} -> {path}")
        except Exception as e:
            self._log(f"Failed to add model path: {e}", "error")


    def register_in_memory_model(self, label: str, model_obj: Any):
        """Allow Step 9 to register trained models here."""
        self._models[label] = ModelEntry(name=label, source="memory", obj=model_obj, path=None)
        self._set_items(self.dlg.cBTrainedModelInterpretation, sorted(self._models.keys()))
    def refresh_models(self):
        """Load discoverable models (in-memory and on-disk) and populate related combos."""
        # Start with existing in-memory models
        labels = sorted(self._models.keys())

        # Discover models from common folders (project home + XGeoAI_Outputs)
        try:
            discover_dirs = []
            try:
                proj_home = QgsProject.instance().homePath()
                if proj_home and os.path.isdir(proj_home):
                    discover_dirs.append(proj_home)
            except Exception:
                pass

            user_base = os.path.join(os.path.expanduser("~"), "XGeoAI_Outputs")
            for sub in ("PredictiveModels", "InterpretResults", "Models"):
                p = os.path.join(user_base, sub)
                if os.path.isdir(p):
                    discover_dirs.append(p)

            discovered_paths = []
            exts = (".joblib", ".pkl", ".pickle", ".json")
            for d in discover_dirs:
                try:
                    for fn in os.listdir(d):
                        if fn.lower().endswith(exts):
                            discovered_paths.append(os.path.join(d, fn))
                except Exception:
                    continue

            # Make them available in SHAP/LIME model selection (full paths)
            discovered_paths = sorted(list(dict.fromkeys(discovered_paths)))
            self.set_shap_model_paths(discovered_paths)
            self.set_lime_model_paths(discovered_paths)

            # Also register them as selectable labels in the trained-model combo
            for pth in discovered_paths:
                base = os.path.basename(pth)
                label = os.path.splitext(base)[0]
                if label not in self._models:
                    self._models[label] = ModelEntry(name=label, source="disk", obj=None, path=pth)
            labels = sorted(self._models.keys())
        except Exception as e:
            self._log(f"Model discovery failed: {e}", "warning")

        # Populate the trained-model combo
        self._set_items(self.dlg.cBTrainedModelInterpretation, labels)
    def refresh_csv_layers(self):
        """Populate CSV combos from QGIS project and prefill RSU-related combos."""
        items = []
        for lyr in QgsProject.instance().mapLayers().values():
            try:
                if isinstance(lyr, QgsVectorLayer) and lyr.providerType() in ("delimitedtext", "ogr"):
                    items.append(lyr.name())
            except Exception:
                continue

        items = sorted(items)
        self._set_items(self.dlg.cBInputCSVForInterpretation, items)
        self._set_items(self.dlg.cBSelectCSVLayerforRSAnalysis, items)

        # Prefill RSU combos using the first available CSV
        try:
            if items:
                df, _ = self._get_csv_layer_df(items[0])
                cols = list(df.columns)
                self._set_items(self.dlg.cBRSUVariableSelection, cols)
        except Exception as e:
            self._log(f"Could not prefill RSU variable list: {e}", "warning")

        # Ensure parameters and visualization combos have sensible defaults
        try:
            self._set_items(
                self.dlg.cBRSUParameters,
                [
                    '{"var_col":"variable","f_col":"f","I_col":"I","Ex_col":"Ex","rank_desc":{"f":true,"I":true,"Ex":true}}',
                    '{"var_col":"variable","f_col":"f","I_col":"I","Ex_col":"Ex","rank_desc":{"f":false,"I":false,"Ex":false}}'
                ]
            )
        except Exception:
            pass
        try:
            self._set_items(self.dlg.cBRSUVisualizationType, ["bar_top20", "full_table"])
        except Exception:
            pass

    def _connect(self):


        # SHAP feature list reacts to CSV selection (guarded if widget exists)
        try:
            if hasattr(self.dlg, "cBInputCSVForInterpretation") and hasattr(self.dlg, "lWFeatureColumnsSHAP"):
                self.dlg.cBInputCSVForInterpretation.currentTextChanged.connect(self._on_csv_for_shap_changed)
        except Exception as _e:
            self._log(f"SHAP column-picker wiring failed: {_e}", "warning")
        
        d = self.dlg

        # Analysis type toggles
        d.rBSHAP.toggled.connect(self._on_mode_toggle)
        d.rBLIME.toggled.connect(self._on_mode_toggle)

        # Folder choosers
        d.tBChooseFolderStatisticalAnalysisSHAPLIME.clicked.connect(self._choose_out_folder_shap_lime)
        d.tBChooseFolderStatisticalAnalysisRankSum.clicked.connect(self._choose_out_folder_rsu)

        # Save/Run
        d.pBStatisticalAnalysisSaveSHAPLIME.clicked.connect(self._run_shap_or_lime)
        d.pBStatisticalAnalysisSaveRankSum.clicked.connect(self._run_rsu)

        # Cancel
        d.pBStatisticalAnalysisCancel.clicked.connect(self._cancel_current)

        # Populate dependent combos defaults
        self._init_defaults()

        self._wire_csv_change_handlers()

        self.dlg.pBRefreshLayersInterpretResults.clicked.connect(self.refresh_layers_interpret_results)



    def _init_defaults(self):
        # SHAP visualization types
        self._set_items(self.dlg.cBSHAPVisualizationType, [
            "summary_beeswarm", "bar_importance", "dependence",
            "force_single", "force_multi", "decision_plot", "heatmap"
        ])
        # LIME subset and visualization types
        self._set_items(self.dlg.cBLIMEFeatureSubset, [
            "All features", "Top-K by model importances", "Custom list"
        ])
        self._set_items(self.dlg.cBLIMEVisualizationType, [
            "local_explanation_bar", "textual_explanation", "HTML"
        ])
        # Defaults
        self.dlg.rBSHAP.setChecked(True)
        self._on_mode_toggle()

    # ---------- UI helpers ----------
    def _set_items(self, combo, items: List[str]):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.blockSignals(False)

    def _log(self, msg: str, level: str = "info"):
        # tELog text area + message bar
        try:
            if hasattr(self.dlg, "tELog") and self.dlg.tELog is not None:
                self.dlg.tELog.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
        except Exception:
            pass
        qgis_log(msg, level)

    def _set_progress(self, v: int):
        if hasattr(self.dlg, "pBStatisticalAnalysis") and self.dlg.pBStatisticalAnalysis is not None:
            self.dlg.pBStatisticalAnalysis.setValue(int(max(0, min(100, v))))

    def _on_mode_toggle(self):
        is_shap = self.dlg.rBSHAP.isChecked()
        # Enable/disable SHAP/LIME areas
        for w in [self.dlg.cBSHAPModelSelection, self.dlg.cBSHAPVisualizationType]:
            w.setEnabled(is_shap)
        for w in [self.dlg.cBLIMEModelSelection, self.dlg.cBLIMEFeatureSubset, self.dlg.cBLIMEVisualizationType]:
            w.setEnabled(not is_shap)

    def _choose_out_folder_shap_lime(self):
        folder = self._ask_folder("Choose output folder for SHAP/LIME")
        if folder:
            self._last_output_dir = folder
            self._log(f"Output folder set for SHAP/LIME: {folder}")

    def _choose_out_folder_rsu(self):
        folder = self._ask_folder("Choose output folder for RSU")
        if folder:
            self._last_output_dir = folder
            self._log(f"Output folder set for RSU: {folder}")

    def _ask_folder(self, title: str) -> Optional[str]:
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            folder = QFileDialog.getExistingDirectory(self.dlg, title, os.path.expanduser("~"))
            return folder if folder else None
        except Exception as e:
            self._log(f"Folder selection failed: {e}", "error")
            return None

    def _get_csv_layer_df(self, layer_name: str) -> Tuple[pd.DataFrame, QgsVectorLayer]:
        lyr = None
        for L in QgsProject.instance().mapLayers().values():
            if isinstance(L, QgsVectorLayer) and L.name() == layer_name:
                lyr = L
                break
        if lyr is None:
            raise ValueError(f"CSV layer not found: {layer_name}")

        data = []
        fields = [f.name() for f in lyr.fields()]
        for f in lyr.getFeatures():
            row = [f[name] for name in fields]
            data.append(row)
        df = pd.DataFrame(data, columns=fields)
        return df, lyr

    def _load_csv_as_table_layer(self, csv_path: str, layer_name: Optional[str] = None):
        uri = f"file:///{csv_path}?encoding=UTF-8&delimiter=,&detectTypes=yes&geomType=none"
        vl = QgsVectorLayer(uri, layer_name or os.path.splitext(os.path.basename(csv_path))[0], "delimitedtext")
        if not vl.isValid():
            raise RuntimeError(f"Failed to load CSV as layer: {csv_path}")
        QgsProject.instance().addMapLayer(vl)
        return vl

    def _resolve_model(self, label: str) -> Any:
        if label in self._models and self._models[label].source == "memory":
            return self._models[label].obj
        # otherwise assume disk path encoded in label or selection combos for SHAP/LIME model selections
        path = getattr(self.dlg, "cBSHAPModelSelection").currentText() if self.dlg.rBSHAP.isChecked() else getattr(self.dlg, "cBLIMEModelSelection").currentText()
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in (".joblib", ".pkl", ".pickle"):
                return joblib.load(path)
            if ext == ".json" and xgb is not None:
                booster = xgb.Booster()
                booster.load_model(path)
                return booster
        # fallback to trained-model combo’s selected label
        entry = self._models.get(label)
        if entry and entry.source == "disk" and entry.path:
            return joblib.load(entry.path)
        if entry and entry.obj is not None:
            return entry.obj
        raise ValueError("Selected model could not be resolved (memory or disk).")

    # ---------- Actions ----------
    def _on_csv_for_shap_changed(self, layer_name: str):
        """Populate the SHAP feature list from the selected CSV (numeric cols excluding X/Y/fid/target)."""
        try:
            if not hasattr(self.dlg, "lWFeatureColumnsSHAP"):
                return
            if not layer_name:
                self.dlg.lWFeatureColumnsSHAP.clear()
                return
            df, _ = self._get_csv_layer_df(layer_name)
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            blacklist = {"x", "y", "fid", "target", "label", "dn"}
            numeric_cols = [c for c in numeric_cols if c.lower() not in blacklist]
            self.dlg.lWFeatureColumnsSHAP.clear()
            self.dlg.lWFeatureColumnsSHAP.addItems(numeric_cols)
            self._log(f"SHAP feature picker populated with {len(numeric_cols)} numeric columns.")
        except Exception as e:
            self._log(f"Failed to populate SHAP columns: {e}", "warning")

    def _get_selected_shap_features(self, df: pd.DataFrame):
        """Return user-selected SHAP predictors if available; otherwise None (fall back to auto)."""
        try:
            if not hasattr(self.dlg, "lWFeatureColumnsSHAP"):
                return None
            items_fn = getattr(self.dlg.lWFeatureColumnsSHAP, "selectedItems", None)
            if items_fn is None:
                return None
            cols = [it.text() for it in items_fn()]
            blacklist = {"x", "y", "fid", "target", "label", "dn"}
            cols = [c for c in cols if c.lower() not in blacklist]
            cols = [c for c in cols if c in df.columns]
            return cols if cols else None
        except Exception:
            return None
    
    def _run_shap_or_lime(self):

        try:
            self._set_progress(0)
            mode = "shap" if self.dlg.rBSHAP.isChecked() else "lime"
            out_dir = self._last_output_dir or os.path.join(
                os.path.expanduser("~"), "XGeoAI_Outputs", "InterpretResults"
            )
            ensure_dir(out_dir)

            # Inputs
            model_label = self.dlg.cBTrainedModelInterpretation.currentText()
            csv_layer_name = self.dlg.cBInputCSVForInterpretation.currentText()
            if not model_label:
                self._log("Please select a trained model.", "warning")
                return
            if not csv_layer_name:
                self._log("Please select an input CSV layer.", "warning")
                return

            # Data
            df, _ = self._get_csv_layer_df(csv_layer_name)

            # === Build X for SHAP/LIME (manual selection supported) ===
            selected_cols = self._get_selected_shap_features(df)
            if selected_cols:
                X = df[selected_cols].copy()
                self._log(f"Using user-selected SHAP predictors: {selected_cols}")
            else:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                numeric_cols = [c for c in numeric_cols if c.lower() not in {"x","y","fid","target","label","dn"}]
                X = df[numeric_cols].copy()
                self._log(f"Using all numeric columns for SHAP (excluding X,Y,fid,target,label,DN): {numeric_cols}")

            if X.shape[1] == 0:
                raise ValueError("No valid predictor columns for SHAP after applying your selections and exclusions.")

            # Try to pick y if exists
            y = None
            y_name = None
            for candidate in ["target", "y", "label", "DN", "dn"]:
                if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
                    y = df[candidate]
                    y_name = candidate
                    break

            # Ensure the target is NOT in X
            if y_name:
                X = X.drop(columns=[y_name], errors="ignore")

            # Model
            model_obj = self._resolve_model(model_label)

            # Config
            if mode == "shap":
                cfg = {
                    "shap_visual": self.dlg.cBSHAPVisualizationType.currentText() or "summary_beeswarm",
                    "sample_n": 200,
                    "local_count": 10,
                    "dependence_feature": None,
                }
                worker = ShapLimeWorker("shap", model_obj, X, y, cfg, out_dir)
            else:
                cfg = {
                    "lime_visual": self.dlg.cBLIMEVisualizationType.currentText() or "local_explanation_bar",
                    "lime_subset": self.dlg.cBLIMEFeatureSubset.currentText() or "All features",
                    "lime_top_k": 10,
                    "lime_local_count": 8,
                    "lime_custom_features": None,
                }
                worker = ShapLimeWorker("lime", model_obj, X, y, cfg, out_dir)

            self._attach_worker(worker)
            worker.start()

        except Exception as e:
            self._log(f"Failed to start {('SHAP' if self.dlg.rBSHAP.isChecked() else 'LIME')} analysis: {e}", "error")



    def _wire_csv_change_handlers(self):
        try:
            self.dlg.cBSelectCSVLayerforRSAnalysis.currentTextChanged.connect(self._on_rsu_csv_changed)
        except Exception:
            pass


    def _on_rsu_csv_changed(self, layer_name: str):
        """When RSU CSV selection changes, refresh the variable list combo."""
        if not layer_name:
            return
        try:
            df, _ = self._get_csv_layer_df(layer_name)
            self._set_items(self.dlg.cBRSUVariableSelection, list(df.columns))
        except Exception as e:
            self._log(f"Failed to update RSU variable list: {e}", "warning")

    
    def _run_rsu(self):
        try:
            self._set_progress(0)
            out_dir = self._last_output_dir or os.path.join(os.path.expanduser("~"), "XGeoAI_Outputs", "InterpretResults")
            ensure_dir(out_dir)

            # Selected CSV (used when not combining)
            csv_layer_name = self.dlg.cBSelectCSVLayerforRSAnalysis.currentText()

            # If there are no items, bail early
            if self.dlg.cBSelectCSVLayerforRSAnalysis.count() == 0:
                self._log("No CSV layers found for RSU.", "warning")
                return

            # Base params
            params_map = {
                "var_col": (self.dlg.cBRSUVariableSelection.currentText() or None),
                "f_col": "f",
                "I_col": "I",
                "Ex_col": "Ex",
                "rank_desc": {"f": True, "I": True, "Ex": True},
                "csv_layer_name": csv_layer_name
            }

            # If user provided a mapping in cBRSUParameters (e.g., JSON)
            try:
                raw = self.dlg.cBRSUParameters.currentText().strip()
                if raw.startswith("{") and raw.endswith("}"):
                    params_map.update(json.loads(raw))
            except Exception:
                pass

            # New: allow combining multiple CSV layers automatically
            combine = bool(params_map.get("combine_project_csv_layers", True))
            use_basename = bool(params_map.get("variable_label_use_basename", True))

            def _layer_label(name: str) -> str:
                if not use_basename:
                    return name
                # best-effort shorten "file.csv" -> "file"
                return os.path.splitext(os.path.basename(name))[0] if name else name

            # Helper to load one CSV layer and return (df, label) with f/I/Ex means if needed
            def _prepare_single_layer(layer_name: str, params: dict):
                df, _ = self._get_csv_layer_df(layer_name)
                var_col = params.get("var_col")
                f_col = params.get("f_col", "f")
                i_col = params.get("I_col", "I")
                ex_col = params.get("Ex_col", "Ex")
                
                # Param: expand rows as variables when no explicit variable column (default True)
                expand_rows = bool(params.get("expand_rows_as_variables", True))
                row_id_candidates = params.get("row_id_candidates", ["variable","name","id","ID","grid_id","cell_id","feature","FID","fid","index"])
                
                # If a variable column exists in this layer, keep rows as-is
                if var_col and var_col in df.columns:
                    df2 = df[[var_col, f_col, i_col, ex_col]].rename(columns={var_col: "variable"}).copy()
                    return df2, layer_name
                
                # Coerce metrics if present
                for metric in [f_col, i_col, ex_col]:
                    if metric in df.columns and not pd.api.types.is_numeric_dtype(df[metric]):
                        df[metric] = pd.to_numeric(df[metric], errors="coerce")
                
                # If we should expand each row into its own 'variable'
                if expand_rows and all(col in df.columns for col in [f_col, i_col, ex_col]):
                    # Try to pick a reasonable ID column for the variable label
                    chosen_id = None
                    for c in row_id_candidates:
                        if c in df.columns:
                            chosen_id = c
                            break
                    df2 = df[[f_col, i_col, ex_col]].copy()
                    if chosen_id is not None:
                        df2.insert(0, "variable", df[chosen_id].astype(str).values)
                    else:
                        # fallback to layer name + row number
                        df2.insert(0, "variable", [f"{os.path.splitext(os.path.basename(layer_name))[0]}_row{i}" for i in range(1, len(df2)+1)])
                    # drop rows with all-NaN metrics
                    df2 = df2.dropna(subset=[f_col, i_col, ex_col], how="all")
                    return df2, layer_name
                
                # Otherwise compute a single representative row (mean of metrics)
                # and use the layer name as 'variable'
                means = df[[f_col, i_col, ex_col]].mean(numeric_only=True)
                row = {
                    "variable": os.path.splitext(os.path.basename(layer_name))[0] if layer_name else "RSU",
                    f_col: means.get(f_col, float("nan")),
                    i_col: means.get(i_col, float("nan")),
                    ex_col: means.get(ex_col, float("nan")),
                }
                return pd.DataFrame([row]), layer_name

            # Build the working dataframe
            if combine:
                # Combine all items currently present in the combo
                all_names = [self.dlg.cBSelectCSVLayerforRSAnalysis.itemText(i) for i in range(self.dlg.cBSelectCSVLayerforRSAnalysis.count())]
                frames = []
                for nm in all_names:
                    try:
                        df_part, _ = _prepare_single_layer(nm, params_map)
                        frames.append(df_part)
                    except Exception as e:
                        self._log(f"Skipping layer '{nm}' for RSU combine: {e}", "warning")
                if not frames:
                    self._log("No valid CSV layers to combine for RSU.", "warning")
                    return
                df_combined = pd.concat(frames, ignore_index=True)
                # NEW: collapse duplicates: group by 'variable' and average f/I/Ex
                if 'variable' in df_combined.columns:
                    metric_cols = [c for c in ['f','I','Ex'] if c in df_combined.columns]
                    if metric_cols:
                        df_combined = df_combined.groupby('variable', as_index=False)[metric_cols].mean()
                # Force var_col to 'variable' since we created it
                params_map["var_col"] = "variable"
                worker_df = df_combined
                self._log(f"RSU combining {len(frames)} CSV layers from project (unique variables: {df_combined['variable'].nunique()}).", "info")
            else:
                if not csv_layer_name:
                    self._log("Please select a CSV layer for RSU.", "warning")
                    return
                worker_df, _ = self._get_csv_layer_df(csv_layer_name)

            worker = RSUWorker(worker_df, params_map, out_dir)
            self._attach_worker(worker)
            worker.start()
        except Exception as e:
            self._log(f"Failed to start RSU analysis: {e}", "error")
    
    def _attach_worker(self, worker: CancellableWorker):
        
        # Wire logging + progress + completion
        worker.progressed.connect(self._on_worker_progress)
        worker.message.connect(self._on_worker_message)
        worker.finished_ok.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._workers.append(worker)
        self._log("Task started.")
        self._set_progress(1)

    def _on_worker_progress(self, v: int):
        # Progress bar only reflects computation (your spec)
        self._set_progress(v)

    def _on_worker_message(self, msg: str, level: str):
        self._log(msg, level)

    def _on_worker_done(self, results: Dict[str, Any]):
        self._log("Task finished successfully.")
        self._set_progress(100)

        # Register CSV outputs in QGIS (when present)
        csv_keys = ["importances_csv", "csv_weights", "csv"]
        for k in csv_keys:
            path = results.get(k)
            if path and os.path.isfile(path):
                try:
                    self._load_csv_as_table_layer(path)
                    self._log(f"Added CSV to QGIS: {path}")
                except Exception as e:
                    self._log(f"Could not add CSV to QGIS: {e}", "warning")

    def _on_worker_failed(self, err: str):
        self._log(err, "error")
        self._set_progress(0)

    def _cancel_current(self):
        # Stop all running workers
        any_active = False
        for w in list(self._workers):
            if w.isRunning():
                w.request_cancel()
                any_active = True
        if any_active:
            self._log("Cancellation requested. Stopping current task ...", "warning")
        else:
            self._log("No active task to cancel.", "info")

    # ---------- Optional helpers to pre-fill SHAP/LIME model combos ----------
    def set_shap_model_paths(self, paths: List[str]):
        """Populate cBSHAPModelSelection with disk paths (joblib/pickle/xgboost JSON)."""
        self._set_items(self.dlg.cBSHAPModelSelection, paths)
        self._log(f"SHAP model paths set: {paths}")

    def set_lime_model_paths(self, paths: List[str]):
        """Populate cBLIMEModelSelection with disk paths (joblib/pickle/xgboost JSON)."""
        self._set_items(self.dlg.cBLIMEModelSelection, paths)
        self._log(f"LIME model paths set: {paths}")

    from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer

    def _load_output_to_qgis(self, path):
        """Add SHAP/LIME output to QGIS layers panel."""
        if not os.path.exists(path):
            self._log(f"Output not found: {path}", "warning")
            return

        ext = os.path.splitext(path)[1].lower()
        layer = None
        if ext in (".gpkg", ".shp"):
            layer = QgsVectorLayer(path, os.path.basename(path), "ogr")
        elif ext in (".tif", ".tiff"):
            layer = QgsRasterLayer(path, os.path.basename(path))
        elif ext == ".csv":
            uri = f'file:///{path}?delimiter=,&detectTypes=yes&geomType=none'
            layer = QgsVectorLayer(uri, os.path.basename(path), "delimitedtext")

        if layer and layer.isValid():
            QgsProject.instance().addMapLayer(layer)
            self._log(f"Loaded output into QGIS: {os.path.basename(path)}", "info")
        else:
            self._log(f"Failed to load output layer: {path}", "error")

    def refresh_layers_interpret_results(self):
        """Refresh trained-model and CSV combos from the current QGIS context."""
        try:
            # Remember current selections
            prev_model = ""
            prev_csv = ""
            if hasattr(self.dlg, "cBTrainedModelInterpretation"):
                prev_model = self.dlg.cBTrainedModelInterpretation.currentText() or ""
            if hasattr(self.dlg, "cBInputCSVForInterpretation"):
                prev_csv = self.dlg.cBInputCSVForInterpretation.currentText() or ""

            # Rebuild the lists using existing helpers
            self.refresh_models()      # repopulates cBTrainedModelInterpretation
            self.refresh_csv_layers()  # repopulates cBInputCSVForInterpretation (and RSU combos)

            # Restore previous selections when possible
            try:
                if prev_model:
                    idx = self.dlg.cBTrainedModelInterpretation.findText(prev_model, Qt.MatchExactly)
                    if idx >= 0:
                        self.dlg.cBTrainedModelInterpretation.setCurrentIndex(idx)
                if prev_csv:
                    idx = self.dlg.cBInputCSVForInterpretation.findText(prev_csv, Qt.MatchExactly)
                    if idx >= 0:
                        self.dlg.cBInputCSVForInterpretation.setCurrentIndex(idx)
            except Exception:
                pass

            self._log("Refreshed Interpret Results inputs (models + CSV layers).")
        except Exception as e:
            self._log(f"Failed to refresh Interpret Results inputs: {e}", "error")
