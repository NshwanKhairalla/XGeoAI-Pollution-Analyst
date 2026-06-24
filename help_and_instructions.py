# -*- coding: utf-8 -*-
"""
Help and Instructions module for the AGeoAI Pollution Analyst plugin.
This file renders a rich, structured Help page inside a QTextEdit.
"""

from typing import Optional
import datetime as _dt  # robust import (avoid NameError)

class HelpAndInstructions:
    @staticmethod
    def set_help_text(
        text_edit_widget,
        plugin_name: str = "XGeoAI Pollution Analyst",
        version: str = "v1.0",
        build_date: Optional[str] = None
    ) -> None:
        """
        Populate the given QTextEdit with formatted HTML Help & Instructions.

        Parameters
        ----------
        text_edit_widget : QTextEdit
            The widget where the help text will be displayed via setHtml().
        plugin_name : str, optional
            Display name for the plugin header.
        version : str, optional
            Version label to show in the header.
        build_date : str, optional
            Build date string (e.g., '2025-10-05'). If None, uses today's date.
        """
        if build_date is None:
            build_date = str(_dt.date.today())  # fix: use datetime import safely

        # Inline CSS to keep a clean readable style inside QTextEdit
        css = """
        <style>
          body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.45; color: #1f2937; }
          h1, h2, h3 { color: #111827; margin-top: 1.0em; }
          h1 { font-size: 20px; margin-bottom: 0.2em; }
          h2 { font-size: 16px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
          h3 { font-size: 14px; }
          p { margin: 0.35em 0; }
          ul, ol { margin: 0.2em 1.2em; }
          code, kbd { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
          .tag { display:inline-block; padding:2px 6px; margin-left:6px; font-size:11px; background:#eef2ff; color:#3730a3; border-radius:999px; }
          .note { background: #fef3c7; color:#92400e; padding:8px; border-left:4px solid #f59e0b; border-radius:4px; }
          .ok { background: #ecfdf5; color:#065f46; padding:8px; border-left:4px solid #10b981; border-radius:4px; }
          .warn { background: #fee2e2; color:#991b1b; padding:8px; border-left:4px solid #ef4444; border-radius:4px; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
          .small { color:#6b7280; font-size: 12px; }
          hr { border: none; border-top: 1px solid #e5e7eb; margin: 10px 0; }
          .kbd { border: 1px solid #d1d5db; border-bottom-width: 3px; padding: 2px 6px; border-radius: 6px; background: #fff; }
          .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
        </style>
        """

        # Main HTML content
        help_html = f"""
        {css}
        <body>
          <h1>Welcome to {plugin_name} <span class="tag">{version}</span></h1>
          <p class="small">Build: {build_date}. This plugin provides an end-to-end, explainable, geospatial AI pipeline for pollution data.</p>
          <div class="note">
            <strong>Tip:</strong> Work left → right across tabs. Each tab writes outputs that next tabs can read.
          </div>

          <h2>1. What this plugin does</h2>
          <p>
            {plugin_name} helps you import, prepare, refine, analyze, model, and interpret
            air-pollution datasets inside QGIS. The workflow is inspired by the D-DUST / MGWR / RF methodology
            (e.g., Gianquintieri et&nbsp;al., 2024) with explainability via SHAP / LIME and rank-sum summaries.
          </p>

          <h2>2. Tabs at a glance</h2>
          <ul>
            <li><strong>Help &amp; Instructions</strong> — You are here.</li>
            <li><strong>Data Import</strong> — Load rasters (GeoTIFF/NetCDF) and CSVs with coordinates.</li>
            <li><strong>Data Preparation</strong> — CRS checks, reprojection, temporal alignment &amp; interpolation.</li>
            <li><strong>Data Spatial Refinement</strong> — Clipping by vector, vectorization, grid resampling.</li>
            <li><strong>Data Quality Cleaning</strong> — Missing values, outliers, types, normalization, collinearity.</li>
            <li><strong>Time Series Analysis</strong> — Decomposition (LOESS), correlations, anomalies, aggregation.</li>
            <li><strong>Spatial Metrics Aggregation</strong> — Frequency (f), Intensity (I), Exposure (Ex), filters.</li>
            <li><strong>Spatial Analysis (MGWR)</strong> — MGWR/GWR with diagnostics and stability safeguards.</li>
            <li><strong>Predictive Modeling</strong> — RF / XGBoost training &amp; validation; export models.</li>
            <li><strong>Interpret Results</strong> — SHAP/LIME visuals, rank-sum plots, compare scenarios.</li>
            <li><strong>Log</strong> — Live progress, info/warnings/errors for all steps.</li>
          </ul>

          <h2>3. Quick Start (5 minutes)</h2>
          <ol>
            <li><strong>Import</strong>: Add a pollutant time-series raster (NetCDF or GeoTIFF) and any land-use layers or spatial CSVs.</li>
            <li><strong>Prepare</strong>: Check CRS; run <em>Temporal Alignment</em> to harmonize timestamps and sampling interval.</li>
            <li><strong>Refine</strong>: Clip to study area; (optionally) vectorize; resample grids to a common resolution.</li>
            <li><strong>Clean</strong>: Handle missing/outliers; ensure correct types; optionally normalize.</li>
            <li><strong>Analyze</strong>: Time-series decomposition &amp; correlation; compute <span class="mono">f / I / Ex</span> in Spatial Metrics Aggregation.</li>
            <li><strong>Model</strong>: Run MGWR; train RF/XGBoost; inspect explainability in <em>Interpret Results</em>.</li>
            <li><strong>Export</strong>: Save plots, CSVs, and GeoPackages. Outputs appear in QGIS and the Log tab confirms paths.</li>
          </ol>

          <h2>4. Inputs &amp; Outputs</h2>
          <div class="grid">
            <div>
              <h3>Supported Inputs</h3>
              <ul>
                <li>Raster: <em>GeoTIFF</em>, <em>NetCDF</em> (with time dimension where applicable)</li>
                <li>Vector: <em>GeoPackage</em>, <em>Shapefile</em> (project CRS recommended)</li>
                <li>Tables: <em>CSV</em> with <code>X</code>/<code>Y</code> and timestamp (<code>YYYY-MM-DD HH:MM:SS</code>)</li>
              </ul>
            </div>
            <div>
              <h3>Key Outputs</h3>
              <ul>
                <li>Aligned rasters (multi-band GeoTIFF) + sidecar timestamp CSV</li>
                <li>Clipped/Resampled rasters; polygonized vectors</li>
                <li>Cleaned/aggregated CSVs reloaded as spatial layers</li>
                <li>MGWR diagnostics, model files, feature importances</li>
                <li>Explainability plots (SHAP/LIME) &amp; rank-sum charts</li>
              </ul>
            </div>
          </div>

          <h2>5. Best Practices</h2>
          <ul>
            <li>Use a consistent CRS (e.g., project all inputs to your target CRS before MGWR).</li>
            <li>Keep file paths short and ASCII-only to avoid GDAL/QGIS path issues.</li>
            <li>Name outputs with timestamps (the plugin often does this automatically).</li>
            <li>Document thresholds (e.g., EU PM2.5 daily 25&nbsp;µg/m³) and keep units consistent.</li>
          </ul>

          <h2>6. Troubleshooting</h2>
          <ul>
            <li><strong>Layer not visible</strong> — Check CRS, symbology, and that the layer is within the map extent.</li>
            <li><strong>Temporal alignment crashes</strong> — Ensure the time variable is present/valid; try linear interpolation first.</li>
            <li><strong>MGWR fails</strong> — The plugin will fall back to stable GWR settings; inspect the Log for bandwidth/diagnostics.</li>
            <li><strong>Plots empty</strong> — Verify selected columns, timestamp parsing, and that there are no duplicate index labels.</li>
          </ul>

          <h2>7. Tips &amp; Shortcuts</h2>
          <ul>
            <li>Refresh layer lists after creating new outputs.</li>
            <li>Use <span class="kbd">Ctrl</span>+<span class="kbd">S</span> in file dialogs to quickly save derived datasets.</li>
            <li>Look at the <strong>Log</strong> tab first whenever something seems off.</li>
          </ul>

          <h2>8. Credits &amp; Contact</h2>
          <p>
            Methodology inspired by peer-reviewed work (Gianquintieri. Implementation of a GEOAI model to assess the impact of agricultural land on the spatial distribution of PM2.5 concentration 2024) and open-source geospatial tooling.
            For questions, email: <a href="mailto:nshwan.khairalla@gmail.com">nshwan.khairalla@gmail.com</a>.
          </p>
          <hr/>
          <p class="small">© {build_date[:4]} {plugin_name}. All rights reserved.</p>
        </body>
        """

        text_edit_widget.setHtml(help_html)
