import os
import csv
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProject
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog
from qgis.PyQt.QtCore import QDir

class Vectorization:
    def __init__(self, iface, lWInputPollutantRasterClip, tBChooseFolderDataRefinementVectorPollutantLayer, 
                 pBConvertPollutantLayertoVector, pBDataRefinement, log):
        self.iface = iface
        self.lWInputPollutantRasterClip = lWInputPollutantRasterClip
        self.tBChooseFolderDataRefinementVectorPollutantLayer = tBChooseFolderDataRefinementVectorPollutantLayer
        self.pBConvertPollutantLayertoVector = pBConvertPollutantLayertoVector
        self.pBDataRefinement = pBDataRefinement
        self.log = log
        self.output_folder = None

        # Connect signals
        self.tBChooseFolderDataRefinementVectorPollutantLayer.clicked.connect(self.choose_output_folder)
        self.pBConvertPollutantLayertoVector.clicked.connect(self.convert_raster_to_vector)

    def choose_output_folder(self):
        """Open a dialog to choose the output folder for vectorization."""
        self.output_folder = QFileDialog.getExistingDirectory(
            None, "Select Output Folder", QDir.homePath()
        )
        if self.output_folder:
            self.log.append(f"Output folder selected: {self.output_folder}")
        else:
            self.log.append("No output folder selected.")

    def load_timestamp_mapping(self, raster_name):
        """Improved version to handle complex filenames with spaces/special characters"""
        timestamp_mapping = {}

        # Step 1: Better cleaning with ordered suffix removal
        suffixes_to_remove = [
            r'_clipped', r'_resampled', r'_Reprojected', r'_Aligned', 
            r'_masked', r'_\d{8,14}$', r'_grid.*'  # Remove grid patterns
        ]
        
        base_raster_name = raster_name
        for suffix in suffixes_to_remove:
            base_raster_name = re.sub(suffix, '', base_raster_name, flags=re.IGNORECASE)

        # Step 2: Sanitize filename for glob pattern
        sanitized_name = re.sub(r'[ ()]', '*', base_raster_name)  # Handle spaces/parentheses
        csv_pattern = os.path.join(self.output_folder, f"*{sanitized_name}*_timestamps.csv")
        
        self.log.append(f"Searching for CSVs with pattern: {csv_pattern}")
        
        # Step 3: Find all potential matches
        potential_csvs = glob.glob(csv_pattern, recursive=False)
        self.log.append(f"Found {len(potential_csvs)} potential CSVs: {potential_csvs}")
        
        if not potential_csvs:
            self.log.append(f"WARNING: No timestamp CSV found matching pattern: {csv_pattern}")
            return timestamp_mapping

        matching_csv = None
        for csv_path in potential_csvs:
            if base_raster_name in os.path.basename(csv_path):
                matching_csv = csv_path
                break

        if not matching_csv:
            self.log.append(f"WARNING: No valid CSV found for {base_raster_name}")
            return timestamp_mapping

        try:
            with open(matching_csv, "r") as f:
                reader = csv.DictReader(f)
                if not all(col in reader.fieldnames for col in ["BandIndex", "Timestamp"]):
                    self.log.append(f"ERROR: CSV {matching_csv} missing required columns")
                    return timestamp_mapping
                    
                for row in reader:
                    band_idx = int(row["BandIndex"])
                    timestamp_mapping[band_idx] = row["Timestamp"]
            self.log.append(f"Loaded {len(timestamp_mapping)} timestamps from {matching_csv}")
        except Exception as e:
            self.log.append(f"ERROR loading CSV: {str(e)}")

        return timestamp_mapping

    def convert_raster_to_vector(self):
        if not self.output_folder:
            QMessageBox.warning(None, "Warning", "Please select an output folder first.")
            return
        selected_items = self.lWInputPollutantRasterClip.selectedItems()
        if not selected_items:
            QMessageBox.warning(None, "Warning", "No raster layers selected.")
            return
        self.pBDataRefinement.setMaximum(len(selected_items))
        self.pBDataRefinement.setValue(0)
        for i, item in enumerate(selected_items):
            layer_name = item.text()
            raster_layer = QgsProject.instance().mapLayersByName(layer_name)[0]
            if not isinstance(raster_layer, QgsRasterLayer):
                self.log.append(f"Skipping non-raster layer: {layer_name}")
                continue
            raster_path = raster_layer.source()
            original_filename = os.path.splitext(os.path.basename(raster_path))[0]
            vector_output = os.path.join(self.output_folder, f"{original_filename}_vector.gpkg")
            try:
                self.log.append(f"\nProcessing: {original_filename}")
                timestamp_mapping = self.load_timestamp_mapping(original_filename)
                with rasterio.open(raster_path) as src:
                    crs = src.crs.to_string()
                    transform = src.transform
                    all_bands = []
                    for band_idx in range(1, src.count + 1):
                        self.log.append(f"Processing band {band_idx}")
                        array = src.read(band_idx)
                        mask = array != src.nodata
                        if not np.any(mask):
                            continue
                        features = []
                        for geom, value in shapes(array, mask=mask, transform=transform):
                            shapely_geom = shape(geom)
                            centroid = shapely_geom.centroid
                            timestamp = timestamp_mapping.get(band_idx, "Unknown")
                            features.append({
                                'geometry': shapely_geom,
                                'x': centroid.x,
                                'y': centroid.y,
                                'DN': value,
                                'Band': band_idx,
                                'Timestamp': timestamp
                            })
                        if features:
                            gdf = gpd.GeoDataFrame(features, crs=crs)
                            all_bands.append(gdf)
                    if all_bands:
                        combined_gdf = gpd.GeoDataFrame(pd.concat(all_bands, ignore_index=True), crs=crs)
                        combined_gdf.to_file(vector_output, driver='GPKG', layer=original_filename)
                        self.log.append(f"Saved vector data to: {vector_output}")
                        vector_layer = QgsVectorLayer(f"{vector_output}|layername={original_filename}",
                                                      f"{original_filename}_vector", "ogr")
                        if vector_layer.isValid():
                            QgsProject.instance().addMapLayer(vector_layer)
                        else:
                            self.log.append("ERROR: Failed to load vector layer")
            except Exception as e:
                self.log.append(f"ERROR processing {original_filename}: {str(e)}")
                continue
            self.pBDataRefinement.setValue(i + 1)
        self.log.append("\nVectorization process completed")
        QMessageBox.information(None, "Success", "Conversion completed successfully")
        self.pBDataRefinement.setValue(0)
