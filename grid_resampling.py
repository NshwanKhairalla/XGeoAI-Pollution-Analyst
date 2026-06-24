
import os
import rasterio
import shutil
import logging
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from qgis.core import QgsProject, QgsRasterLayer
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class GridResampling:
    def __init__(self, dialog):
        self.dialog = dialog
        self.setup_ui()

    def setup_ui(self):
        self.dialog.cBGridResamplingMethod.clear()
        self.dialog.cBGridResamplingMethod.addItems([
            "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "max", "min", "med", "q1", "q3"
        ])

        self.dialog.rBTargetResolutionGridResampling.toggled.connect(self.toggle_target_resolution_mode)
        self.dialog.rBRefernceRasterGridResampling.toggled.connect(self.toggle_reference_raster_mode)
        self.dialog.tBChooseFolderGridResampling.clicked.connect(self.choose_output_folder)
        self.dialog.pBGridResampling.clicked.connect(self.run_grid_resampling)
        populate_reference_raster_combobox(self.dialog.cBAligntoReferenceRaster)

    def toggle_target_resolution_mode(self):
        self.dialog.lETargetResolutionGridResampling.setEnabled(self.dialog.rBTargetResolutionGridResampling.isChecked())

    def toggle_reference_raster_mode(self):
        is_checked = self.dialog.rBRefernceRasterGridResampling.isChecked()
        self.dialog.cBAligntoReferenceRaster.setEnabled(is_checked)

        if is_checked:
            self.dialog.cBAligntoReferenceRaster.setVisible(True)
            self.dialog.cBAligntoReferenceRaster.show()

            # Populate the combo box with raster layers from both QGIS and plugin memory
            plugin_memory_layers = self.dialog.plugin_memory.get("grid_resampled_layers", {})
            populate_reference_raster_combobox(
                self.dialog.cBAligntoReferenceRaster,
                plugin_memory=plugin_memory_layers
            )
       

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.tBChooseFolderGridResampling.setText(folder)

    def run_grid_resampling(self):
        """
        Executes grid resampling based on user options and adds output to QGIS and plugin memory.
        Also copies .aux.xml and _timestamps.csv sidecars if they exist.
        """
        dialog = self.dialog
        try:
            if not dialog.cBEnableGridResampling.isChecked():
                dialog.tELog.append("INFO: Grid resampling is disabled. Skipping step.")
                return

            selected_items = dialog.lWInputPollutantRasterClip.selectedItems()
            if not selected_items:
                QMessageBox.warning(dialog, "Warning", "No raster layer selected for resampling.")
                return

            output_folder = dialog.tBChooseFolderGridResampling.text()
            if not output_folder or not os.path.exists(output_folder):
                QMessageBox.warning(dialog, "Warning", "Please choose a valid output folder.")
                return

            resampling_method = dialog.cBGridResamplingMethod.currentText()
            resampling_enum = {
                "nearest": Resampling.nearest,
                "bilinear": Resampling.bilinear,
                "cubic": Resampling.cubic,
                "cubic_spline": Resampling.cubic_spline,
                "lanczos": Resampling.lanczos,
                "average": Resampling.average,
                "mode": Resampling.mode,
                "max": Resampling.max,
                "min": Resampling.min,
                "med": Resampling.med,
                "q1": Resampling.q1,
                "q3": Resampling.q3
            }.get(resampling_method.lower(), Resampling.nearest)

            dialog.pBDataRefinement.setMaximum(len(selected_items))
            dialog.pBDataRefinement.setValue(0)

            for idx, item in enumerate(selected_items):
                raster_name = item.text()
                original_layer = QgsProject.instance().mapLayersByName(raster_name)[0]
                input_path = original_layer.source()
                output_path = os.path.join(output_folder, f"{raster_name}_resampled.tif")

                with rasterio.open(input_path) as src:
                    transform = src.transform
                    profile = src.profile.copy()
                    nodata = src.nodata

                    if dialog.rBTargetResolutionGridResampling.isChecked():
                        try:
                            res_m = float(dialog.lETargetResolutionGridResampling.text())
                        except ValueError:
                            QMessageBox.critical(dialog, "Error", "Invalid resolution format.")
                            return

                        scale_x = transform.a / res_m
                        scale_y = -transform.e / res_m
                        new_width = int(src.width * scale_x)
                        new_height = int(src.height * scale_y)

                        dst_transform = rasterio.Affine(
                            res_m, transform.b, transform.c,
                            transform.d, -res_m, transform.f
                        )
                        resampled_data = src.read(
                            out_shape=(src.count, new_height, new_width),
                            resampling=resampling_enum
                        )

                    elif dialog.rBRefernceRasterGridResampling.isChecked():
                        ref_name = dialog.cBAligntoReferenceRaster.currentText()
                        ref_layer = QgsProject.instance().mapLayersByName(ref_name)[0]
                        with rasterio.open(ref_layer.source()) as ref_src:
                            ref_transform = ref_src.transform
                            ref_shape = (ref_src.height, ref_src.width)

                            dst_transform = ref_transform
                            new_width, new_height = ref_shape[1], ref_shape[0]

                            resampled_data = src.read(
                                out_shape=(src.count, new_height, new_width),
                                resampling=resampling_enum
                            )
                    else:
                        QMessageBox.warning(dialog, "Warning", "No resampling method selected.")
                        return

                    profile.update({
                        "height": new_height,
                        "width": new_width,
                        "transform": dst_transform,
                        "nodata": nodata
                    })

                    with rasterio.open(output_path, "w", **profile) as dst:
                        dst.write(resampled_data)

                # Add resampled raster to QGIS and plugin memory
                if hasattr(dialog, "plugin_memory"):
                    layer = add_raster_to_qgis_and_plugin_memory(
                        output_path,
                        layer_name=f"Resampled_{raster_name}",
                        plugin_memory_dict=dialog.plugin_memory["grid_resampled_layers"]
                    )
                    if layer:
                        dialog.tELog.append(f"INFO: Added resampled layer to QGIS and plugin memory: {layer.name()}")
                    else:
                        dialog.tELog.append(f"<font color='red'>ERROR: Failed to add resampled layer for {raster_name}</font>")

                # Copy .aux.xml sidecar
                sidecar_src = input_path + ".aux.xml"
                sidecar_dst = output_path + ".aux.xml"
                if os.path.exists(sidecar_src):
                    try:
                        shutil.copyfile(sidecar_src, sidecar_dst)
                        dialog.tELog.append(f"INFO: Copied aux.xml metadata sidecar: {sidecar_dst}")
                    except Exception as e:
                        dialog.tELog.append(f"<font color='red'>WARNING: Failed to copy aux.xml sidecar: {e}</font>")

                # Copy _timestamps.csv sidecar
                timestamp_csv_src = os.path.splitext(input_path)[0] + "_timestamps.csv"
                timestamp_csv_dst = os.path.splitext(output_path)[0] + "_timestamps.csv"
                if os.path.exists(timestamp_csv_src):
                    try:
                        shutil.copyfile(timestamp_csv_src, timestamp_csv_dst)
                        dialog.tELog.append(f"INFO: Copied timestamps CSV: {timestamp_csv_dst}")
                    except Exception as e:
                        dialog.tELog.append(f"<font color='red'>WARNING: Failed to copy timestamps CSV: {e}</font>")

                dialog.tELog.append(f"INFO: Resampled {raster_name} saved to {output_path}")
                dialog.pBDataRefinement.setValue(idx + 1)

            dialog.tELog.append("INFO: Grid resampling completed.")
            QMessageBox.information(dialog, "Success", "Grid resampling completed successfully.")

        except Exception as e:
            dialog.tELog.append(f"<font color='red'>ERROR: Grid resampling failed: {e}</font>")
            QMessageBox.critical(dialog, "Error", f"Grid resampling failed: {e}")



from qgis.core import QgsRasterLayer, QgsMapLayer, QgsProject
from PyQt5.QtWidgets import QMessageBox

def populate_reference_raster_combobox(combo_box, plugin_memory=None):
    """
    Populates the combo box with all valid raster layers in the QGIS project
    and from plugin memory (if provided).
    """
    try:
        combo_box.clear()
        seen_names = set()

        # From QGIS layer tree (Layers panel)
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                combo_box.addItem(layer.name())
                seen_names.add(layer.name())

        # From plugin memory
        if plugin_memory and isinstance(plugin_memory, dict):
            for name, layer in plugin_memory.items():
                if isinstance(layer, QgsRasterLayer) and layer.isValid() and name not in seen_names:
                    combo_box.addItem(name)

        if combo_box.count() == 0:
            combo_box.addItem("No raster layers found")
            combo_box.setEnabled(False)
        else:
            combo_box.setEnabled(True)
            combo_box.setCurrentIndex(0)

    except Exception as e:
        QMessageBox.warning(None, "Error", f"Failed to populate raster combo box: {e}")


from qgis.core import QgsRasterLayer, QgsProject, QgsLayerTreeLayer
from PyQt5.QtWidgets import QMessageBox
import os

def add_raster_to_qgis_and_plugin_memory(raster_path, layer_name, plugin_memory_dict):
    """
    Adds a raster layer to QGIS Layers panel and plugin memory. Ensures visibility.
    """
    try:
        full_path = os.path.abspath(raster_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Raster file not found at: {full_path}")

        # Create and validate layer
        layer = QgsRasterLayer(full_path, layer_name)
        if not layer.isValid():
            raise Exception(f"Raster layer is invalid. Path: {full_path}")

        # Add layer to the project and legend
        QgsProject.instance().addMapLayer(layer)  # addToLegend=True by default

        # Store in plugin memory
        plugin_memory_dict[layer_name] = layer

        print(f"[DEBUG] Successfully added layer to QGIS and memory: {layer_name} from {full_path}")
        return layer

    except Exception as e:
        error_msg = f"Failed to add raster layer to QGIS: {e}"
        print(f"[ERROR] {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
        return None