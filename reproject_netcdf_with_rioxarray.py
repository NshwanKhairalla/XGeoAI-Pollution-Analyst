import os
import logging
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pyproj import Transformer
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def detect_main_variable(ds):
    for var in ds.data_vars:
        dims = ds[var].dims
        if "time" in dims and ("latitude" in dims or "lat" in dims) and ("longitude" in dims or "lon" in dims):
            return var
    raise ValueError("No suitable variable with time, lat/lon dimensions found.")


def prettify_xml(elem):
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def write_aux_xml_with_global(tif_path, band_metadatas, global_metadata):
    aux_xml_path = tif_path + ".aux.xml"
    root = Element("PAMDataset")

    if global_metadata:
        metadata_global = SubElement(root, "Metadata")
        for k, v in global_metadata.items():
            if v is not None and v != "":
                item = SubElement(metadata_global, "MDI", key=str(k))
                item.text = str(v)

    for i, band_metadata in enumerate(band_metadatas, start=1):
        band = SubElement(root, "VRTRasterBand", dataType="Float32", band=str(i))
        metadata_band = SubElement(band, "Metadata")
        for k, v in band_metadata.items():
            if v is not None and v != "":
                item = SubElement(metadata_band, "MDI", key=str(k))
                item.text = str(v)

    with open(aux_xml_path, "w", encoding="utf-8") as f:
        f.write(prettify_xml(root))


def reproject_netcdf_with_rioxarray(input_path, target_crs="EPSG:4326", output_path=None):
    import rioxarray
    import os
    import tempfile
    import pandas as pd
    from pyproj import Transformer
    import logging

    logger = logging.getLogger(__name__)

    try:
        ds = xr.open_dataset(input_path, engine="netcdf4")
        variable = detect_main_variable(ds)
        da = ds[variable]

        for dim in [d for d in da.dims if d not in ["latitude", "longitude", "time"]]:
            da = da.isel({dim: 0})
            logger.warning(f"Dropped dimension '{dim}' for reprojection.")

        da = da.rename({"latitude": "y", "longitude": "x"})

        if (da.x > 180).any():
            da = da.assign_coords(x=(((da.x + 180) % 360) - 180))
            da = da.sortby("x")
            logger.info("Normalized longitudes from 0–360 to -180–180.")

        if not da.rio.crs:
            logger.warning("No CRS found, assuming EPSG:4326 for input.")
            da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.set_spatial_dims("x", "y", inplace=True)

        xmin, xmax = float(da.x.min()), float(da.x.max())
        ymin, ymax = float(da.y.min()), float(da.y.max())
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        minx_proj, miny_proj = transformer.transform(xmin, ymin)
        maxx_proj, maxy_proj = transformer.transform(xmax, ymax)

        target_resolution = 10000
        width = int((maxx_proj - minx_proj) / target_resolution)
        height = int((maxy_proj - miny_proj) / target_resolution)

        reprojected_slices = []
        for t in range(len(da.time)):
            single_slice = da.isel(time=t)
            reprojected_slice = single_slice.rio.reproject(
                target_crs,
                shape=(height, width),
                resampling=Resampling.nearest
            )
            reprojected_slices.append(reprojected_slice.expand_dims(dim={"time": [da.time.values[t]]}))

        reprojected = xr.concat(reprojected_slices, dim="time")

        timestamps = [
            str((pd.Timestamp("1970-01-01") + pd.to_timedelta(val)).to_pydatetime())
            for val in da.time.values
        ]

        # 🔥 HERE IS THE IMPORTANT CHANGE 🔥
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
                output_path = tmpfile.name

        reprojected.rio.to_raster(output_path, driver="GTiff")

        band_metadatas = []
        for t, ts in enumerate(timestamps):
            band_metadatas.append({
                "BandName": ts,
                "NETCDF_VARNAME": variable,
                "species": getattr(da, "species", ""),
                "standard_name": getattr(da, "standard_name", ""),
                "units": getattr(da, "units", ""),
                "timestamp": ts,
            })

        global_metadata = {}
        if "value" in da.attrs:
            global_metadata["value"] = da.attrs["value"]
        if "AREA_OR_POINT" in ds.attrs:
            global_metadata["AREA_OR_POINT"] = ds.attrs["AREA_OR_POINT"]

        write_aux_xml_with_global(output_path, band_metadatas, global_metadata)

        logger.info(f"Created auxiliary XML for {output_path}")
        logger.info(f"Saved reprojected raster to: {output_path}")
        return output_path, variable

    except Exception as e:
        logger.error(f"Error during NetCDF reprojection: {e}", exc_info=True)
        return None, None