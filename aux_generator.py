import os
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def generate_aux_xml(tif_path, band_labels):
    """
    Generates a .aux.xml file next to the given .tif file to store band metadata.
    
    Parameters:
        tif_path (str): Full path to the .tif raster.
        band_labels (list): List of band names (usually timestamps as strings).
    
    Returns:
        str: Path to the generated .aux.xml file.
    """
    try:
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"TIFF file not found: {tif_path}")

        aux_path = f"{tif_path}.aux.xml"
        logger.debug(f"Generating .aux.xml metadata for {tif_path}")

        # Build XML structure
        root = ET.Element("PAMDataset")
        metadata = ET.SubElement(root, "Metadata", domain="IMAGE_STRUCTURE")

        for i, label in enumerate(band_labels):
            ET.SubElement(metadata, "MDI", key=f"Band_{i+1}_description").text = label

        # Save XML
        tree = ET.ElementTree(root)
        tree.write(aux_path, encoding="UTF-8", xml_declaration=True)

        logger.info(f"Generated .aux.xml metadata at: {aux_path}")
        return aux_path

    except Exception as e:
        logger.error(f"Error generating .aux.xml for {tif_path}: {e}", exc_info=True)
        return None
