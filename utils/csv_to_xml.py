import csv
import xml.etree.ElementTree as ET

def load_meta_block(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return f.read()

def create_annotation_from_csv(csv_path, meta_xml_path, output_xml_path):
    # Load meta section as a string
    meta_content = load_meta_block(meta_xml_path)

    # Root <annotations>
    annotations = ET.Element("annotations")

    # Append <version>
    version = ET.SubElement(annotations, "version")
    version.text = "1.1"

    # Append <meta> from file (parsed)
    meta_element = ET.fromstring(meta_content)
    annotations.append(meta_element)

    # Read CSV
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        bbox_fields = {"bbox_tl", "bbox_br"}
        keypoints = sorted(set(
            col[:-2] for col in reader.fieldnames
            if col.endswith("-x") and col[:-2] not in bbox_fields
        ))
        rows = list(reader)


    for img_id, row in enumerate(rows):
        image = ET.SubElement(annotations, "image", {
            "id": str(img_id),
            "name": row["filename"],
            "subset": "default",
            "task_id": "2",
            "width": "1280",
            "height": "720"
        })

        # Add bounding box if available
        if all(k in row for k in ["bbox_tl-x", "bbox_tl-y", "bbox_br-x", "bbox_br-y"]):
            if row["bbox_tl-x"] and row["bbox_tl-y"] and row["bbox_br-x"] and row["bbox_br-y"]:
                ET.SubElement(image, "box", {
                    "label": "Bounding Box",
                    "source": "file",
                    "occluded": "0",
                    "xtl": row["bbox_tl-x"],
                    "ytl": row["bbox_tl-y"],
                    "xbr": row["bbox_br-x"],
                    "ybr": row["bbox_br-y"],
                    "z_order": "0"
                })

        # Add skeleton with points
        skeleton = ET.SubElement(image, "skeleton", {
            "label": "RatSkeleton",
            "source": "file",
            "z_order": "0"
        })

        for kp in keypoints:
            x = row.get(f"{kp}-x", "")
            y = row.get(f"{kp}-y", "")
            outside = "1" if not x or not y else "0"
            points_str = f"{x},{y}" if x and y else "0.0,0.0"
            ET.SubElement(skeleton, "points", {
                "label": kp,
                "source": "file",
                "outside": outside,
                "occluded": "0",
                "points": points_str
            })

    # Write XML to file
    tree = ET.ElementTree(annotations)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"CVAT XML written to: {output_xml_path}")

# Example usage
csv_path = "/Users/jonasmucke/Desktop/pose_estimation/Rat/side2194/annotations.csv"
meta_xml_path = "/Users/jonasmucke/Desktop/merged_output/annotations_meta.xml"  # this file should contain only the <meta>...</meta> block
output_xml_path = "/Users/jonasmucke/Desktop/pose_estimation/Rat/side2194/annotations.xml"

create_annotation_from_csv(csv_path, meta_xml_path, output_xml_path)
