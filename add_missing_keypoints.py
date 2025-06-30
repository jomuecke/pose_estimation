import xml.etree.ElementTree as ET

groups = [
    ["left_eye", "left_ear_tip", "left_ear_base", "head_midpoint"],
    ["right_eye", "right_ear_base", "right_ear_tip"],
    ["chest", "throat", "lower_jaw"],
    ["back_withers", "back_midpoint", "back_croup"],
    ["tail_base", "tail_upper_midpoint", "tail_midpoint", "tail_lower_midpoint", "tail_end"],
    ["back_right_hip", "back_right_knee", "back_right_wrist", "back_right_paw"],
    ["back_left_hip", "back_left_knee", "back_left_wrist", "back_left_paw"],
    ["front_left_shoulder", "front_left_elbow", "front_left_paw", "front_left_wrist"],
    ["front_right_shoulder", "front_right_elbow", "front_right_paw", "front_right_wrist"],
]

file_in = "/Users/jonasmucke/Desktop/annotations_2.xml"
file_out = "/Users/jonasmucke/Desktop/annotations_filled_upwards.xml"

tree = ET.parse(file_in)
root = tree.getroot()

for image in root.findall('image'):
    # Get the top-left of the bounding box
    box = image.find('box[@label="Bounding Box"]')
    if box is None:
        continue  # No bounding box for this image, skip

    xtl = float(box.attrib['xtl'])
    ytl = float(box.attrib['ytl'])

    for skeleton in image.findall('skeleton'):
        # For each group, process missing keypoints
        for group_idx, group in enumerate(groups):
            group_x = xtl + group_idx * 40  # 40px right per group
            group_y = ytl
            points_in_group = [kp for kp in skeleton.findall('points')
                               if kp.attrib['label'] in group and
                                  kp.attrib.get('outside', '0') == '1' and
                                  kp.attrib.get('points', '0.00,0.00') == '0.00,0.00']
            for j, kp in enumerate(points_in_group):
                kp.set('outside', '0')
                # Stack upwards
                kp.set('points', f"{group_x:.2f},{group_y - 15*j:.2f}")

tree.write(file_out, encoding="utf-8", xml_declaration=True)
print(f"Done! Saved to {file_out}")
