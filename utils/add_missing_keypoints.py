## Changes the coordinates of keypoints that have not been predicted during inference
## Puts them above the bounding box, which makes it faster to correct in CVAT

import xml.etree.ElementTree as ET

groups = [
    ["nose", "left_eye", "left_ear_tip", "left_ear_base", "head_midpoint"],
    ["right_eye", "right_ear_base", "right_ear_tip"],
    ["chest", "throat", "lower_jaw"],
    ["back_withers", "back_midpoint", "back_croup"],
    ["tail_base", "tail_upper_midpoint", "tail_midpoint", "tail_lower_midpoint", "tail_end"],
    ["back_right_hip", "back_right_knee", "back_right_wrist", "back_right_paw"],
    ["back_left_hip", "back_left_knee", "back_left_wrist", "back_left_paw"],
    ["front_left_shoulder", "front_left_elbow", "front_left_paw", "front_left_wrist"],
    ["front_right_shoulder", "front_right_elbow", "front_right_paw", "front_right_wrist"],
]

input_file = "/Users/jonasmucke/Downloads/predictions_250703_100551.xml"
output_file = "/Users/jonasmucke/Downloads/predictions_side75_added_keypoints.xml"

tree = ET.parse(input_file)
root = tree.getroot()

for image in root.findall('image'):
    box = image.find('box[@label="Bounding Box"]')
    if box is None:
        continue

    xtl = float(box.attrib['xtl'])
    ytl = float(box.attrib['ytl'])

    for skeleton in image.findall('skeleton'):
        # For each group
        for group_idx, group in enumerate(groups):
            group_x = xtl + group_idx * 40
            group_y = ytl
            missing = [
                kp for kp in skeleton.findall('points')
                if kp.attrib['label'] in group
                and kp.attrib.get('outside', '0') == '1'
                and kp.attrib.get('points', '0.0,0.0').replace(" ", "") in ['0.0,0.0', '0.00,0.00']
            ]
            for j, kp in enumerate(missing):
                kp.set('points', f"{group_x:.2f},{group_y - 15*j:.2f}")
                kp.set('outside', '1')  # keep hidden!

tree.write(output_file, encoding='utf-8', xml_declaration=True)
print(f"Saved to {output_file}")
