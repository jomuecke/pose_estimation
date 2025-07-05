import os
import datetime
import pandas as pd
import h5py
import shutil
import FreeSimpleGUI as sg

# Constants
KEYPOINT_SUFFIX = ["-x", "-y"]


def extract_id(filename):
    """
    Extract subject ID from filename: parts 1-4 joined by underscores.
    """
    parts = filename.split('_')
    if len(parts) >= 5:
        return '_'.join(parts[1:5])
    return None


def read_annotations(csv_path):
    """
    Read annotations CSV and drop only bounding-box columns.
    Preserve all keypoint columns, even if they contain only NaNs.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith('bbox_')], errors='ignore')
    return df


def write_config(project_path, task, scorer, date_str, bodyparts, subject_ids):
    """
    Generate config.yaml with specified sections and defaults.
    """
    cfg_path = os.path.join(project_path, 'config.yaml')
    with open(cfg_path, 'w') as f:
        f.write("# Project definitions (do not edit)\n")
        f.write(f"Task: {task}\n")
        f.write(f"scorer: {scorer}\n")
        f.write(f"date: {date_str}\n")
        f.write("multianimalproject: false\n")
        f.write("identity: false\n\n")

        f.write("# Project path (change when moving around)\n")
        f.write(f"project_path: {project_path}\n\n")

        f.write("# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)\n")
        f.write("engine: pytorch\n\n")

        f.write("# Plotting configuration\n")
        f.write("skeleton: []\n")
        f.write("skeleton_color: black\n")
        f.write("pcutoff: 0.6\n")
        f.write("dotsize: 8\n")
        f.write("alphavalue: 0.7\n")
        f.write("colormap: jet\n\n")

        f.write("# Annotation data set configuration (and individual video cropping parameters)\n")
        f.write("video_sets:\n")
        for sid in subject_ids:
            vid = os.path.join(project_path, 'videos', f"{sid}.mkv")
            f.write(f"  {vid}: null\n")
        f.write("\n")

        f.write("# Bodyparts to track\n")
        f.write("bodyparts:\n")
        for bp in bodyparts:
            f.write(f"- {bp}\n")
        f.write("\n")

        f.write("# Training,Evaluation and Analysis configuration\n")
        f.write("TrainingFraction:\n- 0.9\n")
        f.write("iteration: 0\n")
        f.write("default_net_type: resnet_50\n")
        f.write("default_augmenter: imgaug\n")
        f.write("snapshotindex: -1\n")
        f.write("detector_snapshotindex: -1\n")
        f.write("batch_size: 4\n")
        f.write("detector_batch_size: 1\n\n")

        f.write("# Cropping Parameters (for analysis and outlier frame detection)\n")
        f.write("cropping: false\n")
        f.write("# if cropping is true for analysis, then set the values here:\n")
        f.write("x1: 0\n")
        f.write("x2: 640\n")
        f.write("y1: 277\n")
        f.write("y2: 624\n\n")

        f.write("# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)\n")
        f.write("corner2move2:\n- 50\n- 50\n")
        f.write("move2corner: true\n")


def create_dlc_structure(base_folder, annotations_df, scorer, view, animal):
    """
    Build DeepLabCut project: write config and export per-subject keypoint files
    preserving original column order and alignment.
    """
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    project_name = f"{view}{animal}-{scorer}-{date_str}"
    project_path = os.path.join(base_folder, project_name)
    ld = os.path.join(project_path, 'labeled-data')
    vid_dir = os.path.join(project_path, 'videos')
    os.makedirs(ld, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    # Determine bodyparts in original order
    bp_list = []
    for col in annotations_df.columns:
        if col.endswith('-x'):
            bp = col[:-2]
            if bp not in bp_list:
                bp_list.append(bp)
    bodyparts = bp_list
    task = f"{view}{animal}"

    # Prepare subject IDs
    annotations_df['ID'] = annotations_df['filename'].apply(extract_id)
    subject_ids = sorted(annotations_df['ID'].dropna().unique())

    # Write config with correct bodyparts order
    write_config(project_path, task, scorer, date_str, bodyparts, subject_ids)

    # Export data for each subject
    for sid, group in annotations_df.groupby('ID'):
        subfld = os.path.join(ld, sid)
        os.makedirs(subfld, exist_ok=True)
        # Copy images into subject folder
        for fname in group['filename']:
            src_img = os.path.join(base_folder, 'Images', fname)
            dst_img = os.path.join(subfld, fname)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
        # Index by filename to preserve original column alignment
        grp = group.set_index('filename', drop=True)
        # Build ordered columns list: x then y for each bodypart in original sequence
        ordered_cols = []
        for bp in bodyparts:
            ordered_cols.append(f"{bp}-x")
            ordered_cols.append(f"{bp}-y")
        # Slice DataFrame directly
        df_out = grp.reindex(columns=ordered_cols)
        # Create MultiIndex header
        mi = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords']
        )
        df_out.columns = mi
        # Set index to relative paths
        df_out.index = [os.path.join('labeled-data', sid, fn) for fn in df_out.index]
        # Save CSV
        csv_path = os.path.join(subfld, f"CollectedData_{scorer}.csv")
        df_out.to_csv(csv_path, index=True)
        # Save H5
        h5_path = os.path.join(subfld, f"CollectedData_{scorer}.h5")
        df_out.to_hdf(h5_path, key='collected_data', mode='w', format='table')

    sg.popup(f"DeepLabCut project created at: {project_path}")
    return project_path


def build_gui():
    try: sg.theme('DarkBlue')
    except: sg.ChangeLookAndFeel('Dark')
    layout = [
        [sg.Text('Annotation folder:'), sg.Input(key='-F-'), sg.FolderBrowse()],
        [sg.Text('Scorer:'), sg.Input('jm', key='-S-')],
        [sg.Text('View:'), sg.Combo(['top','side','bottom'], default_value='top', key='-V-')],
        [sg.Text('Animal:'), sg.Combo(['rat','mouse'], default_value='mouse', key='-A-')],
        [sg.Button('Analyze Keypoints'), sg.Button('Create DLC'), sg.Button('Exit')]
    ]
    return sg.Window('DLC Converter', layout)


def main():
    window = build_gui()
    while True:
        event, vals = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        base = vals['-F-'] or ''
        csvp = os.path.join(base, 'annotations.csv')
        if event == 'Analyze Keypoints':
            if not os.path.isfile(csvp):
                sg.popup_error('Select folder with annotations.csv')
                continue
            raw = pd.read_csv(csvp)
            parts = []
            for col in raw.columns:
                if col.endswith('-x'):
                    bp = col[:-2]
                    if bp not in parts:
                        parts.append(bp)
            labeled = [p for p in parts if raw[f"{p}-x"].notna().any()]
            missing = [p for p in parts if p not in labeled]
            msg = f"Labeled ({len(labeled)}): " + ", ".join(labeled)
            msg += f"\nMissing ({len(missing)}): " + ", ".join(missing)
            sg.popup_scrolled(msg, title='Keypoints')
        elif event == 'Create DLC':
            if not os.path.isfile(csvp):
                sg.popup_error('annotations.csv missing')
                continue
            df_ann = read_annotations(csvp)
            create_dlc_structure(
                base,
                df_ann,
                vals['-S-'],
                vals['-V-'],
                vals['-A-']
            )
    window.close()

if __name__ == '__main__':
    main()
