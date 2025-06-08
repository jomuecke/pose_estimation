import os
import datetime
import pandas as pd
import numpy as np
import h5py
import shutil
import PySimpleGUI as sg

# Constants
KEYPOINT_SUFFIX = ["-x", "-y"]


def extract_id(filename):
    """
    Extract ID from filename: everything after first underscore until fifth underscore.
    """
    parts = filename.split('_')
    if len(parts) >= 5:
        return '_'.join(parts[1:5])
    return None


def read_annotations(csv_path):
    """
    Read the annotation CSV and drop only bounding-box columns.
    Preserves columns even if they contain only NaNs.
    """
    df = pd.read_csv(csv_path)
    # Drop bbox columns
    df = df.drop(columns=[c for c in df.columns if c.startswith('bbox_')], errors='ignore')
    return df


def write_config(project_path, task, scorer, date_str, bodyparts, subject_ids):
    """
    Generate config.yaml matching DeepLabCut format, including all bodyparts.
    """
    cfg_path = os.path.join(project_path, 'config.yaml')
    with open(cfg_path, 'w') as f:
        # Project definitions
        f.write("# Project definitions (do not edit)\n")
        f.write(f"Task: {task}\n")
        f.write(f"scorer: {scorer}\n")
        f.write(f"date: {date_str}\n")
        f.write("multianimalproject: false\n")
        f.write("identity: false\n\n")
        # Project path
        f.write("# Project path (change when moving around)\n")
        f.write(f"project_path: {project_path}\n\n")
        # Engine
        f.write("# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)\n")
        f.write("engine: pytorch\n\n")
        # Video sets
        f.write("# Annotation data set configuration (and individual video cropping parameters)\n")
        f.write("video_sets:\n")
        for sid in subject_ids:
            vid_path = os.path.join(project_path, 'videos', f"{sid}.mkv")
            f.write(f"  {vid_path}:\n    null\n")
        f.write("\n")
        # Bodyparts list
        f.write("bodyparts:\n")
        for bp in bodyparts:
            f.write(f"- {bp}\n")
        f.write("\n")
        # Conversion tables placeholder
        f.write("# Conversion tables to fine-tune SuperAnimal weights\n")
        f.write("SuperAnimalConversionTables:\n  {}\n")


def create_dlc_structure(base_folder, annotations_df, scorer, view, animal):
    """
    Build DeepLabCut structure, write config, and export full keypoint files,
    including bodyparts with no current annotations (all NaNs).
    """
    # Compute project identifiers
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    project_name = f"{view}{animal}-{scorer}-{date_str}"
    project_path = os.path.join(base_folder, project_name)
    # Prepare directories
    ld = os.path.join(project_path, 'labeled-data')
    vid_dir = os.path.join(project_path, 'videos')
    os.makedirs(ld, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    # Extract all bodyparts from columns ending in -x
    all_parts = sorted({col[:-2] for col in annotations_df.columns if col.endswith('-x')})
    task = f"{view}{animal}"
    # Subject IDs for config and folder creation
    annotations_df['ID'] = annotations_df['filename'].apply(extract_id)
    subject_ids = sorted(annotations_df['ID'].dropna().unique())
    # Write config.yaml with full bodyparts list
    write_config(project_path, task, scorer, date_str, all_parts, subject_ids)
    # Export per-subject CollectedData
    for sid, group in annotations_df.groupby('ID'):
        subfld = os.path.join(ld, sid)
        os.makedirs(subfld, exist_ok=True)
        records, paths = [], []
        for _, row in group.iterrows():
            fn = row['filename']
            src = os.path.join(base_folder, 'Images', fn)
            dst = os.path.join(subfld, fn)
            if os.path.exists(src): shutil.copy(src, dst)
            rel = os.path.join('labeled-data', sid, fn)
            paths.append(rel)
            records.append(row.drop(['filename', 'ID']).to_dict())
        df_out = pd.DataFrame(records)
        # Ensure all_parts columns exist, even if all NaN
        for bp in all_parts:
            for suf in ['-x','-y']:
                col = bp + suf
                if col not in df_out.columns:
                    df_out[col] = ''  # leave empty entries blank
        # Build MultiIndex for columns
        mi = pd.MultiIndex.from_product([[scorer], all_parts, ['x','y']],
                                        names=['scorer','bodyparts','coords'])
        ordered_cols = [f"{bp}-x" for bp in all_parts] + [f"{bp}-y" for bp in all_parts]
        df_out = df_out[ordered_cols]
        df_out.columns = mi
        # Index with image paths
        df_out.index = paths
        # Save CSV and H5
        csv_path = os.path.join(subfld, f"CollectedData_{scorer}.csv")
        df_out.to_csv(csv_path, index=True)
        h5_path = os.path.join(subfld, f"CollectedData_{scorer}.h5")
        with h5py.File(h5_path,'w') as hf:
            grp = hf.create_group('collected_data')
            grp.create_dataset('coords', data=df_out.values)
            dt = h5py.string_dtype(encoding='utf-8')
            grp.create_dataset('file_paths', data=np.array(paths, dtype=dt))
            grp.attrs['scorer'] = scorer
            grp.attrs['bodyparts'] = all_parts
            grp.attrs['coords'] = ['x','y']
    sg.popup(f"Project created at: {project_path}")
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
    win = build_gui()
    while True:
        ev, vals = win.read()
        if ev in (sg.WIN_CLOSED,'Exit'): break
        base = vals['-F-'] or ''
        csvp = os.path.join(base,'annotations.csv')
        if ev=='Analyze Keypoints':
            if not os.path.isfile(csvp): sg.popup_error('Select folder with annotations.csv'); continue
            raw = pd.read_csv(csvp)
            parts = sorted({c[:-2] for c in raw.columns if c.endswith('-x') and not c.startswith('bbox_')})
            labeled = [p for p in parts if raw[f"{p}-x"].notna().any()]
            miss = [p for p in parts if p not in labeled]
            msg = f"Labeled ({len(labeled)}): " + ",".join(labeled)
            msg+= f"\nMissing ({len(miss)}): " + ",".join(miss)
            sg.popup_scrolled(msg, title='Keypoints')
        elif ev=='Create DLC':
            if not os.path.isfile(csvp): sg.popup_error('annotations.csv missing'); continue
            df_ann = read_annotations(csvp)
            create_dlc_structure(base, df_ann, vals['-S-'], vals['-V-'], vals['-A-'])
    win.close()

if __name__=='__main__': main()
