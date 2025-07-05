import os
import glob
import pandas as pd

# 1) List of bodyparts to KEEP
KEEP_BPS = [
    "nose",
    "left_ear_base",
    "right_ear_base",
    "left_ear_tip",
    "right_ear_tip",
    "back_withers",
    "back_midpoint",
    "back_croup",
    "tail_base",
    "tail_upper_midpoint",
    "tail_midpoint",
    "tail_lower_midpoint",
    "tail_end",
    "head_midpoint",
]

def filter_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with a 3‐level column index (scorer, bodypart, coord),
    returns a new df keeping only columns whose bodypart is in KEEP_BPS.
    """
    cols = [col for col in df.columns if col[1] in KEEP_BPS]
    return df.loc[:, cols]

def process_subject_folder(subject_folder: str):
    """
    In `subject_folder`, finds CollectedData_*.csv and .h5, filters them,
    and writes them back.
    """
    # find all CSVs
    csv_paths = glob.glob(os.path.join(subject_folder, "CollectedData_*.csv"))
    for csv_path in csv_paths:
        # derive scorer from filename if you like; not needed here
        # load CSV (3 header rows: scorer/bodyparts/coords, index is first column)
        df = pd.read_csv(csv_path, header=[0,1,2], index_col=0)
        df_filt = filter_keypoints(df)
        # overwrite CSV
        df_filt.to_csv(csv_path)

        # HDF5 sidecar
        h5_path = csv_path[:-4] + ".h5"
        if os.path.isfile(h5_path):
            # use same key name as you wrote originally
            df_h5 = pd.read_hdf(h5_path, key="collected_data")
            df_h5_filt = filter_keypoints(df_h5)
            # overwrite H5
            df_h5_filt.to_hdf(h5_path, key="collected_data", mode="w", format="table")

def main(project_folder: str):
    """
    project_folder should be the root of your DLC project,
    i.e. the parent of `labeled-data/`.
    """
    ld = os.path.join(project_folder, "labeled-data")
    if not os.path.isdir(ld):
        raise RuntimeError(f"No labeled-data/ under {project_folder}")
    # each subfolder is a subject ID
    for subject in os.listdir(ld):
        subj_folder = os.path.join(ld, subject)
        if os.path.isdir(subj_folder):
            print("Processing", subj_folder)
            process_subject_folder(subj_folder)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python filter_bodyparts.py /path/to/project_folder")
        sys.exit(1)
    main(sys.argv[1])
