# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Cluster Browser")

st.title("🎥 Pose Cluster Browser")

# 1. Sidebar: controls for loading data, selecting cluster, etc.
st.sidebar.header("Load & Settings")

csv_path = st.sidebar.text_input(
    "Path to annotations CSV:",
    value="Rat/side/annotations_with_clusters.csv",
    help="This CSV must have columns: 'filename', 'cluster', (optionally 'tsne1','tsne2')."
)
image_root = st.sidebar.text_input(
    "Image folder/root path:",
    value="Rat/side/images",
    help="Where to look for each image filename in the CSV."
)

# If user does not fill, warn and stop
if not csv_path:
    st.sidebar.error("Please specify the CSV path.")
    st.stop()

# 1a. Read CSV (once)
@st.cache_data
def load_df(path):
    return pd.read_csv(path)

try:
    df = load_df(csv_path)
except FileNotFoundError:
    st.sidebar.error(f"Could not find CSV at '{csv_path}'.")
    st.stop()

# Check essential columns
required_cols = {"filename", "cluster"}
if not required_cols.issubset(set(df.columns)):
    st.sidebar.error(f"CSV must contain columns: {required_cols}")
    st.sidebar.write("Columns found:", df.columns.tolist())
    st.stop()

# 2. Show overall stats
n_samples = len(df)
n_clusters = int(df["cluster"].nunique())
st.sidebar.markdown(f"**Total frames:** {n_samples}")
st.sidebar.markdown(f"**Number of clusters:** {n_clusters}")

# 3. If tsne columns exist, show a small 2D scatter colored by cluster
if {"tsne1", "tsne2"}.issubset(set(df.columns)):
    st.sidebar.markdown("### t-SNE overview")
    # We’ll draw a small thumbnail of the entire scatter 
    # so users get a sense of how clusters spread.
    fig, ax = plt.subplots(figsize=(3, 3))
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    for cl in range(n_clusters):
        pts = df[df["cluster"] == cl]
        ax.scatter(
            pts["tsne1"],
            pts["tsne2"],
            s=5,
            alpha=0.6,
            color=cmap(cl),
            label=f"{cl}"
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE clusters")
    st.sidebar.pyplot(fig)
    st.sidebar.caption("Colored by `cluster`")

# 4. Let user pick a cluster ID
cluster_ids = sorted(df["cluster"].unique())
sel_cluster = st.sidebar.selectbox(
    "Select a cluster to inspect:",
    cluster_ids,
    format_func=lambda x: f"Cluster {x}",
)

# Re-plot t-SNE here, greying out all clusters except the selected one
if {"tsne1", "tsne2"}.issubset(df.columns):
    st.write("### t-SNE: only the selected cluster is colored")

    # 1. Create a matplotlib figure
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # 2. Plot all points in light gray
    ax2.scatter(
        df["tsne1"],
        df["tsne2"],
        s=5,
        color="lightgray",
        alpha=0.5,
        label="_nolegend_",  # no legend entry for gray points
    )

    # 3. Overlay only the selected cluster in a distinct color
    highlight = df[df["cluster"] == sel_cluster]
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    color_sel = cmap(sel_cluster % 10)  # pick a color for your selected cluster

    ax2.scatter(
        highlight["tsne1"],
        highlight["tsne2"],
        s=15,
        color=color_sel,
        alpha=0.8,
        label=f"Cluster {sel_cluster}",
    )

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"t-SNE (highlighting Cluster {sel_cluster})")
    ax2.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig2)


# 5. Filter df to that cluster
cluster_df = df[df["cluster"] == sel_cluster]
n_in_cluster = len(cluster_df)
st.sidebar.markdown(f"Frames in Cluster {sel_cluster}: **{n_in_cluster}**")

if n_in_cluster == 0:
    st.warning(f"No frames found in cluster {sel_cluster}.")
    st.stop()

# 6. Show a random subset of sample images (up to 9)
n_display = st.sidebar.slider(
    "Number of samples to show:",
    min_value=1,
    max_value=min(25, n_in_cluster),
    value=min(9, n_in_cluster),
    step=1,
)

sample_paths = cluster_df["filename"].sample(n=n_display, random_state=42).tolist()

# 7. Create columns to display images side by side
cols = st.columns(min(n_display, 5))  # up to 5 columns per row

for idx, img_name in enumerate(sample_paths):
    col_idx = idx % len(cols)
    with cols[col_idx]:
        abs_path = os.path.join(image_root, img_name)
        if os.path.isfile(abs_path):
            st.image(
                abs_path,
                caption=os.path.basename(img_name),
                use_container_width=True,
            )
        else:
            st.write(f"❌ Not found: {img_name}")
            st.write(f"Expected at: {abs_path}")

# 8. (Optional) Show filenames & any additional metadata
if st.sidebar.checkbox("Show filenames & metadata table", value=False):
    st.write(
        cluster_df[
            ["filename", "cluster", "tsne1", "tsne2"]
            if {"tsne1", "tsne2"}.issubset(df.columns)
            else ["filename", "cluster"]
        ].reset_index(drop=True)
    )
