import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN
import folium

# ---------------------- Load and Prepare Data ----------------------
df = pd.read_csv("../Assignment7/Power_Plants.csv")

df = df.dropna(subset=["PrimSource", "Longitude", "Latitude", "Total_MW"]).copy()

# Coordinates for Menomonie, WI
menomonie_coords = (44.8755, -91.9194)

df["Distance_km"] = df.apply(
    lambda row: great_circle(menomonie_coords, (row["Latitude"], row["Longitude"])).km,
    axis=1
)

# ---------------------- Cluster with DBSCAN ----------------------
coords_rad = np.radians(df[["Latitude", "Longitude"]])
db = DBSCAN(eps=200 / 6371, min_samples=3, algorithm='ball_tree', metric='haversine')
df["Cluster"] = db.fit_predict(coords_rad)

# ---------------------- Calculate Distance to Cluster Center ----------------------
cluster_centers = df[df["Cluster"] != -1].groupby("Cluster")[["Latitude", "Longitude"]].mean()

def dist_to_center(row):
    if row["Cluster"] == -1:
        return None
    center = cluster_centers.loc[row["Cluster"]]
    return great_circle((row["Latitude"], row["Longitude"]), (center["Latitude"], center["Longitude"])).km

df["Dist_to_ClusterCenter"] = df.apply(dist_to_center, axis=1)

# ---------------------- Export Per-Cluster CSVs with Summary ----------------------
summary_cols = ["Plant_Name", "PrimSource", "Total_MW", "Distance_km", "Cluster", "Dist_to_ClusterCenter"]

clusters = df[df["Cluster"] != -1]["Cluster"].unique()
clusters.sort()

os.makedirs("clusters_output", exist_ok=True)

for cluster_id in clusters:
    cluster_df = df[df["Cluster"] == cluster_id]
    total_mw = cluster_df["Total_MW"].sum()
    num_plants = cluster_df.shape[0]
    avg_dist_center = cluster_df["Dist_to_ClusterCenter"].mean()
    source_counts = cluster_df["PrimSource"].value_counts().to_dict()
    closest_row = cluster_df.loc[cluster_df["Distance_km"].idxmin()]
    closest_name = closest_row["Plant_Name"]
    closest_dist = closest_row["Distance_km"]

    csv_filename = f"clusters_output/cluster_{cluster_id}.csv"

    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write(f"Cluster {cluster_id} Summary\n")
        f.write(f"Number of Plants: {num_plants}\n")
        f.write(f"Total Capacity: {total_mw:,.1f} MW\n")
        f.write(f"Average Distance to Cluster Center: {avg_dist_center:.2f} km\n")
        f.write(f"Closest Plant to Menomonie: \"{closest_name}\" ({closest_dist:.1f} km)\n")
        f.write("Primary Sources:\n")
        for source, count in source_counts.items():
            f.write(f"  - {source}: {count}\n")
        f.write("\nPlant-Level Data:\n")
        f.write(",".join(summary_cols) + "\n")
        for _, row in cluster_df[summary_cols].iterrows():
            f.write(",".join(str(row[col]) for col in summary_cols) + "\n")

    print(f"\nCluster {cluster_id}")
    print("-" * 40)
    print(f"Number of Plants: {num_plants}")
    print(f"Total Capacity: {total_mw:,.1f} MW")
    print("Primary Sources:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")
    print(f"Average Distance to Cluster Center: {avg_dist_center:.2f} km")
    print(f"Closest Plant to Menomonie: \"{closest_name}\" ({closest_dist:.1f} km)")
    print(f"CSV exported to: {csv_filename}")

# ---------------------- Generate Interactive Folium Map ----------------------
html_filename = "powerplant_clusters.html"
if not os.path.exists(html_filename):
    m = folium.Map(location=menomonie_coords, zoom_start=5)

    folium.Marker(
        menomonie_coords,
        tooltip="Menomonie, WI",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Color by PrimSource using matplotlib's tab20
    sources = df["PrimSource"].unique()
    cmap = plt.cm.get_cmap('tab20', len(sources))
    color_map = {
        source: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        for source, (r, g, b, _) in zip(sources, cmap.colors)
    }

    for _, row in df.iterrows():
        popup_text = (
            f"{row['Plant_Name']}<br>"
            f"Source: {row['PrimSource']}<br>"
            f"MW: {row['Total_MW']}<br>"
            f"Dist to Menomonie: {row['Distance_km']:.1f} km<br>"
            f"Cluster: {row['Cluster']}<br>"
            f"Dist to Cluster Center: {row['Dist_to_ClusterCenter']:.1f} km"
            if row["Cluster"] != -1 else
            f"{row['Plant_Name']}<br>Unclustered"
        )

        marker_color = color_map.get(row["PrimSource"], "gray")

        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=3,
            color=marker_color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)

    # Add color legend
    legend_html = """
    <div style='position: fixed; bottom: 30px; left: 30px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px;'>
    <b>Primary Source Legend</b><br>
    """ + "".join([
        f"<i style='background:{color};width:12px;height:12px;display:inline-block;'></i> {src}<br>"
        for src, color in color_map.items()
    ]) + "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(html_filename)
    print(f"\nMap saved to: {html_filename}")
else:
    print(f"\nMap already exists at: {html_filename}, skipping regeneration.")
