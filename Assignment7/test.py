import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# ========== 1. Load & Preprocess ==========
# Replace with your actual file path
df = pd.read_csv("Assignment7/Power_Plants.csv")

# Drop rows missing key fields
df = df.dropna(subset=["PrimSource", "Longitude", "Latitude"])
df.reset_index(drop=True, inplace=True)

# Fill NaNs in energy columns with 0
energy_cols = ["Install_MW", "Total_MW", "Solar_MW", "Wind_MW", 
               "Coal_MW", "NG_MW", "Nuclear_MW"]
df[energy_cols] = df[energy_cols].fillna(0)

# ========== 2. KMeans Clustering ==========

# Cluster by geographic location
geo_coords = df[["Longitude", "Latitude"]]
geo_kmeans = KMeans(n_clusters=10, random_state=42)
df["GeoCluster"] = geo_kmeans.fit_predict(geo_coords)

# Cluster by energy profile
X_energy = df[energy_cols]
energy_kmeans = KMeans(n_clusters=5, random_state=42)
df["EnergyCluster"] = energy_kmeans.fit_predict(X_energy)

# ========== 3. Interactive Map ==========
map_center = [39.5, -98.35]  # Center of USA
m = folium.Map(location=map_center, zoom_start=5)

color_map = {
    source: f'#{np.random.randint(0, 0xFFFFFF):06x}' for source in df["PrimSource"].unique()
}

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4,
        color=color_map[row["PrimSource"]],
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(f"""
            <b>Plant:</b> {row['Plant_Name']}<br>
            <b>Source:</b> {row['PrimSource']}<br>
            <b>Total MW:</b> {row['Total_MW']}<br>
            <b>City:</b> {row['City']}
        """, max_width=300),
    ).add_to(m)

m.save("../Assignment7/interactive_power_plants_map.html")
print("Saved interactive map to 'interactive_power_plants_map.html'")


# ========== 5. Predict PrimSource ==========
X = X_energy
y = df["PrimSource"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
