import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# https://atlas.eia.gov/datasets/eia::power-plants/
df = pd.read_csv("../Assignment7/Power_Plants.csv")  # Replace with your actual file
# Drop rows with missing coordinates or primary source
df = df.dropna(subset=["PrimSource", "Longitude", "Latitude"])

# Extract coordinates
coords = df[["Longitude", "Latitude"]]

# Fit KMeans
n_clusters = len(df["PrimSource"].unique())  # One cluster per primary source
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(coords)

# Plot
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Color by PrimSource
sources = df["PrimSource"].unique()
colors = plt.cm.get_cmap('tab20', len(sources))

for i, source in enumerate(sources):
    sub_df = df[df["PrimSource"] == source]
    ax.scatter(
        sub_df["Longitude"], sub_df["Latitude"],
        color=colors(i),
        label=source,
        s=10,
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )

plt.legend(title="Primary Source", loc="lower left")
plt.title("Power Plants Clustered by Primary Source")
plt.show()
