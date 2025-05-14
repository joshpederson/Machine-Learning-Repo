import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_geojson(country_code):
    """Load GeoJSON data using a country code."""
    json_path = f"countries/{country_code}.geo.json"
    with open(json_path, 'r') as json_file:
        return json.load(json_file)  # Ensure correct JSON loading


def extract_coordinates(geojson_data):
    """Extracts coordinates from GeoJSON features."""
    pts = []
    for feature in geojson_data['features']:
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            pts.extend(geometry['coordinates'][0])
            pts.append([None, None])  # Separate polygons
        elif geometry.get('type') == 'MultiPolygon':
            for polyg in geometry['coordinates']:
                pts.extend(polyg[0])
                pts.append([None, None])  # Separate multiple polygons
    if pts:
        x, y = zip(*pts)
        return x, y
    return [], []  # Return empty lists if no coordinates are found


def plot_country_map(country_code, country_name, df):
    """Generates an interactive map for power plants in a given country."""
    geojson_data = load_geojson(country_code)
    x, y = extract_coordinates(geojson_data)

    df_country = df[df["country_long"] == country_name]  # Ensure column name is correct

    if df_country.empty:
        print(f"No data available for {country_name}. Check your dataset.")
        return

    resources_contribution = df_country.groupby('primary_fuel')['capacity in MW'].sum().to_frame()
    resources_contribution['capacity in GW'] = resources_contribution['capacity in MW'] / 1e3

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'domain'}]])

    # Plot country boundary
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='#999999', legendgroup='2', line_width=1), row=1, col=1)

    # Plot power plants
    for resource in df_country['primary_fuel'].unique():
        df_resource = df_country[df_country['primary_fuel'] == resource]
        fig.add_trace(go.Scatter(x=df_resource['longitude'],
                                 y=df_resource['latitude'],
                                 hovertext=df_resource['name of powerplant'],
                                 name=resource,
                                 mode='markers',
                                 marker=dict(size=6)),  # Adjust marker size
                      row=1, col=1)

    # Plot resource contribution pie chart
    fig.add_trace(go.Pie(labels=resources_contribution.index,
                         values=resources_contribution['capacity in GW'],
                         title=country_name.upper(),
                         showlegend=True,
                         textinfo='percent+label',
                         hole=.4),  # Adjust hole size
                  row=1, col=2)

    fig.update_layout(title=f'{country_name} Power Plants Map')
    fig.show()

# Example usage:
# df = pd.read_csv('power_plants.csv')  # Ensure your dataset is loaded
# plot_country_map('EG', 'Egypt', df)
