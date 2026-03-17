import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# --- Load necessary data and files ---
BFS_FILE = "data/bfs_municipality_and_tax_data.csv"
APT_FILE = "data/original_apartment_data_analytics_hs24_with_lat_lon.csv"
CENTROIDS_FILE = "data/municipality_centroids.csv"

# --- Define features and price ---
FEATURES = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income", "distance_to_center_km"]
TARGET = "price"

ZURICH_CENTER_LAT = 47.3780
ZURICH_CENTER_LON = 8.5402

#--- calculate distance to center ---
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))

# --- Train model at startup ---
apt_df = pd.read_csv(APT_FILE).dropna().drop_duplicates()
apt_df["distance_to_center_km"] = haversine_km(
    apt_df["lat"], apt_df["lon"],
    ZURICH_CENTER_LAT, ZURICH_CENTER_LON,
)
X = apt_df[FEATURES]
y = apt_df[TARGET]

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X, y)

# --- Load BFS + municipality centroids ---
bfs_df = pd.read_csv(BFS_FILE)
bfs_df["tax_income"] = (
    bfs_df["tax_income"].astype(str).str.replace("'", "", regex=False).astype(float)
)
centroids = pd.read_csv(CENTROIDS_FILE)
bfs_df = bfs_df.merge(centroids, on="bfs_name", how="left")
bfs_df["distance_to_center_km"] = haversine_km(
    bfs_df["lat"], bfs_df["lon"],
    ZURICH_CENTER_LAT, ZURICH_CENTER_LON,
)

towns = sorted(bfs_df["bfs_name"].dropna().unique().tolist())

# --- predict target price ---
def predict_rent(rooms, area, town):
    row = bfs_df[bfs_df["bfs_name"] == town].copy()

    if row.empty:
        return "Municipality not found."

    row = row.iloc[[0]].copy()
    row["rooms"] = rooms
    row["area"] = area

    X_input = row[FEATURES]
    prediction = model.predict(X_input)[0]

    return f"Predicted monthly rent: CHF {prediction:,.0f}"

# --- GUI ---
demo = gr.Interface(
    fn=predict_rent,
    inputs=[
        gr.Number(label="Rooms", value=3.5),
        gr.Number(label="Area (m²)", value=75),
        gr.Dropdown(choices=towns, label="Municipality", value="Zürich"),
    ],
    outputs="text",
    title="Apartment Rent Prediction – Canton of Zurich",
    description="Predict apartment rent using apartment features and municipality statistics.",
)

demo.launch()