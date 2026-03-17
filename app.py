import pickle
import gradio as gr
import pandas as pd

MODEL_FILE = "model.pkl"
BFS_FILE = "data/bfs_municipality_and_tax_data.csv"

FEATURES = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income"]

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bfs_df = pd.read_csv(BFS_FILE)
bfs_df["tax_income"] = (
    bfs_df["tax_income"].astype(str).str.replace("'", "", regex=False).astype(float)
)

towns = sorted(bfs_df["bfs_name"].dropna().unique().tolist())


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