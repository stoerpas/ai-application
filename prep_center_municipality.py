import pandas as pd

apt_df = pd.read_csv("data/original_apartment_data_analytics_hs24_with_lat_lon.csv")
centroids = apt_df.groupby("bfs_name")[["lat", "lon"]].mean().reset_index()
centroids.to_csv("data/municipality_centroids.csv", index=False)