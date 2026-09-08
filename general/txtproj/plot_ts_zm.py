"""
Geospatial plot with a time slider, using Plotly.

Plotly Express's `animation_frame` argument automatically builds a slider
(plus play/pause buttons) that steps through your time dimension. This
script demonstrates it with synthetic data for a handful of cities whose
"value" (e.g. temperature, sales, cases...) changes over 12 months.

Swap in your own DataFrame with columns: lat, lon, time, value, (label)
and it will work the same way.
"""

import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# 1. Build (or load) your data.
#    Replace this block with pd.read_csv(...) or however you get real data.
#    Required shape: one row per (location, time) combination.
# ---------------------------------------------------------------------------
cities = pd.DataFrame({
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Miami", "Seattle", "Denver", "Boston", "Atlanta"],
    "lat":  [40.71, 34.05, 41.88, 29.76, 33.45, 25.76, 47.61, 39.74, 42.36, 33.75],
    "lon":  [-74.01, -118.24, -87.63, -95.37, -112.07, -80.19, -122.33, -104.99, -71.06, -84.39],
})

months = pd.date_range("2025-01-01", periods=12, freq="MS")

rng = np.random.default_rng(42)
rows = []
for _, city in cities.iterrows():
    base = rng.uniform(20, 100)
    for i, month in enumerate(months):
        # a smooth-ish synthetic trend + noise, just for demonstration
        value = base + 15 * np.sin(i / 12 * 2 * np.pi) + rng.normal(0, 5)
        rows.append({
            "city": city["city"],
            "lat": city["lat"],
            "lon": city["lon"],
            "month": month.strftime("%Y-%m"),
            "value": round(value, 1),
        })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. Plot: scatter_geo with animation_frame -> gives you the time slider.
# ---------------------------------------------------------------------------
fig = px.scatter_geo(
    df,
    lat="lat",
    lon="lon",
    color="value",
    size="value",
    hover_name="city",
    animation_frame="month",          # <- this is what creates the slider
    scope="usa",                      # remove/change for a world map
    color_continuous_scale="Viridis",
    size_max=40,
    title="Example metric by city over time",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=60, b=0),
    coloraxis_colorbar=dict(title="Value"),
)

# Slightly slower animation so it's easier to watch
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

# ---------------------------------------------------------------------------
# 3. Save as a standalone interactive HTML file (opens in any browser,
#    no Python needed to view it) and also try to display if run in a
#    notebook / interactive environment.
# ---------------------------------------------------------------------------
output_path = "geo_time_slider.html"
fig.write_html(output_path, auto_play=False)
print(f"Saved interactive plot to {output_path}")

if __name__ == "__main__":
    # Uncomment if running locally with a GUI/browser available:
    # fig.show()
    pass
