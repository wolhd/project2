# Step 1: interpolate direction (horizontal position) using ECEF on a unit sphere/ellipsoid
x1, y1, z1 = geodetic_to_ecef(lat1, lon1, 0)   # altitude=0, i.e. project to surface
x2, y2, z2 = geodetic_to_ecef(lat2, lon2, 0)

x = x1 + fraction * (x2 - x1)
y = y1 + fraction * (y2 - y1)
z = z1 + fraction * (z2 - z1)

lat, lon, _ = ecef_to_geodetic(x, y, z)   # discard the bogus altitude from this step

# Step 2: interpolate altitude independently, linearly in time
alt = alt1 + fraction * (alt2 - alt1)
#----------

# store lat lon time points and query by time window in pandas

import pandas as pd

# 1. Create a sample DataFrame
data = {
    "lat": [42.36, 42.37, 42.38],
    "lon": [-71.05, -71.06, -71.07],
    "time": ["2026-06-01 10:00:00", "2026-06-01 12:30:00", "2026-06-01 15:00:00"],
}
df = pd.DataFrame(data)

# 2. Convert the time column to datetime
df["time"] = pd.to_datetime(df["time"])

# 3. Define your start and end times
start_time = "2026-06-01 11:00:00"
end_time = "2026-06-01 16:00:00"

# 4. Query the DataFrame for the time window
filtered_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]

print(filtered_df)

-----

# convert epoch time to datetime

import pandas as pd

# Sample DataFrame with epoch timestamps in seconds
data = {'epoch_time': [1718600000, 1718686400, 1718772800]}
df = pd.DataFrame(data)

# 1. Convert to Datetime (Defaults to UTC timezone)
df['datetime_utc'] = pd.to_datetime(df['epoch_time'], unit='s')

# 2. Optional: Convert to a specific local timezone (e.g., US/Eastern)
df['datetime_local'] = df['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

print(df)

---
# calling main.py puts the directory of main.py in sys.path regardless of
# you current working directoy
# calling python -m dirA/libA.py puts current working dir in sys.path, not dirA

---
# read 3 files and sort

from datetime import datetime

# 1. Dummy data representing rows from three different files
file1_data = [{"timestamp": "2026-08-23T10:00:00Z", "val": 1}]
file2_data = [{"time": "2026-08-23T09:30:00Z", "val": 2}]
file3_data = [{"created_at": "2026-08-23T11:15:00Z", "val": 3}]

combined_data = []

# 2. Map and normalize each file's specific time field
for item in file1_data:
  combined_data.append({
      "time": datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
      "original": item,
  })

for item in file2_data:
  combined_data.append({
      "time": datetime.fromisoformat(item["time"].replace("Z", "+00:00")),
      "original": item,
  })

for item in file3_data:
  combined_data.append({
      "time": datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
      "original": item,
  })

# 3. Sort by the unified time key (oldest first)
combined_data.sort(key=lambda x: x["time"])

# 4. Send data to your API in order
for entry in combined_data:
  actual_data = entry["original"]
  # call_your_api(actual_data)
  print("Sending:", actual_data)
