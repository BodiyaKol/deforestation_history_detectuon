import numpy as np
import pandas as pd
from pystac_client import Client
import odc.stac

bbox = [23.55, 48.50, 23.60, 48.55]   # ~5×5 км → 500×500 = 250к пікселів
date_range = "2024-06-01/2024-07-31"  # 2 місяці
MAX_IMAGES_PER_MONTH = 3              # 3 знімки на місяць = ~6 знімків

client = Client.open("https://earth-search.aws.element84.com/v1")

search = client.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=date_range,
    query={"eo:cloud_cover": {"lt": 50}}
)

items = list(search.items())

data_info = []
for item in items:
    date = item.datetime
    data_info.append({
        'item': item,
        'date': date,
        'month': date.month,
        'cloud_cover': item.properties['eo:cloud_cover']
    })

df = pd.DataFrame(data_info)
df = df.sort_values(by=['month', 'cloud_cover'])

best_df = df.groupby('month').head(MAX_IMAGES_PER_MONTH)
selected_items = best_df['item'].tolist()
selected_dates = [d.strftime("%Y-%m-%d") for d in best_df['date']]

dataset = odc.stac.load(
    selected_items,
    bands=["red", "nir"],
    bbox=bbox,
    resolution=10,
    crs="EPSG:3857"
)

red = dataset.red.astype(float)
nir = dataset.nir.astype(float)
ndvi = (nir - red) / (nir + red + 1e-8)

ndvi_array = ndvi.values
T, H, W = ndvi_array.shape 

X = ndvi_array.reshape(T, H * W).T

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

np.save("X.npy", X)
np.save("meta.npy", {
    "dates": selected_dates,
    "height": H,
    "width": W,
    "bbox": bbox
})
