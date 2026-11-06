import torch
import pandas as pd
from pathlib import Path


pt_path = Path("/home/guest/Desktop/lxy/MyModel/data_provider/Dataset/physionet/processed/set-c_0.0.pt")
out_path = Path("/home/guest/Desktop/lxy/MyModel/data_provider/Dataset/physionet/processed/set-c.csv")

data = torch.load(pt_path, map_location="cpu")

rows = []

split_name = pt_path.stem.replace("_0.0", "")

for idx, item in enumerate(data):
    sample_id = item[0]
    times = item[1].detach().cpu()
    values = item[2].detach().cpu()
    mask = item[3].detach().cpu()

    # values: [T, D]
    T, D = values.shape

    for t in range(T):
        time_value = times[t].item()

        for d in range(D):
            m = mask[t, d].item()

            rows.append({
                "sample_id": sample_id,
                "time": time_value,
                "variable": d,
                "value": values[t, d].item(),
                "mask": m,
                "split": split_name,
            })

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("保存完成:", out_path)
print("CSV shape:", df.shape)
print(df.head())