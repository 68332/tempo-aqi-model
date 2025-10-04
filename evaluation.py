import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


# 污染物與對應的 history 檔案
pollutants = {
    "CO": "models/co_lstm_history.npy",
    "NO2": "models/no2_lstm_history.npy",
    "SO2": "models/so2_lstm_history.npy",
    "PM2.5": "models/pm25_lstm_history.npy",
    "PM10": "models/pm10_lstm_history.npy"
}

# 想顯示的指標
metrics = ["loss", "mae", "rmse"]

n_pollutants = len(pollutants)
fig, axes = plt.subplots(
    nrows=len(metrics), ncols=n_pollutants,
    figsize=(4*n_pollutants, 4*len(metrics))
)

for row, metric in enumerate(metrics):
    for col, (name, path) in enumerate(pollutants.items()):
        ax = axes[row, col] if len(metrics) > 1 else axes[col]
        history = np.load(path, allow_pickle=True).item()
        print(f"{name} history keys: {list(history.keys())}")

        if metric in history:
            ax.plot(history[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], label="val", linestyle="--")

        ax.set_title(f"{name} - {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

plt.subplots_adjust(hspace=1) 
plt.show()