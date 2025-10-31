import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("logs/detections.csv")
if df.empty:
    print("No logs yet.")
else:
    counts = df['detected_object'].value_counts()
    counts.plot(kind='bar')
    plt.title("Detected Object Counts")
    plt.xlabel("Object Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("logs/detection_summary.png")
    plt.show()
