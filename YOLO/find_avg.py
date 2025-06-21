import pandas as pd

df = pd.read_csv("dataset/train/_annotations.csv")

print((df['xmax'] - df['xmin']).mean())
print((df['ymax'] - df['ymin']).mean())
