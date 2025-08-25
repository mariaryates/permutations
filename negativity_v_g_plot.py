import sys
import csv
import matplotlib.pyplot as plt
import numpy as np 

ntls = int(sys.argv[1])
nphot = int(sys.argv[2])
gmin = float(sys.argv[3])
gmax = float(sys.argv[4])
num_points = int(sys.argv[5])

g_values = np.linspace(gmin, gmax, num_points)
summed_negativities = []

for g in g_values:
    file_path = f"data.tmp/negativity_pt_{ntls}_{nphot}_{g}.txt"
    print(file_path)
    total_neg = 0.0
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                print(row)
                negativity = float(row[1])
                print(negativity)
                total_neg += negativity
        summed_negativities.append(total_neg)

    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
        summed_negativities.append(0.0)

# Plot
plt.figure(figsize=(6,4))
plt.plot(g_values, summed_negativities, marker='o', linestyle='-')
plt.xlabel("λ")  # g is labeled as λ
plt.ylabel("Negativity")
plt.title(f"Negativity vs λ (ntls={ntls}, nphot={nphot})")
plt.grid(True)
plt.tight_layout()
plt.show()
