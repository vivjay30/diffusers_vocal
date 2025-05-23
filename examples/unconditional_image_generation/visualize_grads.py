import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the directory containing the JSON files
directory = '.'
algorithm = "Algorithm 3 with 4x superres"

# Initialize a dictionary to aggregate values based on x
data = defaultdict(list)

# Iterate through each file in the directory
for filename in os.listdir(directory):
    seen_x = set()
    if filename.endswith('grads.json'):
        with open(os.path.join(directory, filename), 'r') as f:
            content = json.load(f)
            for x, y in content:
                data[x].append(y)
                # if x not in seen_x:
                #     data[x].append(y)
                #     seen_x.add(x)

# Convert the aggregated data to lists of x, y, mean, and std
x_values = sorted(data.keys())
y_values = [data[x] for x in x_values]
means = [np.sqrt(np.mean(np.array(y)**2)) for y in y_values]
stds = [np.std(y) for y in y_values]
print({x_values[i]: means[i] for i in range(len(x_values))})

# Scatter plot of all values
plt.figure(figsize=(10, 6))
for x, y in zip(x_values, y_values):
    plt.scatter([x] * len(y), y, alpha=0.5)
plt.xlabel('t')
plt.ylabel('eta / sigma^2 *||grad||')
# plt.ylim(0, 10)
plt.title(f'{algorithm}, All Values')
plt.savefig('scatter_all_values.png')

# Scatter plot of mean values
plt.figure(figsize=(10, 6))
plt.scatter(x_values, means, color='blue')
plt.xlabel('t')
plt.ylabel('||grad||')
# plt.ylim(0, 10)
plt.title(f'{algorithm}, Mean Values')
plt.savefig('scatter_mean_values.png')

# Scatter plot of mean values with 1 std deviation
plt.figure(figsize=(10, 6))
plt.errorbar(x_values, means, yerr=stds, fmt='o', color='blue', ecolor='red', capsize=5)
plt.xlabel('t')
plt.ylabel('||grad||')
plt.yscale('log')
# plt.ylim(0, 10)
plt.title(f'{algorithm}, Mean Values with 1 Std Deviation')
plt.savefig('scatter_mean_std_values.png')
