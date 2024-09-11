from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Load the images
gt = Image.open("gt_3.png")
partial = Image.open("partial_3.png")
mask = Image.open("mask_3.png")

# Compute the differences where partial > 3
indices = np.array(mask) > 0
diffs = np.array(partial).astype(np.float32)[indices] - np.array(gt).astype(np.float32)[indices]
print((diffs / 127.5).std())
np.save("diffs.npy", diffs)

# Plot the histogram
plt.figure(figsize=(5, 3))
count, bins, ignored = plt.hist(diffs, bins=30, color='blue', edgecolor='black', alpha=0.75, density=True)

# Fit a Poisson distribution to the data
# mu = np.mean(diffs)
# x = np.arange(np.min(diffs), np.max(diffs)+1)
# poisson_fit = poisson.pmf(x, mu)

# Plot the Poisson distribution
# plt.plot(x, poisson_fit, 'r-', lw=2, label='Poisson fit')

# Add title and labels
plt.title('Histogram of Differences')
plt.xlabel('Difference')
plt.ylabel('Density')
# plt.legend()

# Save the plot to disk
plt.savefig('difference_histogram.png', dpi=300)

# Optionally, display the plot
plt.show()
