from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

SIZE = 512
START = [0, 350]

# Load the images
projected = np.array(Image.open("render_dense/vivek/project_h512/seq1-000269_rot40.png"))
mask = np.array(Image.open("render_dense/vivek/mask_h512/seq1-000269_rot40.png"))
gt = np.array(Image.open("render_dense/vivek/gt_h512/seq1-000269_rot40.png"))

target_slice = np.s_[START[0]:START[0] + SIZE, START[1]:START[1] + SIZE]
projected = projected[target_slice]
mask = mask[target_slice]
gt = gt[target_slice]

Image.fromarray(projected).resize((256, 256), Image.NEAREST).save("partial_5.png")
Image.fromarray(mask).resize((256, 256), Image.NEAREST).save("mask_5.png")
Image.fromarray(gt).resize((256, 256), Image.NEAREST).save("gt_5.png")