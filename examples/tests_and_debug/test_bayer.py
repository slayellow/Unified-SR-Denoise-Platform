import cv2
import numpy as np

# Create a small Bayer RG image
H, W = 4, 4
bayer = np.zeros((H, W), dtype=np.uint16)
# BayerRG means:
# R G
# G B
# Let's set R=65535, B=0
bayer[0::2, 0::2] = 65535

# Convert with BayerRG2RGB
rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)

print("RGB [0,0] (should be pure red):", rgb[0,0])

# What if we use BayerRG2BGR?
bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
print("BGR [0,0] (should be pure red in BGR, meaning [0,0,65535]):", bgr[0,0])
