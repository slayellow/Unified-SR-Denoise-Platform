import cv2
import numpy as np

H, W = 4, 4
bayer = np.zeros((H, W), dtype=np.uint16)
bayer[0::2, 0::2] = 65535 # We put 65535 at [0,0]

print("--- Testing Bayer flags to RGB ---")
rgb1 = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
print("BayerRG2RGB ->", rgb1[0,0])

rgb2 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
print("BayerBG2RGB ->", rgb2[0,0])
