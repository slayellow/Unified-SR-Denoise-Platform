import cv2
import numpy as np

H, W = 4, 4
bayer = np.zeros((H, W), dtype=np.uint16)
bayer[0::2, 0::2] = 65535 # We put 65535 at [0,0]

print("--- Testing Bayer flags to find which one outputs native BGR (Red at index 2) ---")
bgr1 = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
print("BayerBG2BGR ->", bgr1[0,0])

bgr2 = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)
print("BayerGB2BGR ->", bgr2[0,0])

bgr3 = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
print("BayerRG2BGR ->", bgr3[0,0])

bgr4 = cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)
print("BayerGR2BGR ->", bgr4[0,0])
