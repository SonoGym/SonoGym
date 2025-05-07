import cv2
import numpy as np

img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()