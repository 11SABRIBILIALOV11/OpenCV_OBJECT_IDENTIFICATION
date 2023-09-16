import cv2
import numpy as np

img = cv2.imread("images/Photo.JPEG")

# img = cv2.flip(img, -1)


def rotate(img_param, angle):
    height, width = img_param.shape[:2]
    point = (width // 2, height // 2)

    mat = cv2.getRotationMatrix2D(point, angle, 1)
    return cv2.warpAffine(img, mat, (width, height))


# img = rotate(img, 90)

def tranform(img_param, x, y):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))


img = tranform(img, 30, 200)

cv2.imshow("Result", img)

cv2.waitKey(0)

