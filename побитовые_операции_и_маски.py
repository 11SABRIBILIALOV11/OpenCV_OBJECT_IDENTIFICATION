#операции по объединению изображений и способу обЪединений
import cv2
import numpy

photo = cv2.imread("images/Photo.JPEG")
img = numpy.zeros(photo.shape[:2], dtype='uint8')

circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
square = cv2.rectangle(img.copy(), (25, 25), (250, 150), 255, -1)

img = cv2.bitwise_and(photo, photo, mask=square)

#bitwise_and-находит общие части и черты у изображений
#bitwise_or-объединяет все изображения
#bitwise_xor-вырезает общие части изображений
#bitwise_not-инверсия изображения, вырезает изображение, но оствляет все вокруг


#img = cv2.bitwise_or(circle, square)

#img = cv2.bitwise_not(circle, square)

#img = cv2.bitwise_xor(circle, square)

cv2.imshow("Result", img)

cv2.waitKey(0)

