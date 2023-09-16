import cv2
#easyocr - считывет инфу с картинки
#imutils-доступ к базовым функциям в работе с изображениями
#matpllotlib - выводит изображение на экран
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as pl
from PIL import Image

img = cv2.imread("images/plate_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filter, 30, 200)
#RETR_TREE-контуры в иерархичном порядке
#CHAIN_APPROX_SIMPLE-позволяет найти первую и конечную точки

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
#contourArea-метод сортировки, поиск контуров
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)

    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

y, x = np.where(mask == 255)
x1, y1 = (np.min(x), np.min(y))
x2, y2 = (np.max(x), np.max(y))
crop = gray[y1:y2, x1:x2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)

res = text[0][-2]
final_image = cv2.putText(img, res, (x2 + 10, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
final_image = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

#min-находит минимальный элемент
pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
pl.show()

