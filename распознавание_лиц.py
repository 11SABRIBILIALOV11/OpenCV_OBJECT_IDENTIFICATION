import cv2

img = cv2.imread('images/people_3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces.xml')
#метод вытягивает файл как натренированную модель


results = faces.detectMultiScale(gray, scaleFactor=2.2, minNeighbors=4)
#метод находит координаты всех найденных объектов(лица)

for(x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

cv2.imshow("Result", img)
cv2.waitKey(0)



