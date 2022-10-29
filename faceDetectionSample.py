import cv2

# NOTE: 顔検出したい画像の読み込み
faceImg = cv2.imread('./sample-face.png')

# NOTE: 白と黒のグレースケール
grayFaceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./grayFaceImg.jpg', grayFaceImg)

# NOTE: 顔検出処理
faceCascade = cv2.CascadeClassifier(
  '/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
)
faceData = faceCascade.detectMultiScale(grayFaceImg)

# NOTE: 顔を四角で囲む処理
for (x, y, w, h) in faceData:
  faceImg = cv2.rectangle(faceImg, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imwrite('./face.jpg', faceImg)
