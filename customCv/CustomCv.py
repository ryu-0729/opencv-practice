from dataclasses import dataclass
import cv2


@dataclass
class CustomCv:
    __filePath: str
    __downloadPath: str
    __grayImgPath: str = ''

    @property
    def filePath(self):
        return self.__filePath

    @property
    def downloadPath(self):
        return self.__downloadPath

    @property
    def grayImgPath(self):
        return self.__grayImgPath

    def imgRead(self):
        """ 画像の読み込み """
        return cv2.imread(self.__filePath)

    def grayScale(self, img):
        """ 白と黒でグレースケールした画像を返却 """
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.__grayImgPath:
            cv2.imwrite(self.__grayImgPath, grayImg)
        return grayImg

    def imgDownload(self, img):
        """ 指定したパスに画像を出力 """
        return cv2.imwrite(self.__downloadPath, img)

    def faceDetection(self, grayImg):
        """ 顔検出処理 """
        faceCascade = cv2.CascadeClassifier(
            '/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        )
        return faceCascade.detectMultiScale(grayImg)

    def multipleRectangles(self, rectanglesImg, imgData):
        """ 複数の対象を四角で囲み出力する処理 """
        for (x, y, w, h) in imgData:
            rectanglesImg = cv2.rectangle(rectanglesImg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return cv2.imwrite(self.__downloadPath, rectanglesImg)
