from dataclasses import dataclass
import cv2


@dataclass
class Detection:
    __detectionPath: str
    __templatePath: str
    __downloadPath: str
    
    def twoImgRead(self):
        """ 検索元とテンプレート画像の読み込み """
        detectionImg = cv2.imread(self.__detectionPath)
        templatePath = cv2.imread(self.__templatePath)
        return (detectionImg, templatePath)

    def matchData(self, detectionImg, templateImg):
        """ 画像の類似度と類似度のデータを取得する処理 """
        matchRes = cv2.matchTemplate(detectionImg, templateImg, cv2.TM_CCOEFF_NORMED)
        return cv2.minMaxLoc(matchRes)

    def rectangles(self, img, x, y, w, h):
        """ 対象を四角で囲み出力する処理 """
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
        return cv2.imwrite(self.__downloadPath, img)
