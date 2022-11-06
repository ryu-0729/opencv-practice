import sys
sys.path.append('./common/CustomCommon.py')
import common.CustomCommon as customCommon

custom = customCommon.CustomCommon('./sample-face.png', './face.jpg', './grayFaceImg.jpg')

faceImg = custom.imgRead()
grayFaceImg = custom.grayScale(faceImg)
faceData = custom.faceDetection(grayFaceImg)
custom.multipleRectangles(faceImg, faceData)
