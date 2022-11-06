import sys
sys.path.append('./customCv/CustomCommon.py')
import customCv.CustomCv as customCv

custom = customCv.CustomCv('./sample-face.png', './face.jpg', './grayFaceImg.jpg')

faceImg = custom.imgRead()
grayFaceImg = custom.grayScale(faceImg)
faceData = custom.faceDetection(grayFaceImg)
custom.multipleRectangles(faceImg, faceData)
