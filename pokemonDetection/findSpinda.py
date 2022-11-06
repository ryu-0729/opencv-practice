import sys
sys.path.append('./Detection.py')
from Detection import Detection


detection = Detection('./img/spinda_index.png', './img/spinda.png', './img/find_spinda.jpg')

spindaIndexImg, spindaImg = detection.twoImgRead()
minVal, maxVal, minLoc, maxLoc = detection.matchData(spindaIndexImg, spindaImg)

detection.rectangles(
  spindaIndexImg,
  maxLoc[0],
  maxLoc[1],
  spindaImg.shape[1],
  spindaImg.shape[0],
)
