import sys
sys.path.append('./Detection.py')
from Detection import Detection


detection = Detection(
    './img/spinda_index.png',
    './img/spinda.png',
    './img/find_spinda02.jpg',
    0.7 # NOTE: デフォルト値では閾値を超えるものがない。。
)

# TODO: 2値化処理を行なった場合を試してみたい。
spindaIndexImg, spindaImg = detection.twoImgRead()
matchRes = detection.matchData(spindaIndexImg, spindaImg, False)
detection.multipleRectangles(
    spindaIndexImg,
    matchRes,
    spindaImg.shape[1],
    spindaImg.shape[0],
)
