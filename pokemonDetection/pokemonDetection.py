import sys
sys.path.append('./Detection.py')
from Detection import Detection

detection = Detection('./pokedex.png', './snorlax.png', './pokemon-detection.jpg')

pokedexImg, templetePokemonImg = detection.twoImgRead()
minVal, maxVal, minLoc, maxLoc = detection.matchData(pokedexImg, templetePokemonImg)

detection.rectangles(
  pokedexImg,
  maxLoc[0],
  maxLoc[1],
  templetePokemonImg.shape[1],
  templetePokemonImg.shape[0],
)
