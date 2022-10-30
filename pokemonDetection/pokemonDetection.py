import cv2

# NOTE: 検索する画像と検索対象の画像の読み込み
pokedexImg = cv2.imread('./pokedex.png')
templetePokemonImg = cv2.imread('./snorlax.png')

# NOTE: 画像の類似度を取得
matchRes = cv2.matchTemplate(pokedexImg, templetePokemonImg, cv2.TM_CCOEFF_NORMED)

# NOTE: 類似度のデータを取得
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchRes)

# NOTE: 検出した箇所の右下座標を計算(x + w, y + h)
rightLower = maxLoc[0] + templetePokemonImg.shape[1], maxLoc[1] + templetePokemonImg.shape[0]

# NOTE: 対象のポケモンを四角で囲む
findPokemon = cv2.rectangle(pokedexImg, maxLoc, rightLower, (0, 0, 255), 10)

cv2.imwrite('./pokemon-detection.jpg', findPokemon)
