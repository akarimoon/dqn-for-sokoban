import pyxel
pyxel.init(64, 64, caption='soko')
pyxel.load('soko2.pyxres')

def map2array(shape):
    tilearray = []
    for i in range(shape[0] // 8):
        temp = []
        for j in range(shape[1] // 8):
            if pyxel.tilemap(0).data[i][j] == 64:
                temp.append(64)
            else:
                temp.append(0)
        tilearray.append(temp)
    return tilearray

array = map2array(shape=(48, 48))
for i in range(len(array)):
    print(array[i])
