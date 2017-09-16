def getSicknessLvl(mmseScore):
    mmseScore = int(float(mmseScore))

    lvl = 0
    label = '0-10'

    if mmseScore > 10 and mmseScore <= 18:
        lvl = 1
        label = '11-18'
    if mmseScore > 18 and mmseScore <= 23:
        lvl = 2
        label = '19-23'
    if mmseScore > 23 and mmseScore <= 26:
        lvl = 3
        label = '24-26'
    if mmseScore > 26:
        lvl = 4
        label = '26-30'

    return {'level': lvl, 'label': label}
