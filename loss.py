import os
import matplotlib.pylab as pyl
file = open("loss.txt")
loss = []
iter = []
i = 0
loss.append(6.3)
iter.append(0)
while True:
    text = file.readline()  # 只读取一行内容

    # 判断是否读取到内容
    if not text:
        break
    if text[0] == 'E':
        i = i+1
        list = text.split(' ')
        loss.append(float(list[3]))
        iter.append(i)

file.close()
pyl.plot(iter, loss)
pyl.show()


