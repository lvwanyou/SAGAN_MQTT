#!/usr/bin/python
# coding=gbk
import matplotlib.pyplot as plt
from pylab import *

class model_EFVD:
    TT = []
    TAT = []
    EFVD = []

    def __init__(self):
        self.TT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.TAT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.EFVD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# model init
GPF = model_EFVD()
CNN_1D = model_EFVD()
LSTM = model_EFVD()
WGAN = model_EFVD()
SAGAN = model_EFVD()


# model para_setting
GPF.TT = [50	,50,	50,	50,	50,	50,	50,	50,	50,	50]
GPF.TAT = [180,	180,	180,	180,	180,	180,	180,	180,	180,	180]

CNN_1D.TT = [5,	10,	15,	20,	25,	30,	35,	40,	45,	50]
CNN_1D.TAT = [250,	240,	220,	210,	170,	160,	140,	130,	110,	105]

LSTM.TT = [7,	12,	18,	25,	34,	40,	47,	59,	68,	80]
LSTM.TAT = [257,	240,	220,	190,	165,	152,	140,	120,	100,	90]

WGAN.TT = [10,	20,	29,	38,	49,	57,	70,	80,	90,	100]
WGAN.TAT = [257,	241,	200,	170,	150,	130,	120,	110,	95,	87]

SAGAN.TT = [12,	21,	27,	35,	45,	52,	60,	67,	74,	83]
SAGAN.TAT = [251,	242,	197,	160,	140,	120,	110,	100,	92,	80]
for i in range(10):
    TT_min = min(GPF.TT[i], CNN_1D.TT[i], LSTM.TT[i], WGAN.TT[i], SAGAN.TT[i])
    TT_max = max(GPF.TT[i], CNN_1D.TT[i], LSTM.TT[i], WGAN.TT[i], SAGAN.TT[i])
    TAT_min = min(GPF.TAT[i], CNN_1D.TAT[i], LSTM.TAT[i], WGAN.TAT[i], SAGAN.TAT[i])
    TAT_max = max(GPF.TAT[i], CNN_1D.TAT[i], LSTM.TAT[i], WGAN.TAT[i], SAGAN.TAT[i])

    GPF.EFVD[i] = ((GPF.TT[i] - TT_min) / (TT_max - TT_min)) + ((GPF.TAT[i] - TAT_min) / (TAT_max - TAT_min))
    CNN_1D.EFVD[i] = ((CNN_1D.TT[i] - TT_min) / (TT_max - TT_min)) + ((CNN_1D.TAT[i] - TAT_min) / (TAT_max - TAT_min))
    LSTM.EFVD[i] = ((LSTM.TT[i] - TT_min) / (TT_max - TT_min)) + ((LSTM.TAT[i] - TAT_min) / (TAT_max - TAT_min))
    WGAN.EFVD[i] = ((WGAN.TT[i] - TT_min) / (TT_max - TT_min)) + ((WGAN.TAT[i] - TAT_min) / (TAT_max - TAT_min))
    SAGAN.EFVD[i] = ((SAGAN.TT[i] - TT_min) / (TT_max - TT_min)) + ((SAGAN.TAT[i] - TAT_min) / (TAT_max - TAT_min))


print("GPF EFVD : " + str(GPF.EFVD))
print("CNN EFVD : " + str(CNN_1D.EFVD))
print("LSTM EFVD : " + str(LSTM.EFVD))
print("WGAN EFVD : " + str(WGAN.EFVD))
print("SAGAN EFVD : " + str(SAGAN.EFVD))


############   Painting start ##################
names = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
x = range(len(names))


plt.xlim(-0.5, 9.5)  # 限定纵轴的范围
plt.ylim(0.10, 5)  # 限定横轴的范围
plt.xticks(size=13)
plt.yticks(size=13)

#color 线条颜色 ms 点的大小，mec 点的外色，mfc点的内色，marker点的形式，
# plt.plot(x, y1, marker='s', mec='b', mfc='lightblue',ms='6',label='WGAN-based Method',linewidth=1,color='lightblue')
plt.plot(x, SAGAN.EFVD, marker='s', mec='b', mfc='lightblue',ms='6',label='BLSTM-DCNNFuzz',linewidth=1,)
plt.plot(x, WGAN.EFVD, marker='o',label='WGAN-based model', linewidth=1)
plt.plot(x, LSTM.EFVD, marker='v',label='LSTM-based model', linewidth=1)
plt.plot(x, CNN_1D.EFVD, marker='^',label='CNN-1D model', linewidth=1)
plt.plot(x, GPF.EFVD, marker='.',label='GPF', linewidth=1)
# plt.plot(x, y2, marker='*', ms=10,label=u'another line')

plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Training Epochs",size=13) #X轴标签
plt.ylabel("TCRR",size=13) #Y轴标签
#plt.title("WGAN vs GAN on TIAR") #标题
plt.grid(True,linestyle='-.')
# plt.plot(y[:,0], 'ro')
# plt.axis('tight')


def to_percent(temp, position):
    return '%2.1f' % (100 * temp) + '%'

plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.savefig("TCRR.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()