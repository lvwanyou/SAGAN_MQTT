# import matplotlib.pyplot as plt
#
# import numpy as np
#
# # a=np.linspace(-np.pi,np.pi,100)
# # b=np.sin(a)
# # c=np.cos(a)
# a = [0,50]
# b = [1,100]
#
# s=plt.plot(a,b)
#
# plt.show()
# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
# names = ['1E', '2E', '3E', '4E', '5E']
# x = range(len(names))
# y = ['30%', '40%', '20%', '50%', '10%']
# plt.plot(x, y, 'ro-')
# plt.xticks(x, names, rotation=45)
# plt.margins(0.08)
# plt.subplots_adjust(bottom=0.15)
# plt.show()

from matplotlib.pyplot import plot,savefig
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
#mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
x = range(len(names))

y1 = [0.49, 0.52, 0.62, 0.75, 0.81, 0.79, 0.83, 0.86, 0.84, 0.85]
y2 = [0.51, 0.55, 0.56, 0.65, 0.76, 0.83, 0.77, 0.73, 0.76, 0.74]
y3 = [0.64, 0.68, 0.57, 0.63, 0.59, 0.60, 0.57, 0.55, 0.64, 0.58]
y4 = [0.61, 0.58, 0.50, 0.53, 0.52, 0.47, 0.50, 0.57, 0.547, 0.55]
# y5 = [0.57, 0.43, 0.35, 0.30, 0.21, 0.19, 0.18, 0.17, 0.17, 0.165]
#y1 = [0.15, 0.22, 0.40, 0.45, 0.47, 0.45, 0.50, 0.52]
# y2 = [0.77,0.66,0.555,0.33,0.22]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(-0.5, 9.5)  # 限定纵轴的范围
plt.ylim(0.10, 1)  # 限定横轴的范围
plt.xticks(size=13)
plt.yticks(size=13)


#color 线条颜色 ms 点的大小，mec 点的外色，mfc点的内色，marker点的形式，
# plt.plot(x, y1, marker='s', mec='b', mfc='lightblue',ms='6',label='WGAN-based Method',linewidth=1,color='lightblue')
plt.plot(x, y1, marker='s', mec='b', mfc='lightblue',ms='6',label='HexGANFuzzer',linewidth=1,)
plt.plot(x, y2, marker='o',label='WGAN-based model', linewidth=1)
plt.plot(x, y3, marker='v',label='LSTM-based model', linewidth=1)
plt.plot(x, y4, marker='^',label='CNN-1D', linewidth=1)
# plt.plot(x, y5, marker='.',label='GPF', linewidth=1)


# plt.plot(x, y2, marker='*', ms=10,label=u'another line')

plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Training Epochs",size=13) #X轴标签
plt.ylabel("EFVD",size=13) #Y轴标签
#plt.title("WGAN vs GAN on TIAR") #标题
plt.grid(True,linestyle='-.')
# plt.plot(y[:,0], 'ro')
# plt.axis('tight')x


def to_percent(temp, position):
    return '%2.1f' % (100 * temp) + '%'
savefig("C:\\Users\\11442\\Desktop\\2.jpg")
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.savefig("TCRR.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()
