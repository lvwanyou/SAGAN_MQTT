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

y1 = [0.12, 0.33, 0.51, 0.74, 0.85, 0.89, 0.90, 0.95, 0.90, 0.92]
y2 = [0.14, 0.37, 0.57, 0.65, 0.72, 0.83, 0.80, 0.83, 0.86, 0.84]
y3 = [0.20, 0.39, 0.57, 0.63, 0.59, 0.60, 0.57, 0.65, 0.64, 0.68]
y4 = [0.18, 0.47, 0.50, 0.53, 0.52, 0.57, 0.60, 0.58, 0.63, 0.61]
y5 = [0.57, 0.43, 0.35, 0.30, 0.21, 0.19, 0.18, 0.17, 0.17, 0.165]
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
plt.plot(x, y1, marker='s',markersize=6.5, mec='b', mfc='lightblue',ms='6',label='HexGANFuzzer',linewidth=2)
plt.plot(x, y2, marker='o',markersize=5,label='WGAN-based model', linewidth=0.8)
plt.plot(x, y3, marker='v',markersize=5,label='LSTM-based model', linewidth=0.8)
plt.plot(x, y4, marker='^',markersize=5,label='CNN-1D', linewidth=0.8)
plt.plot(x, y5, marker='.',markersize=5,label='GPF', linewidth=0.8)


# plt.plot(x, y2, marker='*', ms=10,label=u'another line')

plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Training Epochs",size=13) #X轴标签
plt.ylabel("EVD",size=13) #Y轴标签
#plt.title("WGAN vs GAN on TIAR") #标题
plt.grid(True,linestyle='-.')
# plt.plot(y[:,0], 'ro')
# plt.axis('tight')


def to_percent(temp, position):
    return '%2.1f' % (100 * temp) + '%'
savefig("C:\\Users\\11442\\Desktop\\EVD.jpg")
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.savefig("TCRR.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()
