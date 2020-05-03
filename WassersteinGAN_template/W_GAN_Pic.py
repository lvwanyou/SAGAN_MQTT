import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
from load_normal_data import DataLoad

sess = tf.InteractiveSession()#互动session 可以在运算图的时候插入一些运算

mb_size = 128 # batch的大小
Z_dim = 100   #所取噪声的列


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#读取mnist数据 分为 训练集，测试集，放入固定的类对象中 图像尺寸为 784列 的一行 0-1的数据（原来的每个数字乘以 1/255）


def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())# w 矩阵生成函数


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))# b矩阵生成函数


# discriminater net
#discriminater net 的参数

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')#shape中的none意味着可以是任意的维度 即：任意行 784列

D_W1 = weight_var([784, 128], 'D_W1')#调用weight_var 函数生成 变量，其中有行数，列数，784行 128列 。也有名字
D_b1 = bias_var([128], 'D_b1')#同上 128维的向量

D_W2 = weight_var([128, 1], 'D_W2')#同上 128维，1列的向量 是个列向量
D_b2 = bias_var([1], 'D_b2')#同上是个数字


theta_D = [D_W1, D_W2, D_b1, D_b2]#参数的列表


# generator net
# net 的参数
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')#shape中的none意味着可以是任意的维度 即：任意行 100列

G_W1 = weight_var([100, 128], 'G_W1')#调用weight_var 函数生成 变量，其中有行数，列数，100行 128列 。也有名字
G_b1 = bias_var([128], 'G_B1')#同上 128维的向量

G_W2 = weight_var([128, 784], 'G_W2')#同上 128行，784列的矩阵
G_b2 = bias_var([784], 'G_B2')#同上 784维，1列的向量 是个列向量

theta_G = [G_W1, G_W2, G_b1, G_b2]#参数的列表


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)#tf.matmul()是矩阵乘法，tf.multiply()是点乘法   G_h1是 x行128列  G_W2 是 128行 784列
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2#      一层矩阵相乘                                           G_log_prob是x行784列
    G_prob = tf.nn.sigmoid(G_log_prob)#              sigmoid激活函数                                         G_prob还是 x行784列

    return G_prob#                                                                          G_prob还是 x行784列


def discriminator(x):#x行784列
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)# x，x行784列，D_W1，784行128列，D_h1 x行128列
    D_logit = tf.matmul(D_h1, D_W2) + D_b2#  D_h1 x行128列， D_W2，128行1列  D_logit x行1列
    D_prob = tf.nn.sigmoid(D_logit)#D_prob 同 D_logit x行1列   sigmoid函数激活
    return D_prob, D_logit #都是 x行1列                                                                     #判别器最后一层去掉sigmoid

G_sample = generator(Z)# G_sample 生成一些样本
D_real, D_logit_real = discriminator(X)# 判别器 判别一个batch真实样本 X 后返回自己的判断
D_fake, D_logit_fake = discriminator(G_sample)# 判别器判别 一个batch 假的样本后 返回自己的判断 （G_sample有生成器生成）

#生成器判别器的loss不取log
D_loss_real = tf.reduce_mean(tf.scalar_mul(-1,D_logit_real))#将判别器返回的结果乘以-1 再求平均值作为判别器的损失
D_loss_fake = tf.reduce_mean(D_logit_fake)# 将判别器判别的假的样本的返回结果 求平均值，作为假样本的损失
D_loss = D_loss_real + D_loss_fake# 两个损失之和为判别器的总损失
G_loss = tf.reduce_mean(tf.scalar_mul(-1,D_logit_fake))  #生成器的损失为假样本的损失乘以-1 再求个平均值

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)#优化器 通过是D_optimizer最小来确定 theta_D中的参数
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)#优化器 通过是G_optimizer最小来确定 ttheta_G中的参数

#不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行 ****************************
#D_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3).minimize(D_loss, var_list=theta_D)# lr=0.005
#G_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-3).minimize(G_loss,var_list=theta_G)


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])#产生-1到1的的区间随机采样数字 形成m行n列的矩阵

def plot(samples):
    fig = plt.figure(figsize=(4, 4))#绘制 4行4列的图片
    gs = gridspec.GridSpec(4, 4)#确定子图的位置
    gs.update(wspace=0.05, hspace=0.05)###更新子图的位置

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])#字面意思就是绘制子图的意思，其实就是绘制子图的意思，ax为返回子图的轴
        #plt.axis('off')#不显示坐标尺寸
        ax.set_xticklabels([])#设定x轴的标签文字
        ax.set_aspect('equal')#####################设置平面相等
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')# 展示图片

    return fig


sess.run(tf.global_variables_initializer())

if not os.path.exists('outc/'):
    os.makedirs('outc/')

i = 0

for it in range(1000000):#循环执行1000000 次
    if it % 1000 == 0:
        print("g_sample...")
        print(G_sample)
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})  # 16*z_dim是100 ，运行生成器
        #数据可以从这个samples中可以取出来
        print("samples....")
        # samples.shape的形状为[16,784]
        print(samples)
        fig = plot(samples)#调用函数绘制出来samples
        plt.savefig('outc/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')#{}.png其中的大括号是个占位符，str是自带的函数
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)#mb_size = 128  读取下一个batch###
    #X_mb的形状为[128,784] 打印中可以看出来。。。。
    print("x_mb.......")
    print(X_mb)

    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)}) # mb_size=128 Z_dim = 100运行图，运行判别器的网络，调整判别器的损失
                                                    #feed进去两个一个x，另一个是z是高斯噪声
    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})#feed进去一个高斯噪声，运行生成器的图调整生成器的损失, mb_size128,   Z_dim 100

    if it % 1000 == 0:
        print('Iter: {}'.format(it))# 第几次循环 即 for里面生命的it
        print('D loss: {:.4}'.format(D_loss_curr))#把返回的D_loss_curr 格式化打印出来
        print('G_loss: {:.4}'.format(G_loss_curr))#把返回的G_loss_curr 格式化打印出来
        print()#换行

        #wgan的特点
        # 判别器最后一层去掉sigmoid----------------------------------------
        # 生成器和判别器的loss不取log-----------------------------------------
        # 每次更新判别器的参数之后把它们的值截断到不超过一个固定常数c
        # 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行---------------------------------------------
        #sess.run()可以根据fetch的名字匹配出图中的名字
