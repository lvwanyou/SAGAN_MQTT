from datetime import timedelta

import tensorflow as tf
from utils import *
import numpy as np
import time
from ops import *
import os
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import matplotlib.pyplot as plt

class SA_GAN_SEQ(object):
    def __init__(self, sess, args, w2i, i2w):
        self.model_name = "SAGAN_SEQ"
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir

        self.w2i = w2i
        self.i2w = i2w

        self.LAMBDA = args.LAMBDA
        self.epoch = args.epoch
        self.iteration = args.iteration
        self.critic_iters = args.critic_iters
        self.batch_size = args.batch_size
        self.seq_size = args.seq_size
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        self.d_ff = args.d_ff

        self.z_dim = args.z_dim

        self.sample_num = args.sample_num
        self.vocab_size = args.vocab_size

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.d_model = args.d_model # embedding_dim

        #self.data = load_data(args.data_file, self.w2i)# [total_num * seq_size]

        self.embeddings = tf.get_variable('weight_mat',
                                          dtype=tf.float32,
                                          shape=(self.vocab_size, self.d_model),
                                          initializer=tf.contrib.layers.xavier_initializer())

        #################################

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_size])
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.vocab_size)
        self.fake_inputs = self.generator(self.z)# shape (batch_size, seq_size, vocab_size), there is probablity
        self.fake_inputs_discrete = tf.argmax(self.fake_inputs, self.fake_inputs.get_shape().ndims-1)

        self.fake_inputs_test = self.generator(self.z, False)
        self.fake_inputs_discrete_test = tf.argmax(self.fake_inputs_test, self.fake_inputs_test.get_shape().ndims-1)

        self.real_logits = self.discriminator(self.real_inputs)
        self.fake_logits = self.discriminator(self.fake_inputs)

        self.disc_cost = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
        self.gen_cost = -tf.reduce_mean(self.fake_logits)
        self.w_distance = tf.reduce_mean(self.real_logits) - tf.reduce_mean(self.fake_logits)

        self.gen_prob = tf.reduce_mean(tf.sigmoid(self.fake_logits))
        self.real_prob = tf.reduce_mean(tf.sigmoid(self.real_logits))

        # WGAN lipschitz-penalty
        self.alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.fake_inputs - self.real_inputs
        self.interpolates = self.real_inputs + (self.alpha * self.differences)
        #self.disc_inter = self.discriminator(self.interpolates)
        #self.grad = tf.gradients(self.disc_inter, [self.interpolates])
        self.gradients = tf.gradients(self.discriminator(self.interpolates), [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.LAMBDA * self.gradient_penalty

        self.variable = tf.trainable_variables()
        self.gen_params = [v for v in self.variable if 'Generator' in v.op.name]
        self.disc_params = [v for v in self.variable if 'discriminator' in v.op.name]

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.gen_cost,
                                                                                                 var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.disc_cost,
                                                                                                  var_list=self.disc_params)
        self.d_sum = tf.summary.scalar("gen_cost", self.gen_cost)
        self.g_sum = tf.summary.scalar("disc_cost", self.disc_cost)

    def generator(self, z, is_training=True):
        """
        achitecture:

        [one] linear layer
        [num_blocks] transformer block
        [one] linear layer
        [one] softmax layer
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):# z's shape (batch_size, z_dim)
            x = tf.layers.dense(z, units=self.seq_size * self.d_model) # batch_size, seq_size
            x = tf.nn.relu(x)
            x = tf.reshape(x, [self.batch_size, self.seq_size, self.d_model])
            x *= self.d_model**0.5
            x += positional_encoding(x, self.seq_size)
            x = tf.layers.dropout(x, self.dropout_rate, training=is_training)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    x = multihead_attention(queries=x,
                                              keys=x,
                                              values=x,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=is_training,
                                              causality=False)
                    x = ff(x, num_units=[self.d_ff, self.d_model])
            # x = tf.transpose(x, [0, 2, 1])   # before:(batch_size, d_model , seq_len)  ---->   after transpose: (batch_size, d_model , seq_len)
            # x = self.ResBlock('ResBlock', x)
            # x = tf.transpose(x, [0, 2, 1])

            weights = tf.Variable(tf.random_normal([self.d_model, self.vocab_size], stddev=0.1),
                                  name="weights")
            logits = tf.einsum('ntd,dk->ntk', x, weights) #(batch_size, seq_size, vocab_size)
            #res = tf.reshape(tf.argmax(logits, axis=2), [self.batch_size, self.seq_size])
        return tf.nn.softmax(logits)

    def self_attention(self, input, is_training=True):
        return multihead_attention(queries=input,
                                  keys=input,
                                  values=input,
                                  num_heads=self.num_heads,
                                  dropout_rate=self.dropout_rate,
                                  training=is_training,
                                  causality=False)

    def discriminator(self, x, is_training=True):
        """
        achitecture:
                                               [one] conv1d
        positional encoding |     conv1d     |       conv1d   |     conv1d     |     conv1d
        self-attention      | self-attention | self-attention | self-attention | self-attention
                                                concanate
                                                [one] linear
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            #原版
            output = tf.transpose(x, [0, 2, 1])  # (batch_size, vocab_size, seq_len)
            output = lib.ops.conv1d.Conv1D('Conv1d.1', self.vocab_size, self.vocab_size, 5, output)
            output = lib.ops.conv1d.Conv1D('Conv1d.2', self.vocab_size, self.vocab_size, 5, output)
            output = lib.ops.conv1d.Conv1D('Conv1d.3', self.vocab_size, self.vocab_size, 5, output)
            output = tf.transpose(output, [0, 2, 1])
            output = self.self_attention(output, is_training)
            output = tf.transpose(output, [0, 2, 1])
            output = lib.ops.conv1d.Conv1D('Conv1d.4', self.vocab_size, self.vocab_size, 5, output)
            output = tf.transpose(output, [0, 2, 1])
            output = self.self_attention(output, is_training)
            enc = tf.reshape(output, [self.batch_size, self.vocab_size * self.seq_size])
            res = tf.layers.dense(enc, units=1)

            #新版
            # output = tf.transpose(x, [0, 2, 1])
            # output = lib.ops.conv1d.Conv1D('Conv1d.1', self.vocab_size, self.d_model, 1, output) #(batch_size, d_model, seq_size)
            # branch_1 = tf.transpose(output, [0, 2, 1])#(batch_size, seq_size, d_model)
            # branch_1 = positional_encoding(branch_1, self.seq_size)
            # branch_1 = self.self_attention(branch_1, is_training=is_training)#(batch_size, seq_size, d_model)
            #
            # branch_2 = lib.ops.conv1d.Conv1D('Conv1d.2', self.d_model, self.d_model, 2, output)
            # branch_2 = tf.transpose(branch_2, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # branch_2 = self.self_attention(branch_2, is_training=is_training)
            #
            #
            # branch_3 = lib.ops.conv1d.Conv1D('Conv1d.3', self.d_model, self.d_model, 3, output)
            # branch_3 = tf.transpose(branch_3, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # branch_3 = self.self_attention(branch_3, is_training=is_training)
            #
            # branch_4 = lib.ops.conv1d.Conv1D('Conv1d.4', self.d_model, self.d_model, 4, output)
            # branch_4 = tf.transpose(branch_4, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # branch_4 = self.self_attention(branch_4, is_training=is_training)
            #
            # branch_5 = lib.ops.conv1d.Conv1D('Conv1d.5', self.d_model, self.d_model, 5, output)
            # branch_5 = tf.transpose(branch_5, [0, 2, 1])  # (batch_size, seq_size, d_model)
            # branch_5 = self.self_attention(branch_5, is_training=is_training)
            #
            # res = tf.concat([branch_1, branch_2, branch_3, branch_4, branch_5], 1)
            # print("after concat:", res.shape)
            # res = tf.reshape(res, [self.batch_size, self.d_model * self.seq_size * 5])
            # res = tf.layers.dense(res, units=1)
            # print("after linear", res.shape)
        return res

    def ResBlock(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.1', self.d_model, self.d_model, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.2', self.d_model, self.d_model, 5, output)
        return inputs + (0.3 * output)

    def discriminator_(self, x):
        """
        WGANGP原版discriminator
        :param x:
        :return:
        """
        output = tf.transpose(x, [0, 2, 1])# (batch_size, vocab_size, seq_len)
        output = lib.ops.conv1d.Conv1D('Discriminator.Input', self.vocab_size, self.d_model, 1, output)
        output = self.ResBlock('Discriminator.1', output)
        output = self.ResBlock('Discriminator.2', output)
        output = self.ResBlock('Discriminator.3', output)
        output = self.ResBlock('Discriminator.4', output)
        output = self.ResBlock('Discriminator.5', output)
        output = tf.reshape(output, [-1, self.seq_size * self.d_model])
        output = lib.ops.linear.Linear('Discriminator.Output', self.seq_size * self.d_model, 1, output)
        return output

    def train(self, data):
        batch = 0
        epoch_need_record = [10, 20, 30, 40, 50]
        n_batch = len(data)//self.batch_size #总batch数
        total_batch = n_batch // 100 * self.epoch
        print("n_batch:", n_batch)
        print("total_batch", n_batch // 100 * self.epoch)

        # saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # 画图相关
        fig_w_distance = np.zeros([total_batch])
        fig_d_loss_trains = np.zeros([total_batch])
        fig_g_loss_trains = np.zeros([total_batch])

        train_start_time = time.time()
        for e in range(self.epoch):
            real_data = inf_train_gen(data, self.batch_size)
            epoch_start_time = time.time()
            iter = 0
            epoch_over = False
            while not epoch_over:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                self.sess.run(self.gen_train_op, feed_dict={self.z: z})
                for _ in range(self.critic_iters):
                    iter += 1
                    real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                    z = make_noise([self.batch_size, self.z_dim])
                    _disc_cost, _gen_cost, _ = self.sess.run([self.disc_cost, self.gen_cost, self.disc_train_op],
                                                             feed_dict={self.real_inputs_discrete: real_inputs_discrete,
                                                                        self.z: z})
                    if iter % 100 == 99:
                        z = make_noise([self.batch_size, self.z_dim])
                        iter += 1
                        real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
                        gen_samples, gen_prob, real_prob, disc_cost, gen_cost, w_distance = \
                            self.sess.run([self.fake_inputs_discrete, self.gen_prob, self.real_prob,self.disc_cost, self.gen_cost, self.w_distance],
                                          feed_dict={self.real_inputs_discrete: real_inputs_discrete,self.z: z})
                        fig_w_distance[batch] = w_distance
                        fig_d_loss_trains[batch] = disc_cost
                        fig_g_loss_trains[batch] = gen_cost
                        batch += 1

                        translate(gen_samples, self.i2w)
                        print("Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime{}"
                              .format(e+1, iter, disc_cost, real_prob, gen_cost, gen_prob, w_distance, timedelta(seconds=time.time() - epoch_start_time)))

                    print("iter:", iter)

                    if iter % n_batch == 0:

                        # 某些epoch存checkpoint并且保存生成的数据
                        if e+1 in epoch_need_record:
                            # saver.save(self.sess, self.checkpoint_dir+"/epoch_"+str(e+1))
                            data_to_write = []
                            for i in range(3):
                                z = make_noise([self.batch_size, self.z_dim])
                                gen_samples = self.sess.run(self.fake_inputs_discrete_test, feed_dict={self.z: z})
                                data_to_write.append(gen_samples)
                            save_gen_samples(data_to_write, self.i2w, e+1)


                        epoch_over = True
                        break

        print("After training, total time:", timedelta(seconds=time.time() - train_start_time))

        print("w_sitance")
        print(fig_w_distance)
        print("\nd_loss")
        print(fig_d_loss_trains)
        print("\ng_loss")
        print(fig_g_loss_trains)
        ###########################   绘图 start   #######################################
        # 绘制曲线
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(np.arange(total_batch), fig_w_distance, label="w_distance")
        # 按一定间隔显示实现方法
        # ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
        lns2 = ax2.plot(np.arange(total_batch), fig_d_loss_trains, 'r', label="d_loss")
        lns3 = ax2.plot(np.arange(total_batch), fig_g_loss_trains, 'g', label="g_loss")
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('w_distance')
        ax2.set_ylabel('d_loss & g_loss')

        # 合并图例
        lns = lns1 + lns2 + lns3
        labels = ["w_distance", "D_Loss", "G_Loss"]
        # labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=7)
        plt.show()
        ###########################   绘图 end   #######################################


    #舍弃版本
    def train_(self, data):
        # #数据一共有x行
        # #那么epoch_batch数有n_batch = x//batch_size
        # #total_batch为总的batch数，total_epoch = epoch * epoch_batch
        # batch = 0
        # epoch_batch = len(data)//self.batch_size #data可以转换的batch数
        # total_batch = epoch_batch * self.epoch
        # print("n_batch:", epoch_batch)
        # print("total_batch:", total_batch)
        #
        # self.sess.run(tf.global_variables_initializer())
        # fig_w_distance = np.zeros([epoch_batch//100 * self.epoch])
        # fig_d_loss_trains = np.zeros([epoch_batch//100 * self.epoch])
        # fig_g_loss_trains = np.zeros([epoch_batch//100 * self.epoch])
        #
        #
        #
        #
        # for e in range(self.epoch):
        #     real_data = inf_train_gen(data, self.batch_size)
        #     epoch_start_time = time.time()
        #     iter = 0
        #     epoch_over = False
        #     while not epoch_over:
        #         #训练G
        #         z = make_noise(shape=[self.batch_size, self.z_dim])
        #         self.sess.run(self.gen_train_op, feed_dict={self.z: z})
        #
        #         #训练D
        #         for _ in range(self.critic_iters):
        #             iter += 1
        #             real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
        #             z = make_noise([self.batch_size, self.z_dim])
        #             _disc_cost, _gen_cost, _ = self.sess.run([self.disc_cost, self.gen_cost, self.disc_train_op],
        #                                                      feed_dict={self.real_inputs_discrete: real_inputs_discrete,
        #                                                                 self.z: z})
        #             if iter % 100 == 99: #每一百次迭代输出一下结果
        #                 z = make_noise([self.batch_size, self.z_dim])
        #                 iter += 1
        #                 real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
        #                 gen_samples, gen_prob, real_prob, disc_cost, gen_cost, w_distance = \
        #                     self.sess.run([self.fake_inputs_discrete, self.gen_prob, self.real_prob,self.disc_cost, self.gen_cost, self.w_distance],
        #                                   feed_dict={self.real_inputs_discrete: real_inputs_discrete,self.z: z})
        #                 fig_w_distance[batch] = w_distance
        #                 fig_d_loss_trains[batch] = disc_cost
        #                 fig_g_loss_trains[batch] = gen_cost
        #                 batch += 1
        #
        #                 translate(gen_samples, self.i2w)
        #                 print("Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime{}"
        #                       .format(e+1, iter, disc_cost, real_prob, gen_cost, gen_prob, w_distance, timedelta(seconds=time.time() - epoch_start_time)))
        #
        #             if iter % epoch_batch == epoch_batch-1:
        #                 epoch_over = True
        #                 break

        epoch_batch = len(data) // self.batch_size
        total_batch = epoch_batch * self.epoch
        print("epoch_batch:", epoch_batch)
        print("total_batch:", total_batch)

        self.sess.run(tf.global_variables_initializer())

        fig_points = epoch_batch // 100 * self.epoch
        fig_w_distance = np.zeros([fig_points])
        fig_d_loss_trains = np.zeros([fig_points])
        fig_g_loss_trains = np.zeros([fig_points])
        train_start_time = epoch_start_time = batch_100_start_time = time.time()
        fig_batch = 0
        for iter in range(total_batch):
            epoch = iter // epoch_batch + 1
            real_data = inf_train_gen(data, self.batch_size)



            # G训练critic_iters次
            if iter % self.critic_iters == 0 and iter != 0:
                z = make_noise(shape=[self.batch_size, self.z_dim])
                self.sess.run(self.gen_train_op, feed_dict={self.z: z})

            real_inputs_discrete = next(real_data)  # (batch_size, seq_len)
            z = make_noise([self.batch_size, self.z_dim])
            #训练D
            gen_samples, gen_prob, real_prob, disc_cost, gen_cost, w_distance = \
                    self.sess.run([self.fake_inputs_discrete, self.gen_prob, self.real_prob,
                                   self.disc_cost, self.gen_cost, self.w_distance],
                    feed_dict={self.real_inputs_discrete: real_inputs_discrete,self.z: z})
            # 每训练100个batch记录一下，为画图作准备
            if iter % 100 == 99:
                fig_w_distance[fig_batch] = w_distance
                fig_d_loss_trains[fig_batch] = disc_cost
                fig_g_loss_trains[fig_batch] = gen_cost
                fig_batch += 1
                print('------------------------------')
                translate(gen_samples, self.i2w)
                print("Every 100 batch:")
                print("Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime {}"
                      .format(epoch, iter+1 % epoch_batch, disc_cost, real_prob, gen_cost, gen_prob, w_distance,
                              timedelta(seconds=time.time() - batch_100_start_time)))
                batch_100_start_time = time.time()

            # 每个epoch更新时间
            if iter % epoch_batch == epoch_batch-1:
                print('------------------------------')
                print("Every epoch:")
                print("Epoch {}\ncurrent iter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime {}"
                      .format(epoch, iter+1 , disc_cost, real_prob, gen_cost, gen_prob, w_distance,
                              timedelta(seconds=time.time() - epoch_start_time)))
                epoch_start_time = time.time()
                epoch_need_reocrd = [10, 20, 30, 40, 50]
                if epoch in epoch_need_reocrd:
                    print("Current epoch:", epoch)


        print("============================")
        print("After training, total time:", timedelta(seconds=time.time() - train_start_time))

        print("w_distance:")
        print(fig_w_distance)
        print("\ng_loss:")
        print(fig_g_loss_trains)
        print("\nd_loss")
        print(fig_d_loss_trains)
        ###########################   绘图 start   #######################################
        # 绘制曲线
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(np.arange(fig_points), fig_w_distance, label="w_distance")
        # 按一定间隔显示实现方法
        # ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
        lns2 = ax2.plot(np.arange(fig_points), fig_d_loss_trains, 'r', label="d_loss")
        lns3 = ax2.plot(np.arange(fig_points), fig_g_loss_trains, 'g', label="g_loss")
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('w_distance')
        ax2.set_ylabel('d_loss & g_loss')

        # 合并图例
        lns = lns1 + lns2 + lns3
        labels = ["w_distance", "D_Loss", "G_Loss"]
        # labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=7)
        plt.show()
        ###########################   绘图 end   #######################################

    def eval(self, z):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.seq_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

