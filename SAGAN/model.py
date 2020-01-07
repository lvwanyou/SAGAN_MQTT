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
        right now, one ff layer and one self-attention layer
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
            x = tf.transpose(x, [0, 2, 1])# (batch_size, d_model , seq_len)
            x = self.ResBlock('ResBlock', x)
            x = tf.transpose(x, [0, 2, 1])

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

        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
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

            # enc *= self.d_model ** 0.5
            # enc += positional_encoding(enc, self.seq_size)
            # enc = tf.layers.dropout(enc, self.dropout_rate, training=is_training)
            #
            # for i in range(self.num_blocks):
            #     with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
            #         enc = multihead_attention(queries=enc,
            #                                   keys=enc,
            #                                   values=enc,
            #                                   num_heads=self.num_heads,
            #                                   dropout_rate=self.dropout_rate,
            #                                   training=is_training,
            #                                   causality=False)
            #         enc = ff(enc, num_units=[self.d_ff, self.d_model])
            enc = tf.reshape(output, [self.batch_size, self.vocab_size * self.seq_size])
            res = tf.layers.dense(enc, units=1)
        return res

    def ResBlock(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.1', self.d_model, self.d_model, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name + '.2', self.d_model, self.d_model, 5, output)
        return inputs + (0.3 * output)

    def discriminator_(self, x):
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
        iter = 0
        n_batch = len(data)//self.batch_size
        self.sess.run(tf.global_variables_initializer())

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
                        translate(gen_samples, self.i2w)
                        print("Epoch {}\niter {}\ndisc cost {}, real prob {}\ngen cost {}, gen prob {}\nw-distance {}\ntime{}"
                              .format(e, iter, disc_cost, real_prob, gen_cost, gen_prob, w_distance, timedelta(seconds=time.time() - epoch_start_time)))

                    if iter % n_batch == n_batch-1:
                        epoch_over = True
                        break





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

