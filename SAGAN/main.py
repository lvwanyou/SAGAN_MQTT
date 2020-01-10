import tensorflow as tf
import argparse
from model import *
from utils import *

def parse_args():
    desc = "Tensorflow implementation of Self-Attention GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='mode')

    parser.add_argument('--data_file', type=str,
                        default='data/generated_write_single_register_1.txt')
    parser.add_argument('--vocab_file', type=str, default='vocab')
    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument('--z_dim', type=int, default=10)

    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--LAMBDA', type=int, default=10)
    parser.add_argument('--seq_size', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--iteration', type=int, default=10000)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=20)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--d_ff', type=int, default=64)

    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--print_freq', type=int, default=500, help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=500, help='The number of ckpt_save_freq')

    return parser.parse_args()

def main():
    args = parse_args()

    if args is None:
        exit()

    w2i, i2w = read_vocab(args.vocab_file)

    with tf.Session() as sess:
        gan = SA_GAN_SEQ(sess, args, w2i, i2w)
        #gan.build_model()
        data = load_data(args.data_file, w2i)

        # show network architecture
        #show_all_variables()

        gan.train(data)


if __name__ == '__main__':
    main()