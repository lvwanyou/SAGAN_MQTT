import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def load_data(filename, w2i):
    data = []
    with open(filename, 'r') as f:
        lines = [list(line[:-1]) for line in f]
        for i in range(len(lines)):
            data.append(np.array([w2i[word] for word in lines[i]], dtype=int))
    return np.array(data)

def inf_train_gen(data, batch_size):
    np.random.shuffle(data)
    n_batches = len(data)//batch_size
    for i in range(n_batches):
        yield data[i * batch_size:(i+1) * batch_size]

def save_gen_samples(generate_data, i2w, e):

    res = []
    for data in generate_data:
        for i in range(len(data)):
            res.append([i2w[int(index)] for index in data[i]])

    with open('output/epoch' + str(e) + '_generate_data.txt', 'w+') as f:
        for i in range(len(res)):
            f.write(''.join(res[i])+'\n')
    print("Saved epoch " + str(e) + " generate data.")

def translate(data, i2w):
    res = []
    for i in range(len(data)):
        res.append([i2w[int(index)] for index in data[i]])
    for i in range(len(res)):
        print(''.join(res[i]))
    return res

def generate_batch(data):
    return None

def read_vocab(file_name):
    word_to_id = {}
    id_to_word = {}
    with open(file_name, 'r') as f:
        for line in f:
            word, index = line.strip().split('\t')
            iden = int(index)
            word_to_id[word] = index
            id_to_word[iden] = word
    return word_to_id, id_to_word

def make_noise(shape):
    noise = np.random.normal(size=shape)
    return noise

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def seq2index(seq, w2i):
    data = []
    for i in seq:
        data.append([w2i[word] for word in seq[i]])
    return data