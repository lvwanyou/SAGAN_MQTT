import numpy as np


class DataLoad(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.vector_array = []
        self.num_batch = 0
        self.vector_matrix = []
        self.pointer = 0

    def create_batches(self, file_name):
        self.vector_array = []  #每一行的数据读到这个 list中，最后通过np.array把他转变为矩阵

        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                int_line = [int(x) for x in line]
                print("int_line........")
                print(int_line)
                self.vector_array.append(int_line)

        # 此处进行分块处理
        self.num_batch = int(len(self.vector_array) / self.batch_size)
        self.vector_array = self.vector_array[:self.num_batch * self.batch_size]#这一行好像没什么用，有用，应该是凑够整数个batch，不够的就丢掉了
        self.vector_matrix = np.split(np.array(self.vector_array), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.vector_matrix[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0