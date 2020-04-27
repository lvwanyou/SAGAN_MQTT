import numpy as np

n = [1, 2, 3, 4, 5]

b = 1
str_sea = ""
seq_count = 0


def func( n, str_seq):
    count = 0
    for i in range(len(n)):
        if n[i] == 0:
            count += 1
        if count == len(n):
            print(str_seq)
            print()
            global seq_count
            seq_count += 1
            break

    for j in range(len(n)):
        if n[j] != 0:
            n_temp = n.copy
            str_seq_temp = str_seq
            if str_seq_temp == "":
                str_seq_temp = str(n_temp[j])
            else:
                str_seq_temp = str_seq_temp + '->' + str(n_temp[j])
            n_temp[j] = 0
            func(n_temp, str_seq_temp)


func(n, str_sea)
print("Number of Sequence is : " + str(seq_count))