import socket
from MQTT.mqtt_data_transmission.utils.handle_data_util import dataSwitch
import time
import os
import sys
import subprocess
import signal
from tqdm import tqdm
""" 
#####   执行目录为 ..\SAGAN_MQTT; 
"""


def TCP_collect2server():
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1883
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(0)
    s.settimeout(2)
    s.connect((TCP_IP, TCP_PORT))


def MQTT_collect2server():
    global s
    TCP_collect2server()
    req_val = "104100044d51545404e6870000147061686f3131323432353032373331383130303000036c7774000b48656c6c6f20576f726421000561646d696e0006313233343536"
    req_string = dataSwitch(req_val.strip('\n'))  # switch hex to bit for sending it to the simulations
    s.send(req_string)
    time.sleep(0.1)
    req_ack = s.recv(BUFFER_SIZE).hex()  # 获取连接请求回应数据包


if __name__ == '__main__':
    py_location = "MQTT/mqtt_data_transmission/FrameSenderReciver_B.py"
    train_data_path = 'Data/output'
    seek_number = 10

    global sum_count
    global BUFFER_SIZE
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1883
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    BUFFER_SIZE = 10000

    files_dict = {}
    if os.path.isdir(train_data_path):
        print("it's a directory")
        for file_name in os.listdir(train_data_path):
            file_path = train_data_path + "/" + file_name
            files_dict[file_path] = "MQTT/log_data_communications/logfirst_" + file_name + ".txt"
    elif os.path.isfile(train_data_path):
        files_dict[train_data_path] = "MQTT/log_data_communications/logfirst_" + time.strftime("%H.%M.%S#%Y-%m-%d",
                                                                                 time.localtime()) + ".txt"
        print("it's a normal file")
    else:
        print("it's a special file(socket,FIFO,device file)")
        exit()

    for file_path in files_dict.keys():
        fatal_error_count = 0
        conventional_error_count = 0
        fatal_error_list = []
        conventional_error_list = []
        log_file_name = files_dict[file_path]
        print("tested file path : " + file_path + "\n" + "log file name : " + log_file_name)
        with open(log_file_name,
                  "r+") as f_log:
            f_log.seek(0)
            f_log.truncate()
            try:
                with open(file_path, 'r') as f:
                    try:
                        MQTT_collect2server()
                        content = f.readlines()
                        sum_count = seek_number
                        for i in tqdm(range(seek_number)):
                            val = content[i]
                            if val is not None and val != '\n':
                                val = val.strip()
                                string = dataSwitch(val.strip('\n'))    # switch hex to bit for sending it to the simulations
                                s.send(string)
                                f_log.write(" No." +str(i) + ' TX :' + val)

                                time.sleep(0.1)
                                try:
                                    data = s.recv(BUFFER_SIZE)
                                except IOError as e:
                                    fatal_error_count += 1
                                    fatal_error_list.append(val)
                                    f_log.write('\n Fatal error! ########################################## \n')
                                    f_log.write('\n')
                                    print('No. ' + str(i) + " rejected by server; content: " + val)
                                    MQTT_collect2server()
                                    continue

                                result = data.hex()
                                if result == '' or result is None:
                                    conventional_error_count += 1
                                    conventional_error_list.append(val)
                                    f_log.write('\n Conventional error! **********************************************  \n')
                                    f_log.write('\n')
                                    print('No. ' + str(i) + " Malformed packet; content: " + val)
                                else:
                                    f_log.write('\n')
                                    f_log.write("==> No." + str(i) + ' RX :' + result + '\n')
                                    f_log.write('\n')
                    finally:
                        if i >= sum_count-1:
                            log_sum = "Total tested packed number : " + str(seek_number) + "\n"\
                                "Fatal error count : " + str(fatal_error_count) + "\nConventional error count :" + str(conventional_error_count) + "\n" \
                                      "Error rate :" + str(fatal_error_count + conventional_error_count) + "/" + str(sum_count) + "; \n" \
                                     "Acceptance(1 - Fatal/Sum) : {:.3%}  \nFormed packet rate(1 - Malformed packet/Sum) :{:.3%}".format(1 - (fatal_error_count / sum_count), 1 - (conventional_error_count / sum_count))
                            print(log_sum)
                            f_log.write(log_sum)
            except e:
                None
            finally:
                f_log.close()

#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
