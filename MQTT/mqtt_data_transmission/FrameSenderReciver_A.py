import socket
from MQTT.mqtt_data_transmission.utils.handle_data_util import dataSwitch
import time
import os
import sys
import subprocess
import signal
from tqdm import tqdm

if __name__ == '__main__':
    py_location = "MQTT/mqtt_data_transmission/FrameSenderReciver_B.py"
    # FrameSenderReciver_before_pid = int(subprocess.run(['pgrep', '-f', py_location]).strip())
    # os.kill(FrameSenderReciver_before_pid, signal.SIGTERM)
    global sum_count
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1883
    BUFFER_SIZE = 10000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(0)
    s.settimeout(2)
    s.connect((TCP_IP, TCP_PORT))



    """
    count 为 两个脚本发送脚本共有的index;
    log_file_name 为两个脚本共同保存log 的文件名;
    error_count 为两个脚本共享的错误的log的个数;
    error_list 为两个脚本共享的log list
    """
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
        log_file_name = str(sys.argv[2])
        error_count = int(sys.argv[3])
        error_list = str(sys.argv[4]).split('.')
    else:
        count = 0
        log_file_name = "MQTT/log_data_communications/logfirst_" + time.strftime("%H.%M.%S#%Y-%m-%d",
                                                                                 time.localtime()) + ".txt"
        error_count = 0
        error_list = []

    with open(log_file_name,
              "a") as f_log:
        try:
            with open('Data/mqtt_template.txt', 'r') as f:
                try:

                    content = f.readlines()
                    req_val = "104100044d51545404e6870000147061686f3131323432353032373331383130303000036c7774000b48656c6c6f20576f726421000561646d696e0006313233343536"
                    req_string = dataSwitch(req_val.strip('\n'))  # switch hex to bit for sending it to the simulations
                    s.send(req_string)
                    sum_count = len(content)
                    req_ack = s.recv(BUFFER_SIZE).hex()  # 获取连接请求回应数据包
                    for i in tqdm(range(len(content))):
                        if i >= count:
                            val = content[i]
                            if val is not None and val != '\n':
                                val = val.strip()
                                count = count + 1
                                string = dataSwitch(val.strip('\n'))    # switch hex to bit for sending it to the simulations
                                s.send(string)
                                f_log.write(" No." +str(i) + ' TX :' + val)

                                # time.sleep(0.1)
                                data = s.recv(BUFFER_SIZE)

                                # result = data.encode('hex')########################################################################
                                result = data.hex()
                                f_log.write('\n')
                                f_log.write("==> No." + str(i) + ' RX :' + result + '\n')
                                f_log.write('\n')
                except IOError as e:
                    s.close()
                    f.close()
                    error_count += 1
                    error_list.append(val)
                    f_log.write('\n error! **********************************************  \n')
                    f_log.write('\n')
                    f_log.close()
                    print('No. ' + str(i) + " go wrong; content: " + val)
                    error_list = '.'.join(error_list)
                    os.system("python " + py_location + " %i %s %i %s" % (count, log_file_name, error_count, error_list))
                    exit()
                finally:
                    if count >= sum_count:
                        log_sum = "Error ratio :" + str(error_count) + "/" + str(sum_count) + ";  Acceptance : {:.3%}".format(
                            1 - (error_count / sum_count))
                        print(log_sum)
                        f_log.write(log_sum)
        except:
            None
        finally:
            f_log.close()

#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
