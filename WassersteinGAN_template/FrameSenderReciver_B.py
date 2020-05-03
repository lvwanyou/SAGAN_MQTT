import socket
from handle_data_util import dataSwitch
import time
import os
import sys

if __name__ == '__main__':
    TCP_IP = '127.0.0.1'
    TCP_PORT = 502
    BUFFER_SIZE = 10000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(1)
    s.settimeout(2)
    s.connect((TCP_IP, TCP_PORT))
    logs = []
    # count = int(sys.argv[1])
    if len(sys.argv) > 1:
        count = int(sys.argv[1])##############################################################################################
    else:
        count = 0
    with open('GeneratedDataModbus/generated_write_single_register_25.txt', 'r') as f:
        try:
            content = f.readlines()
            for i, val in enumerate(content):
                if i >= count:
                    if val is not None and val != '\n':
                        count = count + 1
                        string = dataSwitch(val.strip('\n'))
                        s.send(string)
                        logs.append(str(i) + '  TX  ' + val)
                        time.sleep(0.1)
                        data = s.recv(BUFFER_SIZE)
                        # result = data.encode('hex')#################################################################################
                        result = data.hex()
                        logs.append(str(i) + '  RX  ' + result + '\n')
        except IOError as e:
            s.close()
            f.close()
            os.system("python FrameSenderReciver_A.py " + str(count))
            print('can not read the file!')
        finally:
            with open("logfirst36.txt", "a") as f:
                f.write(" ".join(logs))