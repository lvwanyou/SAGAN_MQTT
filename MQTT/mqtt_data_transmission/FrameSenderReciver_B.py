import socket
from MQTT.mqtt_data_transmission.utils.handle_data_util import dataSwitch
import time
import os
import sys
from tqdm import tqdm
if __name__ == '__main__':
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1883
    BUFFER_SIZE = 10000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(0)
    s.settimeout(2)
    s.connect((TCP_IP, TCP_PORT))
    logs = []

    # count 为 两个脚本发送脚本共有的index
    if len(sys.argv) > 1:
        count = int(sys.argv[1]) ##############################################################################################
    else:
        count = 0
    #with open('GeneratedDataModbus/generated_data_write_single_register_16.txt', 'r') as f:
    #with open('dataseven1.txt', 'r') as f:
    with open('Data/mqtt_template.txt', 'r') as f:
        try:
            content = f.readlines()
            req_val = "104100044d51545404e6870000147061686f3131323432353032373331383130303000036c7774000b48656c6c6f20576f726421000561646d696e0006313233343536"
            req_string = dataSwitch(req_val.strip('\n'))  # switch hex to bit for sending it to the simulations
            s.send(req_string)
            for i in tqdm(range(len(content))):
                if i >= count:
                    val = content[i]
                    if val is not None and val != '\n':
                        val = val.strip()
                        count = count + 1
                        string = dataSwitch(val.strip('\n'))    # switch hex to bit for sending it to the simulations
                        s.send(string)
                        logs.append("No." +str(i) + ' TX :' + val)
                        time.sleep(0.1)
                        data = s.recv(BUFFER_SIZE)

                        # result = data.encode('hex')########################################################################
                        result = data.hex()
                        logs.append('\n')
                        logs.append("==> No." + str(i) + ' RX :' + result + '\n')
        except IOError as e:
            s.close()
            f.close()
            os.system("python MQTT/mqtt_data_transmission/FrameSenderReciver_A.py " + str(count))
            print('can not read the file!')
        finally:
            with open("MQTT/log_data_communications/logfirst_0.txt", "a") as f:
                f.write(" ".join(logs))
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data
#TX Sending Data
#RX Reciving Data