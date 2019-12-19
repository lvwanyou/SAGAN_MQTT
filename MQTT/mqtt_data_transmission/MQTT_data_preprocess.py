import json
import os
import MQTT.mqtt_data_transmission.MQTT_const as MQTT_const


# 读取 [{字典},{字典},{字典},{字典}] 类型的 json 文件：
# 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
def json_dict():
    f = open("MQTT/preprocessing_data/origin_data.json", encoding='utf-8')
    f_s2c = open("MQTT/preprocessing_data/cases_server2clients.txt", "w+")
    f_c2s = open("MQTT/preprocessing_data/cases_clients2server.txt", "w+")
    pop_datas = json.load(f)

    mqtt_msg_c2s = []
    mqtt_msg_s2c = []
    for pop_dict in pop_datas:
        # 注意多重结构的读取语法
        src_port = int(pop_dict['_source']['layers']['tcp']['tcp.srcport'])
        dst_port = int(pop_dict['_source']['layers']['tcp']['tcp.dstport'])
        mqtt_msg = pop_dict['_source']['layers']['tcp']['tcp.payload']   #  eg:10:11:00:04:4d:51:54:54:04:02:00:3c:00:05:74:65:73:74:31
        mqtt_msg = str(mqtt_msg).replace(':', '')
        if dst_port == MQTT_const.port:
            #   数据存入到cases_clients2server.txt
            mqtt_msg_c2s.append(mqtt_msg)
            f_c2s.writelines(str(mqtt_msg) + "\n")
        elif src_port == MQTT_const.port:
            # 数据存入到cases_server2clients.txt
            mqtt_msg_s2c.append(mqtt_msg)
            f_s2c.writelines(str(mqtt_msg) + "\n")
        # print(mqtt_msg)
    f.close()
    f_s2c.close()
    f_c2s.close()


json_dict()