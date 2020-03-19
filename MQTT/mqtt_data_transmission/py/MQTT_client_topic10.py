#!/usr/bin/python

import MQTT.mqtt_data_transmission.MQTT_const as MQTT_const
import numpy as np
import time
import sys
from tqdm import tqdm
# PINGREQ 心跳请求
# ======================================================
try:
    import paho.mqtt.client as mqtt
    import paho.mqtt.subscribe as subscribe
    import paho.mqtt.publish as publish
except ImportError:
    print("MQTT client not find. Please install as follow:")
    print("git clone http://git.eclipse.org/gitroot/paho/org.eclipse.paho.mqtt.python.git")
    print("cd org.eclipse.paho.mqtt.python")
    print("sudo python setup.py install")

self_topic_array = []
for i in range(MQTT_const.total_topic_num):
    self_topic_array.append("")


if __name__ == '__main__':
    mqttc = mqtt.Client("test5")
    # mqttc.on_message = MQTT_const.on_message
    # mqttc.on_connect = MQTT_const.on_connect
    # mqttc.on_publish = MQTT_const.on_publish
    # mqttc.on_subscribe = MQTT_const.on_subscribe
    # mqttc.on_log = MQTT_const.on_log
    mqttc.connect(MQTT_const.strBroker, MQTT_const.port, 60)
    mqttc.loop_start()
    loop_count = MQTT_const.loop_count
    loop_flag = MQTT_const.flag

    if loop_flag:
        loop_count = sys.maxint
    # 循环操作
    for i in tqdm(range(loop_count)):
        Qos = np.random.randint(0, MQTT_const.qos_top)  # 随机选择一个Qos;  [0,1,2]

        """
        根据随机令牌的值判定是否需要订阅消息
        """
        subscribe_token = np.random.random()  # set random_token
        subscribe_token_statue = True if(subscribe_token >= MQTT_const.token_threshold) else False
        if subscribe_token_statue:
            self_topic_index = np.random.randint(0, MQTT_const.total_topic_num)   # 左闭右开
            if self_topic_array[self_topic_index] == "":
                topic = "topic" + str(self_topic_index)
                self_topic_array[self_topic_index] = topic
                mqttc.subscribe(topic, Qos)
        time.sleep(MQTT_const.sleep_time)
        sub_self_topic_array = list(filter(lambda x: x != "", self_topic_array))

        """
        根据随机令牌的值判定是否需要发布消息
        """
        publish_token = np.random.random()  # set random_token
        publish_token_statue = True if(publish_token >= MQTT_const.token_threshold) else False
        if publish_token_statue:
            if len(sub_self_topic_array) > 0:
                sub_self_topic_index = np.random.randint(0, len(sub_self_topic_array))  # 左闭右开
                mqttc.publish(sub_self_topic_array[sub_self_topic_index], str(time.time()), Qos)
        time.sleep(MQTT_const.sleep_time)

        """
        根据随机令牌的值判定是否需要取消订阅主题
        """
        cancel_subscribe_token = np.random.random()  # set random_token
        cancel_subscribe_token_statue = True if(cancel_subscribe_token >= MQTT_const.token_threshold) else False
        flag = False
        for item in self_topic_array:
            if item != "":
                flag = True
                break
        if flag and cancel_subscribe_token_statue:
            if len(sub_self_topic_array) > 0:
                sub_self_topic_index = np.random.randint(0, len(sub_self_topic_array))  # 左闭右开
                mqttc.unsubscribe(sub_self_topic_array[sub_self_topic_index])
        time.sleep(MQTT_const.sleep_time)











