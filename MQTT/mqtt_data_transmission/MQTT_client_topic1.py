#!/usr/bin/python

import sys
import datetime
import socket, sys

# PINGREQ 心跳请求

# ======================================================
try:
    import paho.mqtt.client as mqtt
    import paho.mqtt.subscribe as subscribe
except ImportError:
    print("MQTT client not find. Please install as follow:")
    print("git clone http://git.eclipse.org/gitroot/paho/org.eclipse.paho.mqtt.python.git")
    print("cd org.eclipse.paho.mqtt.python")
    print("sudo python setup.py install")

# 服务器地址
strBroker = "localhost"
# 通信端口
port = 1883
# 用户名
username = 'username'
# 密码
password = 'password'
# 订阅主题名
topic = 'topic1'


# ======================================================
def on_connect(mqttc, obj, rc):
    print("OnConnetc, rc: " + str(rc))


def on_publish(mqttc, obj, mid):
    print("OnPublish, mid: " + str(mid))


def on_subscribe(mqttc, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log(mqttc, obj, level, string):
    print("Log:" + string)


def on_message(mqttc, obj, msg):
    curtime = datetime.datetime.now()
    strcurtime = curtime.strftime("%Y-%m-%d %H:%M:%S")
    print(strcurtime + ": " + msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    on_exec(str(msg.payload))


def on_exec(strcmd):
    print("Exec:", strcmd)

def on_message_print(client, userdata, message):
    print("%s %s" % (message.topic, message.payload))

# =====================================================
if __name__ == '__main__':
    mqttc = mqtt.Client("test1")
    mqttc.on_message = on_message
    mqttc.on_connect = on_connect
    mqttc.on_publish = on_publish
    mqttc.on_subscribe = on_subscribe
    mqttc.on_log = on_log

    # 设置账号密码（如果需要的话）
    # mqttc.username_pw_set(username, password=password)

    mqttc.connect(strBroker, port, 60)
    mqttc.subscribe(topic, 0)   # the second para is: QOS(Quality of Service)
    # subscribe.callback(on_message_print, "paho/test/callback", hostname="iot.eclipse.org")
    mqttc.loop_forever()