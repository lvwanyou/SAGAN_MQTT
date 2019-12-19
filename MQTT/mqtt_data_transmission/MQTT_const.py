import sys
import datetime
import socket, sys


total_topic_pool = []
total_topic_num = 10
for i in range(total_topic_num):
    total_topic_pool.append("topic" + str(i))

token_threshold = 0.7

sleep_time = 0.1  # unit : second

# 服务器地址
strBroker = "localhost"
# 通信端口
port = 1883
# 用户名
username = 'username'
# 密码
password = 'password'
# 定义每个client loop 次数
loop_count = 10000
# 定义每个client 是否永真loop
flag = False

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
