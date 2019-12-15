#!/usr/bin/python

import sys
import datetime
import socket, sys
import paho.mqtt.publish as publish


def transmitMQTT(strMsg):
    strMqttBroker = "localhost"
    strMqttChannel = "topic1"
    print(strMsg)
    publish.single(strMqttChannel, strMsg, hostname=strMqttBroker)


if __name__ == '__main__':
    transmitMQTT("Hello,MQTT11")
    print("Send msg ok.")
