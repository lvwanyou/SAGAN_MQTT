"""
Helper for data pre-processing!
"""
import re
import os
import sys
import struct
import importlib
hex_pattern = re.compile(r'^[a-fA-F0-9]+$')
importlib.reload(sys)####################################################################################################
# sys.setdefaultencoding( "ISO-8859-1" )#################################################################################
# sys.setdefaultencoding( "utf-8" )


def dec2hex(string_num):
    """
    Convert int number to a hex string.
    :param string_num: 
    :return: 
    """
    # if string_num.isdigit():
    hex_str = hex(string_num)
    hex_str = hex_str.replace('0x', '')
    if len(hex_str) < 2:
        hex_str = '0' + hex_str
    return hex_str


def hex2dec(string_num):
    """
    Convert a hex string to a int number
    :param string_num: 
    :return: -1 if the string_num given is illegal
    """
    if hex_pattern.match(string_num):
        return int(string_num.upper(), 16)
    else:
        return -1


def hex2dec_on_list(lst):
    """
    Conduct the hex2dec operation on a list.
    :param lst: 
    :return: 
    """
    data = []
    for i, val in enumerate(lst):
        data.append(hex2dec(val))
    return data

def transfer(path):
    """
    Transfer all the str in the file to separated version.(Separated by space per two chars)
    :param path: 
    :return: 
    """
    file_name = 'separated_modbus.txt'
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, file_name)
    new_content = ''
    with open(path, 'r') as f:
        try:
            text = f.readlines()
            for index, val in enumerate(text):
                new_str = divide_str(val)
                new_content += new_str
        except IOError:
            print('can not read the file!')


def divide_str(string):
    """
    Use space to separate the str per two chars.
    :param string: 
    :return: 
    """
    new_str = ''
    for i in range(0, int(len(string)), 2):
        new_str += string[i:i + 2] + ' '
    return new_str


def str2list(string):
    """
    Break the str into sub str of length 2 and form a list.
    :param string: 
    :return: 
    """
    return [string[i:i + 2] for i in range(0, len(string), 2)]

def convert_list_to_unicode_str(data):
    """
    Convert string to unicode str.
    :param data:
    :return:
    """
    string = ''
    for i, val in enumerate(data):
        # string = string + unicode(unichr(int(val)))
        string = string + str(int(val))
    return string

def make_request_string(string):
    """
    Wrap the string to a request string.
    :param string:
    :return:
    """
    message = str2list(string)
    modbus = hex2dec_on_list(message)
    return convert_list_to_unicode_str(modbus)


def dataSwitch(data):
    str1 = ''
    str2 = b''#############################################################################################################
    while data:
        str1 = data[0:2]
        s = int(str1,16)
        str2 += struct.pack('B',s)
        data = data[2:]
    return str2