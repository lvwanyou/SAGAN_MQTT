val_all = {}
with open('C:/Users/11442/Desktop/merge_academicPaper/merge2010-2015_分词后_词频.txt', 'r') as f:
    try:
        content = f.readlines()
        count = 0
        for i, val in enumerate(content):
            if count < 30:
                if val is not None and val != '\n':
                    temp = val.strip().split('\t')
                    if temp[0] == '发展' or temp[0] == '政府' or temp[0] == '教师' or temp[0] == '政策' or temp[0] == '投入'\
                        or temp[0] == '社会' or temp[0] == '管理' or temp[0] == '质量' or temp[0] == '服务' or temp[0] == '建设'\
                        or temp[0] == '经费' or temp[0] == '标准' or temp[0] == '收费' or temp[0] == '公共' or temp[0] == '农村'\
                        or temp[0] == '入园' or temp[0] == '国家' or temp[0] == '资源' or temp[0] == '保障' or temp[0] == '机构' \
                        or temp[0] == '资金' or temp[0] == '提高' or temp[0] == '机制' or temp[0] == '扶持' or temp[0] == '水平' \
                        or temp[0] == '条件' or temp[0] == '改革' or temp[0] == '公平' or temp[0] == '公益性' or temp[0] == '经济':
                            val_all[temp[0]] = int(temp[1])
                            count += 1

    except IOError as e:
        f.close()
        print('can not read the file!')

val_temp1 = {}
with open('C:/Users/11442/Desktop/merge_academicPaper/merge2016-2018_分词后_词频.txt', 'r') as f:
    try:
        content = f.readlines()
        for val in enumerate(content):
                if val is not None and val != '\n':
                    temp = val[1].strip().split('\t')
                    val_temp1[temp[0]] = int(temp[1])
    except IOError as e:
        f.close()
        print('can not read the file!')

val_temp2 = {}
with open('C:/Users/11442/Desktop/merge_academicPaper/2019_分词后_词频.txt', 'r') as f:
    try:
        content = f.readlines()
        for i, val in enumerate(content):
                if val is not None and val != '\n':
                    temp = val.strip().split('\t')
                    val_temp2[temp[0]] = int(temp[1])
    except IOError as e:
        f.close()
        print('can not read the file!')

for key in val_all:
    if key in val_temp1 :
        val_all[key] = val_all[key] + val_temp1[key]
    if key in val_temp2:
        val_all[key] = val_all[key] + val_temp2[key]
    print ( key +"   :   "+ str(val_all[key]))
        #with open("C:/Users/11442/Desktop/merge_academicPaper/mergeDeal.txt", "a") as f:

