import csv
import linecache
import numpy.random


# 第一步处理，仅保留不舒适的数据，整理成gcforest能识别的格式
def init(file_name):
    directions = {'E': '1', 'N': '2', 'NE': '3', 'NW': '4', 'S': '5', 'SE': '6', 'SW': '7', 'W': '8'}
    thickness = {'BLM': '1', 'BWSJ': '2', 'EPS': '3', 'JAZ': '4', 'JBB': '5', 'SMXPS': '6', 'XPS': '7', 'YM': '8'}

    headers = ['year', 'month', 'day', 'hour', 'sm_set_temp', 'wt_set_temp', 'wall_material', 'wall_thickness',
               'roof_material', 'roof_thickness', 'floor_material', 'floor_thickness', 'out_temp',
               'rlt_humidity', 'solar_scatter', 'wind_dir', 'wind_speed', 'room_1', 'uncomfortable_hours']
    # 年份,月,日,时刻,朝向,夏季设定温度,冬季设定温度,墙体保温材料,墙体保温厚度,屋面保温材料,屋面保温厚度,楼板保温材料,
    # 楼板保温厚度,室外天气干球温度,相对湿度,太阳散射水平,风向,风速,1号房间,不舒适小时数
    print(file_name)

    with open('../original_data/第%d个子文件.csv' % file_name, mode='r', encoding='utf8') as original_data:
        # lines = csv.reader(original_data)
        original_data.__next__()
        lines = original_data.readlines()

        with open('./room_1_less/subfile_%d.csv' % file_name, mode='w', encoding='utf8', newline='') as processed_data:
            processed_data.write("out_temp,rlt_humidity,solar_scatter,wind_dir,wind_speed,room_temp,set_temp\n")

            for i in range(1, 2):   # 1、2号房间作为训练集
                col = 16 + 2 * i

                # random_index = numpy.random.randint(1, len(lines), int(len(lines)/10))

                # for index in random_index:
                    # print(line)
                    # line = lines[index].split(',')

                for row in lines:
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col+1] != '0' and line[col+1] != '#':

                        for x in range(13, 18):
                            temp += line[x] + ','

                        if float(line[col]) > float(line[5]):
                            temp += line[col] + ',' + line[5]
                        elif float(line[col]) < float(line[6]):
                            temp += line[col] + ',' + line[6]
                        else:
                            continue

                        processed_data.write(temp + "\n")

                    else:
                        continue


        """
        with open('./subfile_' + file_name + '_test.csv', mode='a+', encoding='utf8',
                  newline='') as processed_data:

            processed_data.write("dir,wall_material,wall_thickness,roof_material,roof_thickness,"
                                 "floor_material,floor_thickness,out_temp,rlt_humidity,solar_scatter,"
                                 "wind_dir,wind_speed,room_temp,set_temp\n")

            for i in range(3, 6):  # 3-5号房间作为测试集
                col = 16 + 2 * i
                for row in lines:
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col + 1] != '0' or line[col + 1] != '#':

                        temp += directions[line[4]]

                        for x in range(7, 18):
                            if x == 7:
                                line[x] = thickness[line[x]]
                            elif x == 9:
                                line[x] = thickness[line[x]]
                            elif x == 11:
                                line[x] = thickness[line[x]]

                            temp += "," + line[x]

                        if float(line[col]) > float(line[5]):
                            temp += ',' + line[col] + ',' + line[5]
                        elif float(line[col]) < float(line[6]):
                            temp += ',' + line[col] + ',' + line[6]
                        else:
                            continue

                        processed_data.write(temp + "\n")

                    else:
                        continue
            """


def init2(file_name, room):
    directions = {'E': '1', 'N': '2', 'NE': '3', 'NW': '4', 'S': '5', 'SE': '6', 'SW': '7', 'W': '8'}
    thickness = {'BLM': '1', 'BWSJ': '2', 'EPS': '3', 'JAZ': '4', 'JBB': '5', 'SMXPS': '6', 'XPS': '7', 'YM': '8'}

    headers = ['year', 'month', 'day', 'hour', 'sm_set_temp', 'wt_set_temp', 'wall_material', 'wall_thickness',
               'roof_material', 'roof_thickness', 'floor_material', 'floor_thickness', 'out_temp',
               'rlt_humidity', 'solar_scatter', 'wind_dir', 'wind_speed', 'room_1', 'uncomfortable_hours']
    # 年份,月,日,时刻,朝向,夏季设定温度,冬季设定温度,墙体保温材料,墙体保温厚度,屋面保温材料,屋面保温厚度,楼板保温材料,
    # 楼板保温厚度,室外天气干球温度,相对湿度,太阳散射水平,风向,风速,1号房间,不舒适小时数
    print('第' + file_name + '个文件')

    with open('../original_data/第' + file_name + '个子文件.csv', mode='r', encoding='utf8') as original_data:
        # lines = csv.reader(original_data)
        original_data.__next__()
        lines = original_data.readlines()

        with open('./room_' + str(room) + '/subfile_' + file_name + '.csv', mode='w', encoding='utf8', newline='') as processed_data:

            processed_data.write("dir,wall_material,wall_thickness,roof_material,roof_thickness,"
                                 "floor_material,floor_thickness,out_temp,rlt_humidity,solar_scatter,"
                                 "wind_dir,wind_speed,room_temp,set_temp\n")

            col = 16 + 2 * room

            # random_index = numpy.random.randint(1, len(lines), int(len(lines)/10))

            for row in lines:
                # print(line)
                line = row.split(',')

                temp = ""

                if line[col+1] != '0' and line[col+1] != '#':

                    temp += directions[line[4]]

                    for x in range(7, 18):
                        if x == 7:
                            line[x] = thickness[line[x]]
                        elif x == 9:
                            line[x] = thickness[line[x]]
                        elif x == 11:
                            line[x] = thickness[line[x]]

                        temp += ',' + line[x]

                    if float(line[col]) > float(line[5]):
                        temp += ',' + line[col] + ',' + line[5]
                    elif float(line[col]) < float(line[6]):
                        temp += ',' + line[col] + ',' + line[6]
                    else:
                        continue

                    processed_data.write(temp + "\n")

                else:
                    continue


        """
        with open('./subfile_' + file_name + '_test.csv', mode='a+', encoding='utf8',
                  newline='') as processed_data:

            processed_data.write("dir,wall_material,wall_thickness,roof_material,roof_thickness,"
                                 "floor_material,floor_thickness,out_temp,rlt_humidity,solar_scatter,"
                                 "wind_dir,wind_speed,room_temp,set_temp\n")

            for i in range(3, 6):  # 3-5号房间作为测试集
                col = 16 + 2 * i
                for row in lines:
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col + 1] != '0' or line[col + 1] != '#':

                        temp += directions[line[4]]

                        for x in range(7, 18):
                            if x == 7:
                                line[x] = thickness[line[x]]
                            elif x == 9:
                                line[x] = thickness[line[x]]
                            elif x == 11:
                                line[x] = thickness[line[x]]

                            temp += "," + line[x]

                        if float(line[col]) > float(line[5]):
                            temp += ',' + line[col] + ',' + line[5]
                        elif float(line[col]) < float(line[6]):
                            temp += ',' + line[col] + ',' + line[6]
                        else:
                            continue

                        processed_data.write(temp + "\n")

                    else:
                        continue
            """


def func2():
    with open('../data.csv', mode='r', encoding='utf8') as original_data:
        lines = original_data.readlines(1000)
        for line in lines:
            print(line)


# 算一下分类个数
def func3():
    classes = []
    for x in range(1, 101):
        with open('../original_data/第' + str(x) + '个子文件.csv', mode='r', encoding='utf8') as original_data:
            # lines = csv.reader(original_data)
            original_data.__next__()
            lines = csv.reader(original_data)

            for line in lines:
                if line[5] not in classes:
                    classes.append(line[5])

                if line[6] not in classes:
                    classes.append(line[6])

    print(classes)


# 计算每个文件样本数
def func4():
    for x in range(1, 101):
        with open('./room_1/subfile_' + str(x) + '.csv', mode='r', encoding='utf8') as original_data:
            original_data.__next__()
            lines = original_data.readlines()

            with open('./instance_numbers.txt', mode='a+', newline='') as ftow:
                ftow.write(str(len(lines)) + '\n')


def balance_data():
    for x in range(1, 101):
        print(x)
        with open('../original_data/第%d个子文件.csv' % x, mode='r', encoding='utf8') as original_data:
            original_data.__next__()
            lines = original_data.readlines()

            lines_num = len(lines)
            positive_array = [0 for j in range(1, 131)]
            useful_array = [0 for j in range(1, 131)]
            ftow = open('./balance_rates.csv', mode='a+', newline='')
            csv_writer = csv.writer(ftow)

            for l in lines:
                line = l.split(',')
                for c in range(0, 130):
                    col = 18 + 2 * c
                    if line[col + 1] != '0' and line[col + 1] != '#':
                        useful_array[c] += 1
                        if line[col] >= line[5]:
                            positive_array[c] += 1
                    else:
                        continue

            for (p, u, n) in zip(positive_array, useful_array, range(0, 130)):
                positive_array[n] = (4 * p * (u - p)) / (u ** 2 + 1)

            if x == 1:
                headers = [i for i in range(1, 131)]
                csv_writer.writerow(headers)

            csv_writer.writerow(positive_array)
            ftow.close()


if __name__ == '__main__':

    # for x in range(1, 101):
    #     init(x)
        # func2()
    # func3()

    # for y in range(1, 2):
    #     for x in range(1, 101):  # 取1、2子文件
#
    #     # for x in [86, 77, 40, 70, 46, 22, 58, 89, 96, 93, 62, 69, 61, 87, 2, 72, 26, 16, 85, 63]:
    #         init2(str(x), y)

    # func4()
    balance_data()
