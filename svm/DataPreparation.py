import csv
import linecache


# 第一步处理，仅保留不舒适的数据，整理成svm能识别的格式
def init(file_name):
    directions = {'E': '1', 'N': '2', 'NE': '3', 'NW': '4', 'S': '5', 'SE': '6', 'SW': '7', 'W': '8'}
    thickness = {'BLM': '1', 'BWSJ': '2', 'EPS': '3', 'JAZ': '4', 'JBB': '5', 'SMXPS': '6', 'XPS': '7', 'YM': '8'}

    headers = ['year', 'month', 'day', 'hour', 'sm_set_temp', 'wt_set_temp', 'wall_material', 'wall_thickness',
               'roof_material', 'roof_thickness', 'floor_material', 'floor_thickness', 'out_temp',
               'rlt_humidity', 'solar_scatter', 'wind_dir', 'wind_speed', 'room_1', 'uncomfortable_hours']
    # 年份,月,日,时刻,朝向,夏季设定温度,冬季设定温度,墙体保温材料,墙体保温厚度,屋面保温材料,屋面保温厚度,楼板保温材料,
    # 楼板保温厚度,室外天气干球温度,相对湿度,太阳散射水平,风向,风速,1号房间,不舒适小时数

    with open('../original_data/第' + file_name + '个子文件.csv', mode='r', encoding='utf8') as original_data:
        # lines = csv.reader(original_data)
        original_data.__next__()
        lines = original_data.readlines()

        with open('./room_1/subfile_' + file_name + '_train.csv', mode='w', encoding='utf8',
                  newline='') as processed_data:

            for row in lines:
                # line_list = line.split(',')
                # print(line)
                line = row.split(',')
                temp = ""
                if line[19] != '0' or line[19] != '#':
                    if float(line[18]) > float(line[5]):
                        temp += "+1"
                    elif float(line[18]) < float(line[6]):
                        temp += "+0"
                    else:
                        continue
                    temp += " 1:" + directions[line[4]]
                    for x in range(7, 19):
                        if x == 7:
                            line[x] = thickness[line[x]]
                        elif x == 9:
                            line[x] = thickness[line[x]]
                        elif x == 11:
                            line[x] = thickness[line[x]]
                        temp += " " + str(x - 5) + ":" + line[x]
                    processed_data.write(temp + "\n")
                else:
                    continue


"""
        with open('./subfile_' + file_name + '_test.csv', mode='w', encoding='utf8',
                  newline='') as processed_data:

            for i in range(3, 131):  # 3-130号房间作为测试集
                col = 16 + 2 * i
                for row in lines:
                    # line_list = line.split(',')
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col+1] != '0' or line[col+1] != '#':
                        if float(line[col]) > 23:
                            temp += "+1"
                        elif float(line[col]) < 17:
                            temp += "+0"
                        else:
                            continue

                        temp += " 1:" + directions[line[4]]

                        for x in range(7, 18):
                            if x == 7:
                                line[x] = thickness[line[x]]
                            elif x == 9:
                                line[x] = thickness[line[x]]
                            elif x == 11:
                                line[x] = thickness[line[x]]

                            temp += " " + str(x - 5) + ":" + line[x]

                        processed_data.write(temp + "\n")

                    else:
                        continue
"""


def func4():
    for x in range(1, 101):
        with open('./room_1/subfile_' + str(x) + '_train.csv', mode='r', encoding='utf8') as original_data:
            original_data.__next__()
            lines = original_data.readlines()

            with open('./instance_numbers.txt', mode='a+', newline='') as ftow:
                ftow.write(str(len(lines)) + '\n')


def func2():
    with open('../data.csv', mode='r', encoding='utf8') as original_data:
        lines = original_data.readlines(1000)
        for line in lines:
            print(line)


if __name__ == '__main__':
    for x in range(1, 101):
        init(str(x))
        # func2()
    # func4()
