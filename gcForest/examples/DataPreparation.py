import csv
import linecache


# 第一步处理，仅保留不舒适的数据，整理成gcforest能识别的格式
def init(file_name):
    directions = {'E': '1', 'N': '2', 'NE': '3', 'NW': '4', 'S': '5', 'SE': '6', 'SW': '7', 'W': '8'}
    thickness = {'BLM': '1', 'BWSJ': '2', 'EPS': '3', 'JAZ': '4', 'JBB': '5', 'SMXPS': '6', 'XPS': '7', 'YM': '8'}

    headers = ['year', 'month', 'day', 'hour', 'sm_set_temp', 'wt_set_temp', 'wall_material', 'wall_thickness',
               'roof_material', 'roof_thickness', 'floor_material', 'floor_thickness', 'out_temp',
               'rlt_humidity', 'solar_scatter', 'wind_dir', 'wind_speed', 'room_1', 'uncomfortable_hours']
    # 年份,月,日,时刻,朝向,夏季设定温度,冬季设定温度,墙体保温材料,墙体保温厚度,屋面保温材料,屋面保温厚度,楼板保温材料,
    # 楼板保温厚度,室外天气干球温度,相对湿度,太阳散射水平,风向,风速,1号房间,不舒适小时数

    with open('../../original_data/第' + file_name + '个子文件.csv', mode='r', encoding='utf8') as original_data:
        # lines = csv.reader(original_data)
        original_data.__next__()
        lines = original_data.readlines()

        with open('./subfile_' + file_name + '_train.csv', mode='w', encoding='utf8',
                  newline='') as processed_data:

            for i in range(1, 3):   # 1、2号房间作为训练集
                col = 16 + 2 * i
                for row in lines:
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col+1] != '0' or line[col+1] != '#':
                        if float(line[col]) > float(line[5]):
                            temp += line[5]
                        elif float(line[col]) < float(line[6]):
                            temp += line[6]
                        else:
                            continue

                        temp += "," + directions[line[4]]

                        for x in range(7, 18):
                            if x == 7:
                                line[x] = thickness[line[x]]
                            elif x == 9:
                                line[x] = thickness[line[x]]
                            elif x == 11:
                                line[x] = thickness[line[x]]

                            temp += "," + line[x]

                        processed_data.write(temp + "\n")

                    else:
                        continue

        with open('./subfile_' + file_name + '_test.csv', mode='w', encoding='utf8',
                  newline='') as processed_data:

            for i in range(3, 6):  # 3-130号房间作为测试集
                col = 16 + 2 * i
                for row in lines:
                    # print(line)
                    line = row.split(',')

                    temp = ""

                    if line[col + 1] != '0' or line[col + 1] != '#':
                        if float(line[col]) > float(line[5]):
                            temp += line[5]
                        elif float(line[col]) < float(line[6]):
                            temp += line[6]
                        else:
                            continue

                        temp += "," + directions[line[4]]

                        for x in range(7, 18):
                            if x == 7:
                                line[x] = thickness[line[x]]
                            elif x == 9:
                                line[x] = thickness[line[x]]
                            elif x == 11:
                                line[x] = thickness[line[x]]

                            temp += "," + line[x]

                        processed_data.write(temp + "\n")

                    else:
                        continue


def func2():
    with open('../data.csv', mode='r', encoding='utf8') as original_data:
        lines = original_data.readlines(1000)
        for line in lines:
            print(line)


if __name__ == '__main__':
    for x in range(1, 3):   # 取1、2子文件
        init(str(x))
        # func2()
