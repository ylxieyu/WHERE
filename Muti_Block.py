# 多个block聚在一起形成的数据结构
import numpy as np


class Muti_Block:
    def __init__(self, block_list):
        area = sum([block_list[i].area for i in range(len(block_list))])
        sample_num = sum([len(block_list[i].sample_list) for i in range(len(block_list))])
        self.density = sample_num / area
        self.sample_list = self.sample_list(block_list)

    @staticmethod
    def sample_list(block_list):
        points = np.array(block_list[0].sample_list)
        if len(block_list) > 1:
            for i in range(1, len(block_list)):
                points = np.append(points, np.array(block_list[i].sample_list), axis=0)
        return points

    # 获得坐标点对应的数据的序号
    def get_data_index(self, dict):
        value = []
        for sample in self.sample_list:
            if sample.__str__() in dict:
                v = dict.get(sample.__str__())
                if v not in value:
                    value.append(v)
        if len(value) >= 1:
            data_index = np.array(value[0])
            for i in range(1, len(value)):
                data_index = np.append(data_index, np.array(value[i]), axis=0)
            return data_index
        else:
            return []
