# 四叉树划分的块数据结构


class Block:
    def __init__(self, vertex_list, sample_list):
        self.vertex_list = vertex_list  # 块四个顶点的坐标列表(顺序：左上->右上->左下->右下)
        self.sample_list = sample_list  # 块中包含的样例的坐标列表
        width = self.vertex_list[1].get_x() - self.vertex_list[0].get_x()
        height = self.vertex_list[0].get_y() - self.vertex_list[2].get_y()
        self.area = width * height  # 面积
        self.density = len(self.sample_list) / self.area  # 初始化时算好
