# WHERE 类
# 用于对数据集中的实例进行聚类
import numpy as np
import math as math
from Point import Point
# import matplotlib.pyplot as plot
from Block import Block
import treelib as treelib
# import threading
from Muti_Block import Muti_Block
import sys
sys.setrecursionlimit(100000)


class WHERE:
    def __init__(self, alpha=0, delta=0.5, depth=4):
        self.alpha = alpha  # stopping rule for quad-tree tree recursion
        self.delta = delta  # stopping rule for clustering
        self.data = None  # 所有的实例
        # self.x_list = None  # 所有实例映射到2D平面的横坐标
        # self.y_list = None  # 所有实例映射到2D平面的纵坐标
        self.dict = dict()  # 保存点与样例的映射
        self.block_list = None  # 保存分好的块列表
        self.not_sort = True  # 是否没对保存的块列表进行过排序
        self.cluster_dict = dict()  # 保存聚类的映射
        self.tree = None  # 四叉树
        self.depth = depth  # 生成四叉树的最大深度

    # data_list 需要聚类的数据集列表
    # data_list的数据结构：外层是list，内层是二维数组
    # 功能：返回聚类好的data,返回的List中，每一个item是一个聚类
    def fit(self, data_list):
        root = self.create_block(data_list)
        self.tree = treelib.Tree()
        self.tree.create_node('root', '0', data=root)
        self.create_leaves()
        print(self.tree)
        leaves_list = self.tree.leaves()
        # 得到所有划分好的块
        self.block_list = [leaf.data for leaf in leaves_list]
        # 聚类
        self.cluster([])
        if len(self.cluster_dict) > 0:
            # 通过聚类解开映射，得到类型相似的样例列表
            muti_block_list = self.im_muti_block()
            # muti_block_list = self.ir_muti_block()
            muti_data_index_list = []
            for muti_block in muti_block_list:
                muti_data_index_list.append(muti_block.get_data_index(self.dict))  # 每一个聚类中的点对应的样例序号
            muti_data_list = []  # 返回的样例
            for muti_data_index in muti_data_index_list:
                muti_data_list.append([self.data[i] for i in muti_data_index])
            return muti_data_list
        else:
            return self.block_list

        # 绘出画好的矩形
        # fig = plot.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # plot.scatter(self.x_list, self.y_list, s=1)
        # plot.xlabel('x')
        # plot.ylabel('y')
        # plot.title('Initial two dimensions')
        # # x_ = [block_list[0].vertex_list[j].get_x() for j in range(len(block_list[0].vertex_list))]
        # # x_[2], x_[3] = x_[3], x_[2]
        # # x_.append(x_[0])
        # # y_ = [block_list[0].vertex_list[j].get_y() for j in range(len(block_list[0].vertex_list))]
        # # y_[2], y_[3] = y_[3], y_[2]
        # # y_.append(y_[0])
        # # plot.plot(x_, y_)
        # for i in range(len(block_list)):
        #     x_ = [block_list[i].vertex_list[j].get_x() for j in range(len(block_list[i].vertex_list))]
        #     x_[2], x_[3] = x_[3], x_[2]
        #     x_.append(x_[0])
        #     y_ = [block_list[i].vertex_list[j].get_y() for j in range(len(block_list[i].vertex_list))]
        #     y_[2], y_[3] = y_[3], y_[2]
        #     y_.append(y_[0])
        #     plot.plot(x_, y_)
        #     x_.clear()
        #     y_.clear()
        # plot.show()

    # 得到原始的块，并将样例投影到块中
    def create_block(self, data_list):
        # 将data_list中的样例合并至一个二维数组
        assert len(data_list) > 0
        data = data_list[0]
        if len(data_list) > 1:
            for i in range(1, len(data_list)):
                data = np.append(data, data_list[i], axis=0)
        self.data = data
        sample_num = len(data)  # 实例总数
        self.alpha = math.sqrt(sample_num)
        rand_index = np.random.randint(0, sample_num)  # 随机选取一个实例
        x_index = self.get_furthest_instance(rand_index)  # 得到x点的序列
        y_index = self.get_furthest_instance(x_index)  # 得到y点的序列
        c = np.linalg.norm(self.data[x_index] - self.data[y_index])  # 得到c
        point_list = self.update_coordinate(c, x_index, y_index)  # 得到实例的二维坐标
        # x_list = [point_list[i].get_x() for i in range(len(point_list))]
        y_list = [point_list[i].get_y() for i in range(len(point_list))]
        # self.x_list = x_list
        # self.y_list = y_list
        # 画初始图
        # fig = plot.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # plot.scatter(x_list, y_list, s=1)
        # plot.xlabel('x')
        # plot.ylabel('y')
        # plot.title('Initial two dimensions')
        # plot.show()
        # 得到初始的块
        vertex_list = [Point(0, max(y_list)), Point(c, max(y_list)), Point(0, 0), Point(0, c)]
        return Block(vertex_list, point_list)

    # 从列表中得到与当前点距离最远的item
    def get_furthest_instance(self, index):
        pre_distance, pre_index = 0, index
        for i in range(len(self.data)):
            cur_distance = np.linalg.norm(self.data[index] - self.data[i])
            if cur_distance > pre_distance:
                pre_distance, pre_index = cur_distance, i
        return pre_index

    # 更新列表坐标
    def update_coordinate(self, c, x_index, y_index):
        point_list = []
        for i in range(len(self.data)):
            a = np.linalg.norm(self.data[i] - self.data[x_index])
            b = np.linalg.norm(self.data[i] - self.data[y_index])
            x = (a**2 + c**2 - b**2) / (2*c)
            y = math.sqrt(a**2 - x**2)
            point_list.append(Point(x, y))
            if Point(x, y).get_str() in self.dict:
                value = self.dict.get(Point(x, y).get_str())
                value.append(i)
                self.dict[Point(x, y).get_str()] = value
            else:
                self.dict[Point(x, y).get_str()] = [i]
        return point_list

    # 递归算出所有的块
    def recursive_block(self, nid):
        if self.tree.get_node(nid).is_leaf():
            block = self.tree.get_node(nid).data
            sample_num = len(block.sample_list)
            # 计算中位数
            x_ = np.median([block.sample_list[i].get_x() for i in range(sample_num)])
            y_ = np.median([block.sample_list[i].get_y() for i in range(sample_num)])
            # 新得到的5个点坐标
            pt_middle = Point(x_, y_)
            pt_up = Point(x_, block.vertex_list[0].get_y())
            pt_left = Point(block.vertex_list[0].get_x(), y_)
            pt_right = Point(block.vertex_list[1].get_x(), y_)
            pt_down = Point(x_, block.vertex_list[2].get_y())
            # 得到新的块中的样例
            first = [block.sample_list[i] for i in range(sample_num)
                     if block.sample_list[i].get_x() < x_ and block.sample_list[i].get_y() > y_]
            second = [block.sample_list[i] for i in range(sample_num)
                      if block.sample_list[i].get_x() >= x_ and block.sample_list[i].get_y() > y_]
            third = [block.sample_list[i] for i in range(sample_num)
                     if block.sample_list[i].get_x() < x_ and block.sample_list[i].get_y() <= y_]
            fourth = [block.sample_list[i] for i in range(sample_num)
                      if block.sample_list[i].get_x() >= x_ and block.sample_list[i].get_y() <= y_]
            # 得到新的块
            block_first = Block([block.vertex_list[0], pt_up, pt_left, pt_middle], first)
            block_second = Block([pt_up, block.vertex_list[1], pt_middle, pt_right], second)
            block_third = Block([pt_left, pt_middle, block.vertex_list[2], pt_down], third)
            block_fourth = Block([pt_middle, pt_right, pt_down, block.vertex_list[3]], fourth)
            # 作为子节点加入树
            self.tree.create_node(identifier=nid+'0', data=block_first, parent=nid)
            self.tree.create_node(identifier=nid+'1', data=block_second, parent=nid)
            self.tree.create_node(identifier=nid+'2', data=block_third, parent=nid)
            self.tree.create_node(identifier=nid+'3', data=block_fourth, parent=nid)
            # # 判断是否要递归
            # if len(block_first.sample_list) > self.alpha:
            #     # t = threading.Thread(target=self.recursive_block, args=(tree, nid + '0'))
            #     # t.start()
            #     self.recursive_block(nid+'0')
            # if len(block_second.sample_list) > self.alpha:
            #     # t = threading.Thread(target=self.recursive_block, args=(tree, nid + '1'))
            #     # t.start()
            #     self.recursive_block(nid+'1')
            # if len(block_third.sample_list) > self.alpha:
            #     # t = threading.Thread(target=self.recursive_block, args=(tree, nid + '2'))
            #     # t.start()
            #     self.recursive_block(nid+'2')
            # if len(block_fourth.sample_list) > self.alpha:
            #     # t = threading.Thread(target=self.recursive_block, args=(tree, nid + '3'))
            #     # t.start()
            #     self.recursive_block(nid+'3')

    # 聚类
    # index_list  # 保存的是block_list中聚类好了的序号列表
    def cluster(self, index_list):
        # 将列表中的块按照从大到小的顺序排序
        if self.not_sort:
            self.block_list.sort(key=lambda x: x.density, reverse=True)
            self.not_sort = False
            index_list = []
        # 去掉了聚类好了的序号的当前block_list
        cur_block_list = [self.block_list.copy()[i] for i in range(len(self.block_list)) if i not in index_list]
        # 当前要操作的block
        cur_block = cur_block_list[0]
        # 记录当前块在self.block_list中的序号
        cur_index = self.block_list.index(cur_block)
        index_list.append(cur_index)
        # 传入外层循环的起始序号
        cur_block_index = self.block_list.index(cur_block)  # 起始操作块序号
        index_list = self.iter_cluster(cur_block, cur_block_list, index_list, cur_block_index)
        # 计算是否还存在未聚类的
        block_list_new = [self.block_list.copy()[i] for i in range(len(self.block_list)) if i not in index_list]
        # 外层迭代
        if len(block_list_new) != 0:
            self.cluster(index_list)

    # 内层迭代
    def iter_cluster(self, cur_block, cur_block_list, index_list, cur_block_index):
        # 找到cur_block的相邻块
        cur_neighbors = self.get_neighbors(cur_block, cur_block_list, index_list)
        for neighbor in cur_neighbors:
            # 如果满足聚类要求
            if (1 - neighbor.density / cur_block.density) < self.delta:
                index_ = self.block_list.index(neighbor)
                if index_ not in index_list:
                    index_list.append(index_)  # 满足聚类的块序号列表
                    if cur_block_index in self.cluster_dict:  # 保存在字典中
                        value = self.cluster_dict.get(cur_block_index)
                        value.append(index_)
                        self.cluster_dict[cur_block_index] = value
                    else:
                        self.cluster_dict[cur_block_index] = [index_]
                    # 进入迭代
                    block_list_new = [self.block_list[i] for i in range(len(self.block_list)) if i not in index_list]
                    self.iter_cluster(neighbor, block_list_new, index_list, cur_block_index)
        return index_list

    # 从列表中选出与中心块相邻的块
    def get_neighbors(self, m_block, block_list, index_list):
        m_points = m_block.vertex_list
        t_index = set()
        for block in block_list:
            for j in range(4):
                if m_points[0].equal(block.vertex_list[j]):
                    t_index.add(block_list.index(block))
                if m_points[1].equal(block.vertex_list[j]):
                    t_index.add(block_list.index(block))
                if m_points[2].equal(block.vertex_list[j]):
                    t_index.add(block_list.index(block))
                if m_points[3].equal(block.vertex_list[j]):
                    t_index.add(block_list.index(block))
        for i in index_list:
            if self.block_list[i] in block_list:
                index = block_list.index(self.block_list[i])  # 获得邻域中已经聚类的在block_list中的序号
                t_index.remove(index)
        t_block = [block_list[i] for i in t_index]
        return t_block

    # 将聚类好了的dict 展开映射，映射到数据实例
    # 返回所有实例（聚类的与未聚类的）
    def ir_muti_block(self):
        muti_block_list = []
        cluster_index = []  # 记录已被聚类的序号
        key_list = self.cluster_dict.keys()
        for key in key_list:
            value = self.cluster_dict.get(key)
            value.append(key)
            cluster_list = [self.block_list[i] for i in value]
            if len(cluster_list) != 0:
                muti_block_list.append(Muti_Block(cluster_list))
                cluster_index.append(value)
        cluster_index_new = np.array(cluster_index[0])
        if len(cluster_index) > 1:
            for i in range(1, len(cluster_index)):
                cluster_index_new = np.append(cluster_index_new, np.array(cluster_index[i]))
        other_blocks = [self.block_list[i] for i in range(len(self.block_list)) if i not in cluster_index_new]
        for other_block in other_blocks:
            if len(other_block.sample_list) != 0:
                muti_block_list.append(Muti_Block([other_block]))  # 将未参与聚类的加入到进来
        return muti_block_list

    # 只返回聚类的block
    def im_muti_block(self):
        muti_block_list = []
        cluster_index = []  # 记录已被聚类的序号
        key_list = self.cluster_dict.keys()
        for key in key_list:
            value = self.cluster_dict.get(key)
            value.append(key)
            cluster_list = [self.block_list[i] for i in value]
            if len(cluster_list) != 0:
                muti_block_list.append(Muti_Block(cluster_list))
                cluster_index.append(value)
        # cluster_index_new = np.array(cluster_index[0])
        # if len(cluster_index) > 1:
        #     for i in range(1, len(cluster_index)):
        #         cluster_index_new = np.append(cluster_index_new, np.array(cluster_index[i]))
        # other_blocks = [self.block_list[i] for i in range(len(self.block_list)) if i not in cluster_index_new]
        return muti_block_list

    # 用循环的方法代替递归，生成四叉树
    # 判断当前树的叶子能否再次划分
    def leaves_can_divide(self):
        depth = self.tree.depth()
        leaves = self.tree.leaves()
        cur_leaves = [leaf for leaf in leaves if self.tree.depth(node=leaf) == depth]
        nid_list = []
        for cur_leaf in cur_leaves:
            if len(cur_leaf.data.sample_list) > self.alpha:
                nid_list.append(cur_leaf.identifier)
        if len(nid_list) != 0:
            return nid_list, True, depth
        else:
            return None, False, depth

    def create_leaves(self):
        nid_list, is_divide, depth = self.leaves_can_divide()
        if is_divide and depth < self.depth:
            for nid in nid_list:
                self.recursive_block(nid)
            return self.create_leaves()
