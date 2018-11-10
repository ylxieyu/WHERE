# 实验
from WHERE import WHERE
from Tools.DataSetTool import DataSetTool

path = 'D:\\data\\txt\\'
x_list, y_list = DataSetTool.init_data(path, 20, is_sample=False, is_normalized=False)
where = WHERE(depth=10)
an = where.fit(x_list)
# print(an)
