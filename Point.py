import math


class Point(object):
    def __init__(self, x_param=0.0, y_param=0.0):
        self.x = x_param
        self.y = y_param

    def __str__(self):
        return str(self.x) + ',' + str(self.y)

    def distance(self, pt):
        x_diff = self.x - pt.x
        y_diff = self.y - pt.y
        return math.sqrt(x_diff ** 2 + y_diff ** 2)

    def sum(self, pt):
        x_new = self.x + pt.x
        y_new = self.y + pt.y
        return Point(x_new, y_new)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_str(self):
        return str(self.x)+','+str(self.y)

    def tuple(self):
        return tuple((self.x, self.y))

    # 比较与目标点是否为同一点
    def equal(self, point):
        if self.x == point.x and self.y == point.y:
            return True
        else:
            return False
