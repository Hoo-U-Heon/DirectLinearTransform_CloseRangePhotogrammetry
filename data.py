import math

PI = 3.1415926


def format_number(value):
    """
    格式化数值输出：如果数值小于0.01或大于100000，使用科学计数法；否则使用普通浮点格式
    """
    try:
        abs_value = abs(value)
        if abs_value < 0.01 or abs_value > 100000:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"
    except:
        return str(value)


class ObjectPt:
    def __init__(self, pid_=-1, X_=0, Y_=0, Z_=0):
        self.pid = pid_
        self.X = X_
        self.Y = Y_
        self.Z = Z_
        self.type = False  # True表示GCP，False表示未知点


class PhotoPt:
    def __init__(self, pid_=-1, x_=0, y_=0):
        self.pid = pid_
        self.x = x_
        self.y = y_
        self.type = False  # True表示GCP，False表示未知点
        self.ischeck = False

    def convert(self):
        # 像素坐标转换为像平面坐标（mm）
        # 图像尺寸：4272 x 2848
        # 传感器尺寸：22.2mm x 14.8mm
        # 先平移到图像中心
        self.x -= 4272 / 2
        self.y = 2848 / 2 - self.y

        # 转换单位（像素 -> mm）
        # 修正：图像宽度4272像素对应传感器宽度22.2mm，图像高度2848像素对应传感器高度14.8mm
        x_sz = 22.2 / 4272  # 宽度方向每像素的mm数
        y_sz = 14.8 / 2848  # 高度方向每像素的mm数

        self.x *= x_sz
        self.y *= y_sz


class RParams:
    def __init__(self):
        self.x0 = 0.0
        self.y0 = 0.0
        self.f = 0.0
        self.Xs = 0.0
        self.Ys = 0.0
        self.Zs = 0.0
        self.phi = 0.0
        self.omega = 0.0
        self.kappa = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0


class LParams:
    def __init__(self):
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.l4 = 0.0
        self.l5 = 0.0
        self.l6 = 0.0
        self.l7 = 0.0
        self.l8 = 0.0
        self.l9 = 0.0
        self.l10 = 0.0
        self.l11 = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0


def readGCP(filepath):
    m = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            line = file.readline()
            ptnum = int(line.strip())
            for _ in range(ptnum):
                line = file.readline()
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        pid = int(parts[0])
                        X = float(parts[1])
                        Y = float(parts[2])
                        Z = float(parts[3])
                        cp = ObjectPt(pid, X, Y, Z)
                        if pid > 110:
                            cp.type = True
                        m[pid] = cp
    except Exception as e:
        print(f"Unable to open file: {filepath}, error: {e}")
    return m


def readMeasure(filepath):
    m = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            if ',' in first_line:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            pid = int(parts[0].strip())
                            x = float(parts[1].strip())
                            y = float(parts[2].strip())
                            pt = PhotoPt(pid, x, y)
                            if pid > 110:
                                pt.type = True
                            m[pid] = pt
            else:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            pid = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            pt = PhotoPt(pid, x, y)
                            if pid > 110:
                                pt.type = True
                            m[pid] = pt
    except Exception as e:
        print(f"Unable to open file: {filepath}, error: {e}")
    return m
