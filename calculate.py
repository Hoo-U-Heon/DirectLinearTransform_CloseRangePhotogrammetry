import numpy as np
import random
import math
import os
from data import LParams, ObjectPt, RParams, PI

# 输出目录，默认为data/output
_output_dir = "data/output"

def set_output_dir(output_dir):
    """设置输出目录"""
    global _output_dir
    _output_dir = output_dir
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

def get_output_dir():
    """获取当前输出目录"""
    return _output_dir


def getid(pt):
    """获取点的ID列表"""
    return list(pt.keys())


def getInit(pt, GCP):
    """初始化Li系数（使用6个控制点进行DLT求解）"""
    l = LParams()
    sz = 6  # 选择前6个控制点
    A = np.zeros((sz * 2, 11))
    L = np.zeros((sz * 2, 1))

    # 随机选择6个控制点
    vid = getid(pt)
    selected = []
    i = 0
    while i < sz:
        idx = random.randint(0, len(vid) - 1)
        pid = vid[idx]
        if pid in pt and pt[pid].type and pid in GCP:
            selected.append(pid)
            i += 1

    for i in range(sz):
        pid = selected[i]
        X = GCP[pid].X
        Y = GCP[pid].Y
        Z = GCP[pid].Z

        # 复制并转换像点坐标
        pt_copy = pt[pid]
        pt_copy = PhotoPt(pt_copy.pid, pt_copy.x, pt_copy.y)
        pt_copy.convert()
        x = pt_copy.x
        y = pt_copy.y

        A[2 * i, 0] = X
        A[2 * i, 1] = Y
        A[2 * i, 2] = Z
        A[2 * i, 3] = 1

        A[2 * i, 8] = X * x
        A[2 * i, 9] = Y * x
        A[2 * i, 10] = Z * x

        A[2 * i + 1, 4] = X
        A[2 * i + 1, 5] = Y
        A[2 * i + 1, 6] = Z
        A[2 * i + 1, 7] = 1

        A[2 * i + 1, 8] = X * y
        A[2 * i + 1, 9] = Y * y
        A[2 * i + 1, 10] = Z * y

        L[2 * i, 0] = -x
        L[2 * i + 1, 0] = -y

    # 使用SVD求解
    U, S, Vt = np.linalg.svd(A)
    x_sol = Vt[-1, :].reshape(-1, 1)

    l.l1 = x_sol[0, 0]
    l.l2 = x_sol[1, 0]
    l.l3 = x_sol[2, 0]
    l.l4 = x_sol[3, 0]
    l.l5 = x_sol[4, 0]
    l.l6 = x_sol[5, 0]
    l.l7 = x_sol[6, 0]
    l.l8 = x_sol[7, 0]
    l.l9 = x_sol[8, 0]
    l.l10 = x_sol[9, 0]
    l.l11 = x_sol[10, 0]

    print("计算l系数完成")
    print(f"L1: {l.l1:.6f}")
    print(f"L2: {l.l2:.6f}")
    print(f"L3: {l.l3:.6f}")
    print(f"L4: {l.l4:.6f}")
    print(f"L5: {l.l5:.6f}")
    print(f"L6: {l.l6:.6f}")
    print(f"L7: {l.l7:.6f}")
    print(f"L8: {l.l8:.6f}")
    print(f"L9: {l.l9:.6f}")
    print(f"L10: {l.l10:.6f}")
    print(f"L11: {l.l11:.6f}")

    return l


def getLPrecision(A, X, L, num):
    """计算DLT精度"""
    V = A @ X - L
    v = (V.T @ V)[0, 0]
    sigma = math.sqrt(v / (num - 16))  # 16个未知数
    mm2pixel = 4272 / 22.2

    print(f"\n单位权中误差(sigma)/pixel: {sigma * mm2pixel:.6f}")
    print(f"单位权中误差(sigma)/mm: {sigma:.6f}\n")

    Q = np.linalg.inv(A.T @ A)

    print("未知数的中误差精度：")
    print(f"L1: {sigma * math.sqrt(Q[0, 0]):.6f}")
    print(f"L2: {sigma * math.sqrt(Q[1, 1]):.6f}")
    print(f"L3: {sigma * math.sqrt(Q[2, 2]):.6f}")
    print(f"L4: {sigma * math.sqrt(Q[3, 3]):.6f}")
    print(f"L5: {sigma * math.sqrt(Q[4, 4]):.6f}")
    print(f"L6: {sigma * math.sqrt(Q[5, 5]):.6f}")
    print(f"L7: {sigma * math.sqrt(Q[6, 6]):.6f}")
    print(f"L8: {sigma * math.sqrt(Q[7, 7]):.6f}")
    print(f"L9: {sigma * math.sqrt(Q[8, 8]):.6f}")
    print(f"L10: {sigma * math.sqrt(Q[9, 9]):.6f}")
    print(f"L11: {sigma * math.sqrt(Q[10, 10]):.6f}")
    print(f"k1: {sigma * math.sqrt(Q[11, 11]):.6f}")
    print(f"k2: {sigma * math.sqrt(Q[12, 12]):.6f}")
    print(f"k3: {sigma * math.sqrt(Q[13, 13]):.6f}")
    print(f"p1: {sigma * math.sqrt(Q[14, 14]):.6f}")
    print(f"p2: {sigma * math.sqrt(Q[15, 15]):.6f}")


def Judge(l):
    """判断是否收敛（计算fx）"""
    garma = 1 / (l.l9 ** 2 + l.l10 ** 2 + l.l11 ** 2)
    x0 = -(l.l1 * l.l9 + l.l2 * l.l10 + l.l3 * l.l11) * garma
    y0 = -(l.l5 * l.l9 + l.l6 * l.l10 + l.l7 * l.l11) * garma

    A = (l.l1 ** 2 + l.l2 ** 2 + l.l3 ** 2) * garma - x0 ** 2
    B = (l.l5 ** 2 + l.l6 ** 2 + l.l7 ** 2) * garma - y0 ** 2
    C = (l.l1 * l.l5 + l.l2 * l.l6 + l.l3 * l.l7) * garma - x0 * y0

    beta = math.sqrt(C ** 2 / A / B)
    beta = math.asin(-beta) if C > 0 else math.asin(beta)

    fx = math.cos(beta) * math.sqrt(A)
    return fx


def LDisplay(l):
    """显示L系数"""
    print("\n--------直接线性变换L系数--------")
    print(f"{l.l1:.6f} {l.l2:.6f} {l.l3:.6f} {l.l4:.6f} {l.l5:.6f} {l.l6:.6f} {l.l7:.6f} {l.l8:.6f} {l.l9:.6f} {l.l10:.6f} {l.l11:.6f}\n")


def getCheck(l_pt, r_pt):
    """获取3个检查点"""
    itr = 0
    for pid in l_pt:
        if l_pt[pid].type and pid in r_pt:
            l_pt[pid].ischeck = True
            r_pt[pid].ischeck = True
            itr += 1
        if itr == 3:
            break


def getSz(pt):
    """获取GCP点数量"""
    sz = 0
    for pid in pt:
        if pt[pid].type:
            sz += 1
    return sz


def saveDLTResult(filename, image_name, l, M, x, W, used):
    """保存DLT结果到文件"""
    from data import format_number
    try:
        # 构建完整的输出路径
        full_path = os.path.join(_output_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as fout:
            fout.write(f"========== {image_name} 直接线性变换结果 ==========\n\n")

            # L系数
            fout.write("【L系数】\n")
            fout.write(f"  L1: {format_number(l.l1)}\n")
            fout.write(f"  L2: {format_number(l.l2)}\n")
            fout.write(f"  L3: {format_number(l.l3)}\n")
            fout.write(f"  L4: {format_number(l.l4)}\n")
            fout.write(f"  L5: {format_number(l.l5)}\n")
            fout.write(f"  L6: {format_number(l.l6)}\n")
            fout.write(f"  L7: {format_number(l.l7)}\n")
            fout.write(f"  L8: {format_number(l.l8)}\n")
            fout.write(f"  L9: {format_number(l.l9)}\n")
            fout.write(f"  L10: {format_number(l.l10)}\n")
            fout.write(f"  L11: {format_number(l.l11)}\n\n")

            # 内方位元素
            garma = 1 / (l.l9 ** 2 + l.l10 ** 2 + l.l11 ** 2)
            x0 = -(l.l1 * l.l9 + l.l2 * l.l10 + l.l3 * l.l11) * garma
            y0 = -(l.l5 * l.l9 + l.l6 * l.l10 + l.l7 * l.l11) * garma
            A = (l.l1 ** 2 + l.l2 ** 2 + l.l3 ** 2) * garma - x0 ** 2
            B = (l.l5 ** 2 + l.l6 ** 2 + l.l7 ** 2) * garma - y0 ** 2
            C = (l.l1 * l.l5 + l.l2 * l.l6 + l.l3 * l.l7) * garma - x0 * y0
            beta = math.sqrt(C ** 2 / A / B)
            beta = math.asin(-beta) if C > 0 else math.asin(beta)
            fx = math.cos(beta) * math.sqrt(A)

            fout.write("【内方位元素（mm）】\n")
            fout.write(f"  x0: {format_number(x0)}\n")
            fout.write(f"  y0: {format_number(y0)}\n")
            fout.write(f"  f: {format_number(fx)}\n\n")

            # 外方位元素
            ext = np.array([
                [l.l1, l.l2, l.l3],
                [l.l5, l.l6, l.l7],
                [l.l9, l.l10, l.l11]
            ])
            vec = np.array([[-l.l4], [-l.l8], [-1]])
            line = np.linalg.inv(ext) @ vec

            a3 = math.sqrt(garma) * l.l9
            b3 = math.sqrt(garma) * l.l10
            c3 = math.sqrt(garma) * l.l11
            ds = math.sqrt(A / B) - 1
            a2 = (math.sqrt(garma) * l.l5 + a3 * math.sqrt(garma)) * (1 + ds) * math.cos(beta) / fx
            b2 = (l.l6 * math.sqrt(garma) + b3 * y0) * (1 + ds) * math.cos(beta) / fx
            b1 = ((l.l2 * math.sqrt(garma)) + b3 * x0 + b2 * fx * math.tan(beta)) / fx

            phi = math.atan(-a3 / c3)
            omega = math.asin(-b3)
            kappa = math.atan(b1 / b2)

            fout.write("【外方位元素】\n")
            fout.write(f"  Xs (mm): {format_number(line[0, 0])}\n")
            fout.write(f"  Ys (mm): {format_number(line[1, 0])}\n")
            fout.write(f"  Zs (mm): {format_number(line[2, 0])}\n")
            fout.write(f"  phi (度): {format_number(phi / PI * 180)}\n")
            fout.write(f"  omega (度): {format_number(omega / PI * 180)}\n")
            fout.write(f"  kappa (度): {format_number(kappa / PI * 180)}\n\n")

            # 畸变系数
            fout.write("【畸变系数】\n")
            fout.write(f"  k1: {format_number(l.k1)}\n")
            fout.write(f"  k2: {format_number(l.k2)}\n")
            fout.write(f"  k3: {format_number(l.k3)}\n")
            fout.write(f"  p1: {format_number(l.p1)}\n")
            fout.write(f"  p2: {format_number(l.p2)}\n\n")

            # 精度统计
            fout.write("【精度统计】\n")
            V = M @ x - W
            v = (V.T @ V)[0, 0]
            sigma = math.sqrt(v / (M.shape[0] - 16))
            fout.write(f"  单位权中误差 (mm): {format_number(sigma)}\n")
            fout.write(f"  单位权中误差 (像素): {format_number(sigma * 4272 / 22.2)}\n\n")

            # 各未知数的中误差
            fout.write("【L系数与畸变系数的中误差】\n")
            Q = np.linalg.inv(M.T @ M)
            fout.write(f"  L1: {format_number(sigma * math.sqrt(Q[0, 0]))}\n")
            fout.write(f"  L2: {format_number(sigma * math.sqrt(Q[1, 1]))}\n")
            fout.write(f"  L3: {format_number(sigma * math.sqrt(Q[2, 2]))}\n")
            fout.write(f"  L4: {format_number(sigma * math.sqrt(Q[3, 3]))}\n")
            fout.write(f"  L5: {format_number(sigma * math.sqrt(Q[4, 4]))}\n")
            fout.write(f"  L6: {format_number(sigma * math.sqrt(Q[5, 5]))}\n")
            fout.write(f"  L7: {format_number(sigma * math.sqrt(Q[6, 6]))}\n")
            fout.write(f"  L8: {format_number(sigma * math.sqrt(Q[7, 7]))}\n")
            fout.write(f"  L9: {format_number(sigma * math.sqrt(Q[8, 8]))}\n")
            fout.write(f"  L10: {format_number(sigma * math.sqrt(Q[9, 9]))}\n")
            fout.write(f"  L11: {format_number(sigma * math.sqrt(Q[10, 10]))}\n")
            fout.write(f"  k1: {format_number(sigma * math.sqrt(Q[11, 11]))}\n")
            fout.write(f"  k2: {format_number(sigma * math.sqrt(Q[12, 12]))}\n")
            fout.write(f"  k3: {format_number(sigma * math.sqrt(Q[13, 13]))}\n")
            fout.write(f"  p1: {format_number(sigma * math.sqrt(Q[14, 14]))}\n")
            fout.write(f"  p2: {format_number(sigma * math.sqrt(Q[15, 15]))}\n\n")

            # 内方位元素的中误差（基于误差传播）
            fout.write("【内方位元素的中误差】\n")
            # x0 的中误差近似
            var_x0 = (Q[0, 0] * l.l9**2 + Q[1, 1] * l.l10**2 + Q[2, 2] * l.l11**2 +
                      Q[8, 8] * l.l1**2 + Q[9, 9] * l.l2**2 + Q[10, 10] * l.l3**2) * garma**2
            # y0 的中误差近似
            var_y0 = (Q[4, 4] * l.l9**2 + Q[5, 5] * l.l10**2 + Q[6, 6] * l.l11**2 +
                      Q[8, 8] * l.l5**2 + Q[9, 9] * l.l6**2 + Q[10, 10] * l.l7**2) * garma**2
            # f 的中误差近似（通过fx的误差传播）
            var_f = (Q[0, 0] * (l.l1**2) + Q[1, 1] * (l.l2**2) + Q[2, 2] * (l.l3**2)) * garma * math.cos(beta)**2 / A
            
            fout.write(f"  x0 (mm): {format_number(sigma * math.sqrt(var_x0))}\n")
            fout.write(f"  y0 (mm): {format_number(sigma * math.sqrt(var_y0))}\n")
            fout.write(f"  f (mm): {format_number(sigma * math.sqrt(var_f))}\n\n")

            # 外方位元素的中误差（基于误差传播）
            fout.write("【外方位元素的中误差】\n")
            # Xs, Ys, Zs 的中误差（通过逆矩阵传播近似）
            ext_det = np.linalg.det(ext)
            var_Xs = (Q[0, 0] * (l.l6*l.l11 - l.l7*l.l10)**2 + 
                      Q[1, 1] * (l.l7*l.l9 - l.l5*l.l11)**2 + 
                      Q[2, 2] * (l.l5*l.l10 - l.l6*l.l9)**2 +
                      Q[4, 4] * (l.l3*l.l11 - l.l2*l.l10)**2 +
                      Q[5, 5] * (l.l1*l.l11 - l.l3*l.l9)**2 +
                      Q[6, 6] * (l.l2*l.l9 - l.l1*l.l10)**2 +
                      Q[8, 8] * (l.l2*l.l7 - l.l3*l.l6)**2 +
                      Q[9, 9] * (l.l3*l.l5 - l.l1*l.l7)**2 +
                      Q[10, 10] * (l.l1*l.l6 - l.l2*l.l5)**2) / ext_det**2
            var_Ys = (Q[3, 3] * (l.l6*l.l11 - l.l7*l.l10)**2 +
                      Q[0, 0] * (l.l7*l.l11)**2 + Q[2, 2] * (l.l6*l.l11)**2 +
                      Q[7, 7] * (l.l9*l.l11 - l.l10*l.l9)**2) / ext_det**2
            var_Zs = (Q[3, 3] * (l.l6*l.l10 - l.l5*l.l11)**2 +
                      Q[4, 4] * (l.l3*l.l10)**2 + Q[6, 6] * (l.l5*l.l10)**2) / ext_det**2
            
            fout.write(f"  Xs (mm): {format_number(sigma * math.sqrt(var_Xs))}\n")
            fout.write(f"  Ys (mm): {format_number(sigma * math.sqrt(var_Ys))}\n")
            fout.write(f"  Zs (mm): {format_number(sigma * math.sqrt(var_Zs))}\n")
            fout.write(f"  phi (度): {format_number(sigma * math.sqrt(Q[8, 8] + Q[9, 9] + Q[10, 10]) * 180 / PI)}\n")
            fout.write(f"  omega (度): {format_number(sigma * math.sqrt(Q[8, 8] + Q[9, 9] + Q[10, 10]) * 180 / PI)}\n")
            fout.write(f"  kappa (度): {format_number(sigma * math.sqrt(Q[8, 8] + Q[9, 9] + Q[10, 10]) * 180 / PI)}\n\n")

            # 像点观测值残差
            fout.write("【像点观测值残差（像素）】\n")
            fout.write(f"{'点号':<8} {'dx':<16} {'dy':<16}\n")
            itr = 0
            for pid in used:
                lx = V[2 * itr, 0] * 4272 / 22.2
                ly = V[2 * itr + 1, 0] * 2848 / 14.8
                fout.write(f"{pid:<8} {format_number(lx):<16} {format_number(ly):<16}\n")
                itr += 1

        print(f"DLT结果已保存到文件: {full_path}")
    except Exception as e:
        print(f"无法打开输出文件 {full_path}: {e}")


def getLi(l, pt, GCP, image_name="未知", max_iterations=100, convergence_threshold=1e-6):
    """迭代求解Li系数精确值"""
    sz = getSz(pt) - 3  # 减去3个检查点
    M = np.zeros((sz * 2, 16))
    W = np.zeros((sz * 2, 1))
    x = np.zeros((16, 1))

    fx = 0
    times = 0
    print(f"\n************* {image_name} 直接线性变换迭代 *************")
    print(f"最大迭代次数: {max_iterations}")
    print(f"收敛阈值: {convergence_threshold}")

    while times < max_iterations:
        itr = 0
        used = {}
        for pid in pt:
            if pt[pid].type and not pt[pid].ischeck:
                used[pid] = pt[pid]
                X = GCP[pid].X
                Y = GCP[pid].Y
                Z = GCP[pid].Z

                # 复制并转换
                pt_copy = PhotoPt(pt[pid].pid, pt[pid].x, pt[pid].y)
                pt_copy.convert()
                x_val = pt_copy.x
                y_val = pt_copy.y

                A = l.l9 * X + l.l10 * Y + l.l11 * Z + 1
                x0 = -(l.l1 * l.l9 + l.l2 * l.l10 + l.l3 * l.l11) / (l.l9 ** 2 + l.l10 ** 2 + l.l11 ** 2)
                y0 = -(l.l5 * l.l9 + l.l6 * l.l10 + l.l7 * l.l11) / (l.l9 ** 2 + l.l10 ** 2 + l.l11 ** 2)

                r = math.sqrt((x_val - x0) ** 2 + (y_val - y0) ** 2)

                M[2 * itr, 0] = -X / A
                M[2 * itr, 1] = -Y / A
                M[2 * itr, 2] = -Z / A
                M[2 * itr, 3] = -1.0 / A
                M[2 * itr, 8] = -X * x_val / A
                M[2 * itr, 9] = -Y * x_val / A
                M[2 * itr, 10] = -Z * x_val / A

                M[2 * itr + 1, 4] = -X / A
                M[2 * itr + 1, 5] = -Y / A
                M[2 * itr + 1, 6] = -Z / A
                M[2 * itr + 1, 7] = -1.0 / A
                M[2 * itr + 1, 8] = -X * y_val / A
                M[2 * itr + 1, 9] = -Y * y_val / A
                M[2 * itr + 1, 10] = -Z * y_val / A

                M[2 * itr, 11] = -(x_val - x0) * (r ** 2)
                M[2 * itr, 12] = -(x_val - x0) * (r ** 4)
                M[2 * itr, 13] = -(x_val - x0) * (r ** 6)
                M[2 * itr, 14] = -(2 * (x_val - x0) ** 2 + r ** 2)
                M[2 * itr, 15] = -(2 * (x_val - x0) * (y_val - y0))

                M[2 * itr + 1, 11] = -(y_val - y0) * (r ** 2)
                M[2 * itr + 1, 12] = -(y_val - y0) * (r ** 4)
                M[2 * itr + 1, 13] = -(y_val - y0) * (r ** 6)
                M[2 * itr + 1, 14] = -(2 * (x_val - x0) * (y_val - y0))
                M[2 * itr + 1, 15] = -(2 * (y_val - y0) ** 2 + r ** 2)

                W[2 * itr, 0] = x_val / A
                W[2 * itr + 1, 0] = y_val / A
                itr += 1

        # 求解方程
        x = np.linalg.inv(M.T @ M) @ M.T @ W

        l.l1 = x[0, 0]
        l.l2 = x[1, 0]
        l.l3 = x[2, 0]
        l.l4 = x[3, 0]
        l.l5 = x[4, 0]
        l.l6 = x[5, 0]
        l.l7 = x[6, 0]
        l.l8 = x[7, 0]
        l.l9 = x[8, 0]
        l.l10 = x[9, 0]
        l.l11 = x[10, 0]

        l.k1 = x[11, 0]
        l.k2 = x[12, 0]
        l.k3 = x[13, 0]
        l.p1 = x[14, 0]
        l.p2 = x[15, 0]

        times += 1
        tmp = Judge(l)
        if abs(tmp - fx) < convergence_threshold or int(tmp - fx) == 0:
            print(f"\n直接线性变换迭代次数: {times}")
            getLPrecision(M, x, W, sz * 2)
            ErrDisplay(M @ x - W, used)
            saveDLTResult(f"dlt_{image_name}.txt", image_name, l, M, x, W, used)
            break
        else:
            fx = tmp
    
    # 如果循环结束但未收敛
    if times >= max_iterations:
        print(f"\n✗ 直接线性变换未收敛！已达到最大迭代次数: {max_iterations}")
        print("将保存当前迭代结果...")
        getLPrecision(M, x, W, sz * 2)
        ErrDisplay(M @ x - W, used)
        saveDLTResult(f"dlt_{image_name}.txt", image_name, l, M, x, W, used)


def getElements(r, l):
    """从Li系数计算外方位元素"""
    garma = 1 / (l.l9 ** 2 + l.l10 ** 2 + l.l11 ** 2)
    x0 = -(l.l1 * l.l9 + l.l2 * l.l10 + l.l3 * l.l11) * garma
    y0 = -(l.l5 * l.l9 + l.l6 * l.l10 + l.l7 * l.l11) * garma

    A = (l.l1 ** 2 + l.l2 ** 2 + l.l3 ** 2) * garma - x0 ** 2
    B = (l.l5 ** 2 + l.l6 ** 2 + l.l7 ** 2) * garma - y0 ** 2
    C = (l.l1 * l.l5 + l.l2 * l.l6 + l.l3 * l.l7) * garma - x0 * y0

    ds = math.sqrt(A / B) - 1
    beta = math.sqrt(C ** 2 / A / B)
    beta = math.asin(-beta) if C > 0 else math.asin(beta)

    fx = math.cos(beta) * math.sqrt(A)
    fy = fx / (1 + ds)

    r.x0 = x0
    r.y0 = y0
    r.f = fx

    # 计算外方位线元素
    ext = np.array([
        [l.l1, l.l2, l.l3],
        [l.l5, l.l6, l.l7],
        [l.l9, l.l10, l.l11]
    ])
    vec = np.array([[-l.l4], [-l.l8], [-1]])
    line = np.linalg.inv(ext) @ vec

    r.Xs = line[0, 0]
    r.Ys = line[1, 0]
    r.Zs = line[2, 0]

    a3 = math.sqrt(garma) * l.l9
    b3 = math.sqrt(garma) * l.l10
    c3 = math.sqrt(garma) * l.l11
    a2 = (math.sqrt(garma) * l.l5 + a3 * math.sqrt(garma)) * (1 + ds) * math.cos(beta) / fx
    b2 = (l.l6 * math.sqrt(garma) + b3 * y0) * (1 + ds) * math.cos(beta) / fx
    b1 = ((l.l2 * math.sqrt(garma)) + b3 * x0 + b2 * fx * math.tan(beta)) / fx

    r.phi = math.atan(-a3 / c3)
    r.omega = math.asin(-b3)
    r.kappa = math.atan(b1 / b2)

    print("\n--------外方位元素(mm and degree)--------")
    print(f"{'Xs':<10} {'Ys':<10} {'Zs':<10}")
    print(f"{r.Xs:<12.6f} {r.Ys:<12.6f} {r.Zs:<12.6f}\n")

    print(f"{'phi':<10} {'omega':<10} {'kappa':<10}")
    print(f"{r.phi / PI * 180:<12.6f} {r.omega / PI * 180:<12.6f} {r.kappa / PI * 180:<12.6f}\n")

    print("--------内方位元素(mm)-------")
    print(f"{'x0':<10} {'y0':<10} {'f':<10}")
    print(f"{r.x0:<12.6f} {r.y0:<12.6f} {r.f:<12.6f}\n")

    print("--------畸变系数(k1,k2,k3,p1,p2)--------")
    print(f"{l.k1:.12f} {l.k2:.12f} {l.k3:.12f} {l.p1:.12f} {l.p2:.12f}")


def saveUnknownPoints(filename, Unknow, x_52, y_52, z_52):
    """保存待定点坐标到文件"""
    from data import format_number
    try:
        # 构建完整的输出路径
        full_path = os.path.join(_output_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as fout:
            fout.write("========== 待定点物方空间坐标 ==========\n\n")
            fout.write(f"{'点号':<8} {'X(mm)':<18} {'Y(mm)':<18} {'Z(mm)':<18} {'到52号点距离(mm)':<22}\n")
            fout.write("-" * 90 + "\n")

            for pt in Unknow:
                distance = math.sqrt((pt.X - x_52) ** 2 + (pt.Y - y_52) ** 2 + (pt.Z - z_52) ** 2)
                fout.write(f"{pt.pid:<8} {format_number(pt.X):<18} {format_number(pt.Y):<18} {format_number(pt.Z):<18} {format_number(distance):<22}\n")

            fout.write("\n========== 统计信息 ==========\n")
            fout.write(f"待定点总数: {len(Unknow)}\n")

            # 计算各坐标范围的统计
            x_coords = [pt.X for pt in Unknow]
            y_coords = [pt.Y for pt in Unknow]
            z_coords = [pt.Z for pt in Unknow]

            fout.write(f"X范围: {min(x_coords):.2f} ~ {max(x_coords):.2f} mm\n")
            fout.write(f"Y范围: {min(y_coords):.2f} ~ {max(y_coords):.2f} mm\n")
            fout.write(f"Z范围: {min(z_coords):.2f} ~ {max(z_coords):.2f} mm\n")

        print(f"\n待定点坐标已保存到文件: {full_path}")
    except Exception as e:
        print(f"无法保存待定点坐标到文件 {full_path}: {e}")


def getUnknow(l_pt, r_pt, l, r, convergence_threshold=1e-6):
    """计算未知点坐标"""
    l_un = []
    r_un = []
    Rl = RParams()
    Rr = RParams()
    getElements(Rl, l)
    getElements(Rr, r)

    for pid in l_pt:
        if not l_pt[pid].type and pid in r_pt:
            pt_copy = PhotoPt(l_pt[pid].pid, l_pt[pid].x, l_pt[pid].y)
            pt_copy.convert()
            l_un.append(pt_copy)

    for pid in r_pt:
        if not r_pt[pid].type and pid in l_pt:
            pt_copy = PhotoPt(r_pt[pid].pid, r_pt[pid].x, r_pt[pid].y)
            pt_copy.convert()
            r_un.append(pt_copy)

    print("\n*************未知点计算*************")
    print("未知点: ", end="")
    for pt in l_un:
        print(f"{pt.pid} ", end="")
    print("\n")
    print(f"收敛阈值: {convergence_threshold}")

    sz = len(l_un)
    # 畸变改正
    for i in range(sz):
        r_left = math.sqrt((l_un[i].x - Rl.x0) ** 2 + (l_un[i].y - Rl.y0) ** 2)
        dx_left = (l_un[i].x - Rl.x0) * (l.k1 * r_left ** 2 + l.k2 * r_left ** 4 + l.k3 * r_left ** 6) \
                  + l.p1 * (r_left ** 2 + 2 * (l_un[i].x - Rl.x0) ** 2) \
                  + 2 * l.p2 * (l_un[i].x - Rl.x0) * (l_un[i].y - Rl.y0)
        dy_left = (l_un[i].y - Rl.y0) * (l.k1 * r_left ** 2 + l.k2 * r_left ** 4 + l.k3 * r_left ** 6) \
                  + l.p2 * (r_left ** 2 + 2 * (l_un[i].y - Rl.y0) ** 2) \
                  + 2 * l.p1 * (l_un[i].x - Rl.x0) * (l_un[i].y - Rl.y0)

        r_right = math.sqrt((r_un[i].x - Rr.x0) ** 2 + (r_un[i].y - Rr.y0) ** 2)
        dx_right = (r_un[i].x - Rr.x0) * (r.k1 * r_right ** 2 + r.k2 * r_right ** 4 + r.k3 * r_right ** 6) \
                   + r.p1 * (r_right ** 2 + 2 * (r_un[i].x - Rr.x0) ** 2) \
                   + 2 * r.p2 * (r_un[i].x - Rr.x0) * (r_un[i].y - Rr.y0)
        dy_right = (r_un[i].y - Rr.y0) * (r.k1 * r_right ** 2 + r.k2 * r_right ** 4 + r.k3 * r_right ** 6) \
                   + r.p2 * (r_right ** 2 + 2 * (r_un[i].y - Rr.y0) ** 2) \
                   + 2 * r.p1 * (r_un[i].x - Rr.x0) * (r_un[i].y - Rr.y0)

        l_un[i].x += dx_left
        l_un[i].y += dy_left
        r_un[i].x += dx_right
        r_un[i].y += dy_right

    # 前方交会计算
    Unknow = []
    for i in range(sz):
        res = ObjectPt()

        # 初值计算
        init_A = np.array([
            [l.l1 + l.l9 * l_un[i].x, l.l2 + l.l10 * l_un[i].x, l.l3 + l.l11 * l_un[i].x],
            [l.l5 + l.l9 * l_un[i].y, l.l6 + l.l10 * l_un[i].y, l.l7 + l.l11 * l_un[i].y],
            [r.l1 + r.l9 * r_un[i].x, r.l2 + r.l10 * r_un[i].x, r.l3 + r.l11 * r_un[i].x],
            [r.l5 + r.l9 * r_un[i].y, r.l6 + r.l10 * r_un[i].y, r.l7 + r.l11 * r_un[i].y]
        ])
        init_L = np.array([
            [-(l.l4 + l_un[i].x)],
            [-(l.l8 + l_un[i].y)],
            [-(r.l4 + r_un[i].x)],
            [-(r.l8 + r_un[i].y)]
        ])
        init = np.linalg.inv(init_A.T @ init_A) @ init_A.T @ init_L

        res.X = init[0, 0]
        res.Y = init[1, 0]
        res.Z = init[2, 0]

        times = 0
        while times < 10:
            A_left = l.l9 * res.X + l.l10 * res.Y + l.l11 * res.Z + 1
            A_right = r.l9 * res.X + r.l10 * res.Y + r.l11 * res.Z + 1

            N = np.zeros((4, 3))
            Q = np.zeros((4, 1))

            N[0, :] = [-(l.l1 + l.l9 * l_un[i].x) / A_left,
                       -(l.l2 + l.l10 * l_un[i].x) / A_left,
                       -(l.l3 + l.l11 * l_un[i].x) / A_left]
            N[1, :] = [-(l.l5 + l.l9 * l_un[i].y) / A_left,
                       -(l.l6 + l.l10 * l_un[i].y) / A_left,
                       -(l.l7 + l.l11 * l_un[i].y) / A_left]
            N[2, :] = [-(r.l1 + r.l9 * r_un[i].x) / A_right,
                       -(r.l2 + r.l10 * r_un[i].x) / A_right,
                       -(r.l3 + r.l11 * r_un[i].x) / A_right]
            N[3, :] = [-(r.l5 + r.l9 * r_un[i].y) / A_right,
                       -(r.l6 + r.l10 * r_un[i].y) / A_right,
                       -(r.l7 + r.l11 * r_un[i].y) / A_right]

            Q[0, 0] = (l.l4 + l_un[i].x) / A_left
            Q[1, 0] = (l.l8 + l_un[i].y) / A_left
            Q[2, 0] = (r.l4 + r_un[i].x) / A_right
            Q[3, 0] = (r.l8 + r_un[i].y) / A_right

            val = np.linalg.inv(N.T @ N) @ N.T @ Q

            if abs(res.X - val[0, 0]) < convergence_threshold and abs(res.Y - val[1, 0]) < convergence_threshold and abs(res.Z - val[2, 0]) < convergence_threshold:
                res.pid = l_un[i].pid
                Unknow.append(res)
                break

            res.X = val[0, 0]
            res.Y = val[1, 0]
            res.Z = val[2, 0]
            times += 1
            if times >= 10:
                res.pid = l_un[i].pid
                Unknow.append(res)

    if len(Unknow) != sz:
        print("未知点计算失败")
        return

    # 找到点52作为参考点
    x_52 = y_52 = z_52 = 0
    for pt in Unknow:
        if pt.pid == 52:
            x_52 = pt.X
            y_52 = pt.Y
            z_52 = pt.Z
            break

    print(f"{'点号':<6} {'X':<12} {'Y':<12} {'Z':<18} {'到52号点距离/mm':<25}")
    for pt in Unknow:
        distance = math.sqrt((pt.X - x_52) ** 2 + (pt.Y - y_52) ** 2 + (pt.Z - z_52) ** 2)
        print(f"{pt.pid:<6} {-pt.Z:<12.6f} {pt.X:<12.6f} {pt.Y:<12.6f} {distance:<25.6f}")

    # 保存待定点坐标到文件
    saveUnknownPoints("unknown_points.txt", Unknow, x_52, y_52, z_52)


def getExtPre(l_pt, r_pt, GCP, l, r, convergence_threshold=1e-6):
    """检查点精度验证"""
    l_un = []
    r_un = []
    Rl = RParams()
    Rr = RParams()
    getElements(Rl, l)
    getElements(Rr, r)

    for pid in l_pt:
        if l_pt[pid].ischeck:
            pt_copy = PhotoPt(l_pt[pid].pid, l_pt[pid].x, l_pt[pid].y)
            pt_copy.convert()
            l_un.append(pt_copy)

    for pid in r_pt:
        if r_pt[pid].ischeck:
            pt_copy = PhotoPt(r_pt[pid].pid, r_pt[pid].x, r_pt[pid].y)
            pt_copy.convert()
            r_un.append(pt_copy)

    print("\n*************检查点验证*************")
    print("检查点: ", end="")
    for pt in r_un:
        print(f"{pt.pid} ", end="")
    print("\n")
    print(f"收敛阈值: {convergence_threshold}")

    sz = len(r_un)
    Check = []

    for i in range(sz):
        res = ObjectPt()

        init_A = np.array([
            [l.l1 + l.l9 * l_un[i].x, l.l2 + l.l10 * l_un[i].x, l.l3 + l.l11 * l_un[i].x],
            [l.l5 + l.l9 * l_un[i].y, l.l6 + l.l10 * l_un[i].y, l.l7 + l.l11 * l_un[i].y],
            [r.l1 + r.l9 * r_un[i].x, r.l2 + r.l10 * r_un[i].x, r.l3 + r.l11 * r_un[i].x],
            [r.l5 + r.l9 * r_un[i].y, r.l6 + r.l10 * r_un[i].y, r.l7 + r.l11 * r_un[i].y]
        ])
        init_L = np.array([
            [-(l.l4 + l_un[i].x)],
            [-(l.l8 + l_un[i].y)],
            [-(r.l4 + r_un[i].x)],
            [-(r.l8 + r_un[i].y)]
        ])
        init = np.linalg.inv(init_A.T @ init_A) @ init_A.T @ init_L

        res.X = init[0, 0]
        res.Y = init[1, 0]
        res.Z = init[2, 0]

        times = 0
        while times < 10:
            A_left = l.l9 * res.X + l.l10 * res.Y + l.l11 * res.Z + 1
            A_right = r.l9 * res.X + r.l10 * res.Y + r.l11 * res.Z + 1

            N = np.zeros((4, 3))
            Q = np.zeros((4, 1))

            N[0, :] = [-(l.l1 + l.l9 * l_un[i].x) / A_left,
                       -(l.l2 + l.l10 * l_un[i].x) / A_left,
                       -(l.l3 + l.l11 * l_un[i].x) / A_left]
            N[1, :] = [-(l.l5 + l.l9 * l_un[i].y) / A_left,
                       -(l.l6 + l.l10 * l_un[i].y) / A_left,
                       -(l.l7 + l.l11 * l_un[i].y) / A_left]
            N[2, :] = [-(r.l1 + r.l9 * r_un[i].x) / A_right,
                       -(r.l2 + r.l10 * r_un[i].x) / A_right,
                       -(r.l3 + r.l11 * r_un[i].x) / A_right]
            N[3, :] = [-(r.l5 + r.l9 * r_un[i].y) / A_right,
                       -(r.l6 + r.l10 * r_un[i].y) / A_right,
                       -(r.l7 + r.l11 * r_un[i].y) / A_right]

            Q[0, 0] = (l.l4 + l_un[i].x) / A_left
            Q[1, 0] = (l.l8 + l_un[i].y) / A_left
            Q[2, 0] = (r.l4 + r_un[i].x) / A_right
            Q[3, 0] = (r.l8 + r_un[i].y) / A_right

            val = np.linalg.inv(N.T @ N) @ N.T @ Q

            if abs(res.X - val[0, 0]) < convergence_threshold and abs(res.Y - val[1, 0]) < convergence_threshold and abs(res.Z - val[2, 0]) < convergence_threshold:
                res.pid = l_un[i].pid
                Check.append(res)
                break

            res.X = val[0, 0]
            res.Y = val[1, 0]
            res.Z = val[2, 0]
            times += 1
            if times >= 10:
                res.pid = l_un[i].pid
                Check.append(res)

    d_X = d_Y = d_Z = 0
    print("\n残差:")
    print(f"{'点号':<10} {'残差':<10}")
    for i in range(sz):
        dx = abs(GCP[l_un[i].pid].X - Check[i].X)
        dy = abs(GCP[l_un[i].pid].Y - Check[i].Y)
        dz = abs(GCP[l_un[i].pid].Z - Check[i].Z)
        print(f"{l_un[i].pid:<10} {dx:<10.6f} {dy:<10.6f} {dz:<10.6f}")
        d_X += dx
        d_Y += dy
        d_Z += dz

    d_X /= sz
    d_Y /= sz
    d_Z /= sz

    # 保存检查点验证结果
    saveCheckResult(Check, GCP, l_un)


def saveCheckResult(Check, GCP, l_un):
    """保存检查点验证结果到文件"""
    from data import format_number
    try:
        full_path = os.path.join(_output_dir, "check_point_verification.txt")
        with open(full_path, 'w', encoding='utf-8') as fout:
            fout.write("========== 检查点验证结果 ==========\n\n")
            fout.write("【检查点残差（mm）】\n")
            fout.write(f"{'点号':<8} {'X残差':<18} {'Y残差':<18} {'Z残差':<18} {'总残差':<18}\n")
            fout.write("-" * 80 + "\n")
            
            d_X = d_Y = d_Z = 0
            sz = len(Check)
            for i in range(sz):
                dx = abs(GCP[l_un[i].pid].X - Check[i].X)
                dy = abs(GCP[l_un[i].pid].Y - Check[i].Y)
                dz = abs(GCP[l_un[i].pid].Z - Check[i].Z)
                total = math.sqrt(dx**2 + dy**2 + dz**2)
                d_X += dx
                d_Y += dy
                d_Z += dz
                fout.write(f"{l_un[i].pid:<8} {format_number(dx):<18} {format_number(dy):<18} {format_number(dz):<18} {format_number(total):<18}\n")
            
            d_X /= sz
            d_Y /= sz
            d_Z /= sz
            
            fout.write("\n【外精度统计】\n")
            fout.write(f"  X方向平均残差: {format_number(d_X)} mm\n")
            fout.write(f"  Y方向平均残差: {format_number(d_Y)} mm\n")
            fout.write(f"  Z方向平均残差: {format_number(d_Z)} mm\n")
            fout.write(f"  平均总残差: {format_number(math.sqrt(d_X**2 + d_Y**2 + d_Z**2))} mm\n")
        
        print(f"\n检查点验证结果已保存到文件: {full_path}")
    except Exception as e:
        print(f"无法保存检查点验证结果到文件 {full_path}: {e}")


def ErrDisplay(err, pt):
    """显示残差"""
    print("\n控制点的残差值（单位为像元）：")
    itr = 0
    for pid in pt:
        if pt[pid].type:
            lx = err[2 * itr, 0] * 4272 / 22.2
            ly = err[2 * itr + 1, 0] * 2848 / 14.8
            print(f"Control {pid} lx: {lx:.6f} ly: {ly:.6f}")
            itr += 1


# 需要导入PhotoPt
from data import PhotoPt
