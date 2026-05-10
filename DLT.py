import numpy as np
import math
from data import RParams, PI


def getR(phi, omega, kappa):
    """计算旋转矩阵"""
    R_phi = np.zeros((3, 3))
    R_omega = np.zeros((3, 3))
    R_kappa = np.zeros((3, 3))

    R_phi[0, 0] = math.cos(phi)
    R_phi[0, 2] = -math.sin(phi)
    R_phi[1, 1] = 1
    R_phi[2, 0] = math.sin(phi)
    R_phi[2, 2] = math.cos(phi)

    R_omega[0, 0] = 1
    R_omega[1, 1] = math.cos(omega)
    R_omega[1, 2] = -math.sin(omega)
    R_omega[2, 1] = math.sin(omega)
    R_omega[2, 2] = math.cos(omega)

    R_kappa[0, 0] = math.cos(kappa)
    R_kappa[0, 1] = -math.sin(kappa)
    R_kappa[1, 0] = math.sin(kappa)
    R_kappa[1, 1] = math.cos(kappa)
    R_kappa[2, 2] = 1

    R = R_phi @ R_omega @ R_kappa
    return R


def getSz(pt):
    """获取GCP点数量"""
    sz = 0
    for pid in pt:
        if pt[pid].type:
            sz += 1
    return sz


def RDisplay(para):
    """显示外方位元素"""
    print("\n--------外方位元素(mm and degree)--------")
    print(f"{'Xs':<10} {'Ys':<10} {'Zs':<10}")
    print(f"{para.Xs:<12.6f} {para.Ys:<12.6f} {para.Zs:<12.6f}\n")

    print(f"{'phi':<10} {'omega':<10} {'kappa':<10}")
    print(f"{para.phi / PI * 180:<12.6f} {para.omega / PI * 180:<12.6f} {para.kappa / PI * 180:<12.6f}\n")

    print("--------内方位元素(mm)-------")
    print(f"{'x0':<10} {'y0':<10} {'f':<10}")
    print(f"{para.x0:<12.6f} {para.y0:<12.6f} {para.f:<12.6f}\n")

    print("--------畸变系数(k1,k2,k3,p1,p2)--------")
    print(f"{para.k1:.12f} {para.k2:.12f} {para.k3:.12f} {para.p1:.12f} {para.p2:.12f}")


def getRPrecision(A, X, L, num):
    """计算后方交会精度"""
    v = A @ X - L
    sigma = math.sqrt((v.T @ v)[0, 0]) / math.sqrt(num - 14)
    mm2pixel = 4272 / 22.2

    print(f"\n单位权中误差(sigma)/pixel: {sigma * mm2pixel:.6f}")
    print(f"单位权中误差(sigma)/mm: {sigma:.6f}\n")

    Q = np.linalg.inv(A.T @ A)

    print("未知数的中误差精度：")
    print(f"Xs: {sigma * math.sqrt(Q[0, 0]):.6f}")
    print(f"Ys: {sigma * math.sqrt(Q[1, 1]):.6f}")
    print(f"Zs: {sigma * math.sqrt(Q[2, 2]):.6f}")
    print(f"phi: {sigma * math.sqrt(Q[3, 3]) / PI * 180:.6f}")
    print(f"omega: {sigma * math.sqrt(Q[4, 4]) / PI * 180:.6f}")
    print(f"kappa: {sigma * math.sqrt(Q[5, 5]) / PI * 180:.6f}")
    print(f"f: {sigma * math.sqrt(Q[6, 6]):.6f}")
    print(f"x0: {sigma * math.sqrt(Q[7, 7]):.6f}")
    print(f"y0: {sigma * math.sqrt(Q[8, 8]):.6f}")
    print(f"k1: {sigma * math.sqrt(Q[9, 9]):.6f}")
    print(f"k2: {sigma * math.sqrt(Q[10, 10]):.6f}")
    print(f"k3: {sigma * math.sqrt(Q[11, 11]):.6f}")
    print(f"p1: {sigma * math.sqrt(Q[12, 12]):.6f}")
    print(f"p2: {sigma * math.sqrt(Q[13, 13]):.6f}")


def ErrDisplay(l, pt):
    """显示残差"""
    print("\n控制点的残差值（单位为像元）：")
    itr = 0
    for pid in pt:
        if pt[pid].type:
            lx = l[2 * itr, 0] * 4272 / 22.2
            ly = l[2 * itr + 1, 0] * 2848 / 14.8
            print(f"Control {pid} lx: {lx:.6f} ly: {ly:.6f}")
            itr += 1


def saveResectionResult(filename, image_name, mypara, A, x, l, pt):
    """保存后方交会结果到文件"""
    from data import format_number
    try:
        with open(filename, 'w', encoding='utf-8') as fout:
            fout.write(f"========== {image_name} 单片后方交会结果 ==========\n\n")

            # 外方位元素
            fout.write("【外方位元素】\n")
            fout.write(f"  Xs (mm): {format_number(mypara.Xs)}\n")
            fout.write(f"  Ys (mm): {format_number(mypara.Ys)}\n")
            fout.write(f"  Zs (mm): {format_number(mypara.Zs)}\n")
            fout.write(f"  phi (度): {format_number(mypara.phi / PI * 180)}\n")
            fout.write(f"  omega (度): {format_number(mypara.omega / PI * 180)}\n")
            fout.write(f"  kappa (度): {format_number(mypara.kappa / PI * 180)}\n\n")

            # 内方位元素
            fout.write("【内方位元素】\n")
            fout.write(f"  x0 (mm): {format_number(mypara.x0)}\n")
            fout.write(f"  y0 (mm): {format_number(mypara.y0)}\n")
            fout.write(f"  f (mm): {format_number(mypara.f)}\n\n")

            # 畸变系数
            fout.write("【畸变系数】\n")
            fout.write(f"  k1: {format_number(mypara.k1)}\n")
            fout.write(f"  k2: {format_number(mypara.k2)}\n")
            fout.write(f"  k3: {format_number(mypara.k3)}\n")
            fout.write(f"  p1: {format_number(mypara.p1)}\n")
            fout.write(f"  p2: {format_number(mypara.p2)}\n\n")

            # 精度统计
            fout.write("【精度统计】\n")
            v = A @ x - l
            sigma = math.sqrt((v.T @ v)[0, 0]) / math.sqrt(l.shape[0] - 14)
            fout.write(f"  单位权中误差 (mm): {format_number(sigma)}\n")
            fout.write(f"  单位权中误差 (像素): {format_number(sigma * 4272 / 22.2)}\n\n")

            # 各未知数的中误差
            fout.write("【各未知数的中误差】\n")
            Q = np.linalg.inv(A.T @ A)
            fout.write(f"  Xs: {format_number(sigma * math.sqrt(Q[0, 0]))} mm\n")
            fout.write(f"  Ys: {format_number(sigma * math.sqrt(Q[1, 1]))} mm\n")
            fout.write(f"  Zs: {format_number(sigma * math.sqrt(Q[2, 2]))} mm\n")
            fout.write(f"  phi: {format_number(sigma * math.sqrt(Q[3, 3]) / PI * 180)} 度\n")
            fout.write(f"  omega: {format_number(sigma * math.sqrt(Q[4, 4]) / PI * 180)} 度\n")
            fout.write(f"  kappa: {format_number(sigma * math.sqrt(Q[5, 5]) / PI * 180)} 度\n")
            fout.write(f"  f: {format_number(sigma * math.sqrt(Q[6, 6]))} mm\n")
            fout.write(f"  x0: {format_number(sigma * math.sqrt(Q[7, 7]))} mm\n")
            fout.write(f"  y0: {format_number(sigma * math.sqrt(Q[8, 8]))} mm\n")
            fout.write(f"  k1: {format_number(sigma * math.sqrt(Q[9, 9]))}\n")
            fout.write(f"  k2: {format_number(sigma * math.sqrt(Q[10, 10]))}\n")
            fout.write(f"  k3: {format_number(sigma * math.sqrt(Q[11, 11]))}\n")
            fout.write(f"  p1: {format_number(sigma * math.sqrt(Q[12, 12]))}\n")
            fout.write(f"  p2: {format_number(sigma * math.sqrt(Q[13, 13]))}\n\n")

            # 像点观测值残差
            fout.write("【像点观测值残差（像素）】\n")
            fout.write(f"{'点号':<8} {'dx':<16} {'dy':<16}\n")
            itr = 0
            for pid in pt:
                if pt[pid].type:
                    lx = (A @ x - l)[2 * itr, 0] * 4272 / 22.2
                    ly = (A @ x - l)[2 * itr + 1, 0] * 2848 / 14.8
                    fout.write(f"{pid:<8} {format_number(lx):<16} {format_number(ly):<16}\n")
                    itr += 1

        print(f"后方交会结果已保存到文件: {filename}")
    except Exception as e:
        print(f"无法打开输出文件 {filename}: {e}")


def resection(pt, GCP, mypara, image_name="未知", max_iterations=100):
    """后方交会主函数"""
    print(f"\n************* {image_name} 空间后方交会 *************")
    sz = getSz(pt)
    print(f"控制点数量: {sz}")
    print(f"最大迭代次数: {max_iterations}")

    iterations = 0
    dxmax = 1.0

    # 输出初始参数
    print(f"初始参数: Xs={mypara.Xs:.2f}, Ys={mypara.Ys:.2f}, Zs={mypara.Zs:.2f}")
    print(f"初始姿态: phi={mypara.phi/PI*180:.2f}, omega={mypara.omega/PI*180:.2f}, kappa={mypara.kappa/PI*180:.2f}")

    while iterations < max_iterations:
        iterations += 1
        phi = mypara.phi
        omega = mypara.omega
        kappa = mypara.kappa
        Xs = mypara.Xs
        Ys = mypara.Ys
        Zs = mypara.Zs
        f = mypara.f

        R = getR(phi, omega, kappa)

        A = np.zeros((2 * sz, 14))
        l = np.zeros((2 * sz, 1))
        used = {}
        itr = 0

        for pid in pt:
            if pt[pid].type:
                # 获取控制点坐标
                X = GCP[pid].X
                Y = GCP[pid].Y
                Z = GCP[pid].Z

                # 转换到像空间辅助坐标系
                xyz = np.array([[X - Xs], [Y - Ys], [Z - Zs]])
                xyz = np.linalg.inv(R) @ xyz

                X_ = xyz[0, 0]
                Y_ = xyz[1, 0]
                Z_ = xyz[2, 0]

                # 复制并转换像点坐标
                pt_copy = PhotoPt(pt[pid].pid, pt[pid].x, pt[pid].y)
                pt_copy.convert()
                x = pt_copy.x - mypara.x0
                y = pt_copy.y - mypara.y0
                r = math.sqrt(x ** 2 + y ** 2)

                # 构建系数矩阵A
                # 外方位元素的偏导数
                A[2 * itr, 0] = 1 / Z_ * (R[0, 0] * f + R[2, 0] * x)
                A[2 * itr, 1] = 1 / Z_ * (R[0, 1] * f + R[2, 1] * x)
                A[2 * itr, 2] = 1 / Z_ * (R[0, 2] * f + R[2, 2] * x)
                A[2 * itr, 3] = y * math.sin(omega) - (x / f * (x * math.cos(kappa) - y * math.sin(kappa)) + f * math.cos(kappa)) * math.cos(omega)
                A[2 * itr, 4] = -f * math.sin(kappa) - x / f * (x * math.sin(kappa) + y * math.cos(kappa))
                A[2 * itr, 5] = y
                A[2 * itr, 6] = x / f
                A[2 * itr, 7] = 1
                A[2 * itr, 8] = 0
                A[2 * itr, 9] = -x * (r ** 2)
                A[2 * itr, 10] = -x * (r ** 4)
                A[2 * itr, 11] = -x * (r ** 6)
                A[2 * itr, 12] = -(r ** 2 + 2 * x ** 2)
                A[2 * itr, 13] = -2 * x * y

                A[2 * itr + 1, 0] = 1 / Z_ * (R[1, 0] * f + R[2, 0] * y)
                A[2 * itr + 1, 1] = 1 / Z_ * (R[1, 1] * f + R[2, 1] * y)
                A[2 * itr + 1, 2] = 1 / Z_ * (R[1, 2] * f + R[2, 2] * y)
                A[2 * itr + 1, 3] = -x * math.sin(omega) - (y / f * (x * math.cos(kappa) - y * math.sin(kappa)) - f * math.sin(kappa)) * math.cos(omega)
                A[2 * itr + 1, 4] = -f * math.cos(kappa) - y / f * (x * math.sin(kappa) + y * math.cos(kappa))
                A[2 * itr + 1, 5] = -x
                A[2 * itr + 1, 6] = y / f
                A[2 * itr + 1, 7] = 0
                A[2 * itr + 1, 8] = 1
                A[2 * itr + 1, 9] = -y * (r ** 2)
                A[2 * itr + 1, 10] = -y * (r ** 4)
                A[2 * itr + 1, 11] = -y * (r ** 6)
                A[2 * itr + 1, 12] = -2 * x * y
                A[2 * itr + 1, 13] = -(r ** 2 + 2 * y ** 2)

                # 构建常数项l（包含畸变改正）
                # 根据共线方程：像点观测值 - 像点计算值 = V
                # 计算值 = -f * X_/Z_ （注意负号！）
                # 误差方程：V = l - A * dx
                # 所以：l = 观测值 - 计算值 = (x + dx) - (-f * X_/Z_) = x + dx + f * X_/Z_
                dx = x * (mypara.k1 * r ** 2 + mypara.k2 * r ** 4 + mypara.k3 * r ** 6) \
                     + mypara.p1 * (r ** 2 + 2 * x ** 2) + 2 * mypara.p2 * x * y
                dy = y * (mypara.k1 * r ** 2 + mypara.k2 * r ** 4 + mypara.k3 * r ** 6) \
                     + mypara.p2 * (r ** 2 + 2 * y ** 2) + 2 * mypara.p1 * x * y

                # 常数项：包含畸变改正的观测值 - 投影计算值
                l[2 * itr, 0] = (x + dx) + f * X_ / Z_
                l[2 * itr + 1, 0] = (y + dy) + f * Y_ / Z_

                used[pid] = pt[pid]
                itr += 1

        # 求解法方程（使用伪逆提高稳定性）
        try:
            ATA = A.T @ A
            cond_num = np.linalg.cond(ATA)
            if cond_num > 1e10:
                lambda_reg = 1e-6 * np.max(np.diag(ATA))
                ATA_reg = ATA + lambda_reg * np.eye(14)
                x = np.linalg.inv(ATA_reg) @ (A.T @ l)
            else:
                x = np.linalg.inv(ATA) @ (A.T @ l)
        except np.linalg.LinAlgError:
            lambda_reg = 1e-4 * np.eye(14)
            x = np.linalg.lstsq(A, l, rcond=lambda_reg)[0]

        dxmax = np.max(np.abs(x))

        # 打印每次迭代的详细信息
        print(f"迭代 {iterations}: dxmax = {dxmax:.6e}")
        print(f"  参数增量: dXs={x[0,0]:.6e}, dYs={x[1,0]:.6e}, dZs={x[2,0]:.6e}")
        print(f"  姿态增量: dPhi={x[3,0]:.6e}, dOmega={x[4,0]:.6e}, dKappa={x[5,0]:.6e}")

        # 更新参数
        mypara.Xs += x[0, 0]
        mypara.Ys += x[1, 0]
        mypara.Zs += x[2, 0]
        mypara.phi += x[3, 0]
        mypara.omega += x[4, 0]
        mypara.kappa += x[5, 0]
        mypara.f += x[6, 0]
        mypara.x0 += x[7, 0]
        mypara.y0 += x[8, 0]
        mypara.k1 += x[9, 0]
        mypara.k2 += x[10, 0]
        mypara.k3 += x[11, 0]
        mypara.p1 += x[12, 0]
        mypara.p2 += x[13, 0]

        print(f"  更新后: Xs={mypara.Xs:.2f}, Ys={mypara.Ys:.2f}, Zs={mypara.Zs:.2f}")

        if dxmax < 1e-4:
            # 输出收敛状态
            print(f"✓ 后方交会收敛成功！迭代次数: {iterations}, 收敛精度: {dxmax:.6e}")
            
            RDisplay(mypara)
            getRPrecision(A, x, l, sz * 2)
            ErrDisplay(A @ x - l, used)
            saveResectionResult(f"resection_{image_name}.txt", image_name, mypara, A, x, l, used)
            break
    
    # 如果循环结束但未收敛
    if iterations >= max_iterations and dxmax >= 1e-4:
        print(f"✗ 后方交会未收敛！已达到最大迭代次数: {max_iterations}, 当前精度: {dxmax:.6e}")
        print("将保存当前迭代结果...")
        RDisplay(mypara)
        getRPrecision(A, x, l, sz * 2)
        ErrDisplay(A @ x - l, used)
        saveResectionResult(f"resection_{image_name}.txt", image_name, mypara, A, x, l, used)


# 需要导入PhotoPt
from data import PhotoPt
