import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import readGCP, readMeasure, LParams
from calculate import getCheck, getInit, getLi, getExtPre, getUnknow


def main():
    # 文件路径设置
    GCPfile = "data/input/控制场坐标.txt"
    lMeasurefile = "data/input/IMG_0015_control_points.txt"
    rMeasurefile = "data/input/IMG_0027_control_points.txt"

    # 读取数据
    print("正在读取数据...")
    mp_GCP = readGCP(GCPfile)
    mp_lpt = readMeasure(lMeasurefile)
    mp_rpt = readMeasure(rMeasurefile)

    # ===== 直接线性变换 =====
    print("\n===== 直接线性变换 =====")
    getCheck(mp_lpt, mp_rpt)

    # 左片DLT
    print("\n--- 左片DLT ---")
    myL_l = getInit(mp_lpt, mp_GCP)
    getLi(myL_l, mp_lpt, mp_GCP, "左片")

    # 右片DLT
    print("\n--- 右片DLT ---")
    myL_r = getInit(mp_rpt, mp_GCP)
    getLi(myL_r, mp_rpt, mp_GCP, "右片")

    # 检查点验证
    getExtPre(mp_lpt, mp_rpt, mp_GCP, myL_l, myL_r)

    # 未知点计算
    getUnknow(mp_lpt, mp_rpt, myL_l, myL_r)


if __name__ == "__main__":
    main()
