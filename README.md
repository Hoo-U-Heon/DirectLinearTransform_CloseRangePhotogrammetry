# 近景摄影测量 DLT 处理系统

基于直接线性变换（DLT）算法的近景摄影测量数据处理工具，支持控制点标定、像点量测、待定点三维坐标计算等功能。

## 功能特性

- **直接线性变换（DLT）**：双片DLT解算内外方位元素和畸变参数
- **检查点验证**：使用检查点评估计算精度
- **待定点计算**：计算目标点的三维空间坐标
- **结果输出**：自动保存DLT结果、检查点验证结果、待定点坐标到文件
- **GUI界面**：基于PySide6的图形化操作界面
- **数值格式化**：根据数值大小自动切换普通浮点与科学计数法显示

## 文件结构

```
Python/
├── data/                   # 数据目录
│   ├── input/              # 输入数据
│   │   ├── 控制场坐标.txt         # 控制场坐标文件
│   │   ├── IMG_0015_control_points.txt  # 左片像点量测文件
│   │   └── IMG_0027_control_points.txt  # 右片像点量测文件
│   └── output/             # 输出结果
│       ├── dlt_左片.txt           # 左片DLT结果
│       ├── dlt_右片.txt           # 右片DLT结果
│       ├── check_point_verification.txt  # 检查点验证结果
│       └── unknown_points.txt     # 待定点坐标
├── data.py                 # 数据结构定义
├── calculate.py            # DLT计算核心算法
├── DLT.py                  # 后方交会算法
├── main.py                 # 命令行入口
├── main_gui.py             # GUI图形界面入口
└── requirements.txt        # Python依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### GUI界面

```bash
python main_gui.py
```

### 命令行模式

```bash
python main.py
```

## 输入文件格式

### 控制场坐标文件（控制场坐标.txt）

```
点号 类型 X Y Z
```

- 点号：整数点号
- 类型：0为控制点，1为待定点，2为检查点
- X, Y, Z：物方空间坐标（单位：mm）

### 像点量测文件

```
点号 类型 x y
```

- 点号：整数点号
- 类型：0为控制点，1为待定点，2为检查点
- x, y：像点坐标（单位：mm）

## 输出文件说明

| 文件名 | 内容 |
|--------|------|
| dlt_左片.txt / dlt_右片.txt | DLT计算结果，包含L系数、内/外方位元素、畸变系数、精度统计 |
| check_point_verification.txt | 检查点验证结果，包含各检查点残差及外精度统计 |
| unknown_points.txt | 待定点三维坐标及到参考点距离 |

## 数值输出规则

- 当 `|value| < 0.01` 或 `|value| > 100000` 时使用科学计数法
- 其他情况使用普通浮点格式（6位小数）

## 技术栈

- Python 3.x
- NumPy - 数值计算
- PySide6 - GUI界面
