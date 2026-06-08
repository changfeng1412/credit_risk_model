# 基于 XGBoost & LightGBM 信贷风控预测模型
## 项目简介
本项目基于**传统机器学习算法**搭建信贷风控二分类模型，通过 SQL 提取业务数据、Pandas 完成数据清洗与特征工程，分别使用 XGBoost、LightGBM 构建违约风险预测模型，结合准确率、AUC、召回率多维度评估模型，并输出可视化分析图表。模型可识别高违约风险用户，辅助金融机构风控决策、降低坏账率。

## 技术栈
- 编程语言：Python 3.9~3.11
- 数据抽取：MySQL + `pymysql`
- 数据处理：Pandas、NumPy
- 机器学习：XGBoost、LightGBM
- 模型评估：Scikit-learn
- 可视化：Matplotlib、Seaborn

## 完整目录结构
```
credit_risk_model/
├── data/                   # 本地备用数据集（csv格式，断网/无数据库时使用）
├── pics/                    # 可视化图表输出目录（自动生成）
├── credit_risk.py          # 主程序：数据抽取、清洗、建模、评估、可视化
└── requirements.txt         # 项目依赖清单
```

## 环境准备


### 依赖文件 requirements.txt
```txt
pandas==2.2.2
numpy==1.26.4

pymysql==1.1.0

xgboost==2.0.3
lightgbm==4.3.0

scikit-learn==1.4.2

matplotlib==3.8.4
seaborn==0.13.2
```

### 安装命令
进入项目根目录执行：
```bash
pip install -r requirements.txt
```


## 运行步骤
### 方式1：使用 MySQL 数据库
1. 按照上文表结构在 `credit_db` 中创建 `credit_user` 表，并导入业务数据；
2. 修改代码中 `DB_CONFIG` 里的账号、密码、数据库名；
3. 执行代码：
```bash
python credit_risk.py
```

### 方式2：使用本地 CSV（本地测试）
1. 注释代码中 `df = load_data_from_mysql()`；
2. 取消注释，改为 `df = load_data_from_csv()`；
3. 将 `credit_data.csv` 放入 `data` 文件夹；
4. 执行运行命令：
```bash
python credit_risk.py
```

### 运行输出说明
1. 控制台输出：数据加载状态、训练/测试集数量、两个模型的**准确率、AUC、召回率**；
2. 自动生成 `pics` 文件夹，内部保存 XGBoost、LightGBM 特征重要性图片。

## 核心流程
1. **数据采集**：通过 SQL 从业务数据库抽取用户信贷多维数据，无数据库则使用本地 CSV；
2. **数据清洗**：去重、填充缺失值、过滤异常样本，保证数据质量；
3. **特征工程**：剔除无关ID字段，分层划分训练集与测试集，保证样本分布一致；
4. **模型训练**：并行训练 XGBoost、LightGBM 两大主流树模型；
5. **模型评估**：使用风控核心指标（AUC、召回率）评价模型效果（风控场景召回率尤为重要）；
6. **可视化分析**：输出特征重要性，直观展示哪些因素对用户违约影响最大。

## 常见问题与解决方案
1. **MySQL 连接报错**\
   检查数据库服务是否启动、IP/账号/密码是否正确；\
   本地无数据库直接切换为 CSV 数据源。

2. **图表中文乱码**
   代码已内置中文配置，若仍乱码，补充安装中文字体或修改 `plt.rcParams` 字体名称。

3. **模型效果差、AUC/召回率过低**\
   扩充数据集样本量，保证正负样本均衡；\
   增加特征工程：特征组合、特征筛选、归一化；\
   调整模型超参（学习率、树数量、最大深度）。

4. **缺失值/异常值报错**
   检查原始数据，手动清理脏数据，或修改填充、过滤逻辑。
