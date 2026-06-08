import os
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ===================== 全局配置 =====================
# 可视化图片保存目录
PIC_SAVE_DIR = "./pics"
os.makedirs(PIC_SAVE_DIR, exist_ok=True)

# MySQL数据库配置（有数据库时使用）
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123456",
    "database": "credit_db",
    "charset": "utf8"
}
# 本地CSV路径（无数据库时使用）
CSV_PATH = "./data/credit_data.csv"
# 训练测试集划分比例
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ===================== 1. 数据读取函数 =====================
def load_data_from_mysql():
    """从MySQL数据库读取信贷数据"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        sql = """
        SELECT user_id, age, income, loan_count, overdue, label 
        FROM credit_user;
        """
        df = pd.read_sql(sql, conn)
        conn.close()
        print("✅ 数据从MySQL读取成功")
        return df
    except Exception as e:
        print(f"❌ 数据库连接失败：{e}，切换为本地CSV读取")
        return load_data_from_csv()

def load_data_from_csv():
    """从本地CSV文件读取数据"""
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print("✅ 数据从CSV文件读取成功")
    return df

# ===================== 2. 数据清洗 & 特征工程 =====================
def data_preprocess(df):
    """数据清洗、异常值处理、特征划分"""
    # 1. 去除重复行
    df = df.drop_duplicates()
    # 2. 缺失值填充（均值填充）
    df = df.fillna(df.mean(numeric_only=True))
    # 3. 过滤异常数据：年龄≥18、收入大于0
    df = df[(df["age"] >= 18) & (df["income"] > 0)]

    # 4. 划分特征X 和 标签y，删除无关ID字段
    X = df.drop(["user_id", "label"], axis=1)
    y = df["label"]

    # 5. 划分训练集、测试集（分层抽样，保证正负样本比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")
    return X_train, X_test, y_train, y_test, X.columns

# ===================== 3. 模型训练与评估 =====================
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """训练XGBoost、LightGBM并输出评估指标"""
    # -------- 训练 XGBoost --------
    print("\n===== 开始训练 XGBoost 模型 =====")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # XGBoost 指标
    print("XGBoost 模型评估指标：")
    print(f"准确率: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"AUC值: {roc_auc_score(y_test, y_prob_xgb):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_xgb):.4f}")

    # -------- 训练 LightGBM --------
    print("\n===== 开始训练 LightGBM 模型 =====")
    lgb_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # LightGBM 指标
    print("LightGBM 模型评估指标：")
    print(f"准确率: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print(f"AUC值: {roc_auc_score(y_test, y_prob_lgb):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_lgb):.4f}")

    return xgb_model, lgb_model

# ===================== 4. 特征重要性可视化 =====================
def plot_feature_importance(model, feature_names, pic_name):
    """绘制特征重要性柱状图"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=feature_names)
    plt.title("特征重要性排序", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(PIC_SAVE_DIR, pic_name)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ 可视化图表已保存至：{save_path}")

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    # 1. 读取数据（自动切换数据库/CSV）
    df = load_data_from_mysql()

    # 2. 数据预处理 + 划分数据集
    X_train, X_test, y_train, y_test, feature_names = data_preprocess(df)

    # 3. 训练模型并评估
    xgb_model, lgb_model = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # 4. 绘制特征重要性图
    plot_feature_importance(xgb_model, feature_names, "xgb_feature_importance.png")
    plot_feature_importance(lgb_model, feature_names, "lgb_feature_importance.png")

    print("\n===== 信贷风控模型项目执行完毕 =====")