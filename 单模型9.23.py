import concurrent.futures
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =================================================================
# 1. 打印美化辅助工具
# =================================================================
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def print_line(char="=", length=75):
    print(char * length)


def print_step(step, total, msg):
    print(f"\n[{step}/{total}] ❯❯❯ {msg}")


# =================================================================
# 2. 模型选择 (在这里切换你想跑的单模型！！！)
# =================================================================
MODEL_TYPE = "MLP"  # 选项: "RF", "MLP", "XGB"


def get_single_model(model_type):
    """根据选择返回对应的单模型实例"""
    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1
        )

    elif model_type == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(12, 24),
            alpha=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )

    elif model_type == "XGB":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(
            "不支持的模型类型，请选择 'RF', 'MLP', 或 'XGB'"
        )


# =================================================================
# 3. 并行任务单元 (独立采样 + 单模型训练)
# =================================================================
def run_single_task(sampler_name, X_raw, y_raw, model_type):

    t0 = time.time()

    # --- A. 初始化过采样器 ---
    if sampler_name == "SMOTE":

        over_s = SMOTE(
            sampling_strategy=0.3,
            random_state=42,
            k_neighbors=6
        )

    elif sampler_name == "BorderlineSMOTE":

        over_s = BorderlineSMOTE(
            sampling_strategy=0.3,
            random_state=42,
            k_neighbors=10,
            m_neighbors=10
        )

    elif sampler_name == "ADASYN":

        over_s = ADASYN(
            sampling_strategy=0.3,
            random_state=42,
            n_neighbors=5
        )

    else:
        raise ValueError(
            f"不支持的采样方法: {sampler_name}"
        )

    # --- B. 记录原始数据 ---
    n_pos_raw = np.sum(y_raw == 1)
    n_neg_raw = np.sum(y_raw == 0)

    # --- C. 第一步：过采样 ---
    X_over, y_over = over_s.fit_resample(
        X_raw,
        y_raw
    )

    n_pos_over = np.sum(y_over == 1)
    n_neg_over = np.sum(y_over == 0)

    # --- D. 第二步：随机欠采样 ---
    under_s = RandomUnderSampler(
        sampling_strategy=1.0,
        random_state=42
    )

    X_bal, y_bal = under_s.fit_resample(
        X_over,
        y_over
    )

    n_pos_final = np.sum(y_bal == 1)
    n_neg_final = np.sum(y_bal == 0)

    # --- E. 打印采样报告 ---
    report = (
        f"\n[线程报告: {sampler_name}]\n"
        f"  1. 原始样本 -> 滑坡(1): {n_pos_raw}, "
        f"非滑坡(0): {n_neg_raw}\n"
        f"  2. 过采样后 -> 滑坡(1): {n_pos_over} (增加), "
        f"非滑坡(0): {n_neg_over} (未变)\n"
        f"  3. 欠采样后 -> 滑坡(1): {n_pos_final} (未变), "
        f"非滑坡(0): {n_neg_final} (减少)\n"
        f"  -> 最终比例 1:1, 总样本: {len(y_bal)}"
    )

    print(report)

    # --- F. 划分训练集和测试集 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.3,
        random_state=42,
        stratify=y_bal
    )

    # --- G. 标准化 ---
    sc = StandardScaler()

    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    X_al_s = sc.transform(X_raw)

    # --- H. 单模型训练与预测 ---
    model = get_single_model(model_type)

    model.fit(
        X_tr_s,
        y_train
    )

    test_p = model.predict_proba(
        X_te_s
    )[:, 1]

    all_p = model.predict_proba(
        X_al_s
    )[:, 1]

    y_pred = (
        test_p >= 0.5
    ).astype(int)

    fpr, tpr, _ = roc_curve(
        y_test,
        test_p
    )

    # =================================================================
    # 保存 test_p 和 y_test
    # =================================================================
    test_result_df = pd.DataFrame({
        "sampler": sampler_name,
        "y_test": np.asarray(y_test),
        "test_p": test_p
    })

    return {
        "name": sampler_name,

        "auc": roc_auc_score(
            y_test,
            test_p
        ),

        "f1": f1_score(
            y_test,
            y_pred
        ),

        "recall": recall_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred
        ),

        "fpr": fpr,
        "tpr": tpr,
        "all_prob": all_p,

        # 测试集真实标签和预测概率
        "y_test": np.asarray(y_test),
        "test_p": test_p,

        "test_result_df": test_result_df,

        "time": time.time() - t0
    }


# =================================================================
# 4. 绘图函数
# =================================================================
def plot_roc_curves_scientific(
    results,
    model_type,
    save_path=None
):

    plt.figure(
        figsize=(12, 10),
        dpi=600
    )

    colors = [
        '#E41A1C',
        '#9B30FF',
        '#4DAF4A'
    ]

    line_styles = [
        '-',
        '--',
        ':'
    ]

    line_widths = [
        3,
        3,
        3
    ]

    for idx, res in enumerate(results):

        plt.plot(
            res['fpr'],
            res['tpr'],
            color=colors[idx],
            linestyle=line_styles[idx],
            linewidth=line_widths[idx],
            label=f"{res['name']} (AUC = {res['auc']:.4f})"
        )

    plt.xlim([
        -0.01,
        1.02
    ])

    plt.ylim([
        -0.01,
        1.02
    ])

    ticks = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    plt.xticks(
        ticks,
        fontsize=36,
        fontfamily='Times New Roman'
    )

    plt.yticks(
        ticks,
        fontsize=36,
        fontfamily='Times New Roman'
    )

    plt.xlabel(
        '假正率',
        fontsize=35,
        fontweight='bold',
        fontfamily='SimHei'
    )

    plt.ylabel(
        '真正率',
        fontsize=35,
        fontweight='bold',
        fontfamily='SimHei'
    )

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    legend = plt.legend(
        loc='lower right',
        fontsize=30,
        framealpha=0.95,
        prop={
            'family': 'Times New Roman',
            'size': 30
        }
    )

    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')

    plt.grid(
        True,
        alpha=0.3,
        linestyle='--'
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches='tight',
            facecolor='white'
        )

        print(
            f" [✔] ROC曲线图已导出: {save_path}"
        )

    plt.show()


# =================================================================
# 5. 主程序
# =================================================================
def main():

    print_line()

    print(
        f"      滑坡易发性建模: 单模型 [{MODEL_TYPE}]"
    )

    print_line()

    print_step(
        1,
        4,
        "正在读取原始数据源..."
    )

    data_path = (
        r"C:\Users\knight\Desktop\dian3.csv"
    )

    df = pd.read_csv(
        data_path,
        encoding="gbk"
    )

    X_raw = df.drop(
        columns=[
            "huapo",
            "x",
            "y"
        ]
    )

    y_raw = df["huapo"]

    print_step(
        2,
        4,
        f"启动并行重采样与 "
        f"{MODEL_TYPE} 模型训练..."
    )

    samplers = [
        "SMOTE",
        "BorderlineSMOTE",
        "ADASYN"
    ]

    results = []

    # 并行训练三个采样策略
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=3
    ) as executor:

        futures = {
            executor.submit(
                run_single_task,
                s,
                X_raw,
                y_raw,
                MODEL_TYPE
            ): s
            for s in samplers
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            res = future.result()

            results.append(res)

            print(
                f" [任务完成] {res['name']} | "
                f"{MODEL_TYPE} AUC: "
                f"{res['auc']:.4f}"
            )

    # =================================================================
    # 新增：按照 AUC 从高到低排序
    # 仅影响最终 ROC 图中三种采样策略的显示顺序
    # =================================================================
    results.sort(
        key=lambda x: x['auc'],
        reverse=True
    )

    print_step(
        3,
        4,
        "生成对比图表..."
    )

    # 文件名自动带上模型名字
    plot_path = (
        rf"E:\sy\caoying_模型"
        rf"\{MODEL_TYPE}_Scientific_ROC9222.png"
    )

    plot_roc_curves_scientific(
        results,
        MODEL_TYPE,
        save_path=plot_path
    )

    # =================================================================
    # 保存每个采样方法的 test_p 和 y_test
    # =================================================================
    print_step(
        4,
        4,
        "保存预测概率数据..."
    )

    # 原始数据预测概率
    df_out = df.copy()

    for r in results:

        df_out[
            f"prob_{MODEL_TYPE}_{r['name']}"
        ] = r['all_prob']

    # 保存原始数据 + 全量预测概率
    csv_path = (
        rf"E:\sy\caoying_模型"
        rf"\{MODEL_TYPE}_MultiSampler_Results_Scientific9222.csv"
    )

    df_out.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f" [✔] 原始数据预测概率已保存: "
        f"{csv_path}"
    )

    # =================================================================
    # 合并保存所有采样方法的测试集 y_test 和 test_p
    # =================================================================
    test_results_list = []

    for r in results:

        test_df = pd.DataFrame({
            "sampler": r["name"],
            "y_test": r["y_test"],
            "test_p": r["test_p"]
        })

        test_results_list.append(
            test_df
        )

    # 合并三个采样方法的测试结果
    test_results_all = pd.concat(
        test_results_list,
        ignore_index=True
    )

    # 增加模型名称列
    test_results_all.insert(
        0,
        "model",
        MODEL_TYPE
    )

    # 保存测试集真实标签和预测概率
    test_csv_path = (
        rf"E:\sy\caoying_模型"
        rf"\{MODEL_TYPE}_Test_Probabilities_9222.csv"
    )

    test_results_all.to_csv(
        test_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f" [✔] 测试集 y_test 和 test_p 已保存: "
        f"{test_csv_path}"
    )

    print_line("-")

    print(
        f"流程结束。{MODEL_TYPE} 模型成果存至: "
        f"E:\\sy\\caoying_模型\\"
    )

    print_line("-")


if __name__ == "__main__":

    # 如果你在 Windows 上运行报错，可以加上：
    # multiprocessing.freeze_support()

    main()