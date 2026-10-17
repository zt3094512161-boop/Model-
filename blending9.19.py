import concurrent.futures
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
# 2. 模型架构定义
# =================================================================

def build_base_models():
    return [
        (
            "RF",
            RandomForestClassifier(
                n_estimators=800,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(400, 200),
                alpha=0.1,
                max_iter=2000,
                early_stopping=True,
                random_state=42
            )
        ),
        (
            "XGB",
            XGBClassifier(
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary:logistic",
                random_state=42,
                n_jobs=-1
            )
        )
    ]


# =================================================================
# 3. 核心算法：Blending
# =================================================================

def fit_blending_ensemble(
    X_train_s,
    y_train,
    X_test_s,
    X_all_s,
    base_models,
    blend_ratio=0.3
):
    """
    Blending 集成算法：

    1. 将训练集划分为训练部分（train_part）和验证部分（validation）
    2. 基模型在 train_part 上训练
    3. 基模型对 validation、测试集、全量数据分别预测
    4. 使用 validation 的预测结果作为元特征训练元模型（逻辑回归）
    5. 返回元模型对测试集和全量数据的最终预测概率
    """

    # 处理 y_train 可能为 Series 的情况
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

    # -------------------------------------------------------------
    # 划分训练部分和验证部分
    # -------------------------------------------------------------

    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_s,
        y_train_np,
        test_size=blend_ratio,
        random_state=42,
        stratify=y_train_np
    )

    n_models = len(base_models)

    # -------------------------------------------------------------
    # 存储元特征
    # -------------------------------------------------------------

    val_features = np.zeros((len(X_val), n_models))

    # 测试集特征
    test_features = np.zeros((len(X_test_s), n_models))

    # 原始全量数据特征
    all_features = np.zeros((len(X_all_s), n_models))

    # -------------------------------------------------------------
    # 对每个基模型进行训练
    # -------------------------------------------------------------

    for m_idx, (name, model_template) in enumerate(base_models):

        # 在训练部分上训练基模型
        model = clone(model_template)

        model.fit(
            X_train_part,
            y_train_part
        )

        # ---------------------------------------------------------
        # 对验证集预测
        # 用于训练元模型
        # ---------------------------------------------------------

        val_features[:, m_idx] = model.predict_proba(
            X_val
        )[:, 1]

        # ---------------------------------------------------------
        # 对测试集预测
        # ---------------------------------------------------------

        test_features[:, m_idx] = model.predict_proba(
            X_test_s
        )[:, 1]

        # ---------------------------------------------------------
        # 对原始全量数据预测
        # ---------------------------------------------------------

        all_features[:, m_idx] = model.predict_proba(
            X_all_s
        )[:, 1]

    # -------------------------------------------------------------
    # 元模型：逻辑回归
    # -------------------------------------------------------------

    meta_model = LogisticRegression(
        C=100,
        max_iter=100,
        random_state=42
    )

    # 使用验证集预测结果训练元模型
    meta_model.fit(
        val_features,
        y_val
    )

    # -------------------------------------------------------------
    # 最终测试集预测概率
    # -------------------------------------------------------------

    final_test_prob = meta_model.predict_proba(
        test_features
    )[:, 1]

    # -------------------------------------------------------------
    # 原始全量数据最终预测概率
    # -------------------------------------------------------------

    final_all_prob = meta_model.predict_proba(
        all_features
    )[:, 1]

    return final_test_prob, final_all_prob


# =================================================================
# 4. 并行任务单元
# =================================================================

def run_single_task(name, X_raw, y_raw):

    t0 = time.time()

    # -------------------------------------------------------------
    # A. 初始化过采样器
    # -------------------------------------------------------------

    if name == "SMOTE":

        over_s = SMOTE(
            sampling_strategy=0.3,
            random_state=42,
            k_neighbors=3
        )

    elif name == "BorderlineSMOTE":

        over_s = BorderlineSMOTE(
            sampling_strategy=0.3,
            random_state=42,
            k_neighbors=10,
            m_neighbors=10
        )

    elif name == "ADASYN":

        over_s = ADASYN(
            sampling_strategy=0.3,
            random_state=42,
            n_neighbors=5
        )

    # -------------------------------------------------------------
    # B. 记录原始数据
    # -------------------------------------------------------------

    n_pos_raw = np.sum(y_raw == 1)
    n_neg_raw = np.sum(y_raw == 0)

    # -------------------------------------------------------------
    # C. 第一步：过采样
    # 滑坡 1 增加到非滑坡 0 的 30%
    # -------------------------------------------------------------

    X_over, y_over = over_s.fit_resample(
        X_raw,
        y_raw
    )

    n_pos_over = np.sum(y_over == 1)
    n_neg_over = np.sum(y_over == 0)

    # -------------------------------------------------------------
    # D. 第二步：随机欠采样
    # 非滑坡 0 减少到与滑坡 1 一致，即 1:1
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # E. 打印采样报告
    # -------------------------------------------------------------

    report = (
        f"\n[线程报告: {name}]\n"
        f" 1. 原始样本 -> 滑坡(1): {n_pos_raw}, "
        f"非滑坡(0): {n_neg_raw}\n"
        f" 2. 过采样后 -> 滑坡(1): {n_pos_over} (增加), "
        f"非滑坡(0): {n_neg_over} (未变)\n"
        f" 3. 欠采样后 -> 滑坡(1): {n_pos_final} (未变), "
        f"非滑坡(0): {n_neg_final} (减少)\n"
        f" -> 最终比例 1:1, 总样本: {len(y_bal)}"
    )

    print(report)

    # -------------------------------------------------------------
    # F. 划分训练集和测试集
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.3,
        random_state=42,
        stratify=y_bal
    )

    # -------------------------------------------------------------
    # G. 标准化与模型训练
    # -------------------------------------------------------------

    sc = StandardScaler()

    X_tr_s = sc.fit_transform(X_train)

    X_te_s = sc.transform(X_test)

    # 预测原始全量数据
    X_al_s = sc.transform(X_raw)

    # -------------------------------------------------------------
    # 使用 Blending 集成
    # -------------------------------------------------------------

    test_p, all_p = fit_blending_ensemble(
        X_tr_s,
        y_train,
        X_te_s,
        X_al_s,
        build_base_models(),
        blend_ratio=0.3
    )

    # -------------------------------------------------------------
    # 测试集分类结果
    # -------------------------------------------------------------

    y_pred = (test_p >= 0.5).astype(int)

    # -------------------------------------------------------------
    # ROC
    # -------------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        test_p
    )

    # -------------------------------------------------------------
    # H. 返回结果
    # -------------------------------------------------------------

    return {
        "name": name,

        # 测试集评价指标
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

        # ROC
        "fpr": fpr,
        "tpr": tpr,

        # ---------------------------------------------------------
        # 新增：保存测试集真实标签
        # ---------------------------------------------------------

        "y_test": np.asarray(y_test),

        # ---------------------------------------------------------
        # 新增：保存测试集预测概率
        # ---------------------------------------------------------

        "test_prob": np.asarray(test_p),

        # 原始全量数据预测概率
        "all_prob": all_p,

        # 运行时间
        "time": time.time() - t0
    }


# =================================================================
# 5. ROC绘图
# =================================================================

def plot_roc_curves_scientific(
    results,
    save_path=None
):
    """
    集成用户样式的科研级ROC绘图，
    强制图例顺序：SMOTE、BorderlineSMOTE、ADASYN
    """

    # -------------------------------------------------------------
    # 强制图例顺序
    # -------------------------------------------------------------

    order = [
        "SMOTE",
        "BorderlineSMOTE",
        "ADASYN"
    ]

    # 按指定顺序对 results 进行排序
    results_sorted = sorted(
        results,
        key=lambda x: order.index(x['name'])
    )

    plt.figure(
        figsize=(12, 10),
        dpi=600
    )

    # -------------------------------------------------------------
    # 颜色、线型、粗细定义
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # 绘制ROC
    # -------------------------------------------------------------

    for idx, res in enumerate(results_sorted):

        plt.plot(
            res['fpr'],
            res['tpr'],
            color=colors[idx],
            linestyle=line_styles[idx],
            linewidth=line_widths[idx],
            label=f"{res['name']} (AUC = {res['auc']:.4f})"
        )

    # -------------------------------------------------------------
    # 坐标轴范围
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # 坐标轴标签
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # 坐标轴边框
    # -------------------------------------------------------------

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # -------------------------------------------------------------
    # 图例
    # -------------------------------------------------------------

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

    legend.get_frame().set_edgecolor(
        'black'
    )

    # -------------------------------------------------------------
    # 网格
    # -------------------------------------------------------------

    plt.grid(
        True,
        alpha=0.3,
        linestyle='--'
    )

    plt.tight_layout()

    # -------------------------------------------------------------
    # 保存ROC
    # -------------------------------------------------------------

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
# 6. 主程序
# =================================================================

def main():

    print_line()

    print(
        " 滑坡易发性建模: Blending 集成"
    )

    print_line()

    # =============================================================
    # 第1步：读取数据
    # =============================================================

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

    # -------------------------------------------------------------
    # 注意：
    # 原代码中删除了 "x" 和 "y" 两列
    # 这里保持完全一致
    # -------------------------------------------------------------

    X_raw = df.drop(
        columns=[
            "huapo",
            "x",
            "y"
        ]
    )

    y_raw = df["huapo"]

    # =============================================================
    # 第2步：并行重采样与模型训练
    # =============================================================

    print_step(
        2,
        4,
        "启动并行重采样与模型训练..."
    )

    samplers = [
        "SMOTE",
        "BorderlineSMOTE",
        "ADASYN"
    ]

    results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=3
    ) as executor:

        futures = {
            executor.submit(
                run_single_task,
                s,
                X_raw,
                y_raw
            ): s
            for s in samplers
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            res = future.result()

            results.append(
                res
            )

            print(
                f" [任务完成] "
                f"{res['name']} | "
                f"AUC: {res['auc']:.4f}"
            )

    # =============================================================
    # 第3步：生成ROC图
    # =============================================================

    print_step(
        3,
        4,
        "生成对比图表..."
    )

    plot_path = (
        r"E:\sy\caoying_模型"
        r"\blending_ROC_1111.png"
    )

    plot_roc_curves_scientific(
        results,
        save_path=plot_path
    )

    # =============================================================
    # 第4步：保存预测概率数据
    # =============================================================

    print_step(
        4,
        4,
        "保存预测概率数据..."
    )

    # -------------------------------------------------------------
    # 4.1 保存原始全量数据预测概率
    # -------------------------------------------------------------

    df_out = df.copy()

    for r in results:

        df_out[
            f"prob_{r['name']}"
        ] = r["all_prob"]

    csv_path = (
        r"E:\sy\caoying_模型"
        r"\blending_Results_111.csv"
    )

    df_out.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f" [✔] 全量数据预测概率已保存: "
        f"{csv_path}"
    )

    # =============================================================
    # 4.2 保存测试集真实标签和预测概率
    # =============================================================

    # -------------------------------------------------------------
    # 为了保证输出顺序稳定：
    # SMOTE → BorderlineSMOTE → ADASYN
    # -------------------------------------------------------------

    results_sorted = sorted(
        results,
        key=lambda x: samplers.index(
            x["name"]
        )
    )

    # -------------------------------------------------------------
    # 每个采样策略分别生成自己的 test_df
    # -------------------------------------------------------------

    test_results_list = []

    for r in results_sorted:

        # ---------------------------------------------------------
        # 当前采样策略的测试集结果
        # ---------------------------------------------------------

        test_df = pd.DataFrame({
            "sampler": r["name"],
            "y_test": r["y_test"],
            "test_p": r["test_prob"]
        })

        # ---------------------------------------------------------
        # 保存当前采样策略的 test_df
        # ---------------------------------------------------------

        test_results_list.append(
            test_df
        )

        print(
            f" [测试集] {r['name']} -> "
            f"{len(test_df)} 个样本"
        )

    # -------------------------------------------------------------
    # 纵向合并三个采样策略的测试集结果
    # -------------------------------------------------------------

    df_test_out = pd.concat(
        test_results_list,
        axis=0,
        ignore_index=True
    )

    # -------------------------------------------------------------
    # 测试集结果保存路径
    # -------------------------------------------------------------

    test_csv_path = (
        r"E:\sy\caoying_模型"
        r"\blending_Test_Results_111.csv"
    )

    df_test_out.to_csv(
        test_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f" [✔] 测试集真实标签和预测概率已保存: "
        f"{test_csv_path}"
    )

    # -------------------------------------------------------------
    # 打印最终各采样策略测试集样本数量
    # -------------------------------------------------------------

    print("\n各采样策略测试集样本数量：")

    print(
        df_test_out["sampler"].value_counts()
        .reindex(samplers)
    )

    # =============================================================
    # 最终提示
    # =============================================================

    print_line("-")

    print(
        "流程结束。成果存至: "
        "E:\\sy\\caoying_模型\\"
    )

    print_line()


# =================================================================
# 7. 程序入口
# =================================================================

if __name__ == "__main__":
    main()