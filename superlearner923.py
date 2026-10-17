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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# SuperLearner权重优化
from scipy.optimize import minimize

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

        ("RF", RandomForestClassifier(
            n_estimators=800,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )),

        ("MLP", MLPClassifier(
            hidden_layer_sizes=(400, 200),
            alpha=0.1,
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )),

        ("XGB", XGBClassifier(
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
        ))

    ]


# =================================================================
# 3. SuperLearner核心算法：权重优化
# =================================================================

def optimize_superlearner_weights(
        oof_features,
        y_train_np
):
    """
    SuperLearner权重优化

    输入：
        oof_features：
            训练集5折OOF预测结果
            每一列对应一个基学习器：
                第1列：RF
                第2列：MLP
                第3列：XGB

        y_train_np：
            训练集真实标签

    SuperLearner最终形式：

        P_SL =
            w_RF  × P_RF
          + w_MLP × P_MLP
          + w_XGB × P_XGB

    约束：

        w_RF  >= 0
        w_MLP >= 0
        w_XGB >= 0

        w_RF + w_MLP + w_XGB = 1

    优化目标：

        最小化OOF预测的Log Loss
    """

    n_models = oof_features.shape[1]

    # -------------------------------------------------------------
    # 初始权重
    # -------------------------------------------------------------

    initial_weights = np.ones(
        n_models
    ) / n_models

    # -------------------------------------------------------------
    # 目标函数
    # -------------------------------------------------------------

    def objective(weights):
        # 三个模型概率加权
        combined_prob = np.dot(
            oof_features,
            weights
        )

        # 防止出现0或1
        combined_prob = np.clip(
            combined_prob,
            1e-7,
            1 - 1e-7
        )

        # Log Loss
        loss = log_loss(
            y_train_np,
            combined_prob
        )

        return loss

    # -------------------------------------------------------------
    # 权重范围
    # -------------------------------------------------------------

    bounds = [
        (0.0, 1.0)
        for _ in range(n_models)
    ]

    # -------------------------------------------------------------
    # 权重总和必须等于1
    # -------------------------------------------------------------

    constraints = {
        "type": "eq",
        "fun": lambda weights:
        np.sum(weights) - 1.0
    }

    # -------------------------------------------------------------
    # SLSQP优化
    # -------------------------------------------------------------

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False
        }
    )

    if not result.success:
        raise RuntimeError(
            "SuperLearner权重优化失败："
            + result.message
        )

    optimized_weights = result.x

    # -------------------------------------------------------------
    # 数值修正
    # -------------------------------------------------------------

    optimized_weights = np.maximum(
        optimized_weights,
        0
    )

    optimized_weights = (
            optimized_weights /
            np.sum(optimized_weights)
    )

    return optimized_weights


# =================================================================
# 4. 核心算法：SuperLearner
# =================================================================

def fit_superlearner_ensemble(
        X_train_s,
        y_train,
        X_test_s,
        X_all_s,
        base_models,
        k_folds=5
):
    """
    SuperLearner逻辑：

    1. 通过5折交叉验证获取训练集OOF预测
    2. RF、MLP、XGB分别生成OOF预测
    3. 根据OOF预测优化三个模型的权重
    4. 三个模型在全量训练集上重新训练
    5. 得到测试集预测概率
    6. 得到全量数据预测概率
    7. 使用优化权重进行最终融合

    与原Stacking代码相比：

        原Stacking：
            OOF预测
                ↓
            LogisticRegression
                ↓
            最终预测

        现在SuperLearner：
            OOF预测
                ↓
            Log Loss优化权重
                ↓
            加权平均
                ↓
            最终预测
    """

    # =============================================================
    # 数据转换
    # =============================================================

    skf = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=42
    )

    y_train_np = (
        y_train.values
        if isinstance(y_train, pd.Series)
        else np.asarray(y_train)
    )

    # =============================================================
    # 初始化预测矩阵
    # =============================================================

    # OOF预测
    oof_features = np.zeros(
        (
            len(X_train_s),
            len(base_models)
        )
    )

    # 测试集预测
    test_features = np.zeros(
        (
            len(X_test_s),
            len(base_models)
        )
    )

    # 全量数据预测
    all_features = np.zeros(
        (
            len(X_all_s),
            len(base_models)
        )
    )

    # =============================================================
    # 第一层：三个基学习器
    # =============================================================

    for m_idx, (
            name,
            model_template
    ) in enumerate(base_models):

        print(
            f"\n    → 正在训练基学习器：{name}"
        )

        # ---------------------------------------------------------
        # A. 5折交叉验证生成OOF预测
        # ---------------------------------------------------------

        for fold, (
                train_idx,
                val_idx
        ) in enumerate(
            skf.split(
                X_train_s,
                y_train_np
            ),
            start=1
        ):
            X_fold_tr = (
                X_train_s[train_idx]
            )

            X_fold_val = (
                X_train_s[val_idx]
            )

            y_fold_tr = (
                y_train_np[train_idx]
            )

            # 每一折建立独立模型
            model = clone(
                model_template
            )

            # 训练
            model.fit(
                X_fold_tr,
                y_fold_tr
            )

            # 验证集预测概率
            oof_features[
                val_idx,
                m_idx
            ] = model.predict_proba(
                X_fold_val
            )[:, 1]

        # ---------------------------------------------------------
        # B. 全量训练
        # ---------------------------------------------------------

        full_model = clone(
            model_template
        )

        full_model.fit(
            X_train_s,
            y_train_np
        )

        # ---------------------------------------------------------
        # C. 测试集预测
        # ---------------------------------------------------------

        test_features[
        :,
        m_idx
        ] = full_model.predict_proba(
            X_test_s
        )[:, 1]

        # ---------------------------------------------------------
        # D. 全量数据预测
        # ---------------------------------------------------------

        all_features[
        :,
        m_idx
        ] = full_model.predict_proba(
            X_all_s
        )[:, 1]

    # =============================================================
    # 第二层：SuperLearner权重优化
    # =============================================================

    print(
        "\n    → 正在优化SuperLearner权重..."
    )

    weights = optimize_superlearner_weights(
        oof_features,
        y_train_np
    )

    # =============================================================
    # 输出权重
    # =============================================================

    print("\n    SuperLearner最终权重：")

    for idx, (
            name,
            _
    ) in enumerate(base_models):
        print(
            f"       {name:<8} : "
            f"{weights[idx]:.6f}"
        )

    print(
        f"       权重总和 : "
        f"{np.sum(weights):.6f}"
    )

    # =============================================================
    # 计算OOF融合结果
    # =============================================================

    oof_final_prob = np.dot(
        oof_features,
        weights
    )

    oof_logloss = log_loss(
        y_train_np,
        np.clip(
            oof_final_prob,
            1e-7,
            1 - 1e-7
        )
    )

    print(
        f"       OOF Log Loss : "
        f"{oof_logloss:.6f}"
    )

    # =============================================================
    # SuperLearner最终测试集预测
    # =============================================================

    final_test_prob = np.dot(
        test_features,
        weights
    )

    # =============================================================
    # SuperLearner最终全量预测
    # =============================================================

    final_all_prob = np.dot(
        all_features,
        weights
    )

    # =============================================================
    # 防止概率出现0或1
    # =============================================================

    final_test_prob = np.clip(
        final_test_prob,
        1e-7,
        1 - 1e-7
    )

    final_all_prob = np.clip(
        final_all_prob,
        1e-7,
        1 - 1e-7
    )

    return (
        final_test_prob,
        final_all_prob,
        weights,
        oof_features
    )


# =================================================================
# 5. 并行任务单元
# =================================================================

def run_single_task(
        name,
        X_raw,
        y_raw
):
    t0 = time.time()

    # =============================================================
    # A. 初始化过采样器
    # =============================================================

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

    else:

        raise ValueError(
            f"未知采样方法：{name}"
        )

    # =============================================================
    # B. 记录原始数据
    # =============================================================

    n_pos_raw = np.sum(
        y_raw == 1
    )

    n_neg_raw = np.sum(
        y_raw == 0
    )

    # =============================================================
    # C. 第一步：过采样
    # =============================================================

    X_over, y_over = over_s.fit_resample(
        X_raw,
        y_raw
    )

    n_pos_over = np.sum(
        y_over == 1
    )

    n_neg_over = np.sum(
        y_over == 0
    )

    # =============================================================
    # D. 第二步：随机欠采样
    # =============================================================

    under_s = RandomUnderSampler(
        sampling_strategy=1.0,
        random_state=42
    )

    X_bal, y_bal = under_s.fit_resample(
        X_over,
        y_over
    )

    n_pos_final = np.sum(
        y_bal == 1
    )

    n_neg_final = np.sum(
        y_bal == 0
    )

    # =============================================================
    # E. 打印采样报告
    # =============================================================

    report = (

        f"\n[线程报告: {name}]\n"

        f"  1. 原始样本 -> "
        f"滑坡(1): {n_pos_raw}, "
        f"非滑坡(0): {n_neg_raw}\n"

        f"  2. 过采样后 -> "
        f"滑坡(1): {n_pos_over} (增加), "
        f"非滑坡(0): {n_neg_over} (未变)\n"

        f"  3. 欠采样后 -> "
        f"滑坡(1): {n_pos_final} (未变), "
        f"非滑坡(0): {n_neg_final} (减少)\n"

        f"  -> 最终比例 1:1, "
        f"总样本: {len(y_bal)}"
    )

    print(report)

    # =============================================================
    # F. 划分训练集和测试集
    # =============================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X_bal,
        y_bal,

        test_size=0.3,

        random_state=42,

        stratify=y_bal
    )

    # =============================================================
    # G. 标准化与模型训练
    # =============================================================

    sc = StandardScaler()

    X_tr_s = sc.fit_transform(
        X_train
    )

    X_te_s = sc.transform(
        X_test
    )

    # 原始全量数据
    X_al_s = sc.transform(
        X_raw
    )

    # =============================================================
    # H. SuperLearner
    # =============================================================

    (
        test_p,
        all_p,
        weights,
        oof_features
    ) = fit_superlearner_ensemble(

        X_tr_s,

        y_train,

        X_te_s,

        X_al_s,

        build_base_models(),

        k_folds=5
    )

    # =============================================================
    # I. 0.5阈值分类
    # =============================================================

    y_pred = (
            test_p >= 0.5
    ).astype(int)

    # =============================================================
    # J. ROC
    # =============================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        test_p
    )

    # =============================================================
    # K. 保存test_p和y_test
    # =============================================================

    test_result_df = pd.DataFrame({

        "y_test":
            np.asarray(y_test),

        "test_p":
            test_p
    })

    # =============================================================
    # L. 混淆矩阵
    # =============================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    # =============================================================
    # M. 返回结果
    # =============================================================

    return {

        "name":
            name,

        "auc":
            roc_auc_score(
                y_test,
                test_p
            ),

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "f1":
            f1_score(
                y_test,
                y_pred
            ),

        "recall":
            recall_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "fpr":
            fpr,

        "tpr":
            tpr,

        "all_prob":
            all_p,

        # 测试集真实标签
        "y_test":
            np.asarray(y_test),

        # 测试集预测概率
        "test_p":
            test_p,

        # 测试集结果
        "test_result_df":
            test_result_df,

        # SuperLearner权重
        "weights":
            weights,

        # OOF预测
        "oof_features":
            oof_features,

        "time":
            time.time() - t0
    }


# =================================================================
# 6. ROC绘图函数
# =================================================================

def plot_roc_curves_scientific(
        results,
        save_path=None
):
    """集成用户样式的科研级ROC绘图"""

    plt.figure(
        figsize=(12, 10),
        dpi=600
    )

    # 颜色
    colors = [
        '#E41A1C',
        '#9B30FF',
        '#4DAF4A'
    ]

    # 线型
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

    for idx, res in enumerate(
            results
    ):
        plt.plot(

            res['fpr'],

            res['tpr'],

            color=colors[idx],

            linestyle=line_styles[idx],

            linewidth=line_widths[idx],

            label=(
                f"{res['name']} "
                f"(AUC = {res['auc']:.4f})"
            )
        )

    # -------------------------------------------------------------
    # 坐标轴
    # -------------------------------------------------------------

    plt.xlim(
        [-0.01, 1.02]
    )

    plt.ylim(
        [-0.01, 1.02]
    )

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
    # 标签
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
    # 边框
    # -------------------------------------------------------------

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(
            2
        )

    # -------------------------------------------------------------
    # 图例
    # -------------------------------------------------------------

    legend = plt.legend(

        loc='lower right',

        fontsize=30,

        framealpha=0.95,

        prop={
            'family':
                'Times New Roman',

            'size':
                30
        }
    )

    legend.get_frame().set_linewidth(
        2
    )

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
    # 保存
    # -------------------------------------------------------------

    if save_path:
        plt.savefig(

            save_path,

            dpi=600,

            bbox_inches='tight',

            facecolor='white'
        )

        print(
            f" [✔] ROC曲线图已导出: "
            f"{save_path}"
        )

    plt.show()


# =================================================================
# 7. 主程序
# =================================================================

def main():
    print_line()

    print(
        "      滑坡易发性建模: "
        "SuperLearner"
    )

    print_line()

    # =============================================================
    # Step 1
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
    # 特征
    # -------------------------------------------------------------

    X_raw = df.drop(
        columns=[
            "huapo",
            "x",
            "y"
        ]
    )

    # -------------------------------------------------------------
    # 标签
    # -------------------------------------------------------------

    y_raw = df[
        "huapo"
    ]

    # =============================================================
    # Step 2
    # =============================================================

    print_step(
        2,
        4,
        "启动并行重采样与SuperLearner训练..."
    )

    samplers = [
        "SMOTE",
        "BorderlineSMOTE",
        "ADASYN"
    ]

    results = []

    # =============================================================
    # 并行运行三个采样方法
    # =============================================================

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=3
    ) as executor:

        futures = {

            executor.submit(
                run_single_task,
                s,
                X_raw,
                y_raw
            ):
                s

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

                f"AUC: "
                f"{res['auc']:.4f}"
            )

            # -----------------------------------------------------
            # 输出SuperLearner权重
            # -----------------------------------------------------

            print(
                f"    RF权重: "
                f"{res['weights'][0]:.6f}"
            )

            print(
                f"    MLP权重: "
                f"{res['weights'][1]:.6f}"
            )

            print(
                f"    XGB权重: "
                f"{res['weights'][2]:.6f}"
            )

    # =============================================================
    # Step 3
    # =============================================================

    print_step(
        3,
        4,
        "生成对比图表..."
    )

    plot_path = (
        r"E:\sy\caoying_模型"
        r"\superlearner_ROC1111.png"
    )

    plot_roc_curves_scientific(
        results,
        save_path=plot_path
    )

    # =============================================================
    # Step 4
    # =============================================================

    print_step(
        4,
        4,
        "保存预测概率数据..."
    )

    # =============================================================
    # 保存全量预测概率
    # =============================================================

    df_out = df.copy()

    for r in results:
        df_out[
            f"prob_{r['name']}"
        ] = r[
            'all_prob'
        ]

    csv_path = (
        r"E:\sy\caoying_模型"
        r"\superlearner_Results_Scientific1111.csv"
    )

    df_out.to_csv(

        csv_path,

        index=False,

        encoding="utf-8-sig"
    )

    print(
        f" [✔] 全量预测概率已保存: "
        f"{csv_path}"
    )

    # =============================================================
    # 保存测试集预测概率
    # =============================================================

    test_results_list = []

    for r in results:
        test_df = pd.DataFrame({

            "sampler":
                r["name"],

            "y_test":
                r["y_test"],

            "test_p":
                r["test_p"]
        })

        test_results_list.append(
            test_df
        )

    # 合并
    test_results_all = pd.concat(

        test_results_list,

        ignore_index=True
    )

    test_csv_path = (
        r"E:\sy\caoying_模型"
        r"\superlearner_Test_Probabilities1111.csv"
    )

    test_results_all.to_csv(

        test_csv_path,

        index=False,

        encoding="utf-8-sig"
    )

    print(
        f" [✔] 测试集 "
        f"y_test 和 test_p 已保存: "
        f"{test_csv_path}"
    )

    # =============================================================
    # 保存SuperLearner权重
    # =============================================================

    weight_results = []

    for r in results:
        weight_results.append({

            "sampler":
                r["name"],

            "RF_weight":
                r["weights"][0],

            "MLP_weight":
                r["weights"][1],

            "XGB_weight":
                r["weights"][2],

            "weight_sum":
                np.sum(
                    r["weights"]
                ),

            "AUC":
                r["auc"],

            "F1":
                r["f1"],

            "Recall":
                r["recall"],

            "Precision":
                r["precision"]
        })

    weight_df = pd.DataFrame(
        weight_results
    )

    weight_csv_path = (
        r"E:\sy\caoying_模型"
        r"\superlearner_Weights1111.csv"
    )

    weight_df.to_csv(

        weight_csv_path,

        index=False,

        encoding="utf-8-sig"
    )

    print(
        f" [✔] SuperLearner权重已保存: "
        f"{weight_csv_path}"
    )

    # =============================================================
    # 保存OOF预测
    # =============================================================

    oof_results = []

    for r in results:
        oof_df = pd.DataFrame({

            "sampler":
                r["name"],

            "RF_OOF":
                r["oof_features"][:, 0],

            "MLP_OOF":
                r["oof_features"][:, 1],

            "XGB_OOF":
                r["oof_features"][:, 2]
        })

        oof_results.append(
            oof_df
        )

    # 注意：
    # 三个采样方法的训练集大小相同，
    # 因此可以直接合并。

    oof_all_df = pd.concat(

        oof_results,

        ignore_index=True
    )

    oof_csv_path = (
        r"E:\sy\caoying_模型"
        r"\superlearner_OOF1111.csv"
    )

    oof_all_df.to_csv(

        oof_csv_path,

        index=False,

        encoding="utf-8-sig"
    )

    print(
        f" [✔] OOF预测结果已保存: "
        f"{oof_csv_path}"
    )

    # =============================================================
    # 最终输出
    # =============================================================

    print_line("-")

    print(
        "SuperLearner流程结束。"
    )

    print(
        "成果存至: "
        r"E:\\sy\\caoying_模型\\"
    )

    print_line()


# =================================================================
# 8. 程序入口
# =================================================================

if __name__ == "__main__":
    main()