#!/usr/bin/env python3
"""
Error Analysis + SHAP Analysis for Cancer Classification

Core Story: 如果删除病名关键词，模型还能分类吗？

Data Files:
    - cancer_cleaned_trimmed.csv (原始数据)
    - cancer_cleaned_v2_trimmed.csv (去病名数据)

Author: Your Name
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_score, f1_score
)
from sklearn.preprocessing import label_binarize
import shap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% for test
VAL_SIZE = 0.50   # 50% of remaining = 10% of total = 20%

LABEL_NAMES = {
    0: "Colon Cancer",
    1: "Liver Cancer",
    2: "Lung Cancer",
    3: "Stomach Cancer",
    4: "Thyroid Cancer"
}

# =============================================================================
# Part 1: 加载数据
# =============================================================================
print("=" * 60)
print("Part 1: 加载数据")
print("=" * 60)

# 加载两个数据文件
df_orig = pd.read_csv("cancer_cleaned_trimmed.csv")
df_no_kw = pd.read_csv("cancer_cleaned_v2_trimmed.csv")

print(f"原始数据形状: {df_orig.shape}")
print(f"去病名数据形状: {df_no_kw.shape}")
print(f"列名: {df_orig.columns.tolist()}")

# 检查标签分布
print(f"\n原始数据标签分布:")
print(df_orig.iloc[:, -1].value_counts().sort_index())

# =============================================================================
# Part 2: 数据划分
# =============================================================================
print("\n" + "=" * 60)
print("Part 2: 数据划分 (60/20/20)")
print("=" * 60)

# 原始数据（第一列是标签，第二列是文本）
X_orig = df_orig.iloc[:, 1].astype(str)  # 第二列是文本
y_orig = df_orig.iloc[:, 0].astype(int)  # 第一列是标签

# 去病名数据
X_no_kw = df_no_kw.iloc[:, 1].astype(str)
y_no_kw = df_no_kw.iloc[:, 0].astype(int)

# 验证标签一致
assert (y_orig.values == y_no_kw.values).all(), "标签不一致！"

# 第一次划分: Train + Temp (60% + 40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_orig, y_orig,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_orig
)

# 第二次划分: Validation + Test (50% + 50% of Temp = 20% + 20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# 去病名数据使用相同索引
X_train_no_kw = X_no_kw.iloc[X_train.index]
X_val_no_kw = X_no_kw.iloc[X_val.index]
X_test_no_kw = X_no_kw.iloc[X_test.index]

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")
print(f"Test 标签分布: {y_test.value_counts().sort_index().to_dict()}")

# 合并 Train + Validation 用于最终训练
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
X_trainval_no_kw = pd.concat([X_train_no_kw, X_val_no_kw])

# =============================================================================
# Part 3: 训练 SVM 冠军模型
# =============================================================================
print("\n" + "=" * 60)
print("Part 3: 训练 SVM 冠军模型")
print("=" * 60)

# TF-IDF + SVM Pipeline
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

svm_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", LinearSVC())
])

# 训练原始模型
print("训练原始模型...")
svm_pipeline.fit(X_trainval, y_trainval)
y_pred_orig = svm_pipeline.predict(X_test)

# 训练去病名模型
print("训练去病名模型...")
svm_pipeline_no_kw = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])
svm_pipeline_no_kw.fit(X_trainval_no_kw, y_trainval)
y_pred_no_kw = svm_pipeline_no_kw.predict(X_test_no_kw)

# 准确率
acc_orig = accuracy_score(y_test, y_pred_orig)
acc_no_kw = accuracy_score(y_test, y_pred_no_kw)
acc_drop = (acc_orig - acc_no_kw) / acc_orig * 100

print(f"\n原始数据 Test 准确率: {acc_orig:.4f} ({acc_orig*100:.2f}%)")
print(f"去病名数据 Test 准确率: {acc_no_kw:.4f} ({acc_no_kw*100:.2f}%)")
print(f"准确率下降: {acc_drop:.1f}%")

# =============================================================================
# Part 4: 10-fold Cross Validation（原始数据）
# =============================================================================
print("\n" + "=" * 60)
print("Part 4: 10-fold Cross Validation")
print("=" * 60)

cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])

cv_pipeline_no_kw = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(cv_pipeline, X_trainval, y_trainval, cv=skf, scoring='accuracy')
cv_scores_no_kw = cross_val_score(cv_pipeline_no_kw, X_trainval_no_kw, y_trainval, cv=skf, scoring='accuracy')

print(f"10-fold CV 准确率: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"Std:  {cv_scores.std():.4f}")

print(f"\n去病名数据 10-fold CV 准确率: {cv_scores_no_kw}")
print(f"Mean: {cv_scores_no_kw.mean():.4f} ({cv_scores_no_kw.mean()*100:.2f}%)")
print(f"Std:  {cv_scores_no_kw.std():.4f}")

# =============================================================================
# Part 5: Error Analysis（原始数据）
# =============================================================================
print("\n" + "=" * 60)
print("Part 5: Error Analysis")
print("=" * 60)

# 混淆矩阵
cm_orig = confusion_matrix(y_test, y_pred_orig)

# 分类报告
print("\nClassification Report (Original Data):")
print(classification_report(y_test, y_pred_orig,
                          target_names=[LABEL_NAMES[i] for i in range(5)],
                          digits=3))

# 错误样本分析
error_mask = y_test.values != y_pred_orig  # 用 .values 转成 numpy 数组
errors_X = X_test.iloc[error_mask].reset_index(drop=True)
errors_y_true = y_test.iloc[error_mask].reset_index(drop=True)
errors_y_pred = pd.Series(y_pred_orig)[error_mask].reset_index(drop=True)

print(f"\n错误样本数: {len(errors_X)} / {len(X_test)} ({len(errors_X)/len(X_test)*100:.1f}%)")

# 类别混淆统计
print("\n类别混淆统计:")
for true_label in range(5):
    for pred_label in range(5):
        count = ((errors_y_true == true_label) & (errors_y_pred == pred_label)).sum()
        if count > 0:
            print(f"   {LABEL_NAMES[true_label]} -> {LABEL_NAMES[pred_label]}: {count} 次")

# 保存错误样本
error_df = pd.DataFrame({
    'text': errors_X.values,
    'true_label': errors_y_true.values,
    'pred_label': errors_y_pred.values
})
error_df.to_csv('error_samples.csv', index=False)
print(f"\n错误样本已保存到: error_samples.csv")

# =============================================================================
# Part 6: SHAP 分析（原始数据）
# =============================================================================
print("\n" + "=" * 60)
print("Part 6: SHAP 分析")
print("=" * 60)

# 获取模型组件
vectorizer = svm_pipeline.named_steps["tfidf"]
clf = svm_pipeline.named_steps["clf"]
X_test_tfidf = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

print(f"TF-IDF 特征数: {len(feature_names)}")

# SHAP Explainer
print("计算 SHAP 值（原始数据）...")
explainer = shap.LinearExplainer(clf, X_test_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

# Feature Importance (Top 20) - 原始数据
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_abs_shap = np.mean(mean_abs_shap, axis=1)

# 保存原始数据的特征重要性
feature_importance_orig = mean_abs_shap.copy()

top_n = 20
top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
top_features_orig = [feature_names[i] for i in top_indices]
top_values_orig = mean_abs_shap[top_indices]

# SHAP Summary Plot
print("生成 SHAP Summary Plot...")
plt.figure(figsize=(16, 12))
shap.summary_plot(shap_values, X_test_tfidf, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Summary Plot - Original Data', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary_original.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance 图 - 原始数据
plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
y_pos = np.arange(top_n)
bars = plt.barh(y_pos, top_values_orig[::-1], color=colors)
plt.yticks(y_pos, top_features_orig[::-1])
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('SHAP Feature Importance - Original Data (Top 20)', fontsize=14, pad=20)
for bar, val in zip(bars, top_values_orig[::-1]):
    plt.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nTop 10 最重要特征 (Original Data):")
for i in range(min(10, len(top_features_orig))):
    print(f"   {i+1}. {top_features_orig[i]}: {top_values_orig[i]:.4f}")

# =============================================================================
# Part 6b: SHAP 分析（去病名数据）
# =============================================================================
print("\n" + "=" * 60)
print("Part 6b: SHAP 分析 - 去病名数据")
print("=" * 60)

# 去病名数据的模型
vectorizer_no_kw = svm_pipeline_no_kw.named_steps["tfidf"]
clf_no_kw = svm_pipeline_no_kw.named_steps["clf"]
X_test_tfidf_no_kw = vectorizer_no_kw.transform(X_test_no_kw)
feature_names_no_kw = vectorizer_no_kw.get_feature_names_out()

print(f"TF-IDF 特征数（去病名）: {len(feature_names_no_kw)}")

print("计算 SHAP 值（去病名数据）...")
explainer_no_kw = shap.LinearExplainer(clf_no_kw, X_test_tfidf_no_kw)
shap_values_no_kw = explainer_no_kw.shap_values(X_test_tfidf_no_kw)

# Feature Importance (Top 20) - 去病名数据
mean_abs_shap_no_kw = np.abs(shap_values_no_kw).mean(axis=0)
mean_abs_shap_no_kw = np.mean(mean_abs_shap_no_kw, axis=1)

# 保存去病名数据的特征重要性
feature_importance_no_kw = mean_abs_shap_no_kw.copy()

top_indices_no_kw = np.argsort(mean_abs_shap_no_kw)[-top_n:][::-1]
top_features_no_kw = [feature_names_no_kw[i] for i in top_indices_no_kw]
top_values_no_kw = mean_abs_shap_no_kw[top_indices_no_kw]

print("\nTop 10 最重要特征 (No Keywords Data):")
for i in range(min(10, len(top_features_no_kw))):
    print(f"   {i+1}. {top_features_no_kw[i]}: {top_values_no_kw[i]:.4f}")

# =============================================================================
# Part 6c: SHAP Feature Importance 对比图（优化版：三列对比）
# =============================================================================
print("\n" + "=" * 60)
print("Part 6c: SHAP Feature Importance 对比")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# 左图：原始数据 Top 10（病名）
ax1 = axes[0]
top10_orig = top_features_orig[:10]
top10_values_orig = top_values_orig[:10]
y_pos1 = np.arange(len(top10_orig))
ax1.barh(y_pos1, top10_values_orig[::-1], color='steelblue')
ax1.set_yticks(y_pos1)
ax1.set_yticklabels(top10_orig[::-1])
ax1.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax1.set_title('Original Data\n(Top 10 = Disease Names)', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(ax1.patches, top10_values_orig[::-1])):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
             va='center', fontsize=9)

# 中图：去病名数据 Top 10（顶上来的词）
ax2 = axes[1]
top10_no_kw = top_features_no_kw[:10]
top10_values_no_kw = top_values_no_kw[:10]
y_pos2 = np.arange(len(top10_no_kw))
ax2.barh(y_pos2, top10_values_no_kw[::-1], color='coral')
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top10_no_kw[::-1])
ax2.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax2.set_title('No Keywords Data\n(Top 10 = Clinical/Research Terms)', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(ax2.patches, top10_values_no_kw[::-1])):
    ax2.text(val + 0.0005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
             va='center', fontsize=9)

# 右图：重要性对比（相同位置）
ax3 = axes[2]
# 取原始Top 5病名，看它们去病名后的重要性
top5_disease = top_features_orig[:5]
values_orig_comp = []
values_no_kw_comp = []
labels = []
for f in top5_disease:
    # 原始
    if f in feature_names:
        idx = np.where(feature_names == f)[0][0]
        values_orig_comp.append(feature_importance_orig[idx])
        labels.append(f)
    # 去病名后
    if f in feature_names_no_kw:
        idx = np.where(feature_names_no_kw == f)[0][0]
        values_no_kw_comp.append(feature_importance_no_kw[idx])
    else:
        values_no_kw_comp.append(0)

x = np.arange(len(labels))
width = 0.35
ax3.bar(x - width/2, values_orig_comp, width, label='Original', color='steelblue')
ax3.bar(x + width/2, values_no_kw_comp, width, label='No Keywords', color='coral')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=30, ha='right')
ax3.set_ylabel('Mean |SHAP Value|', fontsize=11)
ax3.set_title('Top 5 Disease Names:\nImportance Drop After Removal', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')

plt.suptitle('SHAP Feature Importance: Original vs No Keywords', fontsize=14, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("SHAP Feature Importance 对比图已保存: shap_feature_importance_comparison.png")

# =============================================================================
# Part 6e: 去病名后的 Top 10 特征 (新增可视化)
# =============================================================================
print("\n" + "=" * 60)
print("Part 6e: 去病名后的 Top 10 特征可视化")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 8))

# 去病名数据的 Top 10
top_10_no_kw = top_features_no_kw[:10]
top_10_values = top_values_no_kw[:10]

y_pos = np.arange(len(top_10_no_kw))
bars = ax.barh(y_pos, top_10_values[::-1], color='coral')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_10_no_kw[::-1])
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('SHAP Top 10 Features - After Removing Disease Names\n(No Keywords Data)', fontsize=14)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, top_10_values[::-1])):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('shap_top10_no_keywords.png', dpi=150, bbox_inches='tight')
plt.close()
print("去病名后的 Top 10 特征图已保存: shap_top10_no_keywords.png")

# =============================================================================
# Part 6f: 顶上来的特征词可视化 (新增)
# =============================================================================
print("\n" + "=" * 60)
print("Part 6f: '顶上来的'特征词可视化")
print("=" * 60)

# 找出新顶上来的词（不在原始 Top 15 中）
new_features = []
new_values = []
orig_top15 = top_features_orig[:15]

for i, f in enumerate(top_features_no_kw):
    if f not in orig_top15:
        new_features.append(f)
        new_values.append(top_values_no_kw[i])
    if len(new_features) >= 10:
        break

if new_features:
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(new_features))
    bars = ax.barh(y_pos, new_values[::-1], color='green')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(new_features[::-1])
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Features That "Rose to Top" After Removing Disease Names\n(Clinical/Research Terms)', fontsize=14)

    for i, (bar, val) in enumerate(zip(bars, new_values[::-1])):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('shap_rising_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("'顶上来的'特征图已保存: shap_rising_features.png")

# =============================================================================
# Part 6d: 哪些词"顶上来了"
# =============================================================================
print("\n" + "=" * 60)
print("Part 6d: 新顶上来的特征词")
print("=" * 60)

# 找出在去病名数据中更重要但在原始数据中不突出的词
all_features_no_kw = list(feature_names_no_kw)
all_features_orig = list(feature_names)

# 去病名数据中新的重要词（不在原始 Top 20 中）
new_important = []
for i, f in enumerate(top_features_no_kw):
    if f not in top_features_orig[:15]:
        new_important.append((f, top_values_no_kw[i]))
        if len(new_important) >= 10:
            break

print("\n去病名后'顶上来的'重要特征词:")
for i, (f, v) in enumerate(new_important[:10]):
    print(f"   {i+1}. {f}: {v:.4f}")

# =============================================================================
# Part 7: 去病名数据对比
# =============================================================================
print("\n" + "=" * 60)
print("Part 7: 去病名数据对比")
print("=" * 60)

# 去病名数据混淆矩阵
cm_no_kw = confusion_matrix(y_test, y_pred_no_kw)

# 对比图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[LABEL_NAMES[i] for i in range(5)],
            yticklabels=[LABEL_NAMES[i] for i in range(5)])
axes[0].set_title(f'Original Data\nAccuracy: {acc_orig*100:.1f}%', fontsize=12)

sns.heatmap(cm_no_kw, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=[LABEL_NAMES[i] for i in range(5)],
            yticklabels=[LABEL_NAMES[i] for i in range(5)])
axes[1].set_title(f'No Keywords Data\nAccuracy: {acc_no_kw*100:.1f}%', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150)
plt.close()

# 准确率对比图（简化版）
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(['Original', 'No Keywords'], [acc_orig*100, acc_no_kw*100],
            color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Comparison: Original vs No Keywords Data', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
for i, v in enumerate([acc_orig*100, acc_no_kw*100]):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)

# 添加下降箭头
ax.annotate('', xy=(1, acc_no_kw*100), xytext=(0, acc_orig*100),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.5, (acc_orig + acc_no_kw)/2 + 5, f'-{acc_drop:.1f}%',
        ha='center', color='red', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
plt.close()

print(f"\n准确率对比:")
print(f"   原始数据: {acc_orig*100:.1f}%")
print(f"   去病名数据: {acc_no_kw*100:.1f}%")
print(f"   下降: {acc_drop:.1f}%")

# =============================================================================
# Part 8: ROC 曲线和 AUC（对比版）
# =============================================================================
print("\n" + "=" * 60)
print("Part 8: ROC 曲线和 AUC 对比")
print("=" * 60)

# 原始数据的 ROC
decision_orig = clf.decision_function(X_test_tfidf)
decision_no_kw = clf_no_kw.decision_function(X_test_tfidf_no_kw)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# 创建2x5的子图（5个类别 x 2种数据）+ AUC对比
fig = plt.figure(figsize=(15, 10))

# 前5个子图：各类别的ROC曲线
colors = ['steelblue', 'coral', 'green', 'purple', 'gold']
auc_orig_list = []
auc_no_kw_list = []

for i in range(5):
    ax = fig.add_subplot(2, 3, i+1)

    # 原始数据
    fpr_o, tpr_o, _ = roc_curve(y_test_bin[:, i], decision_orig[:, i])
    auc_o = auc(fpr_o, tpr_o)
    auc_orig_list.append(auc_o)
    ax.plot(fpr_o, tpr_o, color=colors[i], lw=2.5,
            label=f'Original (AUC={auc_o:.3f})')

    # 去病名数据
    fpr_n, tpr_n, _ = roc_curve(y_test_bin[:, i], decision_no_kw[:, i])
    auc_n = auc(fpr_n, tpr_n)
    auc_no_kw_list.append(auc_n)
    ax.plot(fpr_n, tpr_n, color=colors[i], lw=2, linestyle='--',
            label=f'No Keywords (AUC={auc_n:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'{LABEL_NAMES[i]}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

# 第6个子图：AUC对比柱状图
ax_bar = fig.add_subplot(2, 3, 6)
x = np.arange(5)
width = 0.35
ax_bar.bar(x - width/2, auc_orig_list, width, label='Original', color='steelblue', edgecolor='black')
ax_bar.bar(x + width/2, auc_no_kw_list, width, label='No Keywords', color='coral', edgecolor='black')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(['Colon', 'Liver', 'Lung', 'Stomach', 'Thyroid'], rotation=30, ha='right')
ax_bar.set_xlabel('Cancer Type', fontsize=11)
ax_bar.set_ylabel('AUC', fontsize=11)
ax_bar.set_title('AUC Comparison', fontsize=12, fontweight='bold')
ax_bar.set_ylim(0.8, 1.0)
ax_bar.legend(loc='lower right', fontsize=9)
ax_bar.grid(True, alpha=0.3, axis='y')

plt.suptitle('ROC Curves: Original vs No Keywords Data', fontsize=14, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("AUC 对比:")
for i in range(5):
    print(f"   {LABEL_NAMES[i]}: Original={auc_orig_list[i]:.3f}, No Keywords={auc_no_kw_list[i]:.3f}, Drop={auc_orig_list[i]-auc_no_kw_list[i]:.3f}")

# =============================================================================
# Part 9: 10-fold CV 波动图（对比版）
# =============================================================================
print("\n" + "=" * 60)
print("Part 9: 10-fold CV 波动对比")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始数据CV
ax1 = axes[0]
ax1.plot(range(1, 11), cv_scores * 100, 'o-', color='steelblue', linewidth=2, markersize=8)
ax1.axhline(y=cv_scores.mean() * 100, color='steelblue', linestyle='--',
            label=f'Mean: {cv_scores.mean()*100:.1f}%')
ax1.fill_between(range(1, 11),
                 (cv_scores.mean() - cv_scores.std()) * 100,
                 (cv_scores.mean() + cv_scores.std()) * 100,
                 alpha=0.2, color='steelblue')
ax1.set_xlabel('Fold', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title(f'Original Data\nMean: {cv_scores.mean()*100:.2f}%, Std: {cv_scores.std()*100:.2f}%', fontsize=12, fontweight='bold')
ax1.set_ylim(85, 100)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 右图：去病名数据CV（调整y轴范围）
ax2 = axes[1]
ax2.plot(range(1, 11), cv_scores_no_kw * 100, 'o-', color='coral', linewidth=2, markersize=8)
ax2.axhline(y=cv_scores_no_kw.mean() * 100, color='coral', linestyle='--',
            label=f'Mean: {cv_scores_no_kw.mean()*100:.1f}%')
ax2.fill_between(range(1, 11),
                 max((cv_scores_no_kw.mean() - cv_scores_no_kw.std()) * 100, 50),
                 min((cv_scores_no_kw.mean() + cv_scores_no_kw.std()) * 100, 100),
                 alpha=0.2, color='coral')
ax2.set_xlabel('Fold', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title(f'No Keywords Data\nMean: {cv_scores_no_kw.mean()*100:.2f}%, Std: {cv_scores_no_kw.std()*100:.2f}%', fontsize=12, fontweight='bold')
ax2.set_ylim(50, 85)  # 调整范围以显示66%的数据
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.suptitle('10-Fold CV Variability: Original vs No Keywords', fontsize=14, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_variability.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"CV 波动对比:")
print(f"   Original: Mean={cv_scores.mean()*100:.2f}%, Std={cv_scores.std()*100:.2f}%")
print(f"   No Keywords: Mean={cv_scores_no_kw.mean()*100:.2f}%, Std={cv_scores_no_kw.std()*100:.2f}%")
print(f"   Std增加: {(cv_scores_no_kw.std() - cv_scores.std())*100:.2f}%")

# =============================================================================
# Part 10: 总结报告
# =============================================================================
print("\n" + "=" * 60)
print("总结报告")
print("=" * 60)

print(f"""
【核心发现】

1. 准确率对比:
   - 原始数据: {acc_orig*100:.2f}%
   - 去病名数据: {acc_no_kw*100:.2f}%
   - 准确率下降: {acc_drop:.1f}%

2. 10-fold CV 对比:
   - 原始数据: Mean={cv_scores.mean()*100:.2f}%, Std={cv_scores.std()*100:.2f}%
   - 去病名数据: Mean={cv_scores_no_kw.mean()*100:.2f}%, Std={cv_scores_no_kw.std()*100:.2f}%
   - 稳定性变化: Std增加 {(cv_scores_no_kw.std() - cv_scores.std())*100:.2f}%

3. AUC 对比 (各类别):
""")

for i in range(5):
    print(f"   {LABEL_NAMES[i]}: Original={auc_orig_list[i]:.3f}, No Keywords={auc_no_kw_list[i]:.3f}")

print(f"""
4. 错误分析:
   - 错误样本: {len(errors_X)} / {len(X_test)} ({len(errors_X)/len(X_test)*100:.1f}%)

5. Top 5 重要特征 (原始数据):
""")

for i in range(5):
    print(f"   {i+1}. {top_features_orig[i]}: {top_values_orig[i]:.4f}")

print(f"""
【Story 解读】

删除病名后：
   - 准确率下降 {acc_drop:.1f}% → 模型主要依赖病名关键词
   - 稳定性下降 → 模型变得不稳定
   - AUC 平均下降 → 区分各类别能力减弱

顶上来的特征：
   - metastasis（转移）、mortality（死亡率）、nodule（结节）
   - 这些是临床研究术语，说明模型仍能利用研究语境
""")

print("【生成的图表文件】")
print("   1. confusion_matrix_comparison.png     - 原始vs去病名混淆矩阵对比")
print("   2. shap_summary_original.png           - SHAP Summary Plot（特征正负方向）")
print("   3. shap_feature_importance_comparison.png - SHAP特征对比（3列）")
print("   4. accuracy_comparison.png             - 准确率对比（含下降箭头）")
print("   5. roc_curves.png                      - ROC曲线+各类别AUC对比")
print("   6. cv_variability.png                  - 10-fold CV波动对比")
print("   7. error_samples.csv                   - 错误样本详细列表")

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
