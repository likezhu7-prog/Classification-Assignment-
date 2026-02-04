# Error Analysis + SHAP 计划 (Plan 2)

---

## 数据文件

| 文件 | 用途 |
|------|------|
| `cancer_cleaned_trimmed.csv` | 原始数据（含病名） |
| `cancer_cleaned_v2_trimmed.csv` | 去病名后的数据 |

---

## 你的核心 Story

**问题：** 如果我把"答案写在题目里的词"全部删掉，模型还能靠论文的真正内容来判断吗？

**对比实验：**
1. 原始数据 → 准确率 A
2. 去病名数据 → 准确率 B
3. 准确率下降 = A - B

---

## 你需要做的工作

### 1. 数据准备

```python
# 加载两个数据
df_orig = pd.read_csv("cancer_cleaned_trimmed.csv")
df_no_kw = pd.read_csv("cancer_cleaned_v2_trimmed.csv")

# 数据划分 (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(...)
X_val, X_test, y_val, y_test = train_test_split(...)
```

### 2. 训练 SVM 冠军模型（用 Train+Validation）

```python
# 用原始数据训练
pipeline_orig = Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LinearSVC())
])
pipeline_orig.fit(X_train + X_val, y_train + y_val)
y_pred_orig = pipeline_orig.predict(X_test)

# 用去病名数据训练
pipeline_no_kw = Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LinearSVC())
])
pipeline_no_kw.fit(X_train_no_kw + X_val_no_kw, y_train + y_val)
y_pred_no_kw = pipeline_no_kw.predict(X_test_no_kw)
```

### 3. Error Analysis（用 Test 数据）

```python
# 混淆矩阵
cm_orig = confusion_matrix(y_test, y_pred_orig)
cm_no_kw = confusion_matrix(y_test, y_pred_no_kw)

# 错误样本
errors_orig = X_test[y_test != y_pred_orig]
errors_no_kw = X_test_no_kw[y_test != y_pred_no_kw]

# 分析：哪些类别互相混淆？
for true in range(5):
    for pred in range(5):
        count = ...
        if count > 0:
            print(f"{label[true]} -> {label[pred]}: {count}")
```

**输出：**
- 混淆矩阵热力图（原始 vs 去病名）
- 错误样本分析

### 4. SHAP 可解释性

```python
# 原始模型
vectorizer = pipeline_orig.named_steps["tfidf"]
clf = pipeline_orig.named_steps["clf"]
X_test_tfidf = vectorizer.transform(X_test)

explainer = shap.LinearExplainer(clf, X_test_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

# SHAP Summary Plot
plt.figure(figsize=(16, 12))
shap.summary_plot(shap_values, X_test_tfidf, feature_names=feature_names, max_display=20)
plt.savefig("shap_summary_original.png")

# Feature Importance (Top 20)
mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
top_idx = np.argsort(mean_abs_shap)[-20:][::-1]
top_features = [feature_names[i] for i in top_idx]
plt.barh(top_features, mean_abs_shap[top_idx])
plt.savefig("shap_feature_importance.png")
```

**输出：**
- SHAP Summary Plot
- SHAP Feature Importance 图

### 5. 准确率对比（核心故事）

```python
acc_orig = accuracy_score(y_test, y_pred_orig)
acc_no_kw = accuracy_score(y_test, y_pred_no_kw)
drop = (acc_orig - acc_no_kw) / acc_orig * 100

print(f"原始准确率: {acc_orig:.4f}")
print(f"去病名准确率: {acc_no_kw:.4f}")
print(f"准确率下降: {drop:.1f}%")
```

**输出：**
- 准确率对比柱状图

### 6. Bias and Variability（用 10-fold CV）

```python
# 10-fold CV（用 Train 数据）
cv_scores_orig = cross_val_score(pipeline_orig, X_train, y_train, cv=10)
cv_scores_no_kw = cross_val_score(pipeline_no_kw, X_train_no_kw, y_train, cv=10)

print(f"原始 - Mean: {cv_scores_orig.mean():.4f}, Std: {cv_scores_orig.std():.4f}")
print(f"去病名 - Mean: {cv_scores_no_kw.mean():.4f}, Std: {cv_scores_no_kw.std():.4f}")
```

**输出：**
- 10-fold CV 波动图

### 7. AUC（ROC 曲线）

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4])
decision = clf.decision_function(X_test_tfidf)

for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], decision[:, i])
    plt.plot(fpr, tpr, label=f"{label[i]} (AUC={auc(fpr,tpr):.2f})")
```

**输出：**
- ROC 曲线图

---

## 任务清单

- [ ] 1. 加载两个数据文件
- [ ] 2. 数据划分（60/20/20）
- [ ] 3. 训练 SVM 模型（原始 + 去病名）
- [ ] 4. 混淆矩阵可视化
- [ ] 5. 错误样本分析
- [ ] 6. SHAP Summary Plot
- [ ] 7. SHAP Feature Importance
- [ ] 8. 准确率对比图
- [ ] 9. 10-fold CV 波动图
- [ ] 10. ROC 曲线图
- [ ] 11. 撰写 Story

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `confusion_matrix_orig.png` | 原始数据混淆矩阵 |
| `confusion_matrix_no_kw.png` | 去病名混淆矩阵 |
| `confusion_matrix_comparison.png` | 混淆矩阵对比 |
| `shap_summary_original.png` | SHAP Summary Plot |
| `shap_feature_importance.png` | SHAP 特征重要性 |
| `accuracy_comparison.png` | 准确率对比 |
| `cv_variability.png` | 10-fold CV 波动 |
| `roc_curves.png` | ROC 曲线 |

---

## 对应评分项

| 评分项 | 分值 | 对应输出 |
|--------|------|----------|
| Error Analysis | 2% | 混淆矩阵 + 错误分析 |
| SHAP 可解释性 | 2% | SHAP 图 + 特征重要性 |
| Visualizations | 2% | 所有图表 |
| Bias/Variability | 2% | CV 波动图 |
| AUC | 1% | ROC 曲线 |
| Story | - | 准确率下降分析 |

---

## 预计时间

| 任务 | 时间 |
|------|------|
| 数据准备 + 训练模型 | 30 分钟 |
| Error Analysis | 45 分钟 |
| SHAP 分析 | 45 分钟 |
| Bias/Variability + AUC | 30 分钟 |
| 可视化整理 | 30 分钟 |
| **总计** | **约 3 小时** |
