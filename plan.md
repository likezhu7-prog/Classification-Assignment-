# Error Analysis + SHAP 可解释性分析计划

---

## 核心故事 (Story)

**核心问题：** 如果我把"答案写在题目里的词"（病名、同义词）全部删掉，模型还能不能靠论文的**真正内容**来判断癌症类型？

**实验设计：**
- 原始数据：`merged_cancer.csv`（包含病名关键词）
- 处理后数据：`merged_cancer_no_keywords.csv`（删除病名和同义词）

**预期发现：**
- 某些癌症（如 Lung Cancer）可能有独特的标志性词汇，即使删除病名也能分类
- 某些癌症（如 Colon vs Stomach）可能在研究内容上高度重叠，删除关键词后难以区分
- 这个对比揭示了"表面特征" vs "深层语义"的分类能力差异

---

## 1. 是否可以新开 notebook？

**可以**，完全没问题。`error_analysis.ipynb` 独立运作。

---

## 2. 是否依赖前面代码？

**部分依赖**，但你可以选择：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **方案A: 独立新 notebook** | 干净、独立 | 需要重写数据加载和模型训练代码 |
| **方案B: 在 models.ipynb 继续添加 cell** | 共享变量，不用重写 | 文件变长 |

**推荐方案A**，因为 Error Analysis 是独立的任务。

---

## 3. 具体计划

### 第一阶段：数据准备（原始数据）

```python
# 1. 加载原始数据（带病名关键词）
df_original = pd.read_csv("merged_cancer.csv")
X_orig = df_original["cleaned_text"].astype(str)
y = df_original["label"]

# 2. 划分训练集和测试集（保持和队友一致的 random_state=42）
X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y, test_size=0.2, stratify=y, random_state=42
)

# 3. 训练冠军模型（SVM + TF-IDF）
pipeline_orig = make_tfidf_pipeline(LinearSVC())
pipeline_orig.fit(X_train, y_train)
y_pred_orig = pipeline_orig.predict(X_test)

# 4. 记录原始准确率
accuracy_original = (y_pred_orig == y_test).mean()
print(f"原始数据准确率: {accuracy_original:.4f}")
```

### 第二阶段：Error Analysis（原始数据）

```python
# 1. 获取预测结果
y_pred = pipeline_orig.predict(X_test)

# 2. 找出错误分类的样本
errors = X_test[y_test != y_pred]
error_labels = y_test[y_test != y_pred]
error_preds = y_pred[y_test != y_pred]

# 3. 分析错误模式
# - 哪些类别之间容易混淆？（混淆矩阵可视化）
# - 类别 0 和 4 为什么容易互相混淆？
```

**需要做的事情：**
- [ ] 绘制混淆矩阵热力图
- [ ] 打印错误样本，对比真实标签和预测标签
- [ ] 统计每对类别之间的混淆次数
- [ ] 尝试找出语义原因（例如：某些癌症可能有相似的研究方向）

### 第三阶段：SHAP 可解释性（原始数据）

```python
# 1. 提取 TF-IDF 向量器和模型
vectorizer = pipeline_orig.named_steps["tfidf"]
clf = pipeline_orig.named_steps["clf"]

# 2. 将测试集转为 TF-IDF 特征
X_test_tfidf = vectorizer.transform(X_test)

# 3. SHAP 分析（针对线性模型用 LinearExplainer）
import shap
explainer = shap.LinearExplainer(clf, X_test_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

# 4. 可视化
shap.summary_plot(shap_values, X_test_tfidf,
                  feature_names=vectorizer.get_feature_names_out())
```

**SHAP 需要做的事情：**
- [ ] 特征重要性排序：哪些词对分类影响最大？
- [ ] 每个类别的典型积极/消极特征
- [ ] 解释为什么某些类别容易混淆（共享关键词？）

### 第四阶段：Bias and Variability Analysis

```python
# 1. 从 10-fold CV 结果分析方差
cv_results[name]["all_scores"]  # 10个准确率

# 2. 绘制不同 fold 的准确率变化图
# 3. 分析模型稳定性
```

### 第五阶段：降低准确率实验（核心故事）

> **核心问题：** 删除病名和同义词后，模型还能分类吗？

```python
# ============================================
# 核心实验：使用删除病名后的数据
# ============================================

# 1. 加载删除病名后的数据
df_no_keyword = pd.read_csv("merged_cancer_no_keywords.csv")
X_no_kw = df_no_keyword["cleaned_text"].astype(str)

# 注意：标签 y 保持不变（与原始数据对应）

# 2. 使用相同的训练/测试集划分
# 注意：需要确保 X_no_kw 与 X_orig 的索引对应
X_train_no_kw = X_no_kw.iloc[X_train.index]
X_test_no_kw = X_no_kw.iloc[X_test.index]

# 3. 重新训练模型（相同的 pipeline）
pipeline_no_kw = make_tfidf_pipeline(LinearSVC())
pipeline_no_kw.fit(X_train_no_kw, y_train)
y_pred_no_kw = pipeline_no_kw.predict(X_test_no_kw)

# 4. 计算准确率下降
accuracy_no_keyword = (y_pred_no_kw == y_test).mean()
accuracy_drop = accuracy_original - accuracy_no_keyword

print(f"原始数据准确率: {accuracy_original:.4f}")
print(f"删除病名后准确率: {accuracy_no_keyword:.4f}")
print(f"准确率下降: {accuracy_drop:.4f} ({accuracy_drop/accuracy_original*100:.1f}%)")
```

**分析任务：**
- [ ] 对比两个准确率，计算下降百分比
- [ ] 绘制对比柱状图
- [ ] 分析哪些类别受影响最大/最小
- [ ] **故事化解读：**
  - 为什么某些癌症删除关键词后仍然可分？（有独特的专业术语）
  - 为什么某些癌症删除关键词后几乎无法分？（研究内容高度重叠）

### 第六阶段：SHAP 对比分析（可选，深度 story）

```python
# 对删除关键词后的数据进行 SHAP 分析
# 对比哪些特征现在变成了重要特征
X_test_no_kw_tfidf = pipeline_no_kw.named_steps["tfidf"].transform(X_test_no_kw)
explainer_no_kw = shap.LinearExplainer(
    pipeline_no_kw.named_steps["clf"], X_test_no_kw_tfidf
)
shap_values_no_kw = explainer_no_kw.shap_values(X_test_no_kw_tfidf)

# 对比两个模型的 top 特征
```

---

## 4. 代码依赖图

```
error_analysis.ipynb
      │
      ├── 需要：merged_cancer.csv（原始数据）
      ├── 需要：merged_cancer_no_keywords.csv（删除病名后的数据）
      │         ↓
      ├── 需要：make_tfidf_pipeline 函数（复制粘贴即可）
      │
      └── 需要：random_state=42（与队友保持一致）
```

---

## 5. 时间估算

| 任务 | 预计时间 |
|------|----------|
| 数据加载 + 训练两个模型 | 15 分钟 |
| 混淆矩阵 + 错误样本分析 | 30 分钟 |
| SHAP 分析（原始数据） | 30 分钟 |
| 删除关键词实验 | 20 分钟 |
| SHAP 对比分析（可选） | 30 分钟 |
| Bias and Variability 分析 | 15 分钟 |
| 可视化整理 + 故事撰写 | 30 分钟 |
| **总计** | **约 2.5 小时** |

---

## 6. 待办清单

- [ ] 创建 error_analysis.ipynb
- [ ] 数据加载与两个模型训练（原始 + 删除病名）
- [ ] 原始数据：混淆矩阵可视化
- [ ] 原始数据：错误样本分析与统计
- [ ] 原始数据：SHAP 特征重要性分析
- [ ] 删除病名后的准确率实验
- [ ] 准确率对比可视化
- [ ] 类别受影响程度分析
- [ ] Bias and Variability 分析
- [ ] **撰写 Story**：删除关键词后的发现
- [ ] 整理可视化图表用于报告

---

## 7. 输出要求（对应作业评分项）

| 评分项 | 输出内容 |
|--------|----------|
| Explainable Evaluations (2%) | SHAP summary plot, 特征重要性图表 |
| Error Analysis + SHAP (2%) | 错误样本分析 + SHAP 可解释性 |
| Visualizations (2%) | 混淆矩阵热力图, 准确率对比图, CV 波动图 |
| Bias and Variability (2%) | 方差分析, 模型稳定性评估 |
| AUC (1%) | ROC 曲线, 预测难度分析 |
| Story 讲述 | 删除病名后的发现，语义重叠分析 |

---

## 8. 核心故事框架 (Report/PPT 用)

### 标题：模型是否真的"理解"了癌症论文，还是只是在找关键词？

### 故事线：
1. **现状：** SVM 模型在原始数据上达到 XX% 准确率
2. **问题：** 模型是否依赖病名关键词来分类？
3. **实验：** 删除所有病名和同义词，重新训练和测试
4. **发现：**
   - 整体准确率从 XX% 下降到 XX%（下降 XX%）
   - 某些癌症（如 XXX）仍然可以区分，因为有独特的专业术语
   - 某些癌症（如 XXX 和 XXX）无法区分，因为研究内容高度重叠
5. **洞察：**
   - 表面特征（病名）vs 深层语义（研究内容）
   - 语义重叠：某些癌症的研究方向、治疗方法可能相似
   - 模型的"智能"程度评估

### 图表需求：
- [ ] 原始 vs 删除病名后的准确率对比柱状图
- [ ] 每个类别准确率变化的对比图
- [ ] 混淆矩阵对比（原始 vs 删除病名后）
- [ ] SHAP 特征重要性对比
- [ ] 10-fold CV 准确率波动图
