# Results

```text
results/
├── rq3/          GPT五策略及Gemini/Opus最好策略的历史预测
├── rq4/          预防生成记录、精简测试记录与汇总
├── rq5/          修复生成记录、精简测试记录与汇总
├── discussion/
│   ├── dis2/                原始软件度量CSV与LOC/CC结果表
│   └── dis4/                三模型的四分类预测
├── tables/                 论文14张表格的英文命名LaTeX快照
└── RESULTS_MANIFEST.json    历史预测/测试记录的来源和转换说明
```

RQ1/RQ2的逐样本结果位于 `dataset/type_error_dataset.jsonl`：`label_0` 保存类型错误标签，`label_1` 保存已有的细分类标注。论文汇总表存入 `tables/`。

RQ3保存7组×215条预测；Dis4保存3组×70条预测。RQ4/RQ5保留完整任务记录与精简测试结果，未复制Java或大体积诊断文本。

```powershell
python src/analysis/analyze_experiment_results.py --config configs/analysis/historical_results.json
```

完整Java、原始响应和详细测试诊断可以单独压缩归档，并在获取说明中给出位置。当前Magicoder采用sampling、temperature=0.7；历史日志和预测保持原样。

tables使用当前论文源码，不要求用户再次提供相同数据；若作者终稿已有新版本，应更新该来源快照。CSV数值导出尚未执行。
