# Paper tables

保存当前论文中的14张表格，使用描述性的英文文件名。来源为5RQ小论文/samples/acmcp.tex，精确路径、源文件哈希、表格label和行号见TABLE_SOURCES.json。

本轮保存原始LaTeX表格片段，未修改数值、未重新计算指标，尚未导出CSV。片段依赖论文的LaTeX文档环境，不是独立可编译文档。快照对应当前可用论文，不能代替作者对最新终稿的确认。

RQ1/RQ2的逐样本依据保留在dataset；对应统计表统一存放在这里，不再建立空的results/rq1和results/rq2。RQ3–RQ5与Discussion保留逐样本实验结果，tables保存论文层面的汇总展示。

| File | Paper table |
|---|---|
| [translation_task_counts.tex](translation_task_counts.tex) | Counts of Python source programs and generated Java translations. |
| [baseline_translation_performance.tex](baseline_translation_performance.tex) | Computational Accuracy of Generated Java Translations |
| [type_error_prevalence.tex](type_error_prevalence.tex) | Distribution of Type Errors and Non-Type Errors |
| [type_error_taxonomy.tex](type_error_taxonomy.tex) | Distribution of type-error categories in the analyzed samples. |
| [identification_prompt_comparison.tex](identification_prompt_comparison.tex) | Binary classification performance of GPT-5.4 under different prompting strategies |
| [identification_model_comparison.tex](identification_model_comparison.tex) | Comparison of three models under the selected prompting strategy. |
| [identification_consensus_performance.tex](identification_consensus_performance.tex) | Comparison of single-model and dual-model consensus classification performance. |
| [prevention_overall_performance.tex](prevention_overall_performance.tex) | Overall Results of pre-translation prevention. |
| [prevention_model_performance.tex](prevention_model_performance.tex) | Pre-translation prevention results by model. |
| [repair_overall_performance.tex](repair_overall_performance.tex) | Overall results of post-translation repair. |
| [repair_model_performance.tex](repair_model_performance.tex) | Post-translation repair results by model. |
| [generation_latency.tex](generation_latency.tex) | Mean wall-clock inference time per input file for pre-translation prevention and post-translation repair. |
| [software_metrics_groups.tex](software_metrics_groups.tex) | Type Errors among Translation Failures by Lines of Code (LOC) and Cyclomatic Complexity (CC) |
| [fine_grained_classification.tex](fine_grained_classification.tex) | Fine-grained type-error classification performance ($n=70$). |
