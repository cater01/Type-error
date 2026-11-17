# 统计入口

三个脚本支持--config、--dry-run和显式输入输出。配置在configs/analysis。

- identification_metrics.py：原gold/pred分类指标，可--out保存报告。
- time_statistics.py：elapsed_seconds均值，支持位置参数files或JSON中的files列表。
- analyze_experiment_results.py：分别配置预防/修复测试与任务全集根目录，沿用原分母及论文计数。

RQ1/RQ2的dataset_statistics.py尚未创建。
