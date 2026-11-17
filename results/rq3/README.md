# RQ3历史预测结果

保存GPT-5.4五种策略，以及Gemini 3.1 Pro Preview、Claude Opus 4.7在“定义＋few-shot＋错误反馈”策略下的预测。7份JSONL均为原件，每份215条，样本键与人工gold一致。

Opus采用文件名带“贵”的最终结果，与论文Accuracy≈0.865、F1≈0.794一致。未采用另一份Accuracy≈0.856的note版本。

metrics_original.txt为原报告；metrics.json和summary.csv由保存的最终预测重新计算。本目录保存论文实验结果；重新运行时请通过输出参数指定单独目录。重新运行最新提示不能自动视为复现了这些历史预测。
