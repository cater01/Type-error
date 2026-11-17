# 实验配置

完整入口索引见[INDEX.json](INDEX.json)。每份JSON包含experiment元数据和args参数。元数据保存入口、来源、核对状态及unknown项目；args字段对应CLI参数，例如out_dir对应--out-dir。

参数优先级为CLI > JSON > 环境变量/默认。相对路径均以仓库根目录为基准。错误参数名、无效choices、字符串布尔值及缺失的路径环境变量会被拒绝。

| 配置组 | 数量 | 用途 |
|---|---:|---|
| baseline | 12 | 三模型×四数据集P0；历史设置未完全恢复 |
| prevention | 9 | 三模型×P0/P1/P2，显式采样参数及冻结hint |
| identification/evaluation | 7 | GPT五策略及Gemini/Opus最新实现 |
| identification/prediction | 2 | Gemini/Opus最新批量标注配置 |
| identification/second_round | 2 | 分歧复判输入输出配置 |
| identification/agreement.json | 1 | 预测对齐分组 |
| evaluation | 4 | 四数据集测试 |
| repair | 6 | Qwen7B/32B×R0/R1/R2 |
| analysis | 4 | 识别指标、耗时、跨策略统计 |

共47份可解析配置。prompts/prevention_recorded.json是提示资源，不是执行配置。

## 来源状态

- partially_log_verified：observed_parameters由原日志记录，但仍有unknown。
- source_snapshot_with_saved_prompt_check：参数来自源码，提示与保存样本核对过，不代表所有历史参数均已确认。
- source_snapshot/latest_source_snapshot：冻结当前代码设置，不能声称完全还原历史论文运行。
- recorded_table_audit：保留原任务分母和论文参考计数的统计工具。

原Magicoder预防日志记录sampling、0.7、0.95、10、8192；Qwen预防日志没有完整记录这些字段。Qwen修复日志也记录了相同采样参数。权重revision、历史种子和原环境仍未知，保留null并列入unknown。

预防bundle来自与已保存prompt匹配的enhance_qwen.py，9个模型/策略组合各抽查一份。P0为空hint，其匹配本身不构成完整prompt验证；全部历史生成过程未被认证。默认代码保留最新提示，只有选中bundle才使用冻结hint。

## 外部输入

- 四数据集源码、测试、模板和数据库：外部资源目录或显式CLI路径。
- 最终215条输入、全部待标注失败样本、带源码的二轮分歧输入：本轮未创建，未用AI标签重新抽样替代人工评测集。
- 权重：--model-path指定本地目录，--model指定模型标识，可指定--model-revision。
- API：环境变量提供密钥，配置明确endpoint，可用CLI覆盖。

资源获取与数据准备说明见[README](../README.md#benchmark-resources)，大型文件无需上传。

当前Magicoder的7份配置为user_requested_configuration：greedy=false、do_sample=true、temperature=0.7。原observed_parameters保留历史事实。新增analysis/historical_results.json可复算随仓库保存的RQ4/RQ5轻量结果。

## 已保存的历史结果

RQ3七组预测在results/rq3；Dis4三模型四分类结果在results/discussion/dis4。RQ4/RQ5保存完整生成任务记录、原summary和精简测试记录，不含Java或大体积测试输入输出。结果来源和原件哈希见results/RESULTS_MANIFEST.json。

## Single-file dataset release

The prevention and repair configurations read `dataset/type_error_dataset.jsonl` directly. The `dataset` argument selects cases by their original translation model (`magicoder`, `qwen7b`, or `qwen32b`), using the record's `model` field.
