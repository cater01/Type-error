# Pre-translation prevention

The nine configurations cover three models and P0/P1/P2. All read `dataset/type_error_dataset.jsonl` directly and select the corresponding original translation model through the `dataset` argument and each record's `model` field.

Outputs are saved to `results/rq4/generation/<model>/<strategy>`. Evidence and unresolved historical settings are recorded in each configuration.
