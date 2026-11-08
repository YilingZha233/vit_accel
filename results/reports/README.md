# Measurement reports

- **Latest accuracy evidence:** [1,000-image HPS FP32/native-INT8 report, imported 8 November 2026](hps_accuracy1000_2026-11-08.md), [numeric audit](hps_accuracy1000_2026-11-08_audit.json), [preserved raw import](../raw/hps_accuracy1000_2026-11-08/).
- [Progress tracker](cpu_baseline_progress_audit.md): current A–F status followed by historical audit.
- [October measurement snapshot](current_measurements_2026-10-15.md): kernel, full-model pilots, numerical quantization, operator profiles and engineering analysis.
- [First HPS kernel report](hps_board_run1.md) and [numeric audit](hps_board_run1_audit.json).

The 1,000-image study measured 84.9% FP32 versus 83.6% native INT8 top-1 accuracy (1.3 percentage-point loss) on the same Kaggle ImageNet-mini validation subset. HPS CPU execution is measured; FPGA-fabric acceleration remains pending. Native ncnn INT8 is distinct from Q1–Q4 Python numerical simulations.

Use the [full CPU baseline guide](../../docs/full_cpu_baseline_steps.md) for remaining measurements. Preserve raw evidence, configuration, dataset scope and run IDs with every reported result.
