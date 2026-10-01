# Uncertainty-Guided Knowledge Distillation

Experiment artifacts for **Uncertainty-Guided Knowledge Distillation (UGKD)** on image classification. The available experiment combines **UATM** and **UGSA** on CIFAR-100 with a ResNet32x4 teacher and a ResNet8x4 student.

[Experiment Details](ugkd.md) | [Configs](configs/distillers/cifar100/ugkd/) | [Results](results/summary.csv) | [Checkpoint Notes](checkpoints/README.md)

## Results

| Dataset | Student | Teacher | Baseline Top-1 (%) | +UGKD Top-1 (%) | Epoch | Config | Weight |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CIFAR-100 | ResNet8x4 | ResNet32x4 | Not provided | **77.20** | 240 | [config](configs/distillers/cifar100/ugkd/resnet32x4_resnet8x4.py) | [details](checkpoints/README.md) |

The result is from **one run**, selected by the best per-epoch `accuracy/top1` reported by the validation loop. Its configured validation dataset uses the **CIFAR-100 test split**. An independent test evaluation, student baseline, and multi-seed statistics were not supplied.

## Install

The result summary tool uses the Python standard library. Generating figures additionally requires Matplotlib:

```bash
python -m pip install -r requirements-analysis.txt
```

The recorded training environment and missing training dependencies are listed in [ugkd.md](ugkd.md). This artifact release includes configuration and training records; the UATM/UGSA implementation and training entry points have not been supplied.

## Run

Regenerate the result tables and figures from the included original scalar log:

```bash
python tools/summarize_results.py \
  --scalars results/cifar100/resnet32x4_resnet8x4/20260218_180916/scalars.json \
  --output results/cifar100/resnet32x4_resnet8x4/20260218_180916 \
  --expected-epochs 240 \
  --figures imgs
```

See [ugkd.md](ugkd.md) for the training configuration, checkpoint format, and the files needed for training and evaluation.

## Acknowledgement

The repository organization follows the user-selected [cls_KD](https://github.com/yzd-v/cls_KD/tree/1.0) pattern: a project README, a method-specific document, Python configs under `configs/distillers/`, figures under `imgs/`, and separate model weights. The experiment uses MMClassification/MMEngine interfaces as recorded in its configuration and logs. No third-party implementation has been copied into this artifact package.

## Paper and Citation

The paper URL, author list, venue, and citation metadata have not been supplied. They can be added when the publication information is available.
