# Implementation Layout

The supplied folder contains experiment outputs, not the original implementation. The organization follows [cls_KD](https://github.com/yzd-v/cls_KD/tree/1.0), whose method implementations live in `mmcls/models/dis_losses/` and whose configs live in `configs/distillers/`.

When the original UGKD source is available, include:

| Original component | Intended location |
| :--- | :--- |
| UATM and UGSA losses | `mmcls/models/dis_losses/`, matching the original registrations |
| ORI loss | Original module containing `ORILoss` |
| ClassificationDistiller | Original distiller module and registration |
| Teacher definition | `projects/_models_/cifar100_resnet32x4.py` |
| Student definition | `projects/_models_/cifar100_resnet8x4.py` |
| Training and testing entry points | `tools/`, preserving the original workflow |
| Teacher checkpoint | Separate model download with original file reference |
| Environment | Original dependency file and framework version |

No empty loss functions or inferred algorithm implementations are supplied. The module names in the experiment config are not sufficient to reconstruct the mathematical definitions of UATM or UGSA.
