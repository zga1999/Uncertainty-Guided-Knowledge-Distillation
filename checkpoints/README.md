# Checkpoints

| Dataset | Teacher | Student | Top-1 (%) | Epoch | Format | Publication |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| CIFAR-100 | ResNet32x4 | ResNet8x4 | 77.20 | 240 | Full distillation checkpoint | Prepared locally; not uploaded |

Release asset filename:

```text
ugkd_cifar100_resnet32x4_resnet8x4_uatm_ugsa_20260218_180916_epoch240.pth
```

- Original filename: `best_accuracy_top1_epoch_240.pth`.
- Size: 59,863,977 bytes (approximately 57.09 MiB).
- SHA-256: `662210d7921358a405f2364fcc7914086ecc62a8e5e39c06c7f3bef1f80b1921`.
- Includes both teacher and student parameter names; no student-only export is provided.
- Checkpoint metadata records epoch 240; no model evaluation was executed during preparation.

The weight is prepared as a separate release asset outside the repository package. Publish it through GitHub Releases, then add its real download URL to the result table. No download link is shown until the file is available.

GitHub's [large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) sets a 25 MiB limit for browser uploads to repository files. This model exceeds that limit; a Release asset follows the reference repository's practice of distributing weights separately.
