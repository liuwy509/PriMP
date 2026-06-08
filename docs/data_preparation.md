# Data Preparation

PriMP is evaluated under the cross-domain few-shot object detection setting on six target-domain datasets:

- ArTaxOr
- Clipart1K
- DeepFish
- DIOR
- NEU-DET
- UODD

Due to dataset licenses and file-size limitations, the full datasets and processed annotations are not redistributed in this repository. Users should download the original datasets from their official sources and organize them locally before running PriMP.

## Dataset Organization

A recommended local directory structure is shown below:

```text
data/
├── ArTaxOr/
│   ├── train/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── test.json
│       ├── 1_shot.json
│       ├── 5_shot.json
│       └── 10_shot.json
├── Clipart1K/
│   ├── train/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── test.json
│       ├── 1_shot.json
│       ├── 5_shot.json
│       └── 10_shot.json
├── DeepFish/
│   ├── train/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── test.json
│       ├── 1_shot.json
│       ├── 5_shot.json
│       └── 10_shot.json
├── DIOR/
│   ├── train/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── test.json
│       ├── 1_shot.json
│       ├── 5_shot.json
│       └── 10_shot.json
├── NEU-DET/
│   ├── train/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── test.json
│       ├── 1_shot.json
│       ├── 5_shot.json
│       └── 10_shot.json
└── UODD/
    ├── train/
    ├── test/
    └── annotations/
        ├── train.json
        ├── test.json
        ├── 1_shot.json
        ├── 5_shot.json
        └── 10_shot.json
```

## Annotation Format

PriMP expects COCO-style annotation files. Each annotation file should contain the standard fields:

```text
images
annotations
categories
```

The `test.json` file is used for final evaluation. The `1_shot.json`, `5_shot.json`, and `10_shot.json` files are used to construct the corresponding few-shot support sets.

## Few-Shot Support Sets

For each target dataset, PriMP follows the 1-shot, 5-shot, and 10-shot evaluation settings used in the manuscript.

In general, K-shot means that each target category contains K support images. The same support split should be used for both the baseline and PriMP to ensure fair comparison.

If official few-shot splits are available, users should use the official splits. Otherwise, support sets can be sampled from the training split with a fixed random seed.

## Prototype Construction

After preparing the support annotations, PriMP constructs an offline multimodal prototype bank for each dataset and each shot setting.

A typical prototype-bank organization is:

```text
outputs/
└── prototypes/
    ├── ArTaxOr/
    │   ├── k1.pt
    │   ├── k5.pt
    │   └── k10.pt
    ├── Clipart1K/
    │   ├── k1.pt
    │   ├── k5.pt
    │   └── k10.pt
    ├── DeepFish/
    │   ├── k1.pt
    │   ├── k5.pt
    │   └── k10.pt
    ├── DIOR/
    │   ├── k1.pt
    │   ├── k5.pt
    │   └── k10.pt
    ├── NEU-DET/
    │   ├── k1.pt
    │   ├── k5.pt
    │   └── k10.pt
    └── UODD/
        ├── k1.pt
        ├── k5.pt
        └── k10.pt
```

The generated prototype files are intermediate files and should not be committed to this repository.

## Notes

Dataset-specific paths should be modified according to the local environment.

Large datasets, generated prototype files, processed annotations, checkpoints, logs, and experiment outputs should not be committed to this repository.
