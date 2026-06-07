# PriMP

Official implementation of **PriMP**: *Visual-Semantic Alignment for Cross-Domain Few-Shot Object Detection via Multimodal Prototype Calibration*.

PriMP is a lightweight multimodal prototype calibration framework for cross-domain few-shot object detection. It is built upon vision-language foundation detectors and aims to improve target-domain detection by combining source-domain semantic priors with target-domain visual evidence.

## Overview

This repository provides the core implementation of PriMP, including:

- semantically enhanced prompt construction;
- offline multimodal prototype-bank construction;
- visual prototype and textual prototype utilities;
- uncertainty-aware adaptive gating;
- prototype-based inference-time logit calibration;
- example demo scripts and reproduction notes.

## Planned Repository Structure

This repository focuses on the core implementation of PriMP based on GroundingDINO. The released files are organized around semantically enhanced prompt construction, offline multimodal prototype construction, and inference-time prototype calibration.

```text
PriMP/
├── README.md
├── requirements.txt
├── groundingdino/
│   ├── models/
│   │   └── proto_from_support.py
│   ├── prototype/
│   │   └── prototype_builder.py
│   ├── tools/
│   │   ├── build_mm_proto_from_shot.py
│   │   ├── make_text_protos.py
│   │   └── eval_proto_branch.py
│   └── util/
│       └── syn_def_prompt.py
├── demo/
│   └── test_ap_on_coco_proto.py
└── docs/
    ├── data_preparation.md
    └── reproduction_notes.md
```

## Installation

```bash
git clone https://github.com/liuwy509/PriMP.git
cd PriMP
pip install -r requirements.txt
```

## Main Components

### Semantically Enhanced Prompt Construction

PriMP enriches category prompts by using class names, synonyms, and natural-language definitions. The original class-name prompt is retained as the primary textual input, while additional semantic prompts are used as supplementary evidence when reliable lexical information is available.

The related implementation is mainly provided in:

```text
groundingdino/util/syn_def_prompt.py
groundingdino/tools/make_text_protos.py
```

### Offline Multimodal Prototype Bank

Given a K-shot support set, PriMP constructs class-wise visual prototypes from support-region features and textual prototypes from semantically enhanced prompts. The resulting prototype bank is constructed offline and remains fixed during inference.

The related implementation is mainly provided in:

```text
groundingdino/models/proto_from_support.py
groundingdino/prototype/prototype_builder.py
groundingdino/tools/build_mm_proto_from_shot.py
```

### Uncertainty-Aware Adaptive Gating

PriMP adopts an entropy-margin adaptive gating strategy to dynamically balance the semantic branch and the prototype branch. The gate increases the contribution of target-domain prototype evidence when the text branch is uncertain and the prototype branch provides discriminative evidence.

The related implementation is included in:

```text
groundingdino/tools/eval_proto_branch.py
```

### Inference-Time Logit Calibration

During inference, PriMP fuses semantic-branch logits and prototype-branch logits in the logit space. This calibration process improves visual-semantic alignment under cross-domain few-shot settings while introducing only lightweight additional computation.

The related implementation is included in:

```text
groundingdino/tools/eval_proto_branch.py
```

## Data and Checkpoints

Large pretrained weights, full public datasets, processed annotations, trained checkpoints, and experiment logs are not redistributed in this repository due to dataset licenses and file-size limitations.

Users should download the original datasets and pretrained GroundingDINO or GLIP checkpoints from their official sources.

## Reproduction Notes

This repository provides the core PriMP implementation and example scripts. Exact reproduction of all experimental tables in the paper requires:

1. preparing the corresponding CD-FSOD benchmark datasets;
2. downloading the pretrained detector checkpoints;
3. following the few-shot split protocol described in the manuscript;
4. constructing shot-specific prototype banks;
5. running evaluation under the same COCO-style metrics.

More details are provided in `docs/data_preparation.md` and `docs/reproduction_notes.md`.

## License

This repository is released for academic research purposes only.
