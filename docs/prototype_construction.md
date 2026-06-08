# Prototype Construction

PriMP constructs an offline multimodal prototype bank before inference. The prototype bank provides target-domain visual evidence and textual semantic priors for inference-time calibration.

This document focuses on the prototype construction pipeline. Dataset organization and few-shot split preparation are described in `docs/data_preparation.md`.

## Inputs

Prototype construction requires three types of inputs:

1. A K-shot support annotation file, such as `1_shot.json`, `5_shot.json`, or `10_shot.json`.
2. The corresponding support images.
3. Category names and optional semantic prompt information, including synonyms and natural-language definitions.

The support annotations should follow the COCO-style format described in `docs/data_preparation.md`.

## Visual Prototypes

Visual prototypes are constructed from support-region features.

For each target category, PriMP extracts region-level features from the K-shot support images and aggregates them into a class-wise visual prototype.

The visual prototype represents target-domain appearance evidence.

## Textual Prototypes

Textual prototypes are constructed from semantically enhanced category prompts.

Each prompt keeps the original category name as the primary input. When available, WordNet synonyms and natural-language definitions are added as supplementary semantic evidence.

The textual prototype represents source-domain semantic prior knowledge.

## Multimodal Prototype Bank

The visual and textual prototypes are organized into an offline multimodal prototype bank.

The prototype bank is fixed during inference and is used by PriMP for prototype-based logit calibration.

Generated prototype files are intermediate artifacts and should be kept locally instead of being committed to the repository.

## General Pipeline

A typical prototype construction pipeline is:

```text
K-shot support annotations
        ↓
support-region feature extraction
        ↓
visual prototype construction
        ↓
semantic prompt construction
        ↓
textual prototype construction
        ↓
offline multimodal prototype bank
        ↓
inference-time prototype calibration
```

## Related Implementation

The prototype construction and prompt construction pipeline is mainly related to:

```text
groundingdino/models/proto_from_support.py
groundingdino/tools/build_mm_proto_from_shot.py
groundingdino/tools/make_text_protos.py
groundingdino/util/syn_def_prompt.py
```

The inference-time prototype calibration step is mainly related to:

```text
groundingdino/tools/eval_proto_branch.py
```
## Notes

This repository provides the core implementation of the prototype construction pipeline.

Dataset-specific paths, support annotation paths, checkpoint paths, generated prototype files, and experiment outputs should be configured locally by the user.
