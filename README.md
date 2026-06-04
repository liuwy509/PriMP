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
- example configuration and demo scripts.

## Repository Structure

```text
PriMP/
├── README.md
├── requirements.txt
├── primp/
│   ├── prompt_builder.py
│   ├── prototype_bank.py
│   ├── adaptive_gate.py
│   └── logit_fusion.py
├── configs/
│   └── example_neudet.yaml
├── examples/
│   └── demo_primp_gate.py
└── docs/
    ├── data_preparation.md
    └── reproduction_notes.md
