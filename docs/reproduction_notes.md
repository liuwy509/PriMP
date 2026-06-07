# Reproduction Notes

This repository provides the core implementation of PriMP, including semantic prompt construction, offline multimodal prototype construction, uncertainty-aware adaptive gating, and inference-time prototype calibration.

Exact reproduction of all experimental tables in the paper requires additional preparation steps, including dataset downloading, checkpoint preparation, few-shot split construction, prototype-bank construction, and COCO-style evaluation.

## Required External Resources

The following resources are not redistributed in this repository:

- full public datasets;
- pretrained GroundingDINO or GLIP checkpoints;
- processed annotation files;
- trained checkpoints;
- experiment logs;
- intermediate result files.

Users should download the original datasets and pretrained checkpoints from their official sources.

## General Reproduction Pipeline

A typical PriMP evaluation pipeline contains the following steps:

1. Prepare the target CD-FSOD dataset.
2. Prepare the K-shot support set following the split protocol described in the manuscript.
3. Build visual and textual prototypes from the support set.
4. Save the offline multimodal prototype bank.
5. Run inference-time prototype calibration.
6. Evaluate detection results using COCO-style metrics.

## Notes

This repository focuses on the core PriMP method implementation rather than a one-click reproduction package for all experimental tables.

Dataset-specific paths, large checkpoints, generated prototype files, and experiment logs should be prepared locally by the user.
