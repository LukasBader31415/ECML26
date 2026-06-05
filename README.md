# ECML26 M3GLVQ Repository

This repository contains the code used for the ECML paper on structural analysis of dissimilarity spaces, single-view MGLVQ baselines, and multi-view M3GLVQ.

The workflow is organized into three main blocks that follow the logic of the paper:

1. **Structural analysis of the individual dissimilarity spaces**
2. **Single-view MGLVQ baselines**
3. **Multi-view M3GLVQ analysis**

In addition, the repository contains an analysis layer that combines the outputs of these three blocks and produces the final paper tables and figures.

---

## Repository structure

```text
ecml26-m3glvq/
├── data/
│   └── raw/
├── outputs/
│   ├── structural/
│   ├── single_view_mglvq/
│   ├── m3glvq/
│   └── analysis/
├── src/
│   ├── data_loading/
│   ├── common/
│   ├── structural/
│   ├── single_view_mglvq/
│   ├── m3glvq/
│   └── analysis/
└── scripts/ or notebooks/