# Fourier-Guided Two-Domain Adaptive Optimization for Enhanced Self-Supervised Real-World Image Denoising

![main_fig](./figs/mainFigure.png)


## Abstract
_In the real-world image denoising task, the self-supervised method uses the data’s inherent attributes as the supervised signal to train the model without clean images. However, the existing methods generally rely on pixel-level spatial domain where noise and high-frequency details have similar local statistical characteristics, which makes the model unable to adaptively distinguish and excessively suppress effective features. Moreover, the locality of the convolution makes it difficult to model the global noise with long-range correlation. In this paper, we propose a straight forward approach that utilizes Fourier insights to interact with spatial domains. Firstly, through the Fourier transform we explored the physical property that amplitude mainly encodes noise and phase contains structural information. Based on this prior, Amplitude Spectrum Adaptive Weight(ASAW) module captures global noise distribution and explicitly suppresses the noise-dominated frequency band, enhances the effective low-high frequency components, while maintaining the integrity of the phase to avoid image distortion. Additionally, Multi-Scale Fusion (MSF) dynamically adjusts the contribution of different branches. We also propose Bilateral Filtering Alleviates Artifacts (BFAA), which suppresses artifacts via adaptive low-pass filtering, and preserves edge sharpness through intensity-aware weighting. Extensive experiments on SIDD and DND datasets verify that our method is significantly better than the existing self-supervised denoising methods and achieves high efficiency._
---

## Setup

### Requirements

Our experiments are done with:

- Python 3.8.20
- PyTorch 1.8.2
- numpy 1.24.4
- opencv 4.10.0
- scikit-image 0.21.0

### Directory

Follow the below descriptions to build the code directory.

```
TDC-SSD
├─ ckpt
├─ conf
├─ dataset
│  ├─ DND
│  ├─ SIDD
│  ├─ NIND
│  ├─ prep
├─ figs
├─ output
├─ src
```

### Dataset
- In this paper, we use SIDD validation, SIDD benchmark, DND benchmark datasets to evaluate performance:
  - SIDD validation and SIDD benchmark: https://www.eecs.yorku.ca/~kamel/sidd/
  - DND benchmark: https://noise.visinf.tu-darmstadt.de/

---

## Training & Test

### Training

```
# Train TDC-SSD for the SIDD dataset using gpu:0
python train.py -c TDCSSD_SIDD -g 0
```

### Test

```
# Test SIDD dataset for 25 epoch model in gpu:0
python test.py -c TDCSSD_SIDD -g 0 -e 25
```

---

## Results

### Quantitative results

Here are the reported results of EEFM-BAN.

![results](./figs/quantitative_result.png)

### Qualitative results

![visual](./figs/qualitative_result.png)