# Feature Densities — Uncertainty Estimation

This folder contains the implementation of **feature density–based uncertainty estimation**, where embeddings from Whisper are used to evaluate how “typical” or “atypical” each audio segment is.

## Contents

- **Main experiment notebook**  
  Implements:
  - Extraction of Whisper encoder features  
  - Computing density-based uncertainty scores
  - Correlation with transcription error
  - WER and calibration analysis
  - Visualization of density distributions and error relationships
