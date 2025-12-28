# DavenportAutoencoderClusterABG
Unsupervised deep learning framework for identifying acid–base phenotypes in ARDS using arterial blood gas data. Includes autoencoder-based embeddings, clustering, Davenport diagram visualizations, trajectory analysis, and mortality associations.


## Overview
This repository contains code for an unsupervised deep learning framework designed to characterize acid–base phenotypes in acute respiratory distress syndrome (ARDS) using arterial blood gas (ABG) data.

By embedding classical acid–base physiology into a continuous latent space, this approach identifies distinct physiologic clusters and trajectories that are not fully captured by traditional acid–base categories.

## Key Features
- Autoencoder-based dimensionality reduction of ABG variables
- Unsupervised clustering (k-means / GMM)
- Davenport diagram visualization with machine learning overlays
- Trajectory analysis from day 0 to day 3
- Davenport distance as a continuous measure of acid–base deviation
- Multivariable-adjusted mortality modeling
- Publication-ready tables and figures

## Data
The code is designed to operate on de-identified clinical trial datasets (e.g., ALTA / ARDSNet-style formats).  
**No raw patient data are included in this repository.**

## Methods Summary
- Input variables: pH, PaCO₂, bicarbonate, PaO₂, FiO₂, ventilator parameters
- Latent embeddings learned via autoencoder
- Clustering performed in latent space
- Statistical analyses include ANOVA, χ² tests, Cramér’s V, and adjusted logistic regression

## Outputs
- Table 1: Baseline demographic and physiologic characteristics by cluster
- Davenport diagrams (Day 0, Day 3, trajectories)
- Cluster physiologic signature heatmaps
- Davenport distance boxplots
- Adjusted mortality forest plots

## Requirements
- Python ≥ 3.9
- pandas, numpy, scikit-learn
- statsmodels
- matplotlib, seaborn

## Citation
If you use this code in academic work, please cite the associated manuscript or abstract.

## Author
Samuel Y. Huang, MD  
Pulmonary & Critical Care Medicine

