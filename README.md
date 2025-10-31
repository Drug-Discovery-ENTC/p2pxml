# P2PXML: Deep Geometric Framework to Predict Antibody-Antigen Binding Affinity

[Link for Paper at Journal of Structural Biology](https://authors.elsevier.com/a/1m0Z53kGCFRMZX) | [Preprint Link](https://www.biorxiv.org/content/10.1101/2024.06.09.598103v1) | [Project page for P2PXML](https://drug-discovery-entc.github.io/p2pxml/)

## Data

The link for our curated dataset (P2PXML) can be found [here](https://zenodo.org/records/11531319).
Example PDB files for Colab demonstration can be found [here](https://github.com/Drug-Discovery-ENTC/p2pxml/tree/main/data).

## Colab Minimal Demonstration

The link for minimal demonstration for predicting binding affinity using the proposed combined model is on [Colab](https://colab.research.google.com/drive/1pDAU2Jizu3kZ2skybhhb45UcBJ2skYxY?usp=sharing). 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pDAU2Jizu3kZ2skybhhb45UcBJ2skYxY?usp=sharing)

## Results

<img src="https://github.com/Drug-Discovery-ENTC/p2pxml/blob/main/resources/scatter2.png" width="960" height="540"><br />
log10(IC50) of the predicted values vs log10(IC50) of the target values for the test set of P2PXML-PDB dataset using our best performing model. The Pearson correlation coefficient is 0.8703 and the Spearman’s correlation coefficient is 0.9450 between the predicted and target values.

## Citation

If you find our work, including this repository, geometric models and P2PXML dataset useful, please consider giving a star ⭐ and citing our [paper](https://www.biorxiv.org/content/10.1101/2024.06.09.598103v1).
```bibtex
@article{BANDARA2025108257,
title = {Deep geometric framework to predict antibody–antigen binding affinity},
journal = {Journal of Structural Biology},
volume = {217},
number = {4},
pages = {108257},
year = {2025},
issn = {1047-8477},
doi = {https://doi.org/10.1016/j.jsb.2025.108257},
url = {https://www.sciencedirect.com/science/article/pii/S1047847725000929},
author = {Nuwan Bandara and Dasun Premathilaka and Sachini Chandanayake and Sahan Hettiarachchi and Vithurshan Varenthirarajah and Aravinda Munasinghe and Kaushalya Madhawa and Subodha Charles}
}
```

## Note

This repository will be updated step-wise in the near future. 

## Issues 

If found any issues or need clarifications, please open an issue in the repository or contact the corresponding author (Nuwan) via pmnsribandara@gmail.com.
