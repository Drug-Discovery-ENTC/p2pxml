Requested update: Explicit instructions to train our model on our P2PXML dataset (or any other suitable dataset as structured as the P2PXML dataset)

1) Download the P2PXML datasets (i.e., P2PXML_Seq and P2PXML_Structure) at [this link](https://zenodo.org/records/11531319)
2) Put your dataset paths (to be specific, P2PXML_Structure dataset path after you downloaded it) at "folder_1" (i.e., antibody folder) and "folder_2" (i.e., antigen folder) (and at other locations where necessary: lines 193, 218, etc.) within the integrated_model_v1.py
3) Put your label path (to be specific, P2PXML_Structure dataset label path) at line 61
4) Run integrated_model_v1.py to train the model and inference on the trained model

If you need clarifications, please open an issue in the repository or contact the corresponding author (Nuwan) via pmnsribandara@gmail.com.
