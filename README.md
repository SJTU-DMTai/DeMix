# DeMix
DeMix is an automated framework designed for training data debugging with mixed error types. First, DeMix computes influence vectors for training samples. The influence vectors are then fed into a trained Data Error Classifier (DEC), which identifies suspicious samples and diagnoses their specific error types. Subsequently, DeMix executes type-specific repair on the detected samples.

## Preparation
1. Install dependencies:
```bash
# create conda environment
conda create -n demix python=3.9 -y
conda activate demix
# install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn numpy pandas ucimlrepo tqdm matplotlib
```

2. Create a new directory `demix_files` to store the DEC checkpoints and the dataset.

3. Download DEC checkpoints from [google drive](https://drive.google.com/drive/folders/1_aFP9b-hIDRMHohVP8-ZseVyyRqip075?usp=drive_link) and place them in the `demix_files/dec_ckpts/` directory.

4. The dataset used for DeMix will be automatically downloaded and saved in the `demix_files/data/` directory.

## Running DeMix
```bash
# The script automatically inject errors (with a clean ratio of alpha) to the dataset
# perform data debugging and repair with DeMix, 
# and evaluate the trained model performance after repair.
python scripts/repair_data.py --save_dir demix_files --data_name adult --model_name mlp1 --alpha 0.5
```
