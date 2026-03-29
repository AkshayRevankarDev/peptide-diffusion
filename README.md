# Peptide and Protein Sequencing by Multinomial Diffusion Model

**Team**: Akshay Mohan Revankar, Sanika Nanjan, Vaishak Girish Kumar

## Description
This project implements a de novo peptide sequencing pipeline from tandem mass spectrometry (MS/MS) data, collected from E. coli EV and wastewater samples. The final goal is to develop a Multinomial Diffusion Model for accurate sequencing. This repository currently contains the Checkpoint 1 deliverables: data exploration, a generic preprocessing pipeline, and a simple encoder-decoder LSTM baseline model.

## Data
The dataset includes raw `mzML` files and `xlsx` files containing database search labels (ground truth). The labels are used to train the baseline model on the E. coli EV spectral data.

## Installation
Ensure you have Python 3 installed. Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
The step-by-step notebooks are located in the `notebooks/` directory and should be run in order:
1. `01_eda.ipynb` - Exploratory Data Analysis
2. `02_preprocessing.ipynb` - Preprocessing Pipeline Verification
3. `03_baseline.ipynb` - Training and Evaluation of the Baseline Model
