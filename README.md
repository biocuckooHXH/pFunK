# Hierarchical learning reveals ketogenic diet reprograms cancer metabolism through lysine β-hydroxybutyrylation

We developed a hierarchical learning framework, namely prediction of functionally important lysine modification sites (pFunK), to address the challenge of identifying functionally important lysine β-hydroxybutyrylation (Kbhb) sites in cancer metabolism.

## Model Architecture
pFunK utilizes a novel hierarchical learning framework implemented in three steps to minimize bias and over-fitting:

1. **pFunK-P:** 
    - This pre-training model employs the state-of-the-art Transformer algorithm to learn “in-context” information from a large training dataset of 145,657 non-redundant lysine modification sites spanning 29 types of lysine modifications.
    - By focusing on short sequences around lysine modification sites, pFunK-P captures contextual sequence features crucial for downstream predictions.

2. **pFunK-T:** 
    - The transformer-based model from pFunK-P is fine-tuned to capture Kbhb-specific characteristics through transfer learning.
    - Additionally, pFunK-T integrates 10 types of sequence and structural features, further refined using the Model-Agnostic Meta-Learning (MAML) algorithm.
    - This step uses a benchmark dataset of 6,932 non-redundant Kbhb sites, obtained from 6,318 reported and 5,304 identified sites after homologous elimination.

3. **pFunK:** 
    - In the final step, MAML is employed to fine-tune the model using only 9 functionally important Kbhb sites, ensuring the functional relevance of the predictions.

If you are interested in pFunK and want to know more details, please check the article.

## Usage
- **models:** Pre-trained models referenced in the article.
- **models_buildings_codes:** Scripts to train your own models.

## Quick Start
To predict using your fasta file, simply run `predict.py`.

### Instructions for Use
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/pFunK.git
    cd pFunK
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Predictions:**
    - Prepare your fasta file.
    - Run the prediction script:
      ```bash
      python predict.py --input your_fasta_file.fasta --output results.txt
      ```

4. **Training Your Own Model:**
    - Modify the configuration files as needed.
    - Run the training scripts located in `models_buildings_codes` directory.

### Additional Files
- **Omics Analysis:**
    - We have provided scripts and data for comprehensive omics analysis to complement the predictions made by pFunK.
    - These resources are located in the `omics_analysis` directory.

- **Feature Extraction:**
    - The `feature_extraction` directory contains scripts for extracting various sequence and structural features used in model training and prediction.
    - These scripts ensure the reproducibility of our feature extraction process.



## Code Availability
In addition to pFunK, other custom code used in the study, including scripts for omics analysis and feature extraction, are available in the respective directories. This ensures transparency and facilitates other researchers to reproduce and build upon our work.

### Tutorial
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/pFunK.git
    cd pFunK
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Predictions:**
    - Prepare your fasta file.
    - Run the prediction script:
      ```bash
      python predict.py --input your_fasta_file.fasta --output results.txt
      ```

4. **Training Your Own Model:**
    - Modify the configuration files as needed.
    - Run the training scripts located in `models_buildings_codes` directory.

5. **Omics Analysis:**
    - Navigate to the `omics_analysis` directory for scripts and data related to comprehensive omics analysis.

6. **Feature Extraction:**
    - Use the scripts in the `feature_extraction` directory for extracting sequence and structural features.


## Contact
- **Xinhe Huang:** huangxinhe@hust.edu.cn
- **Dr. Yu Xue:** xueyu@hust.edu.cn
