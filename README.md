# pFunK

**pFunK** (prediction of functionally important lysine modification sites) is a hierarchical deep learning framework designed to identify functionally relevant lysine β-hydroxybutyrylation (Kbhb) sites, revealing how a ketogenic diet reprograms cancer metabolism.

---

## 1. Overview

We developed a three-stage hierarchical learning framework to minimize bias and overfitting and improve predictive interpretability:

1. **pFunK-P:**  
   - Pre-trained on 145,657 non-redundant lysine modification sites spanning 29 modification types.  
   - Captures “in-context” sequence features surrounding lysine residues.

2. **pFunK-T:**  
   - Fine-tuned using transfer learning and MAML to capture Kbhb-specific sequence and structural features.  
   - Trained on 6,932 non-redundant Kbhb sites.

3. **pFunK:**  
   - Final fine-tuning with 9 functionally validated Kbhb sites to maximize biological relevance.

---

## 2. Repository Structure

- **models/** – Pre-trained and fine-tuned models from the study.  
- **models_buildings_codes/** – Scripts for training new models.  
- **omics_analysis/** – Omics analysis scripts and data.  
- **feature_extraction/** – Sequence and structure feature extraction utilities.

---

## 3. Quick Start: Prediction

> ⚠️ **Before running prediction**, unzip the model under `models/pFunKbHb/`:

```bash
cd models/pFunKbHb
unzip pFunK_Transformer.model.zip
```

### Step 1: Install Dependencies

```bash
git clone https://github.com/your-repo/pFunK.git
cd pFunK
pip install -r requirements.txt
```

### Step 2: Input Formats

pFunK supports **two input modes**:

#### A. Pre-extracted sites (Recommended)

Create a TSV file (`sites.tsv`) with two columns:

| id          | fasta_K                                             |
|------------|------------------------------------------------------|
| P12345_145 | XXXMVLSEGEWQLVLHVWAKVE...K...RGLYYYXXX (61 aa total) |
| Q9ABC1_87  | AAAPPPQQQLLLVVV...K...TTTSSSNNNXXXXXXX               |

- **`fasta_K`** must be **61 amino acids**, with the target lysine `K` centered.  
- Use `"X"` as padding for missing residues.

Run prediction:

```bash
python predict.py   --sites_tsv examples/sites.tsv   --model_path models/pFunKbHb/pFunK_Transformer.model.h5   --output results.tsv
```

#### B. FASTA + Site List (Automatic 61-mer extraction)

If you have a full protein FASTA and site list, provide them directly:

- `proteins.fasta`: FASTA file with full sequences  
- `sites_pos.tsv`: TSV with two columns `id` and `pos` (1-based lysine position)

Example `sites_pos.tsv`:
```
id      pos
P12345  145
Q9ABC1   87
```

Run prediction:

```bash
python predict.py   --fasta examples/proteins.fasta   --sites examples/sites_pos.tsv   --model_path models/pFunKbHb/pFunK_Transformer.model.h5   --output results.tsv
```

### Output Format

| id         | score   |
|------------|---------|
| P12345_145 | 0.92351 |
| Q9ABC1_87  | 0.38472 |

Higher scores indicate higher probability of β-hydroxybutyrylation.

---

## 4. Feature Extraction & Omics Analysis

- `feature_extraction/` – Scripts to reproduce sequence and structural feature engineering.  
- `omics_analysis/` – Integrative downstream analysis linking predicted Kbhb sites to metabolic reprogramming.

---

## 5. Training Your Own Model

You can fine-tune or retrain the model:

```bash
python train.py --config configs/your_config.yaml
```

---

## 6. Citation

If you use **pFunK** or related code in your research, please cite:

> **Qin J, Huang X, Gou S, Zhang S, Gou Y, Zhang Q, Chen H, Sun L, Chen M, Liu D, Han C, Tang M, Feng Z, Niu S, Zhao L, Tu Y, Liu Z, Xuan W, Dai L, Jia D, Xue Y.**  
> *Ketogenic diet reshapes cancer metabolism through lysine β-hydroxybutyrylation.*  
> **Nat Metab.** 2024 Aug;6(8):1505–1528. doi: [10.1038/s42255-024-01093-w](https://doi.org/10.1038/s42255-024-01093-w). PMID: 39134903.

---

## 7. Contact

- **Xinhe Huang:** huangxinhe@hust.edu.cn  
- **Dr. Yu Xue:** xueyu@hust.edu.cn
