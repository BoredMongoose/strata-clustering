# Strata Clustering

Workforce diversity analysis of US labour force demographics by **sector** and **industry**, using BLS-style employment data (2020–2023). This repository explores gender balance, ethnic composition, diversity trends, and demographic clustering.

## Branch: `feature/load-data`

This branch adds the data-loading pipeline and exploratory analysis notebook. It loads a CSV of employment demographics, cleans and validates the data, then answers six research questions with visualisations and clustering.

## Project structure

```
strata-clustering/
├── data/
│   └── raw/
│       └── dataset-google.csv   # employment demographics (not tracked in git)
├── notebooks/
│   └── 01_load.ipynb            # load, clean, and analyse the dataset
├── environment.yml                # conda environment export
└── README.md
```

## Dataset

Place `dataset-google.csv` in `data/raw/`. The file is listed in `.gitignore` and must be added locally.

| Column | Description |
|---|---|
| `year` | Survey year (2020–2023) |
| `sector` | Top-level industry sector |
| `subsector` | Subsector (empty for sector-level rows) |
| `industry_group` | Industry group (empty for sector-level rows) |
| `industry` | Specific industry (empty for sector-level rows) |
| `total_employed_in_thousands` | Total employment |
| `percent_women` | Share of women in the workforce |
| `percent_white` | Share of white workers |
| `percent_black_or_african_american` | Share of Black / African American workers |
| `percent_asian` | Share of Asian workers |
| `percent_hispanic_or_latino` | Share of Hispanic / Latino workers |

The raw file contains ~1,270 rows across sector, subsector, industry group, and industry levels. After cleaning, the notebook works with **1,128 rows**, including **52 sector-level rows** across **13 sectors**.

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.11+

### Environment

Create and activate a conda environment with the required packages:

```bash
conda create -n strata-clustering python=3.11 pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel -y
conda activate strata-clustering
```

Alternatively, use the exported environment (note: `environment.yml` includes a machine-specific `prefix` path that you may need to remove):

```bash
conda env create -f environment.yml
conda activate base
```

Register the kernel for Jupyter:

```bash
python -m ipykernel install --user --name strata-clustering
```

## Running the analysis

1. Clone the repository and check out the branch:

   ```bash
   git clone https://github.com/BoredMongoose/strata-clustering.git
   cd strata-clustering
   git checkout feature/load-data
   ```

2. Add `dataset-google.csv` to `data/raw/`.

3. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook notebooks/01_load.ipynb
   ```

4. Run all cells top to bottom. The notebook is self-contained — no other scripts are required.

## Notebook overview

`notebooks/01_load.ipynb` follows this workflow:

1. **Load the data** — read the CSV, inspect shape, dtypes, and missing values
2. **Clean the data** — drop blank rows, remove aggregate totals, normalise percentage columns, filter data-entry errors, and isolate sector-level rows for Q1–Q5
3. **Plot styling** — shared colour palette and seaborn theme for all charts

### Research questions

| # | Question | Method |
|---|---|---|
| Q1 | Which sectors have similar demographic structures? | K-Means clustering (K=3) on standardised demographic features, visualised with PCA |
| Q2 | Which sectors are most gender imbalanced? | Distance from 50/50 parity, weighted by sector employment share |
| Q3 | Which sectors shifted most between 2021 and 2023? | Year-over-year change in demographic composition |
| Q4 | Which sectors are dominated by a single ethnic group? | Maximum ethnic share per sector |
| Q5 | Which sectors are most ethnically diverse, and how does that relate to size? | Shannon entropy of ethnic composition vs. employment |
| Q6 | Which industries share similar ethnic compositions? | K-Means clustering (K=4) on industry-level ethnic features |

Q1–Q5 use **sector-level rows** (where `subsector`, `industry_group`, and `industry` are empty). Q6 uses **industry-level rows** (~110 unique industries).

## Key findings

- Most sectors cluster together demographically; Construction, Agriculture, and Mining are outliers.
- Education & health is the largest and most female-dominated sector.
- Diversity is slowly improving — Mining, Financial, and Professional services shifted most from 2021 to 2023.
- White workers are the majority in every sector; "diversity" here means relative spread, not balance.
- Asian concentration is the most distinctive industry-level outlier.
- Sector size and ethnic diversity are weakly positively related, with notable exceptions.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter / ipykernel
