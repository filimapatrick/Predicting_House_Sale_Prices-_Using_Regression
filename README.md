# House Sale Price Prediction (Regression)

This project explores the Ames Housing dataset and prepares the data for regression modeling. The notebook focuses on data loading, missing-value handling, preprocessing pipelines, and exploratory analysis.

## Overview

The goal is to understand which features drive house prices and to prepare a clean, model-ready dataset. The work is based on 79 explanatory variables describing residential homes in Ames, Iowa.

## Dataset

Files in this repository:

- `train.csv` - Training data with 1,460 rows and the target `SalePrice`
- `test.csv` - Test data for prediction
- `sample_submission.csv` - Submission format template
- `data_description.txt` - Detailed field definitions for all features

Target variable:

- `SalePrice` - Sale price of the property in dollars

Feature groups:

- Property characteristics (lot size, shape, building type)
- Area measurements (living area, basement, garage, porches)
- Quality ratings (overall, kitchen, basement, garage)
- Location and zoning (neighborhood, zoning class, proximity to features)
- Amenities (garage, pool, fence, fireplace, air conditioning)

## Notebook Workflow

The main analysis lives in `Training.ipynb` and follows this flow:

1. Load the data and review basic structure.
2. Separate target and features (`SalePrice` vs. predictors).
3. Identify numeric and categorical columns.
4. Build preprocessing pipelines:
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + one-hot encoding
5. Combine with `ColumnTransformer` and transform the data.
6. Clean key missing values using domain-aware defaults:
   - Pool, fence, alley, fireplace, garage, and basement indicators
   - Masonry veneer type/area
   - `LotFrontage` imputed by neighborhood median
   - `Electrical` imputed by mode
7. Perform EDA:
   - Target distribution and skew
   - Correlations with `SalePrice`
   - Heatmap of top correlated features
   - Scatter plots and box plots for key drivers

## Key EDA Focus

- Top numeric correlations with `SalePrice`
- Size and quality features (e.g., `OverallQual`, `GrLivArea`, `TotalBsmtSF`)
- Neighborhood-level price differences
- Multicollinearity checks among numeric predictors

## Requirements

Python libraries used in the notebook:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

Open and run the notebook:

```bash
jupyter notebook Training.ipynb
```

Or run in VS Code with the Jupyter extension.

## Project Structure

```
Training/
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_description.txt
├── Training.ipynb
└── README.md
```

## Notes

- This notebook prepares data for regression modeling but does not yet train or evaluate a model.
- The dataset is the same as Kaggle's "House Prices: Advanced Regression Techniques" competition.

## Author

Patrick Filima

## License

Open for educational use.

Last updated: 2026-02-06
