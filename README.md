# House Sale Price Prediction Using Regression

A comprehensive machine learning project that predicts residential home sale prices using advanced regression techniques and feature engineering.

## 📊 Project Overview

This project builds predictive models to estimate house sale prices based on 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The goal is to develop accurate regression models that can predict property values from features such as location, size, quality, and amenities.

## 🎯 Objectives

- **Primary Goal**: Build accurate regression models to predict `SalePrice` of residential properties
- **Secondary Goals**:
  - Perform comprehensive exploratory data analysis (EDA)
  - Handle missing values and outliers effectively
  - Engineer relevant features to improve model performance
  - Compare multiple regression algorithms
  - Optimize hyperparameters for best results

## 📁 Dataset Description

### Files
- **`train.csv`** - Training dataset with 1,460 observations and 81 features (including target)
- **`test.csv`** - Test dataset for predictions
- **`sample_submission.csv`** - Format template for competition submissions
- **`data_description.txt`** - Detailed description of all 79 features

### Target Variable
- **`SalePrice`** - The property's sale price in dollars (continuous variable)

### Feature Categories

#### Property Characteristics
- **Physical Attributes**: Lot area, lot shape, land contour, dwelling type
- **Building Details**: Year built, year remodeled, roof style/material, exterior materials
- **Room Information**: Number of bedrooms, bathrooms, kitchens, total rooms
- **Area Measurements**: Living area, basement area, garage area, porch areas

#### Quality Ratings
- **Overall Quality** (`OverallQual`): 1-10 scale rating material and finish
- **Overall Condition** (`OverallCond`): 1-10 scale rating property condition
- **Kitchen Quality**, **Basement Quality**, **Garage Quality**, etc.

#### Location Features
- **Neighborhood**: 25 different neighborhoods within Ames city limits
- **Proximity Conditions**: Distance to arterial streets, railroads, parks
- **Zoning Classification**: Residential, commercial, agricultural zones

#### Amenities & Features
- **Garage**: Type, size, year built, finish, condition
- **Basement**: Type, exposure, finished area, condition
- **Utilities**: Heating type/quality, central air, electrical system
- **Outdoor**: Pool, deck, porch, fence quality

## 🔧 Technical Stack

### Libraries & Tools
```python
- pandas          # Data manipulation
- numpy           # Numerical computations
- scikit-learn    # Machine learning models & preprocessing
- matplotlib      # Data visualization
- seaborn         # Statistical visualizations
```

### Machine Learning Pipeline
```
Data Loading → Data Cleaning → EDA → Feature Engineering → 
Preprocessing → Model Training → Evaluation → Prediction
```

## 🚀 Getting Started

### Installation

1. **Clone the repository**
```bash
cd /path/to/project
```

2. **Install dependencies**
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### Running the Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook Training.ipynb
```

Or use VS Code with Jupyter extension to run cells interactively.

## 📈 Methodology

### 1. Data Loading & Initial Exploration
- Load training and test datasets
- Examine data structure, types, and basic statistics
- Identify data quality issues

### 2. Data Cleaning & Preprocessing

#### Missing Value Treatment
- **Categorical Features**: Fill with meaningful defaults (e.g., 'NoGarage', 'NoBasement')
- **Numerical Features**: Fill with median, mean, or domain-specific values
- **LotFrontage**: Impute by neighborhood median (location-based)
- **Garage/Basement Features**: Group-impute related features

#### Feature Engineering
- Separate numerical and categorical columns
- Create preprocessing pipelines:
  - **Numerical Pipeline**: SimpleImputer (median) → StandardScaler
  - **Categorical Pipeline**: SimpleImputer (most_frequent) → OneHotEncoder

### 3. Exploratory Data Analysis (EDA)

#### Statistical Analysis
- Distribution analysis of `SalePrice` (check for skewness)
- Summary statistics for all numerical features
- Correlation matrix to identify feature relationships

#### Visualizations
- **Histograms**: Target variable distribution
- **Heatmaps**: Correlation between features and target
- **Scatter Plots**: Key predictors (OverallQual, GrLivArea) vs SalePrice
- **Box Plots**: Categorical features (Neighborhood) vs SalePrice

#### Key Insights
- **Top Correlated Features**: 
  - OverallQual (Overall Quality)
  - GrLivArea (Above Ground Living Area)
  - GarageCars (Garage Capacity)
  - GarageArea (Garage Size)
  - TotalBsmtSF (Basement Area)
  
- **Multicollinearity Detection**: Identify highly correlated feature pairs (>0.8)
- **Neighborhood Impact**: Significant price variation across neighborhoods

### 4. Model Development

#### Regression Algorithms
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization, feature selection)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost (if implemented)

#### Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Variance explained by the model
- **Cross-Validation**: K-fold CV for robust evaluation

### 5. Prediction & Submission
- Apply best model to test set
- Generate predictions in submission format
- Export to `sample_submission.csv`

## 📊 Key Features Analysis

### Most Influential Features
1. **OverallQual** - Overall material and finish quality
2. **GrLivArea** - Above ground living area square footage
3. **GarageCars** - Size of garage in car capacity
4. **GarageArea** - Size of garage in square feet
5. **TotalBsmtSF** - Total square feet of basement area
6. **1stFlrSF** - First floor square feet
7. **YearBuilt** - Original construction date
8. **FullBath** - Full bathrooms above grade

### Feature Importance Considerations
- Quality ratings have strong positive correlation with price
- Size-related features (living area, basement, garage) are critical
- Location (neighborhood) significantly affects pricing
- Age-related features (YearBuilt, YearRemodAdd) impact value

## 📝 Project Structure

```
Training/
├── train.csv                 # Training data (1,460 samples)
├── test.csv                  # Test data for predictions
├── sample_submission.csv     # Submission format template
├── data_description.txt      # Feature descriptions
├── Training.ipynb            # Main analysis notebook
└── README.md                 # Project documentation (this file)
```

## 🎓 Learning Outcomes

This project demonstrates:
- **Data Preprocessing**: Handling missing values, encoding, scaling
- **Feature Engineering**: Creating meaningful features from raw data
- **Exploratory Data Analysis**: Understanding data patterns and relationships
- **Pipeline Construction**: Building robust ML pipelines with scikit-learn
- **Model Selection**: Comparing multiple regression algorithms
- **Hyperparameter Tuning**: Optimizing model performance
- **Evaluation**: Using appropriate metrics for regression tasks

## 🔍 Future Enhancements

- [ ] Advanced feature engineering (polynomial features, interactions)
- [ ] Ensemble methods (stacking, blending multiple models)
- [ ] Hyperparameter optimization using GridSearchCV/RandomizedSearchCV
- [ ] Feature selection techniques (RFE, LASSO-based selection)
- [ ] Outlier detection and treatment
- [ ] Log transformation of skewed features
- [ ] Deep learning approaches (Neural Networks)

## 📚 References

- **Dataset**: Ames Housing Dataset (alternative to Boston Housing)
- **Competition**: Based on Kaggle's House Prices: Advanced Regression Techniques
- **Documentation**: [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/)

## 👤 Author

Patrick Filima

## 📄 License

This project is open-source and available for educational purposes.

---

**Last Updated**: February 2026
