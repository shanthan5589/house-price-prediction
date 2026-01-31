# House Price Prediction - Kaggle Competition

## 🏆 Competition Result
- **Kaggle Rank**: 281
- **Score**: 14765.10633
- **Competition**: [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course)

## 📊 Project Overview
This project implements a machine learning solution for predicting house prices using the Kaggle House Prices dataset. The model uses XGBoost regression with comprehensive feature preprocessing to achieve competitive results.

## 🎯 Model Performance
- **Cross-Validation MAE**: 16,492.06
- **Kaggle Public Leaderboard Score**: 14765.10633
- **Evaluation Metric**: Mean Absolute Error (MAE)

## 🛠️ Technical Approach

### Data Preprocessing
The preprocessing pipeline handles three types of features:

1. **Numerical Features**
   - Missing values filled with 0
   - Used `SimpleImputer` with constant strategy

2. **Categorical Features (High Cardinality, >10 unique values)**
   - Missing values imputed with most frequent value
   - Encoded using `OrdinalEncoder`
   - Unknown values handled gracefully with value -1

3. **Categorical Features (Low Cardinality, ≤10 unique values)**
   - Encoded using `OneHotEncoder`
   - Unknown categories ignored during prediction

### Model Architecture
```python
XGBRegressor(
    n_estimators=5400,
    learning_rate=0.01,
    device='cuda'  # GPU acceleration
)
```

### Pipeline Structure
```
Data → Preprocessing (ColumnTransformer) → XGBoost Model → Predictions
```

## 📁 Project Structure
```
house-price-prediction/
│
├── model.ipynb                   # Main Jupyter notebook
├── house_price_prediction.py     # Python script version
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
├── GITHUB_SETUP.md              # GitHub setup guide
│
├── data/                         # Data directory (not tracked)
│   ├── house_price_train.csv
│   └── house_price_test.csv
│
└── submissions/                  # Submission files (not tracked)
    └── submission.csv
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Visit the [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - Download `train.csv` and `test.csv`
   - Rename them to `house_price_train.csv` and `house_price_test.csv`
   - Place them in the `data/` directory

### Usage

**Option 1: Using Jupyter Notebook (Recommended)**
```bash
jupyter notebook model.ipynb
```
Run all cells to:
1. Load and preprocess the training data
2. Perform 4-fold cross-validation
3. Train the model on the full dataset
4. Generate predictions for the test set
5. Create `submission.csv` ready for Kaggle submission

**Option 2: Using Python Script**
```bash
python house_price_prediction.py
```

## 📊 Key Features

- **Automated Feature Type Detection**: Automatically identifies numerical and categorical columns
- **Smart Encoding Strategy**: Different encoding methods for high and low cardinality features
- **Robust Missing Value Handling**: Tailored imputation strategies for different feature types
- **GPU Acceleration**: Utilizes CUDA for faster training (optional)
- **Cross-Validation**: 4-fold CV for reliable performance estimation

## 🔧 Customization

### Using CPU instead of GPU
If you don't have a CUDA-capable GPU, modify the model initialization in the notebook:
```python
model = XGBRegressor(n_estimators=5400, learning_rate=0.01, device='cpu')
```

### Hyperparameter Tuning
Key parameters to experiment with:
- `n_estimators`: Number of boosting rounds (default: 5400)
- `learning_rate`: Step size shrinkage (default: 0.01)
- `max_depth`: Maximum tree depth
- `min_child_weight`: Minimum sum of instance weight
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of columns

## 📈 Results

| Metric | Value |
|--------|-------|
| Cross-Validation MAE | 16,492.06 |
| Kaggle Public Score | 14,765.10633 |
| Kaggle Rank | 281 |

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests for improvements
- Share your experimental results

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Shanthan Yellapragada**

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- The scikit-learn and XGBoost teams for excellent libraries
- The data science community for inspiration and learning resources

---

**Note**: This is a learning project for the Kaggle House Prices competition. Results may vary based on dataset splits and random seeds.
