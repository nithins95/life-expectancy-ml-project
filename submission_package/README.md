Code Quality

To check pylint score run the below commant in bash:
    pylint src

Current score: 9.87/10

📌 Project Overview

This project investigates:

Which economic, demographic, and healthcare indicators most strongly influence life expectancy in EU countries?

The project implements a complete data science pipeline:
	•	Automated data acquisition from public APIs
	•	Data cleaning and integration
	•	Exploratory data analysis
	•	Statistical visualisation
	•	Machine learning modelling
	•	Cross-validation and evaluation
	•	Unit testing
	•	Code quality validation using pylint

This project demonstrates a structured and production-ready ML workflow following best practices to prevent data leakage and ensure robust evaluation.

⸻

📊 Data Sources

Source: Eurostat API
Indicators:
	•	Life expectancy
	•	Doctors per 100k
	•	Hospital capacity
	•	Household expenditure
	•	Government health expenditure

Source: World Bank API
Indicators:
	•	GDP per capita
	•	Fertility rate
	•	Urban population (%)
	•	Population density

Raw datasets are stored in:

data/raw/

The integrated master dataset is stored in:

data/processed/master_dataset.csv

⸻

🏗 Project Structure

life-expectancy-ml-project/

data/
    raw/
    processed/
    figures/

src/
    data_fetcher.py
    eurostat_data_fetcher.py
    world_bank_data_fetcher.py
    data_loader.py
    visualizations.py
    modelling.py
    main.py

tests/
    test_modelling.py
    test_data_validation.py

requirements.txt
README.md

⸻

⚙️ Installation

Clone the repository:

git clone 
cd life-expectancy-ml-project

Create virtual environment:

python -m venv venv
source venv/bin/activate   (macOS/Linux)

Install dependencies:

pip install -r requirements.txt

▶️ Running the Full Pipeline

Run everything from the main module:

python -m src.main

The pipeline executes in the following order:
	1.	Data acquisition (only fetches if files are missing)
	2.	Data integration and cleaning
	3.	Visualisation generation
	4.	Machine learning modelling

⸻

📈 Exploratory Data Analysis

The project generates the following visualisations:
	•	Correlation heatmap
	•	Life expectancy trends over time
	•	GDP vs Life Expectancy (log-scale regression)
	•	Fertility rate vs Life Expectancy
	•	Distribution of Life Expectancy (histogram + KDE)

All figures are saved automatically in:

data/figures/

⸻

🤖 Machine Learning Lifecycle

Model used: Linear Regression

Pipeline implementation includes:
	1.	Train/test split (80/20)
	2.	Outlier removal (IQR method, training data only)
	3.	Mean imputation (fitted on training data only)
	4.	Standard feature scaling (fitted on training data only)
	5.	Model training
	6.	Evaluation using RMSE, R², and Adjusted R²
	7.	5-fold cross-validation

The design explicitly prevents data leakage by ensuring all preprocessing is fitted only on training data.

⸻

📊 Model Results

Observations (train): 456
Features used: 9

TRAIN PERFORMANCE
R²: 0.7648
RMSE: 0.9188

TEST PERFORMANCE
R²: 0.6706
Adjusted R²: 0.6424
RMSE: 1.0013

CROSS-VALIDATION (5-fold)
Mean R²: 0.7480
Std Dev: 0.0187

🔎 Interpretation
	•	GDP per capita shows a strong positive relationship with life expectancy.
	•	Fertility rate shows a strong negative relationship.
	•	Healthcare infrastructure indicators contribute positively.
	•	Small difference between training and test R² suggests limited overfitting.
	•	Cross-validation stability indicates good generalisation performance.

⸻

🧪 Testing

Unit tests are implemented for:
	•	Data splitting
	•	Outlier removal
	•	Preprocessing pipeline
	•	Model training and evaluation
	•	Adjusted R² calculation
	•	Data validation checks

Run tests with:

pytest

⸻

🧹 Code Quality

Code quality is enforced using pylint.

Current score:

src/: 9.7+/10

Check locally with:

pylint src

The project follows:
	•	Proper import ordering
	•	Snake_case naming conventions
	•	Docstrings for all functions
	•	Separation of concerns
	•	Reduced code duplication

⸻

🛡 Data Validation

The project includes explicit validation to ensure:
	•	No duplicate country-year records
	•	No missing target variable values before modelling
	•	Only valid EU ISO3 country codes are retained
	•	Clean numeric dataset before model training

⸻

📌 Key Findings
	•	Economic development is the strongest predictor of life expectancy.
	•	Demographic transition (declining fertility) correlates with increased longevity.
	•	Public health expenditure positively impacts outcomes.
	•	Urbanisation has a moderate but positive association.

⸻

📚 Technologies Used
	•	Python 3
	•	Pandas
	•	NumPy
	•	Scikit-learn
	•	Matplotlib
	•	Requests
	•	Pytest
	•	Pylint

⸻

👤 Authors

Member 1 – Elizaveta Gorshkova (47574)
    •	Developed machine learning pipeline
    •	Performed model evaluation and cross-validation
    •	Wrote unit tests for modelling module
	•	Data validation testing

Member 2 – Adrianna Oleksiewicz (54915)
	•	Built visualisation pipeline
	•	Conducted exploratory data analysis
	•	Assisted with debugging and testing

Member 3 – Nithin Subramanian (54951)
	•	Designed project architecture
	•	Implemented data acquisition modules
    •	API integration improvements
    •	Implemented data integration module
	•	Wrote unit tests for data validation

Member 4 – Jan Piotrowski (55145)
	•	Code quality enforcement (pylint)
    •	Assisted with debugging and testing

Member 5 – Maksym Koshchuk (55147)
    •	Documentation and README writing
    •	Assisted with debugging and testing





