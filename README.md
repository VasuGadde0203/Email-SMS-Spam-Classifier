# Email and SMS Spam Classifier
A machine learning project for classifying email or SMS content as "spam" or "not spam." This project includes data preprocessing, exploratory data analysis (EDA), feature extraction, and model building using various classifiers to determine the best performing algorithm for spam detection. Additionally, this project has been adapted into a Streamlit web application for easy use and interaction.

## Table of Contents
- Project Overview
- Features
- Installation
- Usage
- Model Evaluation
- Contributing

## Project Overview
This spam classifier aims to differentiate between spam and ham (non-spam) messages by training various machine learning models and evaluating their performance. The project explores multiple algorithms, including Naive Bayes, Support Vector Machine, Logistic Regression, Decision Trees, and Ensemble methods. The model selection is based on accuracy and precision scores, with the Multinomial Naive Bayes algorithm achieving high performance and serving as the primary classifier for the Streamlit application.

## Features
- **Data Cleaning and Preprocessing:**
  Handles null values, duplicates, and unwanted columns. Text transformation includes tokenization, removing stop words, and stemming.
- **EDA (Exploratory Data Analysis):**
  Visualizes data characteristics using histograms, word clouds, and heatmaps to understand the distribution of message lengths, word counts, and sentence structures.
- **Model Training:** Trains and evaluates multiple classifiers, including Naive Bayes, SVM, Logistic Regression, and others, to compare precision and accuracy.
- **Model Stacking and Voting:** Combines classifiers in a Voting and Stacking Classifier setup to improve classification performance.
- **Streamlit App:** A user-friendly interface where users can input an email or SMS and get an immediate classification result.

## Installation
- **Clone the Repository**
  - git clone "https://github.com/VasuGadde0203/Email-SMS-Spam-Classifier.git
  - cd Email-SMS-Spam-Classifier
- **Install Dependencies** Ensure you have Python installed. Install necessary packages with:
  - pip install -r requirements.txt

## Usage
- **Training and Testing the Model** You can directly run the email_spam_classifier.ipynb file to explore the analysis and results of various models.

- **Running the Streamlit App** Start the Streamlit app with:
  - streamlit run app.py
  - Once started, input your email or SMS content to get predictions on whether it's spam or not.

## Streamlit App Preview
- **Input:** Email or SMS text.
- **Output:** "Spam" or "Not Spam" label with the model confidence.

## Model Evaluation
The following table summarizes the performance of various algorithms tested during the development of the spam classifier, measured in terms of accuracy and precision:

| Algorithm                         | Accuracy | Precision |
|-----------------------------------|----------|-----------|
| K-Nearest Neighbors (KN)          | 90.72%   | 100.00%   |
| Naive Bayes (NB)                  | 97.39%   | 100.00%   |
| Random Forest (RF)                | 97.20%   | 98.23%    |
| Support Vector Classifier (SVC)   | 97.58%   | 97.48%    |
| Extra Trees Classifier (ETC)      | 97.58%   | 97.48%    |
| Logistic Regression (LR)          | 95.07%   | 94.85%    |
| XGBoost (xgb)                     | 96.81%   | 94.12%    |
| AdaBoost                          | 96.03%   | 93.69%    |
| Gradient Boosting (GBDT)          | 94.78%   | 89.62%    |
| Bagging Classifier (BgC)          | 96.33%   | 88.46%    |
| Decision Tree (DT)                | 93.52%   | 85.86%    |

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## License
This project is licensed under the MIT License.


