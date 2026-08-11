# 📊 Data Science & Machine Learning Portfolio

A comprehensive collection of end-to-end data science projects, machine learning implementations, statistical problem sets, and natural language processing workflows.

---

## 📂 Table of Contents
1. [Statistics & Probability Foundations](#1-statistics--probability-foundations)
2. [Hypothesis Testing](#2-hypothesis-testing)
3. [Linear & Logistic Regression](#3-linear--logistic-regression)
4. [Supervised Classification Models](#4-supervised-classification-models)
5. [Unsupervised Learning & Clustering](#5-unsupervised-learning--clustering)
6. [Dimensionality Reduction & Association Rules](#6-dimensionality-reduction--association-rules)
7. [Time-Series & Forecasting](#7-time-series--forecasting)
8. [Natural Language Processing (NLP)](#8-natural-language-processing-nlp)
9. [Neural Networks & Recommendation Systems](#9-neural-networks--recommendation-systems)

---

## 1. 📈 Statistics & Probability Foundations

### 📄 [Set 1: Descriptive Statistics & Probability](./Set+1_Descriptive+statistics+Probability+(2).docx)
* **Problem Statement:** Analyze corporate return metrics ($\%$) across major firms. Plot distributions, compute central tendency ($\mu$, $\sigma$, $\sigma^2$), identify outliers, and answer probability distributions for business venture returns.
* **Solution Applied:** Applied box-plots and histograms to detect skewness and identify extreme values ($91.36\%$). Computed mean ($\mu = 33.27\%$), standard deviation ($\sigma = 16.94\%$), variance ($\sigma^2 = 287.14$), Interquartile Range ($IQR = 7$), and evaluated binomial/discrete probability outcomes.

### 📄 [Set 2: Normal Distribution & Random Variables](./Set+2_Normal+Distribution+Functions+of+random+variables+(1).docx)
* **Problem Statement:** Solve real-world operational timing and profit distribution problems following normal distributions $N(\mu, \sigma^2)$.
* **Solution Applied:** Applied Z-score transformations $Z = \frac{X - \mu}{\sigma}$ to evaluate probabilities for automobile service delays, employee age distributions, linear combinations of independent normal variables $2X_1$ vs. $X_1 + X_2$, and 95% confidence bounds for multi-division corporate revenue.

### 📄 [Set 3: Confidence Intervals](./Set+3.doc)
* **Problem Statement:** Evaluate sample size sufficiency, market share confidence bounds (e.g., Mozilla market share estimation), and survey sampling biases.
* **Solution Applied:** Constructed 95% and 98% confidence intervals using $Z$-scores and $T$-scores, evaluated sample frame representative requirements, and derived minimum required sample sizes $n = \left(\frac{Z \cdot \sigma}{E}\right)^2$ for bounded margins of error.

### 📄 [Set 4: Sampling Distributions & CLT](./Set+4.docx)
* **Problem Statement:** Analyze sampling variation, standard error calculation, and Central Limit Theorem (CLT) implications across warehouse package weights, ATM transaction audits, and test score distributions.
* **Solution Applied:** Computed Standard Error $SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$, determined audit threshold probabilities via normal approximations, and validated CLT convergence criteria for heavily skewed populations.

---

## 2. 🧪 Hypothesis Testing

### 📓 [A3 Hypothesis.ipynb](./A3%20Hypothesis.ipynb)
* **Problem Statement:** Test operational hypothesis claims across four distinct business datasets:
  1. Cutlet diameter comparison between two manufacturing units.
  2. Turnaround time (TAT) differences across four diagnostic laboratories.
  3. Buyer ratios of male vs. female across four geographic regions.
  4. Defective percentage variations across customer order processing centers.
* **Solution Applied:**
  * **2-Sample t-Test:** Evaluated diameter equality between Unit A and Unit B.
  * **One-Way ANOVA (F-Test):** Evaluated mean TAT variations across 4 laboratories.
  * **Chi-Square ($\chi^2$) Test of Independence:** Evaluated categorical independence for Buyer Ratios and Defective Order proportions across regions/centers at $\alpha = 0.05$.

---

## 3. 📉 Linear & Logistic Regression

### 📓 [A4 Simple Regression.ipynb](./A4%20Simple%20Regression.ipynb)
* **Problem Statement:** Predict single continuous target variables:
  1. Predict `Delivery Time` based on `Sorting Time`.
  2. Predict employee `Salary` based on `Years of Experience`.
* **Solution Applied:** Exploratory Data Analysis (EDA), correlation matrix analysis, log and square-root feature transformations, Ordinary Least Squares (OLS) model fitting, $R^2$ performance evaluation, and Root Mean Squared Error (RMSE) minimization.

### 📓 [A5 Multilinear Regression 1 & 2](./A5%20Multiple%20Regression.ipynb)
* **Problem Statement:** Build multi-predictor linear models:
  1. Predict profit for `50 Startups` using R&D Spend, Administration, and Marketing Spend.
  2. Predict `Toyota Corolla` resale prices using vehicle age, mileage, horsepower, CC, and door count.
* **Solution Applied:** Variance Inflation Factor (VIF) evaluation to remove multicollinearity, Cook's Distance and Leverage diagnostics for outlier removal, residual plot verification, stepwise feature selection, and model validation via Adjusted $R^2$.

### 📓 [logistic regression.ipynb](./logistic%20regression.ipynb)
* **Problem Statement:** Predict whether a banking client will subscribe to a term deposit (`Yes`/`No`) based on demographic and financial campaign attributes.
* **Solution Applied:** Categorical encoding (One-Hot / Label), feature scaling via `StandardScaler`, fitting Logistic Regression classification models, confusion matrix analysis, ROC-AUC curve plotting, and setting optimal decision thresholds.

---

## 4. 🤖 Supervised Classification Models

### 📓 [KNN.ipynb](./KNN.ipynb)
* **Problem Statement:** Classify multi-class instances:
  1. Glass type classification based on refractive index and chemical composition.
  2. Zoo animal categorization based on physical traits and habits.
* **Solution Applied:** Feature standardization, Euclidean distance metric selection, grid search cross-validation to select optimal $k$-neighbors, and accuracy evaluation.

### 📓 [Naive bayes.ipynb](./Naive%20bayes.ipynb)
* **Problem Statement:** Classify salary levels ($>50K$ vs $\le 50K$) using large-scale demographic and employment training/test data.
* **Solution Applied:** Categorical label encoding, Gaussian and Multinomial Naïve Bayes classifier fitting, probability likelihood evaluation, and classification report analysis.

### 📓 [Decision Tree.ipynb](./Decision%20Tree.ipynb)
* **Problem Statement:** 
  1. Classify company cloth sales as high or low using commercial indicators.
  2. Detect fraudulent tax transactions (`Risky` vs `Good`) based on taxable income.
* **Solution Applied:** Target feature binning, tree splitting using Gini Impurity and Entropy criteria, decision tree depth controls (`max_depth`, `min_samples_leaf`) to prevent overfitting, and tree structure visualization.

### 📓 [Random Forest.ipynb](./Random%20Forest.ipynb)
* **Problem Statement:** Improve classification stability and accuracy over single decision trees on company sales and tax fraud datasets.
* **Solution Applied:** Bagging ensemble implementation, random feature subset sampling, hyperparameter tuning (`n_estimators`, `max_features`), Out-Of-Bag (OOB) error evaluation, and feature importance ranking.

### 📓 [SVM.ipynb](./SVM.ipynb)
* **Problem Statement:** Classify complex, non-linearly separable datasets (e.g., Forest Fires burned area prediction and salary category prediction).
* **Solution Applied:** Data normalization, Radial Basis Function (RBF), Linear, and Polynomial SVM kernel applications, Hyperparameter tuning ($C$ and $\gamma$) via GridSearchCV, and support vector decision boundary evaluation.

---

## 5. 🧩 Unsupervised Learning & Clustering

### 📓 [clustering.ipynb](./clustering.ipynb) & [clustering crime data.ipynb](./clustering%20crime%20data.ipynb)
* **Problem Statement:** Group unlabelled records into meaningful segments:
  1. Segment US states based on violent crime statistics (Murder, Assault, UrbanPop, Rape).
  2. Segment airline passengers based on frequent flyer mileage and transaction behavior.
* **Solution Applied:** Feature normalization via `MinMaxScaler` / `StandardScaler`, K-Means clustering with Elbow Method (WCSS plot), Agglomerative Hierarchical Clustering with Dendrogram visualization (Ward's link), DBSCAN density-based clustering, and Silhouette Score validation.

---

## 6. 🔍 Dimensionality Reduction & Association Rules

### 📓 [PCA.ipynb](./PCA.ipynb)
* **Problem Statement:** Reduce high-dimensional wine chemical analysis features (13 variables) into lower dimensions while retaining maximum variance, then compare cluster quality before and after reduction.
* **Solution Applied:** Standardized feature scaling, covariance matrix computation, Eigenvalue decomposition, Scree plot analysis for explained variance ratio, projecting data onto Top-3 Principal Components, and running K-Means / Hierarchical clustering on PC scores.

### 📓 [AssociationRules.ipynb](./AssociationRules.ipynb)
* **Problem Statement:** Identify frequent item combinations and market basket shopping patterns in transaction datasets (e.g., Retail books and movies datasets).
* **Solution Applied:** Applied the Apriori algorithm, configured Support, Confidence, and Lift thresholds, filtered top actionable association rules, and visualized rule distributions via scatter and network plots.

---

## 7. 📈 Time-Series & Forecasting

### 📓 [Forecasting.ipynb](./Forecasting.ipynb)
* **Problem Statement:** Predict future sales trends for Coca-Cola quarterly historical sales and Airlines passenger counts.
* **Solution Applied:** Time-series decomposition (Trend, Seasonality, Residuals), stationarity testing (Augmented Dickey-Fuller test), fitting additive and multiplicative seasonal models, Holt-Winters Exponential Smoothing, ARIMA/SARIMA model fitting, and RMSE comparison across forecast models.

---

## 8. 🔤 Natural Language Processing (NLP)

### 📓 [text emotion.ipynb](./text%20emotion.ipynb)
* **Problem Statement:** Classify textual messages into distinct human emotional states (e.g., Joy, Sadness, Anger, Fear).
* **Solution Applied:** Text normalization, stopword removal, lemmatization, TF-IDF vectorization, feature extraction, and multi-class text classification using Multinomial Naïve Bayes and Logistic Regression.

### 📓 [Text mining sentimental analysis.ipynb](./Text%20mining%20sentimental%20analysis.ipynb)
* **Problem Statement:** Mine product reviews and social feedback data to extract overall polarity (Positive, Negative, Neutral) and key opinion topics.
* **Solution Applied:** Text cleaning (regex HTML/punctuation removal), tokenization, word clouds generation for frequency analysis, sentiment scoring using VADER and TextBlob lexicon engines, and emotion polarity mapping.

---

## 9. 🧠 Neural Networks & Recommendation Systems

### 📓 [neural network.ipynb](./neural%20network.ipynb)
* **Problem Statement:** Build deep learning architectures for complex non-linear prediction tasks (e.g., Gas Turbine energy yield prediction and Forest Fires burn area regression).
* **Solution Applied:** Data normalization, Multi-Layer Perceptron (MLP) construction using Keras/TensorFlow, configuring Dense hidden layers with ReLU activation, Adam optimizer, Mean Squared Error (MSE) loss function, epoch training, and training/validation loss curve plotting.

### 📓 [Reccomendation system.ipynb](./Reccomendation%20system.ipynb)
* **Problem Statement:** Build a personalized recommendation engine for book readers based on user ratings.
* **Solution Applied:** User-Item rating matrix creation (pivot tables), Cosine Similarity matrix calculation, Collaborative Filtering (User-Based and Item-Based filtering), and generating top-N recommended titles for target users.

---

## 🛠️ Tools & Tech Stack
* **Languages:** Python (Pandas, NumPy, SciPy)
* **Machine Learning:** Scikit-Learn, Statsmodels, Keras / TensorFlow
* **NLP & Text Mining:** NLTK, TextBlob, SpaCy, TF-IDF Vectorizer
* **Data Visualization:** Matplotlib, Seaborn, Plotly
