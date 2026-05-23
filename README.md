# Sentiment Analysis on Amazon Food Reviews

## Overview
Natural Language Processing project analyzing 10,000 Amazon food reviews
to predict whether a customer review is positive or negative using 
text classification techniques. This project is a direct continuation 
of my E-Commerce Data Analysis project where we identified that 
customer review text was the missing feature needed to better 
predict customer satisfaction.

## Why This Project
In my previous e-commerce project, a Random Forest model achieved only 
36.50% accuracy when predicting customer satisfaction using numerical 
features like age, price and quantity. The key insight was that what 
customers actually say in their reviews is a far stronger predictor 
of satisfaction than any numerical feature. This project addresses 
that limitation directly using Natural Language Processing.

## Objectives
- Clean and preprocess real Amazon customer review text
- Convert star ratings into positive and negative sentiment labels
- Handle class imbalance in sentiment distribution
- Transform text into numerical features machines can understand
- Build a classification model to predict sentiment from review text
- Compare performance against the previous numerical model

## Dataset
- Source: Amazon Food Reviews dataset
- Size: 10,000 reviews loaded from 568,454 total reviews
- Key columns used: Score (star rating), Summary, Text (full review)
- After cleaning: 3,044 balanced reviews (1,522 positive, 1,522 negative)

## Tools Used
- Python
- Pandas
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Regular Expressions (re)
- Matplotlib
- Google Colab
- Google Drive

## Project Structure
1. Data Preparation
2. Sentiment Labeling
3. Class Balancing
4. Text Cleaning
5. TF-IDF Vectorization
6. Model Building
7. Model Evaluation
8. Influential Words Analysis

---

## Phase 1 - Data Preparation

### Step 1 - Column Selection
Dropped unnecessary columns including UserId, ProductId,
HelpfulnessNumerator, HelpfulnessDenominator and Time.
Retained only Score, Summary and Text as these are the
only columns relevant to sentiment prediction.

### Step 2 - Sentiment Labeling
Converted numerical star ratings into sentiment categories:
- 4 to 5 stars = Positive
- 1 to 2 stars = Negative
- 3 star reviews were removed as they are ambiguous and
  do not clearly represent satisfaction or dissatisfaction

Initial distribution after labeling:
- Positive: 7,616 reviews
- Negative: 1,522 reviews

### Step 3 - Handling Class Imbalance
The dataset was heavily imbalanced with 5 positive reviews
for every 1 negative review. Training a model on imbalanced
data causes it to predict the majority class almost every time,
which was the key limitation identified in the previous project.

Solution: Random undersampling of the positive class to match
the negative class count resulting in a perfectly balanced
dataset of 3,044 reviews with 1,522 of each sentiment.

### Step 4 - Text Cleaning
Raw customer review text contains noise that reduces model
performance. A custom cleaning function was applied to every
review performing the following steps:

- Converted all text to lowercase for uniformity
- Removed all punctuation and numbers using regular expressions
- Removed stopwords using NLTK — common words like "the", "and",
  "is" and "it" that carry no meaningful information for
  sentiment prediction

Example transformation:

BEFORE:
A raw customer review containing punctuation, capital letters,
numbers and common filler words like "the", "and", "it" and "was"

AFTER:
Only the meaningful sentiment carrying words remain after cleaning,
with all punctuation, numbers and stopwords removed

---

## Phase 2 - TF-IDF Vectorization and Model Building

### TF-IDF Vectorization
TF-IDF stands for Term Frequency — Inverse Document Frequency.
It converts cleaned review text into a numerical matrix where
each word gets a score representing how important it is in a
review relative to all other reviews.

- Words appearing frequently in one review but rarely elsewhere
  get higher scores — these are the most meaningful words
- Common words appearing everywhere get lower scores
- Result: a matrix of 3,044 rows and 5,000 columns where every
  review is represented as 5,000 numbers instead of raw text

### Model Used
Logistic Regression with a maximum of 1000 iterations.
Logistic Regression learns which words push a review toward
positive or negative sentiment by assigning weights to each
word in the TF-IDF matrix.

### Train Test Split
- Training set: 2,435 reviews (80%)
- Testing set: 609 reviews (20%)

---

## Model Performance

### Overall Accuracy: 83.42%

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.84 | 0.83 | 0.83 |
| Positive | 0.83 | 0.84 | 0.84 |

### Key Observations
- Both sentiment classes perform almost identically showing
  a well balanced and reliable model
- No class bias present due to dataset balancing in Phase 1
- Consistent F1-scores across both classes confirms the model
  generalizes well to unseen reviews

---

## Project Comparison

| Metric | Project 1 | Project 2 |
|---|---|---|
| Model | Random Forest | Logistic Regression |
| Features | Age, Price, Quantity, Gender | Review Text (TF-IDF) |
| Accuracy | 36.50% | 83.42% |
| Negative Precision | 0.25 | 0.84 |
| Positive Precision | 0.43 | 0.83 |
| Class Balance | Uneven | Perfectly balanced |

The 46.92% improvement in accuracy directly proves the hypothesis
from project 1 — customer review text is a far stronger predictor
of satisfaction than any numerical feature.

---

## Most Influential Words

### Top 10 Positive Words
Words the model learned to associate with satisfied customers:
great, best, love, good, perfect, delicious, favorite, 
find, wonderful, loves

### Top 10 Negative Words
Words the model learned to associate with dissatisfied customers:
disappointed, weak, bad, thought, would, money, 
disappointing, worst, didnt

### Business Insight
The presence of money and didnt in negative reviews suggests
dissatisfied customers frequently feel they did not get value
for what they paid. This is a direct signal for businesses to
focus on delivering on product descriptions and pricing expectations.

---

## Business Applications
This model can be deployed by e-commerce businesses to:
- Automatically flag negative reviews for urgent customer service attention
- Monitor customer satisfaction at scale without reading every review
- Identify unhappy customers before they churn
- Track sentiment trends over time across product categories
- Prioritize product improvements based on recurring negative keywords

---

## Limitations and Future Work
- Dataset limited to 10,000 of 568,454 available reviews
- Future iterations will use the full dataset for improved accuracy
- Deep learning models like LSTM or BERT could push accuracy above 90%
- Adding the Summary column as an additional feature may improve results
- Sentiment analysis could be extended to neutral reviews for 
  three class classification

---

## Project Status
✅ Data loading and exploration complete
✅ Sentiment labeling complete
✅ Class imbalance handled
✅ Text cleaning complete
✅ TF-IDF vectorization complete
✅ Logistic Regression model built and evaluated
✅ Influential words analysis complete
✅ Visualization of influential words complete

## Connection to Previous Project
This project directly addresses the main limitation of the
E-Commerce Data Analysis project where numerical features
produced only 36.50% model accuracy. The 83.42% accuracy
achieved here demonstrates the critical importance of
feature selection and using the right type of data for
the problem at hand.

## Author
Data Science Undergraduate | Passionate about using data to
drive business strategy and decision making
Tools: Python, Pandas, NLTK, Scikit-learn, Matplotlib, Google Colab
Currently exploring the intersection of Data Science, Business and AI
