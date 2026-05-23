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
- Google Colab
- Google Drive

## Project Structure
1. Data Preparation
2. Sentiment Labeling
3. Class Balancing
4. Text Cleaning
5. TF-IDF Vectorization (Phase 2)
6. Model Building (Phase 2)
7. Model Evaluation (Phase 2)

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
review that performed the following steps:

- Converted all text to lowercase for uniformity
- Removed all punctuation and numbers using regular expressions
- Removed stopwords using NLTK — common words like "the", "and", 
  "is" and "it" that carry no meaningful information for 
  sentiment prediction

Example transformation:

BEFORE:
Can't complain. Taste good and had quick delivery! It was my 
first time trying this tea out. I usually drink the peppermint 
one but this gave me energy and sustained me throughout the day.

AFTER:
cant complain taste good quick delivery first time trying tea 
usually drink peppermint one gave energy sustained throughout day

The cleaned text retains only the meaningful words that 
carry sentiment signal, making it significantly easier 
for a machine learning model to learn from.

---

## Key Insights So Far
- Only 15% of reviews were negative indicating Amazon food 
  products are generally well received
- Class imbalance was identified and corrected before modeling
  to prevent biased predictions
- Text cleaning significantly reduces noise and retains only 
  the words that carry real sentiment meaning

---

## Project Status
✅ Data loading and exploration complete
✅ Sentiment labeling complete
✅ Class imbalance handled
✅ Text cleaning complete

🔄 Phase 2 in progress - TF-IDF vectorization and model building

## Connection to Previous Project
This project directly addresses the main limitation of the 
E-Commerce Data Analysis project where numerical features 
produced only 36.50% model accuracy. By using actual customer 
review text as input, we expect significantly higher accuracy
demonstrating the importance of feature selection in 
machine learning.

## Author
Data Science Undergraduate | Passionate about using data to 
drive business strategy and decision making
Tools: Python, Pandas, NLTK, Scikit-learn, Google Colab
Currently exploring the intersection of Data Science, Business and AI
