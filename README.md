# BookingAnalysis
# 📌 Project Overview

This project analyzes booking trends and predicts **booking cancellations** using various machine learning models. By leveraging a dataset containing **1,500 booking records**, we aim to uncover patterns that influence **user booking behavior and cancellation likelihood**. The study incorporates multiple predictive models, including **Random Forest, Gradient Boosting, Logistic Regression, and Support Vector Machine (SVM)**, to determine the best-performing algorithm for cancellation prediction.

## 📂 Dataset Information

The dataset consists of **1,500 records** with the following key variables:

1. **User Information:**
   - **User Age**: The age of the customer making the booking.
   - **Previous Bookings**: The number of previous bookings made by the customer.

2. **Booking Details:**
   - **Booking Price**: The cost of the booking in USD.
   - **Number of Nights**: The length of stay in nights.
   - **Room Type**: The type of room booked (Standard, Deluxe, Suite).
   - **Season**: The season in which the booking was made (Winter, Spring, Summer, Fall).

3. **Geographical Information:**
   - **Country**: The country of the user making the booking.

4. **Target Variable:**
   - **Cancellation**: Whether the booking was canceled (0 = No, 1 = Yes).

## 🚀 Workflow

### Step 1: Load and Clean the Dataset
- Read the dataset into a Pandas DataFrame.
- Handle **missing values** using **mean imputation**.
- Fix **inconsistent values** in categorical variables (e.g., Room Type corrections).

### Step 2: Feature Engineering
- **One-hot encode categorical variables** (Room Type, Country, Season).
- **Scale numerical features** for better model performance.

### Step 3: Model Training & Evaluation
- Train and evaluate the following models:
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
- Perform **cross-validation** to assess model performance.
- Measure **accuracy, precision, recall, and F1-score** for model evaluation.

## 📊Power BI Visualizations

1. **Booking Cancellation Trends** - A bar chart showing the proportion of bookings that were canceled vs. completed.
2. **Impact of Booking Price on Cancellations** - A scatter plot displaying how booking prices influence the likelihood of cancellation.
3. **Seasonal Booking Patterns** - A line graph illustrating booking trends across different seasons.
4. **Room Type Preferences** - A pie chart showing the distribution of bookings across different room types.

This project provides valuable insights into **customer behavior** and enables data-driven strategies to **reduce cancellations and optimize booking experiences**. 🚀

