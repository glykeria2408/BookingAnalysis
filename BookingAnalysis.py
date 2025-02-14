from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Cleaning the dataset
# Fixing inconsistent values in Room_Type
df_booking_cleaned = df_booking.copy()
df_booking_cleaned["Room_Type"] = df_booking_cleaned["Room_Type"].replace("Unknown", "Standard")

# Handling missing values using mean imputation for numerical columns
imputer = SimpleImputer(strategy="mean")
df_booking_cleaned[["Booking_Price", "Number_of_Nights", "Previous_Bookings"]] = imputer.fit_transform(
    df_booking_cleaned[["Booking_Price", "Number_of_Nights", "Previous_Bookings"]]
)

# Splitting dataset into features and target variable
X = df_booking_cleaned.drop(columns=["Booking_ID", "Cancellation"])
y = df_booking_cleaned["Cancellation"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numerical_features = ["User_Age", "Booking_Price", "Number_of_Nights", "Previous_Bookings"]
categorical_features = ["Room_Type", "Country", "Season"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC()
}

# Training and evaluating models
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print("-" * 50)

# Cross-validation for model performance assessment
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy for {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
