# Import ไลบรารีที่จำเป็น
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. โหลดข้อมูล
file_path = 'all_thalassemia.xlsx'  # ใส่ชื่อไฟล์ Excel
data = pd.read_excel(file_path)

# 2. EDA (Exploratory Data Analysis)
print("ข้อมูลเริ่มต้น:")
print(data.head())
print("\nข้อมูลสรุป:")
print(data.info())
print("\nค่าสถิติพื้นฐาน:")
print(data.describe())

# วิเคราะห์การกระจายตัวและความสัมพันธ์
sns.pairplot(data, hue='การแปลผลการวิเคราะห์')  # ใช้คอลัมน์เป้าหมายที่ถูกต้อง
plt.show()

# 3. Data Cleaning
# ตรวจหาค่า missing
print("\nค่าที่หายไปในแต่ละคอลัมน์:")
print(data.isnull().sum())

# เติมหรือกำจัดค่าที่หายไป
data = data.dropna()  # หรือเติมค่าที่เหมาะสมด้วย data.fillna()

# 4. Feature Engineering
# การแปลงหรือสร้างฟีเจอร์ใหม่ (ถ้าจำเป็น)
# เช่น การสร้างฟีเจอร์ Interaction Terms หรือแปลงข้อมูลเป็น Dummy Variables
data = pd.get_dummies(data, drop_first=True)

# 5. Feature Selection
X = data.drop('การแปลผลการวิเคราะห์', axis=1)  # ฟีเจอร์
y = data['การแปลผลการวิเคราะห์']  # คอลัมน์เป้าหมาย

# เลือกฟีเจอร์ที่สำคัญ
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print("ฟีเจอร์ที่ถูกเลือก:", X.columns[selector.get_support()])

# 6. Compare Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {results[name]}")

# 7. Model Selection
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"โมเดลที่ดีที่สุดคือ: {best_model_name} ด้วยความแม่นยำ {results[best_model_name]}")

# 8. Model Evaluation
y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Fine-Tune Model
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print("\nโมเดลที่ปรับแต่งแล้ว:")
print(grid_search.best_params_)

# 10. Save Model
joblib.dump(best_rf_model, 'thalassemia_model.pkl')
print("\nโมเดลถูกบันทึกลงในไฟล์ thalassemia_model.pkl")
