import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

possible_files = ['Gia_Nha_Ha_Noi_v2.xlsx', 'Gia_Nha_Ha_Noi.csv']
file_name = next((name for name in possible_files if os.path.exists(name)), None)

if file_name is None:
    raise FileNotFoundError("Không tìm thấy file dữ liệu. Hãy kiểm tra tên file hoặc đặt file vào cùng thư mục với chương trình.")

if file_name.endswith('.xlsx'):
    df = pd.read_excel(file_name)
else:
    df = pd.read_csv(file_name)

if 'Gia_Nha' not in df.columns:
    target_candidates = ['Gia_Nha_Ty', 'Gia_Nha']
    found_target = next((col for col in target_candidates if col in df.columns), None)
    if found_target is None:
        raise ValueError(f"Không tìm thấy cột mục tiêu trong dữ liệu. Các cột hiện có: {list(df.columns)}")
    df = df.rename(columns={found_target: 'Gia_Nha'})

if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

X = df.drop(columns=['Gia_Nha'])
y = df['Gia_Nha']

numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression()),
    ]
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===')
print(f'Sai số toàn phương trung bình (MSE): {mse:.4f}')
print(f'Sai số tuyệt đối trung bình (MAE): {mae:.4f}')
print(f'Hệ số xác định (R-squared): {r2:.4f}\n')

print('=== PHƯƠNG TRÌNH HỒI QUY TƯƠNG ỨNG ===')
print(f'Hệ số chặn (Bias/Intercept): {model.named_steps["regressor"].intercept_:.4f}')
print('Trọng số (Weights/Coefficients) cho từng đặc trưng:')

X_processed = model.named_steps['preprocessor'].transform(X)
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefficients = model.named_steps['regressor'].coef_

for feature, coef in zip(feature_names, coefficients):
    print(f' - {feature}: {coef:.4f}')
print('=== ĐANG VẼ BIỂU ĐỒ... ===')

# Thiết lập phong cách nền cho biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Lấy lại các biến số học để biểu diễn
numeric_features = model.named_steps['preprocessor'].transformers_[0][2]
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefficients = model.named_steps['regressor'].coef_

# Lọc các hệ số tương ứng với dữ liệu số
num_indices = [i for i, f in enumerate(feature_names) if 'num__' in f]
num_features_names = [feature_names[i].replace('num__', '') for i in num_indices]
num_coefs = [coefficients[i] for i in num_indices]

# --- BIỂU ĐỒ 1: SO SÁNH GIÁ THỰC TẾ & DỰ ĐOÁN ---
# Dự đoán trên toàn bộ tập X để biểu đồ có nhiều điểm hơn
y_pred_all = model.predict(X)

axes[0].scatter(y, y_pred_all, color='blue', alpha=0.7, s=100)
# Vẽ đường chuẩn (y = x)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Giá Thực Tế (Tỷ VNĐ)', fontsize=12)
axes[0].set_ylabel('Giá Dự Đoán (Tỷ VNĐ)', fontsize=12)
axes[0].set_title('So Sánh Giá Thực Tế và Giá Dự Đoán', fontsize=14)

# --- BIỂU ĐỒ 2: TẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG SỐ ---
sns.barplot(x=num_coefs, y=num_features_names, ax=axes[1], palette='viridis')
axes[1].set_xlabel('Trọng Số (Coefficient)', fontsize=12)
axes[1].set_ylabel('Đặc Trưng (Feature)', fontsize=12)
axes[1].set_title('Tầm Quan Trọng Của Các Đặc Trưng Số', fontsize=14)

# Căn chỉnh và hiển thị biểu đồ
plt.tight_layout()
plt.show()