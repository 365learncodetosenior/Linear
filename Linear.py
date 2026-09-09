import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Đọc dữ liệu từ file hiện có trong thư mục
# Hỗ trợ cả file Excel cũ và file CSV đang có trong project.
possible_files = ['Gia_Nha_Ha_Noi_v2.xlsx', 'Gia_Nha_Ha_Noi.csv']
file_name = next((name for name in possible_files if os.path.exists(name)), None)

if file_name is None:
    raise FileNotFoundError("Không tìm thấy file dữ liệu. Hãy kiểm tra tên file hoặc đặt file vào cùng thư mục với chương trình.")

if file_name.endswith('.xlsx'):
    df = pd.read_excel(file_name)
else:
    df = pd.read_csv(file_name)

# 2. Kiểm tra cột quan trọng
if 'Gia_Nha' not in df.columns:
    # Trường hợp file cũ có tên cột khác
    target_candidates = ['Gia_Nha_Ty', 'Gia_Nha']
    found_target = next((col for col in target_candidates if col in df.columns), None)
    if found_target is None:
        raise ValueError(f"Không tìm thấy cột mục tiêu trong dữ liệu. Các cột hiện có: {list(df.columns)}")
    df = df.rename(columns={found_target: 'Gia_Nha'})

# Không dùng cột ID nếu có, vì đây là mã định danh không mang thông tin dự báo.
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# 3. Tách X và y
X = df.drop(columns=['Gia_Nha'])
y = df['Gia_Nha']

# 4. Chuẩn bị dữ liệu cho biến phân loại
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

# 5. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Huấn luyện mô hình
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression()),
    ]
)
model.fit(X_train, y_train)

# 7. Dự đoán và đánh giá
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