import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from download_dataset import download_nslkdd

def save_trained_model():
    print("Đang tải dữ liệu...")
    train_df, test_df = download_nslkdd()
    
    # Chuẩn bị dữ liệu
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    print("Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Đang huấn luyện mô hình Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    print("Đang chọn đặc trưng quan trọng...")
    feature_selector = SelectFromModel(rf_model, prefit=True)
    
    # Tạo thư mục models nếu chưa tồn tại
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("Đang lưu mô hình và các transformer...")
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_selector, 'models/feature_selector.pkl')
    
    print("Đã lưu mô hình thành công!")
    print("Các file đã được lưu trong thư mục 'models/':")
    print("- rf_model.pkl: Mô hình Random Forest")
    print("- scaler.pkl: StandardScaler")
    print("- feature_selector.pkl: Feature Selector")

if __name__ == "__main__":
    save_trained_model()