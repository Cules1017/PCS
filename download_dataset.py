import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import urllib.request
import gzip
import shutil

def download_nslkdd():
    """Tải dataset NSL-KDD"""
    
    # Tạo thư mục data nếu chưa tồn tại
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL của dataset
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    
    # Tải files
    print("Đang tải dataset NSL-KDD...")
    urllib.request.urlretrieve(train_url, "data/KDDTrain+.txt")
    urllib.request.urlretrieve(test_url, "data/KDDTest+.txt")
    
    # Tên các cột trong dataset
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
              'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
              'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
              'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
              'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
              'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
              'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
              'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty']
    
    # Đọc dữ liệu
    print("Đang đọc và xử lý dữ liệu...")
    train_df = pd.read_csv('data/KDDTrain+.txt', names=columns)
    test_df = pd.read_csv('data/KDDTest+.txt', names=columns)
    
    # Xử lý dữ liệu
    def preprocess_data(df):
        # Chuyển đổi các cột categorical
        categorical_columns = ['protocol_type', 'service', 'flag']
        le = LabelEncoder()
        for column in categorical_columns:
            df[column] = le.fit_transform(df[column])
        
        # Chuyển đổi nhãn thành binary (normal = 0, attack = 1)
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Loại bỏ cột difficulty
        df = df.drop('difficulty', axis=1)
        
        return df
    
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Lưu dữ liệu đã xử lý
    print("Đang lưu dữ liệu đã xử lý...")
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("Hoàn thành! Dữ liệu đã được lưu trong thư mục 'data'")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = download_nslkdd()