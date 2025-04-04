from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load model và scaler
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Cấu hình NSL-KDD
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Danh sách categories từ tập train
PROTOCOL_TYPES = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'smtp', 'ftp', 'domain_u', 'auth', 'telnet']  # Cập nhật đầy đủ
FLAGS = ['SF', 'S0', 'REJ', 'RSTO', 'SH']  # Cập nhật đầy đủ

def validate_filepath(filepath):
    upload_path = Path(app.config['UPLOAD_FOLDER']).resolve()
    requested_path = Path(filepath).resolve()
    return requested_path.parent == upload_path

def preprocess_data(df):
    # Tạo encoder động
    encoder = OneHotEncoder(
        categories=[PROTOCOL_TYPES, SERVICES, FLAGS],
        handle_unknown='ignore',
        sparse_output=False
    )
    
    # Fit encoder
    encoder.fit(pd.DataFrame({
        'protocol_type': PROTOCOL_TYPES,
        'service': SERVICES,
        'flag': FLAGS
    }))
    
    # Transform dữ liệu
    encoded = encoder.transform(df[CATEGORICAL_FEATURES])
    numerical = df[NUMERICAL_FEATURES]
    return pd.concat([
        pd.DataFrame(numerical).reset_index(drop=True),
        pd.DataFrame(encoded)
    ], axis=1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'redirect': f'/results/{filename}'
        }), 200
    
    return jsonify({'error': 'Định dạng file không hợp lệ'}), 400
EXPECTED_N_FEATURES =42
@app.route('/results/<filename>')
def show_results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not validate_filepath(filepath):
        abort(403, description="Truy cập file không hợp lệ")

    try:
        # 1. Đọc dữ liệu
        df = pd.read_csv(filepath)
        
        # 2. Kiểm tra cấu trúc file
        required_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes']  # Thêm tất cả 41 features
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            abort(400, description=f"Thiếu các trường bắt buộc: {', '.join(missing)}")

        # 3. Tiền xử lý dữ liệu
        processed = preprocess_data(df)  # Đảm bảo hàm này trả về đúng số lượng features
        
        # 4. Kiểm tra số lượng features
        if processed.shape[1] != EXPECTED_N_FEATURES:  # Thay 42 bằng số features thực tế của model
            abort(400, description=f"Số lượng features không khớp. Yêu cầu: {EXPECTED_N_FEATURES}, Nhận được: {processed.shape[1]}")
            
        # 5. Chuẩn hóa và dự đoán
        scaled = scaler.transform(processed)
        df['prediction'] = model.predict(scaled)
        df['probability'] = model.predict_proba(scaled)[:, 1]
        
        return render_template('results.html', 
                            connections=df.to_dict('records'),
                            filename=filename)
    
    except Exception as e:
        app.logger.error(f"Lỗi xử lý file: {str(e)}")
        abort(500, description=f"Lỗi xử lý dữ liệu: {str(e)}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'txt'}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)