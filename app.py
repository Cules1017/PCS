from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import plotly
import plotly.express as px

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load mô hình đã train
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_selector = joblib.load('models/scaler.pkl')

ALLOWED_EXTENSIONS = {'csv', 'dump'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze/<filename>')
def analyze(filename):
    # Đọc dữ liệu
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    # Chuẩn bị dữ liệu cho phân tích
    ap_list = data.to_dict('records')
    
    return render_template('analyze.html', 
                         ap_list=ap_list,
                         filename=filename)

@app.route('/filter_ap', methods=['POST'])
def filter_ap():
    data = request.json
    filename = data.get('filename')
    filters = data.get('filters', {})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Áp dụng bộ lọc
    if filters.get('ssid'):
        df = df[df['SSID'].str.contains(filters['ssid'], na=False, case=False)]
    if filters.get('mac'):
        df = df[df['MAC'].str.contains(filters['mac'], na=False, case=False)]
    if filters.get('encryption'):
        df = df[df['Encryption'] == filters['encryption']]
    
    return jsonify({'ap_list': df.to_dict('records')})

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    # Chuẩn hóa và chọn đặc trưng
    X = data.drop(['SSID', 'MAC', 'Encryption'], axis=1)  # Điều chỉnh theo cột thực tế
    X_scaled = scaler.transform(X)
    X_selected = feature_selector.transform(X_scaled)
    
    # Dự đoán
    predictions = model.predict(X_selected)
    probabilities = model.predict_proba(X_selected)[:, 1]
    
    # Thêm kết quả vào DataFrame
    data['Risk_Level'] = predictions
    data['Risk_Probability'] = probabilities
    
    # Tạo biểu đồ
    fig = px.scatter(data, 
                    x='Signal_Strength', 
                    y='Risk_Probability',
                    color='Risk_Level',
                    hover_data=['SSID', 'MAC'],
                    title='Access Point Risk Assessment')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('results.html',
                         ap_results=data.to_dict('records'),
                         plot=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)