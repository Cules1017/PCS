from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from datetime import datetime
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

# Cấu hình cho phân tích WiFi traffic
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
# Categories và số features mong đợi
PROTOCOL_TYPES = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'https', 'dns', 'dhcp', 'ftp', 'ssh', 'telnet', 'smtp', 'pop3', 'imap']
FLAGS = ['SF', 'S0', 'REJ', 'RSTO', 'S1', 'S2', 'S3', 'OTH']

# Feature configuration
PROTOCOLS = ['tcp', 'udp', 'icmp']  # 3 protocols
MAIN_SERVICES = ['http', 'https', 'dns', 'smtp', 'ssh', 'telnet', 'ftp', 'pop3', 'imap']  # 9 services
MAIN_FLAGS = ['SF', 'S0', 'REJ', 'RSTO', 'S1']  # 5 flags

# Total expected features
EXPECTED_FEATURES = len(NUMERICAL_FEATURES) + len(PROTOCOLS)  # 38 numerical + 3 protocols

def log_model_config():
    """Log model configuration details."""
    app.logger.info(
        f"Model configuration:\n"
        f"Total features: {EXPECTED_FEATURES}\n"
        f"- Numerical: {len(NUMERICAL_FEATURES)}\n"
        f"- Categorical: {len(PROTOCOL_TYPES) + len(SERVICES) + len(FLAGS)}\n"
        f"  * Protocols: {len(PROTOCOL_TYPES)} {PROTOCOL_TYPES}\n"
        f"  * Services: {len(SERVICES)} {SERVICES}\n"
        f"  * Flags: {len(FLAGS)} {FLAGS}"
    )


def validate_filepath(filepath):
    upload_path = Path(app.config['UPLOAD_FOLDER']).resolve()
    requested_path = Path(filepath).resolve()
    return requested_path.parent == upload_path

def preprocess_data(df):
    """Tiền xử lý dữ liệu cho model."""
    try:
        app.logger.info(f"Processing {df.shape[0]} records")
        
        # 1. Xử lý dữ liệu số
        num_df = pd.DataFrame()
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột số {col}")
            num_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        app.logger.info(f"Numerical features: {len(NUMERICAL_FEATURES)} columns")
        
        # 2. Xử lý dữ liệu categorical
        cat_df = pd.DataFrame()
        cat_stats = {}
        
        for col, valid_values in [
            ('protocol_type', PROTOCOL_TYPES),
            ('service', SERVICES),
            ('flag', FLAGS)
        ]:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột {col}")
                
            # Chuyển đổi và validate giá trị
            series = df[col].astype(str)
            value_counts = series.value_counts()
            cat_stats[col] = value_counts.to_dict()
            
            invalid = set(value_counts.index) - set(valid_values)
            if invalid:
                app.logger.warning(f"Giá trị không hợp lệ trong {col}: {invalid}")
            
            cat_df[col] = series
        
        app.logger.info("Category stats:")
        for col, stats in cat_stats.items():
            app.logger.info(f"{col}: {stats}")
        
        # 3. Mã hóa protocol type (chỉ sử dụng protocol)
        protocol_encoder = OneHotEncoder(
            categories=[PROTOCOLS],
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        # Convert protocol to lowercase and encode
        protocols = cat_df[['protocol_type']].copy()
        protocols['protocol_type'] = protocols['protocol_type'].str.lower()
        
        # Encode protocols only
        cat_encoded = protocol_encoder.fit_transform(protocols)
        app.logger.info(f"Protocol encoding: {cat_encoded.shape[1]} features "
                       f"(protocols: {PROTOCOLS})")
        
        # 4. Validate and combine features
        if num_df.shape[1] != len(NUMERICAL_FEATURES):
            raise ValueError(
                f"Invalid numerical features count: {num_df.shape[1]}, "
                f"expected {len(NUMERICAL_FEATURES)}"
            )
            
        if cat_encoded.shape[1] != len(PROTOCOLS):
            raise ValueError(
                f"Invalid protocol encoding: got {cat_encoded.shape[1]} features, "
                f"expected {len(PROTOCOLS)} ({PROTOCOLS})"
            )
        
        # Combine features
        processed = np.hstack([num_df.values, cat_encoded])
        total_features = processed.shape[1]
        
        # Log feature composition
        app.logger.info(
            f"Feature composition:\n"
            f"- Numerical ({num_df.shape[1]}): {', '.join(NUMERICAL_FEATURES[:5])}...\n"
            f"- Protocol ({cat_encoded.shape[1]}): {', '.join(PROTOCOLS)}\n"
            f"Total: {total_features} features"
        )
        
        # Final validation
        if total_features != EXPECTED_FEATURES:
            error_msg = (
                f"Feature count mismatch: got {total_features}, "
                f"expected {EXPECTED_FEATURES} "
                f"({len(NUMERICAL_FEATURES)} numerical + {len(PROTOCOLS)} protocol)"
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)
            raise ValueError(
                f"Feature count mismatch: got {total_features}, expected {EXPECTED_FEATURES}\n"
                f"- Numerical: {num_df.shape[1]} (expected {len(NUMERICAL_FEATURES)})\n"
                f"- Categorical: {cat_encoded.shape[1]} (expected {len(PROTOCOLS)})"
            )
        
        return processed.astype(float)  # Ensure all values are float
        
    except Exception as e:
        app.logger.error("Preprocessing error:")
        app.logger.error(f"Input shape: {df.shape}")
        if 'num_df' in locals():
            app.logger.error(f"Numerical stats: {num_df.describe().to_dict()}")
        if 'cat_stats' in locals():
            app.logger.error(f"Categorical stats: {cat_stats}")
        app.logger.exception(e)
        raise ValueError(f"Preprocessing failed: {str(e)}")

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
        # Thêm timestamp vào tên file
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': new_filename,
            'redirect': f'/results/{new_filename}'
        }), 200
    
    return jsonify({'error': 'Định dạng file không hợp lệ'}), 400

@app.route('/results/<filename>')
def show_results(filename):
    """Phân tích và hiển thị kết quả từ file traffic."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not validate_filepath(filepath):
        abort(403, description="Truy cập file không hợp lệ")

    try:
        # 1. Load and validate data
        app.logger.info(f"Processing file: {filename}")
        df = pd.read_csv(filepath)
        app.logger.info(f"Loaded data shape: {df.shape}")

        # Check required columns
        missing_cols = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # 2. Clean data
        df_clean = df.copy()
        
        # Convert numerical columns
        for col in NUMERICAL_FEATURES:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        app.logger.info(f"Converted {len(NUMERICAL_FEATURES)} numerical columns")

        # Drop invalid rows
        before_clean = len(df_clean)
        df_clean = df_clean.dropna(subset=NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        rows_dropped = before_clean - len(df_clean)
        app.logger.info(f"Dropped {rows_dropped} rows with invalid data")

        if len(df_clean) == 0:
            raise ValueError("No valid data remained after cleaning")

        # Reset index for consistency
        df_clean = df_clean.reset_index(drop=True)
        app.logger.info(f"Clean data shape: {df_clean.shape}")

        # 3. Process and predict
        try:
            # Feature processing
            processed = preprocess_data(df_clean)
            app.logger.info(f"Processed features: {processed.shape}")

            # Scale data
            scaled = scaler.transform(processed)
            app.logger.info(f"Scaled features: {scaled.shape}")

            # Make predictions
            predictions = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[:, 1]

            # Log prediction results
            normal_count = sum(predictions == 0)
            attack_count = sum(predictions == 1)
            app.logger.info(f"Predictions: {normal_count} normal, {attack_count} attacks")

        except Exception as e:
            app.logger.error("Error in processing/prediction step:")
            app.logger.error(f"- Error: {str(e)}")
            app.logger.error(f"- Data shape: {df_clean.shape}")
            if 'processed' in locals():
                app.logger.error(f"- Processed shape: {processed.shape}")
            raise ValueError(f"Data processing failed: {str(e)}")

        # 4. Prepare visualization data
        df_clean['prediction'] = predictions
        df_clean['probability'] = probabilities

        plot_data = {
            'src_bytes': df_clean['src_bytes'].astype(float).tolist(),
            'dst_bytes': df_clean['dst_bytes'].astype(float).tolist(),
            'predictions': predictions.tolist(),
            'protocol_types': df_clean['protocol_type'].astype(str).tolist(),
            'services': df_clean['service'].astype(str).tolist()
        }

        # Validate plot data
        row_count = len(df_clean)
        for key, arr in plot_data.items():
            if len(arr) != row_count:
                raise ValueError(f"Data length mismatch: {key} has {len(arr)} items, expected {row_count}")
            app.logger.info(f"Validated {key}: {len(arr)} items")

        # Render results
        app.logger.info(f"Rendering results for {row_count} records")
        return render_template(
            'results.html',
            plot_data=plot_data,
            connections=df_clean.to_dict('records'),
            filename=filename,
            record_count=row_count
        )

    except ValueError as e:
        app.logger.error(f"Validation error: {str(e)}")
        app.logger.error("Data state:")
        if 'df' in locals():
            app.logger.error(f"- Input shape: {df.shape}")
            app.logger.error(f"- Input columns: {df.columns.tolist()}")
        if 'df_clean' in locals():
            app.logger.error(f"- Cleaned shape: {df_clean.shape}")
        if 'processed' in locals():
            app.logger.error(f"- Processed shape: {processed.shape}")
        abort(400, description=str(e))

    except Exception as e:
        app.logger.error("Unexpected error:")
        app.logger.exception(e)
        app.logger.error("Processing state:")
        for name, var in [('df', df), ('df_clean', df_clean),
                         ('processed', processed), ('scaled', scaled)]:
            if name in locals():
                app.logger.error(f"- {name} shape: {var.shape}")
        abort(500, description="Lỗi hệ thống. Vui lòng thử lại hoặc liên hệ quản trị viên.")

def allowed_file(filename):
    # Kiểm tra phần mở rộng file
    allowed_extensions = {'csv', 'txt'}
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in allowed_extensions

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        log_model_config()
    app.run(host='0.0.0.0', port=5000, debug=True)