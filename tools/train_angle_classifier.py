import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class AngleClassifier:
    def __init__(self, model_path='angle_classifier_svm.pkl'):
        self.model_path = model_path
        self.model = None

    def _calculate_zenith(self, dir_vectors):
        """Helper to calculate true label from direction vectors"""
        z = np.clip(dir_vectors[:, 2], -1.0, 1.0)
        zenith_rad = np.arccos(z)
        zenith_deg = np.degrees(zenith_rad)
        return zenith_deg

    def extract_features(self, bar_energies_14x22):
        """
        Extract 7 physical features from 14x22 energy matrix.
        Input: (14, 22) numpy array
        Output: (7,) numpy array
        """
        # Ensure input is float
        bar_energies_14x22 = np.array(bar_energies_14x22, dtype=float)
        
        # 1. Basic Statistics
        layer_E = np.sum(bar_energies_14x22, axis=1)
        total_E = np.sum(layer_E)
        shower_max_layer = np.argmax(layer_E)
        
        # 2. LongRMS (Longitudinal Spread)
        if total_E > 0:
            z_coords = np.arange(14)
            z_center = np.average(z_coords, weights=layer_E)
            z_rms = np.sqrt(np.average((z_coords - z_center)**2, weights=layer_E))
        else:
            z_rms = 0
            
        # 3. LatRMS (Lateral Spread) & Centroids
        lat_rms_list = []
        centroids_x, layers_x = [], []
        centroids_y, layers_y = [], []
        
        for L in range(14):
            E_L = bar_energies_14x22[L]
            E_L_sum = np.sum(E_L)
            if E_L_sum > 0:
                bars = np.arange(22)
                centroid = np.average(bars, weights=E_L)
                rms = np.sqrt(np.average((bars - centroid)**2, weights=E_L))
                lat_rms_list.append(rms)
                
                if L % 2 == 0: # Even layers (X)
                    centroids_x.append(centroid)
                    layers_x.append(L)
                else: # Odd layers (Y)
                    centroids_y.append(centroid)
                    layers_y.append(L)
            else:
                lat_rms_list.append(0)
                
        avg_lat_rms = np.mean(lat_rms_list)
        
        # 4. Weighted Slope (Track Direction)
        def get_weighted_slope(layers, centroids, energies):
            if len(layers) > 1 and np.sum(energies) > 0:
                cov = np.cov(layers, centroids, aweights=energies)[0, 1]
                var = np.cov(layers, layers, aweights=energies)[0, 0]
                return cov / var if var > 0 else 0
            return 0

        E_x = [layer_E[L] for L in layers_x]
        slope_x = get_weighted_slope(layers_x, centroids_x, E_x)
        
        E_y = [layer_E[L] for L in layers_y]
        slope_y = get_weighted_slope(layers_y, centroids_y, E_y)
        
        slope_mag = np.sqrt(slope_x**2 + slope_y**2)
        
        # Feature Vector: [Slope, LatRMS, LongRMS, ShowerMax, TotalE, SlopeX, SlopeY]
        return np.array([slope_mag, avg_lat_rms, z_rms, shower_max_layer, total_E, abs(slope_x), abs(slope_y)])

    def train(self, csv_path, save_model=True, max_samples=50000):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Total dataset size: {len(df)}")
        
        # Downsample if too large for SVM, keeping the two angular classes equal
        if len(df) > max_samples:
            print(f"Dataset too large for RBF SVM. Sampling {max_samples} samples (class-balanced)...")

            # Calculate labels for balancing
            pri_dirs = df[['PriDirec_0', 'PriDirec_1', 'PriDirec_2']].values
            norms = np.linalg.norm(pri_dirs, axis=1, keepdims=True)
            pri_dirs = pri_dirs / norms
            true_theta = self._calculate_zenith(pri_dirs)
            temp_y = (true_theta > 60).astype(int)

            # Equal number of events per class (paper Sec. 3.2: class-balanced sample)
            rng = np.random.RandomState(42)
            n_per_class = max_samples // 2
            picked = []
            for cls in (0, 1):
                cls_idx = np.where(temp_y == cls)[0]
                if len(cls_idx) == 0:
                    continue
                take = min(n_per_class, len(cls_idx))
                picked.extend(rng.choice(cls_idx, size=take, replace=False))
            df = df.iloc[np.sort(np.asarray(picked))]

            print(f"Sampled dataset size: {len(df)}")

        # 1. Prepare Labels
        pri_dirs = df[['PriDirec_0', 'PriDirec_1', 'PriDirec_2']].values
        norms = np.linalg.norm(pri_dirs, axis=1, keepdims=True)
        pri_dirs = pri_dirs / norms
        true_theta = self._calculate_zenith(pri_dirs)
        y = (true_theta > 60).astype(int) # 0: <60, 1: >60
        
        # 2. Extract Features
        print("Extracting features...")
        X_features = []
        
        # Check column format
        if 'barE_0' in df.columns:
            print("Detected flat column format (barE_0 ... barE_307)")
            # Flat format: barE_0 to barE_307
            # Assuming order is Layer 0 (0-21), Layer 1 (22-43), ...
            bar_cols = [f'barE_{i}' for i in range(308)]
        else:
            print("Detected structured column format (barE_L_B)")
            # Structured format
            bar_cols = []
            for L in range(14):
                for B in range(22):
                    bar_cols.append(f'barE_{L}_{B}')
        
        # Extract all energy data at once
        all_energies = df[bar_cols].values # (N, 308)
        all_energies = all_energies.reshape(-1, 14, 22) # (N, 14, 22)
        
        for i in range(len(df)):
            feats = self.extract_features(all_energies[i])
            X_features.append(feats)
            if (i+1) % 5000 == 0:
                print(f"Processed {i+1}/{len(df)} samples...")
            
        X = np.array(X_features)
        
        # 3. Train SVM
        print("Training SVM model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Pipeline: Scaler -> SVM
        self.model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale', probability=True))
        self.model.fit(X_train, y_train)
        
        # 4. Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel Trained. Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['0-60 deg', '60-90 deg']))
        
        if save_model:
            print(f"Saving model to {self.model_path}...")
            joblib.dump(self.model, self.model_path)
            
    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        else:
            print(f"Model file {self.model_path} not found.")
            return False

    def predict(self, bar_energies_14x22):
        """
        Predict angle class for a single event.
        Input: (14, 22) numpy array
        Output: 0 (0-60 deg) or 1 (60-90 deg)
        """
        if self.model is None:
            if not self.load():
                raise Exception("Model not loaded and file not found.")
                
        features = self.extract_features(bar_energies_14x22)
        # Reshape for single sample prediction
        return self.model.predict([features])[0]

    def predict_proba(self, bar_energies_14x22):
        """
        Predict probabilities.
        Output: [prob_0_60, prob_60_90]
        """
        if self.model is None:
            if not self.load():
                raise Exception("Model not loaded and file not found.")
                
        features = self.extract_features(bar_energies_14x22)
        return self.model.predict_proba([features])[0]

if __name__ == "__main__":
    # 发布版占位路径：请按本地环境修改，或用环境变量指定
    # 输入 CSV 至少需要 barE_0..barE_307 与 PriDirec_0..2 这些列
    DATA_CSV = os.environ.get('BGO_ANGLE_CSV', 'events.csv')
    MODEL_OUT = os.environ.get('BGO_SVM_OUT', 'angle_classifier_svm.pkl')

    classifier = AngleClassifier(model_path=MODEL_OUT)
    # RBF SVM 不适合全量数据，限制到 50k 样本
    classifier.train(DATA_CSV, max_samples=50000)
    
    print("\n--- Interface Test ---")
    # Test with a dummy input
    dummy_data = np.random.rand(14, 22) * 100
    pred = classifier.predict(dummy_data)
    probs = classifier.predict_proba(dummy_data)
    print(f"Dummy Prediction: {pred} (Probs: {probs})")
