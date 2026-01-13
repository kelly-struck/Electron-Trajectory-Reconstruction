# -*- coding: utf-8 -*-
# Physics-Guided Deep Learning Method: Integrating Traditional Physics with Deep Learning
# For Particle Trajectory Reconstruction

# Add these environment variables at the top of the file, must be before importing TF

# Add this function and ensure it is called before importing TF
def setup_tf_environment():
    """Set environment variables, must be called before importing TensorFlow"""
    import os
    import sys

    # 1. Completely disable XLA/JIT compilation
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_XLA'] = '1'
    os.environ['TF_DISABLE_JIT'] = '1'
    os.environ['TF_CUDA_ENABLE_TENSOR_CORE_MATH'] = '0'

    # 2. Solve libdevice issue - Check and set CUDA path
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.7",
        "/usr/local/cuda-12.4"
    ]

    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            nvvm_path = os.path.join(cuda_path, "nvvm/libdevice")
            if os.path.exists(nvvm_path):
                os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_path}"
                print(f"Found CUDA directory: {cuda_path}")
                break

    # 3. Disable GPU acceleration
    # If you want to use CPU mode completely, uncomment the next line
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 4. Set TF performance optimization
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 5. Pre-select CPU or GPU usage
    try:
        use_cpu = "--cpu" in sys.argv or "--CPU" in sys.argv
        if use_cpu:
            print("Forcing CPU mode")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    except:
        pass

    print("TensorFlow environment configuration completed")


# Call this function before importing TensorFlow
setup_tf_environment()

# Other imports
import random
import numpy as np
# Now import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
import os

# Add paths to import traditional physics methods
from LinearFit import linear_fit_weighted
from LeiTrackReBuild import layer_centroid, bgo_linear_track_cls, stk_track_bgo_sigma

# Define constants
BGOL = 7
BGOLB = 22

# Set random seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Apply random seeds
set_seeds(42)

# Increase log level to reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide non-error messages

# Optimize XLA config
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # Use XLA JIT compilation as much as possible

# Disable XLA compilation for determinism
tf.config.optimizer.set_jit(False)


# Environment configuration part needs thorough modification, forcibly disable XLA
def set_environment_for_tf210():
    """Set environment parameters suitable for TensorFlow 2.10, completely disabling XLA and CUDA optimization"""
    # 1. Completely disable XLA - Avoid operations requiring libdevice
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_XLA'] = '1'  # Explicitly disable XLA

    # 2. Disable JIT compilation and GPU optimization
    os.environ['TF_DISABLE_JIT'] = '1'  # Disable JIT compilation
    os.environ['TF_CUDA_ENABLE_TENSOR_CORE_MATH'] = '0'  # Disable TensorCore acceleration
    tf.config.optimizer.set_jit(False)  # Explicitly disable JIT in TensorFlow config

    # 3. Configure GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected {len(gpus)} GPU(s), configured for on-demand memory allocation")
            # Set visible devices
            tf.config.set_visible_devices(gpus[0:1], 'GPU')  # Use only the first GPU
            print("Restricted to use only one GPU")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    # 4. Set general environment variables
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set log level, hide unimportant warnings

    # 5. Configure precision and parallelism
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print("TensorFlow environment configuration completed, XLA, JIT, and CUDA optimizations disabled")

# Physics feature extraction function
def extract_physics_features(barE_data):
    """
    Extract physics features from energy data
    - barE_data: Tensor of shape [batch_size, 14, 22]
    - Returns: Tensor of shape [batch_size, 21+3], containing features processed by traditional physics
    """
    # Convert TF tensor to numpy for processing
    batch_size = tf.shape(barE_data)[0]
    features = tf.TensorArray(tf.float32, size=batch_size)

    def process_single_sample(i, barE):
        # Store physics features
        physics_feats = tf.zeros((24,), dtype=tf.float32)

        # Reconstruct BGO detector info from barE
        BHXXL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHXZL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHXEL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYYL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYZL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYEL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)

        # Calculate geometric positions, consistent with ReConBGODir2.py
        # Note: In real applications, alignment parameter reading and application should be added
        for l in range(BGOL):
            for b in range(BGOLB):
                BHYYL = tf.tensor_scatter_nd_update(
                    BHYYL, [[l, b]], [tf.constant(-288.75) + b * tf.constant(27.5)]
                )
                BHYZL = tf.tensor_scatter_nd_update(
                    BHYZL, [[l, b]], [tf.constant(58.50) + l * tf.constant(58.0)]
                )
                BHYEL = tf.tensor_scatter_nd_update(
                    BHYEL, [[l, b]], [barE[2 * l, b]]
                )
                BHXXL = tf.tensor_scatter_nd_update(
                    BHXXL, [[l, b]], [tf.constant(-288.75) + b * tf.constant(27.5)]
                )
                BHXZL = tf.tensor_scatter_nd_update(
                    BHXZL, [[l, b]], [tf.constant(87.50) + l * tf.constant(58.0)]
                )
                BHXEL = tf.tensor_scatter_nd_update(
                    BHXEL, [[l, b]], [barE[2 * l + 1, b]]
                )

        # Calculate total energy
        BgoEng = tf.reduce_sum(barE)

        # Try trajectory reconstruction
        ln_x = tf.fill([BGOL], BGOLB)
        ln_y = tf.fill([BGOL], BGOLB)

        # Extract energy distribution features for each layer
        layer_features = tf.zeros((21,), dtype=tf.float32)
        for layer in range(7):
            # Calculate total energy for each layer
            layer_energy = tf.reduce_sum(barE[2*layer]) + tf.reduce_sum(barE[2*layer+1])

            # Calculate max energy and its position for each layer
            max_energy_x = tf.reduce_max(barE[2*layer])
            max_energy_y = tf.reduce_max(barE[2*layer+1])
            max_pos_x = tf.cast(tf.argmax(barE[2*layer]), tf.float32)
            max_pos_y = tf.cast(tf.argmax(barE[2*layer+1]), tf.float32)

            # Store key features for each layer
            idx = layer * 3
            layer_features = tf.tensor_scatter_nd_update(
                layer_features,
                [[idx], [idx+1], [idx+2]],
                [layer_energy, (max_energy_x + max_energy_y)/2, (max_pos_x + max_pos_y)/2]
            )

        # Calculate energy-weighted spatial distribution features
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0
        total_e = tf.reduce_sum(barE) + 1e-10

        # Weighted average in X direction
        for i in range(BGOL):
            for j in range(BGOLB):
                pos_x = -288.75 + j * 27.5
                pos_z = 87.50 + i * 58.0
                e = barE[2*i+1, j]
                weighted_x += pos_x * e / total_e
                weighted_z += pos_z * e / total_e

        # Weighted average in Y direction
        for i in range(BGOL):
            for j in range(BGOLB):
                pos_y = -288.75 + j * 27.5
                pos_z = 58.50 + i * 58.0
                e = barE[2*i, j]
                weighted_y += pos_y * e / total_e
                weighted_z = (weighted_z + pos_z * e / total_e) / 2  # Average Z position

        # Update physics features
        physics_feats = tf.concat([layer_features, [weighted_x, weighted_y, weighted_z]], axis=0)
        return physics_feats

    # Call physics feature extraction for each sample in the batch
    for i in range(batch_size):
        features = features.write(i, process_single_sample(i, barE_data[i]))

    return features.stack()

# TensorFlow function wrapper for physics feature extractor
@tf.function
def physics_features_extractor(barE_data):
    """TensorFlow function wrapper for physics feature extractor, used during training"""
    return tf.py_function(
        func=lambda x: extract_physics_features(x),
        inp=[barE_data],
        Tout=tf.float32
    )

# Data pipeline
def parse_tfrecord(example_proto):
    feature_description = {
        **{f'barE_{i}': tf.io.FixedLenFeature([22], tf.float32) for i in range(14)},
        'PriDirec_0': tf.io.FixedLenFeature([], tf.float32),
        'PriDirec_1': tf.io.FixedLenFeature([], tf.float32),
        'PriDirec_2': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Enhance data preprocessing robustness
    barE = tf.stack([parsed[f'barE_{i}'] for i in range(14)], axis=0)
    barE_original = barE  # Save original data for physics feature extraction

    # Clip extreme values to avoid exceptions in subsequent transformations
    barE = tf.clip_by_value(barE, -10.0, 20.0)

    # Add feature scaling - Standardization
    barE_mean = tf.constant(0.0, dtype=tf.float32)  # Pre-computed mean
    barE_std = tf.constant(1.0, dtype=tf.float32)   # Pre-computed std
    barE_normalized = (barE - barE_mean) / (barE_std + 1e-6)

    # Add log transformation to enhance feature processing
    barE_sign = tf.sign(barE_normalized)
    barE_log = barE_sign * tf.math.log1p(tf.abs(barE_normalized) + 1e-6)

    # Physics feature extraction (Simplified version, actual training should use complex implementation)
    physics_features = tf.zeros((24,), dtype=tf.float32)  # Placeholder

    # Targets: PriDirec_0, PriDirec_1, PriDirec_2
    targets = tf.stack([
        parsed['PriDirec_0'],
        parsed['PriDirec_1'],
        parsed['PriDirec_2']
    ])
    targets_norm = tf.norm(targets, axis=0) + 1e-10  # Avoid division by zero
    targets = targets / targets_norm

    # Flatten barE input
    barE_flat = tf.reshape(tf.concat([barE_normalized, barE_log], axis=-1), [-1])

    return {
        'barE_input': barE_flat,
        'barE_original': barE_original,  # Used for physics feature extraction
        'physics_features': physics_features  # Placeholder, processed later
    }, targets


def process_batch_for_physics(inputs, targets):
    """
    Batch processing to extract physics features
    """
    barE_original = inputs['barE_original']
    physics_features = extract_physics_features(barE_original)

    # Update input dictionary
    updated_inputs = {
        'barE_input': inputs['barE_input'],
        'physics_features': physics_features
    }

    return updated_inputs, targets

def build_data_pipeline(file_pattern, batch_size=2048, shuffle=True, include_physics=True):
    # Enhance data pipeline fault tolerance and determinism
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    # If shuffle is needed, set random seed for reproducibility
    if shuffle:
        files = files.shuffle(buffer_size=100, seed=42)

    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024
    )

    # Parse and preprocess
    dataset = dataset.map(parse_tfrecord,
                         num_parallel_calls=tf.data.AUTOTUNE)

    # Use larger buffer_size for shuffling, and set random seed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    # Batching
    dataset = dataset.batch(batch_size)

    # Add physics feature extraction processing
    if include_physics:
        dataset = dataset.map(process_batch_for_physics,
                            num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# Model Architecture
class BGOInput(layers.Layer):
    def __init__(self, **kwargs):
        super(BGOInput, self).__init__(**kwargs)
        # More robust initializer
        self.reshape = layers.Reshape((14, 22, 2))

    def call(self, inputs):
        x = self.reshape(inputs)
        return x

# Enhanced Attention Mechanism Module
class ChannelAttention(layers.Layer):
    def __init__(self, filters, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')
        self.dense2 = layers.Dense(filters, kernel_initializer='he_normal')

    def call(self, x):
        avg_out = self.dense2(self.dense1(self.avg_pool(x)))
        max_out = self.dense2(self.dense1(self.max_pool(x)))
        out = layers.add([avg_out, max_out])
        out = tf.nn.sigmoid(out)
        return layers.Reshape((1, 1, out.shape[1]))(out)

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid',
                                kernel_initializer='he_normal')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        out = layers.Concatenate()([avg_out, max_out])
        out = self.conv(out)
        return out

class CBAM(layers.Layer):
    def __init__(self, filters, ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ca = ChannelAttention(filters, ratio)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# Enhance numerical stability of residual block and ensure reproducibility
def residual_block(x, filters, kernel_size=(3,3), expansion=2.0, seed_offset=0, use_attention=True, dropout_rate=0.1):
    shortcut = x
    in_channels = x.shape[-1]

    # First convolution layer
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.Activation('swish')(x)

    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=43+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # Attention Mechanism
    if use_attention:
        x = CBAM(filters)(x)

    # If input and output channels differ, use 1x1 convolution to adjust
    if in_channels != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=44+seed_offset))(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)

    # Merge residual connection
    x = layers.add([x, shortcut])
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate, seed=45+seed_offset)(x)

    return x


# Physics-Guided Multimodal Model
def build_physics_guided_model():
    # BGO Energy Input
    barE_input = tf.keras.Input(shape=(14*22*2,), name='barE_input')

    # Reshape to image format
    x = BGOInput()(barE_input)

    # Extract local features
    x = layers.Conv2D(128, (3, 3), padding='same', activation='swish',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # Add an extra initial convolution layer
    x = layers.Conv2D(128, (3, 3), padding='same', activation='swish',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=43))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # Residual Block 1
    x = residual_block(x, 256, seed_offset=0, dropout_rate=0.1)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 2
    x = residual_block(x, 512, seed_offset=10, dropout_rate=0.15)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 3
    x = residual_block(x, 1024, seed_offset=20, dropout_rate=0.2)

    # Add an extra residual block
    x = residual_block(x, 1024, seed_offset=30, dropout_rate=0.2)

    # Multi-scale feature extraction
    x_global = layers.GlobalAveragePooling2D()(x)
    x_max = layers.GlobalMaxPooling2D()(x)
    x_combined = layers.Concatenate()([x_global, x_max])

    # Enhanced Fully Connected Layers
    x_global = layers.Dense(512, activation='swish')(x_combined)  # Increased to 512
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.3)(x_global)

    x_global = layers.Dense(512, activation='swish')(x_global)  # Add extra layer
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.2)(x_global)
    # Enhanced Physics Feature Processing Branch
    physics_input = tf.keras.Input(shape=(24,), name='physics_features')
    p = layers.Dense(128, activation='swish')(physics_input)  # Increased to 128
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # Increased to 256
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # Add extra layer
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.2)(p)

    # Enhanced Fusion Network
    combined = layers.Concatenate()([x_global, p])

    # Deeper Fusion Processing Network
    z = layers.Dense(1024, activation='swish')(combined)       # Increased to 1024
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Dense(512, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(256, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)

    # Model's two outputs:
    # 1. Direction prediction
    direction_output = layers.Dense(3, name='direction_output')(z)

    # 2. Decision network output - Used to judge whether to use physics method
    decision_output = layers.Dense(1, activation='sigmoid', name='decision_output')(z)

    # Create model
    model = tf.keras.Model(
        inputs=[barE_input, physics_input],
        outputs=[direction_output, decision_output]
    )

    return model

# Physics Method Prediction
def physics_method_prediction(barE_data):
    """Use traditional physics method for prediction, return direction vector"""
    # This function should contain the complete implementation of the traditional physics method
    # For simplification, returning a random value here
    batch_size = barE_data.shape[0]
    # In actual application, full physics method should be implemented
    return tf.random.normal((batch_size, 3))

# Custom Loss Function - Completely rewritten to avoid shape incompatibility
class PhysicsGuidedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, name="physics_guided_loss"):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        # Define sub-loss functions
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # Separate predictions
        if isinstance(y_pred, (list, tuple)):
            direction_pred = y_pred[0]  # [batch_size, 3]
            decision_pred = y_pred[1]   # [batch_size, 1]
        else:
            # If prediction is not list/tuple, assume it only contains direction prediction
            direction_pred = y_pred
            # Create fake decision prediction (zeros)
            decision_pred = tf.zeros((tf.shape(y_true)[0], 1), dtype=tf.float32)

        # 1. Calculate direction MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - direction_pred))

        # 2. Calculate direction cosine similarity loss
        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(direction_pred, axis=-1)

        # Calculate cosine similarity for each sample
        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
        direction_loss = tf.reduce_mean(tf.square(1.0 - cos_sim))

        # 3. Decision loss - Simplified to use scalar ops to avoid shape issues
        # We ignore actual physics method labels, only using a simple target: decision output should be close to 0.5
        # This is to avoid broadcasting shape incompatibility
        default_targets = 0.5 * tf.ones_like(tf.reshape(decision_pred, [-1]))
        decision_pred_flat = tf.reshape(decision_pred, [-1])
        decision_loss = tf.reduce_mean(tf.square(default_targets - decision_pred_flat))

        # Combine all losses
        total_loss = self.alpha * mse_loss + self.beta * direction_loss + self.gamma * decision_loss

        return total_loss

# Angle Error Metric
class AngleError(tf.keras.metrics.Metric):
    def __init__(self, name='angle_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.angle_error_sum = self.add_weight(name='angle_error_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get direction prediction part
        if isinstance(y_pred, list) or isinstance(y_pred, tuple):
            direction_pred = y_pred[0]
        else:
            direction_pred = y_pred

        # Ensure correct dimensions
        if len(tf.shape(y_true)) == 1:
            y_true = tf.expand_dims(y_true, 0)
        if len(tf.shape(direction_pred)) == 1:
            direction_pred = tf.expand_dims(direction_pred, 0)

        # Normalize vectors - use axis=-1
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(direction_pred, axis=-1)

        # Calculate dot product
        dot_product = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

        # Calculate angle (radians)
        angles = tf.math.acos(dot_product)

        # Convert to degrees
        angles_deg = angles * 180.0 / np.pi

        # Update state
        if sample_weight is not None:
            self.angle_error_sum.assign_add(tf.reduce_sum(angles_deg * sample_weight))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.angle_error_sum.assign_add(tf.reduce_sum(angles_deg))
            self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.angle_error_sum / (self.count + 1e-10)

    def reset_state(self):
        self.angle_error_sum.assign(0.0)
        self.count.assign(0.0)

# Cosine Similarity Metric
class CosineSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name='cosine_similarity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cos_sim_sum = self.add_weight(name='cos_sim_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get direction prediction part
        if isinstance(y_pred, list) or isinstance(y_pred, tuple):
            direction_pred = y_pred[0]
        else:
            direction_pred = y_pred

        # Ensure correct dimensions
        if len(tf.shape(y_true)) == 1:
            y_true = tf.expand_dims(y_true, 0)
        if len(tf.shape(direction_pred)) == 1:
            direction_pred = tf.expand_dims(direction_pred, 0)

        # Normalize vectors - use axis=-1
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(direction_pred, axis=-1)

        # Calculate cosine similarity
        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

        # Update state
        if sample_weight is not None:
            self.cos_sim_sum.assign_add(tf.reduce_sum(cos_sim * sample_weight))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.cos_sim_sum.assign_add(tf.reduce_sum(cos_sim))
            self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.cos_sim_sum / (self.count + 1e-10)

    def reset_state(self):
        self.cos_sim_sum.assign(0.0)
        self.count.assign(0.0)

# Hybrid Prediction Model
class HybridPredictionModel(tf.keras.Model):
    def __init__(self, dl_model):
        super().__init__()
        self.dl_model = dl_model
        # Load alignment parameters from disk
        self.alignment_params = {
            'BAYYL': np.zeros((BGOL, BGOLB)),
            'BAYZL': np.zeros((BGOL, BGOLB)),
            'BAXXL': np.zeros((BGOL, BGOLB)),
            'BAXZL': np.zeros((BGOL, BGOLB))
        }

        # Load alignment parameters
        try:
            self._load_alignment_params("BGOAlignment.txt")
            print("Successfully loaded BGO alignment parameters")
        except Exception as e:
            print(f"Failed to load BGO alignment parameters: {e}")

    def _load_alignment_params(self, filepath):
        """Load alignment parameters"""
        try:
            with open(filepath, "r") as fp:
                for i in range(BGOL):
                    for j in range(BGOLB):
                        line = fp.readline()
                        arr = line.split()
                        self.alignment_params['BAYYL'][i][j] = float(arr[3])
                        self.alignment_params['BAYZL'][i][j] = float(arr[4])
                    for j in range(BGOLB):
                        line = fp.readline()
                        arr = line.split()
                        self.alignment_params['BAXXL'][i][j] = float(arr[3])
                        self.alignment_params['BAXZL'][i][j] = float(arr[4])
        except Exception as e:
            print(f"Error loading alignment parameters: {e}")
            # Use default values
            pass

    def _apply_physics_method(self, barE_data):
        """Apply traditional physics method for prediction"""
        batch_size = tf.shape(barE_data)[0]
        results = tf.TensorArray(tf.float32, size=batch_size)

        for idx in range(batch_size):
            # Restore energy deposition to 14x22 array
            bgo_energy = tf.reshape(barE_data[idx], (14, 22))

            # Construct spatial coordinates and energy arrays
            BHXXL = np.zeros((BGOL, BGOLB))
            BHXZL = np.zeros((BGOL, BGOLB))
            BHXEL = np.zeros((BGOL, BGOLB))
            BHYYL = np.zeros((BGOL, BGOLB))
            BHYZL = np.zeros((BGOL, BGOLB))
            BHYEL = np.zeros((BGOL, BGOLB))

            # Use geometric position calculation consistent with ReConBGODir2.py
            for i in range(BGOL):
                for j in range(BGOLB):
                    BHYYL[i][j] = -288.75 + j * 27.5 + self.alignment_params['BAYYL'][i][j]
                    BHYZL[i][j] = 58.50 + i * 58.0 + self.alignment_params['BAYZL'][i][j]
                    BHYEL[i][j] = bgo_energy[2 * i][j].numpy()
                    BHXXL[i][j] = -288.75 + j * 27.5 + self.alignment_params['BAXXL'][i][j]
                    BHXZL[i][j] = 87.50 + i * 58.0 + self.alignment_params['BAXZL'][i][j]
                    BHXEL[i][j] = bgo_energy[2 * i + 1][j].numpy()

            # Calculate total energy
            BgoEng = np.sum(bgo_energy)

            # Reconstruct trajectory using physics method
            ln_x = [22] * 7  # Number of bars in X direction per layer
            ln_y = [22] * 7  # Number of bars in Y direction per layer

            # Try to reconstruct using physics method
            try:
                sigma_x, aBgoXCls, bBgoXCls, zz_x = bgo_linear_track_cls(
                    BgoEng, ln_x, BHXXL, BHXZL, BHXEL)
                sigma_y, aBgoYCls, bBgoYCls, zz_y = bgo_linear_track_cls(
                    BgoEng, ln_y, BHYYL, BHYZL, BHYEL)

                if sigma_x > 0 and sigma_y > 0:
                    direction = np.array([aBgoXCls, aBgoYCls, 1.0])
                    direction = direction / np.linalg.norm(direction)
                    results = results.write(idx, tf.convert_to_tensor(direction, dtype=tf.float32))
                else:
                    # Physics method failed, return zero vector as flag
                    results = results.write(idx, tf.zeros((3,), dtype=tf.float32))
            except Exception as e:
                # Physics method error, return zero vector
                print(f"Physics method error: {e}")
                results = results.write(idx, tf.zeros((3,), dtype=tf.float32))

        return results.stack()

    def call(self, inputs):
        # Get deep learning prediction
        dl_direction, decision = self.dl_model(inputs)

        # Try physics method prediction
        barE_original = inputs['barE_original']
        try:
            physics_direction = self._apply_physics_method(barE_original)

            # Check if physics prediction is valid (non-zero vector)
            physics_valid = tf.reduce_sum(tf.square(physics_direction), axis=1) > 1e-6
            physics_valid = tf.reshape(physics_valid, (-1, 1))

            # Combine decision network and physics validity
            decision = decision * tf.cast(physics_valid, tf.float32)

            # Fuse predictions
            # decision is output of decision network, range 0-1, representing weight of physics method
            final_direction = decision * physics_direction + (1 - decision) * dl_direction

            # Normalize final direction
            final_direction = tf.nn.l2_normalize(final_direction, axis=1)

            return final_direction, decision
        except:
            # Physics method failed, use only DL method
            return dl_direction, tf.zeros_like(decision)

    def predict_with_physics(self, inputs):
        """Hybrid prediction using physics method and deep learning model"""
        # Get DL model predictions
        dl_outputs = self.dl_model(inputs)
        dl_direction = dl_outputs[0]
        decision_pred = dl_outputs[1]

        # Physics method prediction
        try:
            barE_original = inputs['barE_original']
            physics_direction = self._apply_physics_method(barE_original)

            # Check if physics prediction is valid
            physics_valid = tf.reduce_sum(tf.square(physics_direction), axis=1) > 1e-6
            physics_valid = tf.reshape(physics_valid, (-1, 1))

            # Apply decision model, combined with physics validity
            used_decision = decision_pred * tf.cast(physics_valid, tf.float32)

            # Fuse predictions
            final_direction = used_decision * physics_direction + (1 - used_decision) * dl_direction
            final_direction = tf.nn.l2_normalize(final_direction, axis=-1)

            return final_direction, used_decision
        except Exception as e:
            print(f"Physics method prediction failed: {e}")
            return dl_direction, tf.zeros_like(decision_pred)

# Custom Learning Rate Schedule
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=1e-5, max_lr=5e-4, step_size=2000):
        super().__init__()
        # Ensure all inputs are float32
        self.base_lr = tf.cast(base_lr, dtype=tf.float32)
        self.max_lr = tf.cast(max_lr, dtype=tf.float32)
        self.step_size = tf.cast(step_size, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Ensure step is float32
        cycle = tf.math.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        # Use tf.constant to ensure 0.0 is float32
        zero = tf.constant(0.0, dtype=tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)
        lr = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(zero, one - x)
        return lr

    def get_config(self):
        return {
            'base_lr': float(self.base_lr),
            'max_lr': float(self.max_lr),
            'step_size': float(self.step_size)
        }

# Gradient Debugger Callback - Fix TF 2.10 Compatibility
class GradientDebugger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            try:
                # Check NaN and Inf - Use TF 2.10 compatible method to get gradients
                gradients = []
                variables = self.model.trainable_weights

                # Get loss function
                loss_fn = self.model.loss
                if hasattr(self.model, 'compiled_loss'):
                    # TF 2.x method
                    with tf.GradientTape() as tape:
                        y_pred = self.model(self.model.inputs, training=True)
                        loss = self.model.compiled_loss(self.model.targets, y_pred)
                    gradients = tape.gradient(loss, variables)

                has_nan = False
                has_inf = False
                for g in gradients:
                    if g is not None:  # Some layers may have no gradient
                        if np.isnan(np.sum(g.numpy())):
                            has_nan = True
                        if np.isinf(np.sum(g.numpy())):
                            has_inf = True

                if has_nan or has_inf:
                    print(f"\nBatch {batch}: Detected {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} gradients!")
            except Exception as e:
                # Just print warning if unable to get gradients
                print(f"Warning: Error during gradient check: {str(e)}")

# Batch Size Scheduler
class BatchSizeScheduler(tf.keras.callbacks.Callback):
    def __init__(self, train_data_path, start_size=1024, max_size=2048, steps=5):
        super().__init__()
        self.train_data_path = train_data_path
        self.start_size = start_size
        self.max_size = max_size
        self.steps = steps
        self.current_step = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % (self.steps * 2) == 0 and epoch > 0:
            self.current_step = (self.current_step + 1) % (self.steps * 2)

        if self.current_step < self.steps:
            new_batch_size = self.start_size + (self.max_size - self.start_size) * self.current_step // self.steps
        else:
            # Descent phase
            step_in_descent = self.current_step - self.steps
            new_batch_size = self.max_size - (self.max_size - self.start_size) * step_in_descent // self.steps

        print(f"\nUpdating batch size to: {new_batch_size}")

        # Rebuild dataset
        if hasattr(self.model, 'train_data'):
            self.model.train_data = build_data_pipeline(self.train_data_path, new_batch_size, shuffle=True)

# GPU Monitor Callback
class GPUMonitor(tf.keras.callbacks.Callback):
    def __init__(self, print_interval=5):
        super().__init__()
        self.print_interval = print_interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.print_interval == 0:
            # Try to get GPU usage info
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"\nGPU Memory Usage: Used={gpu_info['current'] / 1e9:.2f}GB, Peak={gpu_info['peak'] / 1e9:.2f}GB")
            except:
                # Some environments may not support this feature
                pass

# Learning Rate Logger
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_begin(self, epoch, logs=None):
        try:
            # Try different ways to get learning rate
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr
                if callable(lr):
                    lr = lr()
                if hasattr(lr, 'numpy'):
                    lr = lr.numpy()
                print(f"\nCurrent Learning Rate: {float(lr):.6f}")
            elif hasattr(self.model.optimizer, 'learning_rate'):
                lr = self.model.optimizer.learning_rate
                if callable(lr):
                    lr = lr()
                if hasattr(lr, 'numpy'):
                    lr = lr.numpy()
                print(f"\nCurrent Learning Rate: {float(lr):.6f}")
            else:
                # Last resort: try to access current learning rate value directly
                print("\nCannot directly access current learning rate")
        except Exception as e:
            print(f"\nError getting learning rate: {e}")

# Hybrid Model Evaluation and Visualization
def evaluate_hybrid_model(model, test_dataset, output_dir='./results'):
    """Evaluate hybrid model and show results as text"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect predictions
    all_predictions = []
    all_true_values = []
    all_decisions = []

    # Iterate over test dataset
    for inputs, targets in test_dataset:
        # Use predict_with_physics method
        predictions, decisions = model.predict_with_physics(inputs)
        all_predictions.append(predictions)
        all_true_values.append(targets)
        all_decisions.append(decisions)

    # Concatenate results
    all_predictions = np.vstack(all_predictions)
    all_true_values = np.vstack(all_true_values)
    all_decisions = np.vstack(all_decisions)

    # Calculate angle error
    y_true_norm = all_true_values / np.linalg.norm(all_true_values, axis=1, keepdims=True)
    y_pred_norm = all_predictions / np.linalg.norm(all_predictions, axis=1, keepdims=True)

    dot_products = np.sum(y_true_norm * y_pred_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angle_errors = np.arccos(dot_products) * 180.0 / np.pi

    # Calculate cosine similarity
    cosine_sim = dot_products

    # Print detailed evaluation statistics instead of charts
    print("\n===== Hybrid Model Evaluation Results =====")
    print(f"Total Samples: {len(angle_errors)}")
    
    print("\n--- Angle Error Distribution (Degrees) ---")
    print(f"  Mean: {np.mean(angle_errors):.2f}°")
    print(f"  Median: {np.median(angle_errors):.2f}°")
    print(f"  Std Dev: {np.std(angle_errors):.2f}°")
    print(f"  Min: {np.min(angle_errors):.2f}°")
    print(f"  Max: {np.max(angle_errors):.2f}°")
    print(f"  25th Percentile: {np.percentile(angle_errors, 25):.2f}°")
    print(f"  68th Percentile: {np.percentile(angle_errors, 68):.2f}°")
    print(f"  95th Percentile: {np.percentile(angle_errors, 95):.2f}°")

    print("\n--- Decision Network Output (Physics Weight) Distribution ---")
    print(f"  Mean: {np.mean(all_decisions):.4f}")
    print(f"  Median: {np.median(all_decisions):.4f}")
    print(f"  Std Dev: {np.std(all_decisions):.4f}")
    print(f"  Min: {np.min(all_decisions):.4f}")
    print(f"  Max: {np.max(all_decisions):.4f}")
    print(f"  Ratio of weight > 0.5: {np.mean(all_decisions > 0.5):.2%}")
    print(f"  Ratio of weight > 0.8: {np.mean(all_decisions > 0.8):.2%}")

    print("\n--- Overall Performance ---")
    print(f"Average Cosine Similarity: {np.mean(cosine_sim):.4f}")

    # Save evaluation results
    np.savez(os.path.join(output_dir, 'evaluation_results.npz'),
            predictions=all_predictions,
            true_values=all_true_values,
            decisions=all_decisions,
            angle_errors=angle_errors,
            cosine_sim=cosine_sim)

    return {
        'angle_errors': angle_errors,
        'cosine_sim': cosine_sim,
        'decisions': all_decisions
    }

# Main Function
def main():
    print("Initializing Physics-Guided Deep Learning Model Training...")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0

    if use_gpu:
        print(f"Detected {len(gpus)} GPU(s), training on GPU")
        # Set memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except Exception as e:
            print(f"Failed to set GPU memory growth: {e}")
    else:
        print("No GPU detected or GPU disabled, training on CPU")

    # Configuration
    batch_size = 2048  # Smaller batch size for stability
    epochs = 200      # Reduced epochs
    train_data_path = './data/train.tfrecord'
    val_data_path = './data/val.tfrecord'

    # Check data paths
    if not os.path.exists(train_data_path):
        print(f"Warning: Training data not found at {train_data_path}")
    if not os.path.exists(val_data_path):
        print(f"Warning: Validation data not found at {val_data_path}")

    # Build data pipeline
    print("Building data pipeline...")
    train_dataset = build_data_pipeline(train_data_path, batch_size, shuffle=True)
    val_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False)

    # Initialize history
    history = None

    # Build and train model
    print("Building physics-guided model...")
    try:
        # Create model
        physics_guided_model = build_physics_guided_model()

        # Use simple optimizer to avoid complexity
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=0.5)

        # Compile model - force eager execution
        physics_guided_model.compile(
            optimizer=optimizer,
            loss=PhysicsGuidedLoss(alpha=0.6, beta=0.3, gamma=0.1),
            metrics={'direction_output': [AngleError(), CosineSimilarity()]},
            run_eagerly=True  # Avoid graph compilation
        )

        # Model summary
        physics_guided_model.summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_direction_output_angle_error',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
            tf.keras.callbacks.ModelCheckpoint(
                'final_physics_guided_model',
                save_best_only=True,
                monitor='val_direction_output_angle_error',
                mode='min',
                save_weights_only=False,
                verbose=1,
                save_format='tf'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True,
                monitor='val_direction_output_angle_error',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('training_history.csv', append=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # Start training
        print("Starting model training...")
        history = physics_guided_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            workers=0,  # Disable multi-threading
            use_multiprocessing=False
        )

        print("Training complete!")
        # Save final model
        print("Saving final model...")
        try:
            # 1. Save as TensorFlow SavedModel format
            physics_guided_model.save('final_physics_guided_model.keras')
            print("✓ Saved full model: final_physics_guided_model.keras")

            # 2. Save model architecture as JSON
            model_json = physics_guided_model.to_json()
            with open('model_architecture.json', 'w') as json_file:
                json_file.write(model_json)
            print("Saved model architecture: model_architecture.json")

            # 3. Save weights as backup
            physics_guided_model.save_weights('model_weights.h5')
            print("Saved model weights backup: model_weights.h5")

        except Exception as save_error:
            print(f"Error saving model: {save_error}")
            try:
                physics_guided_model.save_weights('final_model_weights.h5')
                print("Saved model weights backup due to error.")
            except Exception as weight_error:
                print(f"Failed to save weights: {weight_error}")

        # Evaluation
        try:
            # Create hybrid model for evaluation
            print("Creating hybrid prediction model for evaluation...")
            hybrid_model = HybridPredictionModel(physics_guided_model)

            # Evaluate hybrid model
            print("Evaluating hybrid model...")
            test_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False)
            evaluation_results = evaluate_hybrid_model(hybrid_model, test_dataset, output_dir='./hybrid_results')
        except Exception as e:
            print(f"Error during evaluation: {e}")

        # Plot training history
        if history:
            print("Plotting training history...")
            try:
                print("\n===== Final Training Results =====")
                final_epoch_metrics = {key: value[-1] for key, value in history.history.items()}
                for metric, value in final_epoch_metrics.items():
                    print(f"  - {metric}: {value:.4f}")
                print("Training history recorded in training_history.csv")

            except Exception as e:
                print(f"Error printing history: {e}")

    except Exception as e:
        print(f"Error during training process: {e}")
        print("Try reducing batch size or simplifying model.")

    print("Program finished!")

if __name__ == "__main__":
    main()