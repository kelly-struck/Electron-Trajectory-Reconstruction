# -*- coding: utf-8 -*-
# Physics-Guided Deep Learning Method: Integrating Traditional Physics with Deep Learning
# For particle trajectory reconstruction

# Add these environment variable settings at the top of the file, must be before importing TF

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

    # 2. Fix libdevice issue - check and set CUDA path
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.7",
        "/usr/local/cuda-12.4",
    ]

    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            nvvm_path = os.path.join(cuda_path, "nvvm/libdevice")
            if os.path.exists(nvvm_path):
                os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_path}"
                print(f"Found CUDA directory: {cuda_path}")
                break

    # 3. Disable GPU acceleration
    # Uncomment next line to use CPU mode entirely
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 4. Set TF performance optimizations
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 5. Pre-select CPU or GPU
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
import argparse
# Now import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
import os

# Physical constants
BGOL = 7
BGOLB = 22

# ====== Adjustable switches and loss hyperparameters (for A/B testing) ======
# Input feature channel switches: whether to use per-event normalized channel and log1p channel
USE_PER_EVENT_NORM = True
USE_LOG1P = True

# Point regression standardization parameters (global mean/std, can be replaced with statistical results later)
PRI_POINT_MEAN = [3.7943913814728782, 2.065510447879724, -307.28289506556894]
PRI_POINT_STD  = [950.2981082528605, 930.3335357602713, 216.40446733015364]

# Point regression Huber loss hyperparameter (stricter but stable)
POINT_HUBER_DELTA = 2.0

# Direction loss weights (Cosine dominant + Light MSE)
DIR_ALPHA = 0.10  # MSE weight
DIR_BETA  = 0.90  # Cosine term weight
DECISION_GAMMA = 0.0  # Decision branch disabled
POINT_DELTA = 0.0      # Stop point head influence on backbone initially to stabilize direction convergence

# Direction "flip" penalty (based on z-component sign consistency)
DIR_FLIP_PENALTY = 0.05

# Sample weight for 60-70 degree samples
ENABLE_ANGLE_WEIGHT = False
FOCUS_THETA_LO, FOCUS_THETA_HI = 60.0, 70.0
ANGLE_FOCUS_ALPHA = 1.0  # Weight = 1 + alpha (in focus bucket)

# Point target: whether to change supervision to "intersection with z=Z0 plane"
USE_POINT_PLANE_TARGET = True
POINT_PLANE_Z = 0.0  # Can be modified according to actual geometry
POINT_STANDARDIZE = True   # Enable standardization to avoid point loss explosion due to large scale
POINT_STD_BOUND = 4.0      # Predicted output is limited by tanh in standardized space (approx +/- 4 sigma)

# Set random seed to ensure reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Apply random seed
set_seeds(42)

# Increase log level to reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide non-error messages

# Optimize XLA config
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Disable XLA compilation for determinism
tf.config.optimizer.set_jit(False)


# Environment configuration: force disable XLA
def set_environment_for_tf210():
    """Set environment parameters suitable for TensorFlow 2.10, completely disable XLA and CUDA optimization"""
    # 1. Completely disable XLA
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_XLA'] = '1'

    # 2. Disable JIT compilation and GPU optimization
    os.environ['TF_DISABLE_JIT'] = '1'
    os.environ['TF_CUDA_ENABLE_TENSOR_CORE_MATH'] = '0'
    tf.config.optimizer.set_jit(False)

    # 3. Configure GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected {len(gpus)} GPUs, configured for on-demand memory allocation")
            # Set visible devices
            tf.config.set_visible_devices(gpus[0:1], 'GPU')  # Only use the first GPU
            print("Restricted to use only one GPU")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    # 4. Set general environment variables
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 5. Configure precision and parallelism
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print("TensorFlow environment configuration completed, XLA, JIT and CUDA optimizations disabled")

# Physics feature extraction function
def extract_physics_features(barE_data):
    """
    Extract physics features from energy data
    - barE_data: Tensor of shape [batch_size, 14, 22]
    - Returns: Tensor of shape [batch_size, 21+3], containing features processed by traditional physics
    """
    # Convert TF tensor to numpy for processing (conceptual, actually using TF ops here)
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

        # Geometric position calculation consistent with ReConBGODir2.py
        # Note: In actual application, alignment parameter reading and application are needed
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

        # Try trajectory reconstruction (Simplified placeholder logic in pure TF)
        ln_x = tf.fill([BGOL], BGOLB)
        ln_y = tf.fill([BGOL], BGOLB)

        # Extract energy distribution features for each layer
        layer_features = tf.zeros((21,), dtype=tf.float32)
        for layer in range(7):
            # Calculate total energy of each layer
            layer_energy = tf.reduce_sum(barE[2*layer]) + tf.reduce_sum(barE[2*layer+1])

            # Calculate max energy and its position for each layer
            max_energy_x = tf.reduce_max(barE[2*layer])
            max_energy_y = tf.reduce_max(barE[2*layer+1])
            max_pos_x = tf.cast(tf.argmax(barE[2*layer]), tf.float32)
            max_pos_y = tf.cast(tf.argmax(barE[2*layer+1]), tf.float32)

            # Store key features of each layer
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

# TensorFlow functional version of physics feature extractor
@tf.function
def physics_features_extractor(barE_data):
    """TensorFlow functional physics feature extractor, used during training"""
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
        'PriDirec_2': tf.io.FixedLenFeature([], tf.float32),
        # Add PriPoint
        'PriPoint_0': tf.io.FixedLenFeature([], tf.float32),
        'PriPoint_1': tf.io.FixedLenFeature([], tf.float32),
        'PriPoint_2': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Enhanced data preprocessing: per-event norm + log1p two channels (switchable)
    barE = tf.stack([parsed[f'barE_{i}'] for i in range(14)], axis=0)  # (14,22)
    # Keep original data for physics feature extraction / visualization
    barE_original = barE

    # Ensure non-negative (energy deposition)
    barE_pos = tf.nn.relu(barE)

    # per-event norm channel
    total_energy = tf.reduce_sum(barE_pos) + 1e-6
    per_event_norm = barE_pos / total_energy  # (14,22)

    # log1p channel (log of energy, preserving dynamic range)
    log1p_map = tf.math.log1p(barE_pos)  # (14,22)

    # Organize two channels based on switches, keeping (14,22,2) layout
    if USE_PER_EVENT_NORM and USE_LOG1P:
        barE_2ch = tf.stack([per_event_norm, log1p_map], axis=-1)
    elif USE_PER_EVENT_NORM and (not USE_LOG1P):
        barE_2ch = tf.stack([per_event_norm, per_event_norm], axis=-1)
    elif (not USE_PER_EVENT_NORM) and USE_LOG1P:
        barE_2ch = tf.stack([log1p_map, log1p_map], axis=-1)
    else:
        # Enable at least one channel; fallback to double copy of per_event_norm
        barE_2ch = tf.stack([per_event_norm, per_event_norm], axis=-1)

    # Physics feature extraction (placeholder/simplified)
    physics_features = tf.zeros((24,), dtype=tf.float32)  # Placeholder

    # Targets: Direction PriDirec_*
    dir_targets = tf.stack([
        parsed['PriDirec_0'],
        parsed['PriDirec_1'],
        parsed['PriDirec_2']
    ])
    dir_targets_norm = tf.norm(dir_targets, axis=0) + 1e-10  # Avoid zero division
    dir_targets = dir_targets / dir_targets_norm

    # New targets: Point -- not using intersection for now (reduce overfitting risk)
    point_targets_raw = tf.stack([
        parsed['PriPoint_0'],
        parsed['PriPoint_1'],
        parsed['PriPoint_2']
    ])
    # If plane intersection needs to be re-enabled later, restore logic and set USE_POINT_PLANE_TARGET=True

    if POINT_STANDARDIZE:
        # Compatibility situation: if PRI_POINT_MEAN/STD is overwritten by external JSON as Python list, convert to Tensor
        pp_mean = tf.convert_to_tensor(PRI_POINT_MEAN, dtype=tf.float32)
        pp_std  = tf.convert_to_tensor(PRI_POINT_STD, dtype=tf.float32)
        point_targets = (point_targets_raw - pp_mean) / (pp_std + 1e-6)
    else:
        point_targets = point_targets_raw

    # Dummy label for decision branch (not supervised, placeholder only)
    decision_dummy = tf.reshape(tf.constant(0.5, dtype=tf.float32), [1])

    # Flatten 2-channel input, keeping memory layout consistent with BGOInput((14,22,2))
    barE_flat = tf.reshape(barE_2ch, [-1])

    # Changed to targets dict, keys consistent with model output
    targets = {
        'direction_output': dir_targets,     # shape: (3,)
        'decision_output': decision_dummy,   # shape: (1,)
        'point_output': point_targets        # shape: (3,)
    }

    return {
        'barE_input': barE_flat,
        'barE_original': barE_original,  # For physics feature extraction
        'physics_features': physics_features  # Placeholder, processed later
    }, targets


def process_batch_for_physics(inputs, targets):
    """
    Batch process data to extract physics features
    Note: This step is used during training/val to provide richer physics_features to DL branch.
         To avoid passing extra keys to DL model, barE_original is not returned.
         Evaluation phase will set include_physics=False to retain barE_original for hybrid model usage.
    """
    barE_original = inputs['barE_original']
    physics_features = extract_physics_features(barE_original)

    # Update input dict (exclude barE_original to avoid Keras extra key issues)
    updated_inputs = {
        'barE_input': inputs['barE_input'],
        'physics_features': physics_features
    }

    return updated_inputs, targets

def build_data_pipeline(file_pattern, batch_size=2048, shuffle=True, include_physics=True):
    """Build data pipeline, load and preprocess data from TFRecord files"""
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

    # Shuffle with larger buffer_size and seed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    # Batching
    dataset = dataset.batch(batch_size)

    # Add physics feature extraction processing
    if include_physics:
        dataset = dataset.map(process_batch_for_physics,
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Angle binning weighting (returns (x, y, sample_weight) triplet)
    if ENABLE_ANGLE_WEIGHT:
        def add_angle_weights(inputs, targets):
            # Extract direction truth and calculate incident angle theta (with Z axis)
            if isinstance(targets, dict):
                dir_true = targets['direction_output']
                decision_true = targets.get('decision_output', tf.zeros((tf.shape(dir_true)[0], 1), dtype=tf.float32))
            else:
                dir_true = targets
                decision_true = tf.zeros((tf.shape(dir_true)[0], 1), dtype=tf.float32)

            # dir_true shape: (batch, 3)
            z = tf.clip_by_value(dir_true[:, 2], -1.0, 1.0)
            theta = tf.acos(z) * 180.0 / np.pi
            focus_mask = tf.cast((theta >= FOCUS_THETA_LO) & (theta < FOCUS_THETA_HI), tf.float32)
            w = 1.0 + ANGLE_FOCUS_ALPHA * focus_mask  # (batch,)

            # Provide dict weights for multi-output model
            sample_weights = {
                'direction_output': w,
                'point_output': w,
                'decision_output': tf.ones_like(tf.reshape(decision_true, [-1]))  # Decouple from direction/point weights
            }
            return inputs, targets, sample_weights

        dataset = dataset.map(add_angle_weights, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# Feature Engineering Layer
class FeatureEngineering(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch, 14, 22, 1)
        
        # 1. Energy normalization (by event total energy)
        total_energy = tf.reduce_sum(inputs, axis=[1, 2], keepdims=True)
        total_energy = tf.maximum(total_energy, 1e-6) # Avoid division by zero
        energy_normalized = inputs / total_energy

        # 2. Layer-wise energy fraction
        layer_energy = tf.reduce_sum(inputs, axis=2, keepdims=True) # (batch, 14, 1, 1)
        layer_fraction = layer_energy / total_energy
        layer_fraction_map = tf.tile(layer_fraction, [1, 1, 22, 1]) # Expand to (batch, 14, 22, 1)

        # 3. Energy Centroid (Shower Centroid)
        coords_x = tf.range(22, dtype=tf.float32)
        coords_y = tf.range(14, dtype=tf.float32)
        
        centroid_x = tf.reduce_sum(inputs * coords_x, axis=2) / tf.reduce_sum(inputs, axis=2)
        centroid_y = tf.reduce_sum(inputs * coords_y, axis=1) / tf.reduce_sum(inputs, axis=1)

        centroid_x = tf.maximum(centroid_x, 0.0)
        centroid_y = tf.maximum(centroid_y, 0.0)

        centroid_x_map = tf.tile(tf.reshape(centroid_x, [-1, 14, 1, 1]), [1, 1, 22, 1])
        centroid_y_map = tf.tile(tf.reshape(centroid_y, [-1, 1, 22, 1]), [1, 14, 1, 1])

        # 4. Find max energy deposition layer (Shower Maximum Layer)
        max_layer_idx = tf.argmax(tf.squeeze(layer_energy, axis=[2,3]), axis=1)
        max_layer_map = tf.one_hot(max_layer_idx, depth=14, dtype=tf.float32)
        max_layer_map = tf.reshape(max_layer_map, [-1, 14, 1, 1])
        max_layer_map = tf.tile(max_layer_map, [1, 1, 22, 1])

        # 5. Combine all new features
        # Original energy, Normalized energy, Layer energy fraction, Centroid X, Centroid Y, Max layer
        combined_features = layers.Concatenate(axis=-1)([
            inputs, 
            energy_normalized, 
            layer_fraction_map,
            centroid_x_map,
            centroid_y_map,
            max_layer_map
        ])
        
        return combined_features

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

# Enhance residual block numerical stability and ensure reproducibility
def residual_block(x, filters, kernel_size=(3,3), expansion=2.0, seed_offset=0, use_attention=True, dropout_rate=0.1):
    shortcut = x
    in_channels = x.shape[-1]

    # First Conv Layer
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.Activation('swish')(x)

    # Second Conv Layer
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=43+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # Attention Mechanism
    if use_attention:
        x = CBAM(filters)(x)

    # If input and output channels differ, use 1x1 conv to adjust
    if in_channels != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=44+seed_offset))(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)

    # Merge residual connection
    x = layers.add([x, shortcut])
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate, seed=45+seed_offset)(x)

    return x


# Physics-Guided Multi-Modal Model
def build_physics_guided_model():
    # BGO Energy Input
    barE_input = tf.keras.Input(shape=(14*22*2,), name='barE_input')

    # Reshape to image format
    x = BGOInput()(barE_input)

    # Extract local features
    x = layers.Conv2D(128, (3, 3), padding='same', activation='swish',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # Add an extra initial conv layer
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
    x_global = layers.Dense(512, activation='swish')(x_combined)  # Increase to 512
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.3)(x_global)

    x_global = layers.Dense(512, activation='swish')(x_global)  # Add extra layer
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.2)(x_global)
    
    # Enhanced Physics Feature Processing Branch
    physics_input = tf.keras.Input(shape=(24,), name='physics_features')

    p = layers.Dense(128, activation='swish')(physics_input)  # Increase to 128
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # Increase to 256
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # Add extra layer
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.2)(p)

    # Enhanced Fusion Network
    combined = layers.Concatenate()([x_global, p])

    # Deeper Fusion Processing Network
    z = layers.Dense(1024, activation='swish')(combined)       # Increase to 1024
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Dense(512, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(256, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)

    # Three outputs of the model:
    # 1. Direction prediction (Linear then normalized)
    direction_raw = layers.Dense(3, name='direction_output_raw')(z)
    direction_output = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=-1), name='direction_output')(direction_raw)

    # 2. Decision network output - used to judge whether to use physics method
    decision_output = layers.Dense(1, activation='sigmoid', name='decision_output')(z)

    # 3. Initial point prediction - Regression back to direct prediction mode
    point_output = layers.Dense(3, name='point_output')(z)

    # Create model
    model = tf.keras.Model(
        inputs=[barE_input, physics_input],
        outputs=[direction_output, decision_output, point_output]
    )

    return model

# Custom Loss Functions - Extended to be compatible with point prediction
class DirectionCosLoss(tf.keras.losses.Loss):
    """Direction loss dominated by cosine similarity, with flip penalty and optional angle weighting."""
    def __init__(self, alpha=0.05, beta=0.95, flip_penalty=0.2, name='direction_cos_loss'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.alpha = alpha
        self.beta = beta
        self.flip_penalty = flip_penalty

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
        loss_cos = 1.0 - cos_sim
        loss_mse = tf.reduce_mean(tf.square(y_true_norm - y_pred_norm), axis=-1)
        flip_penalty = tf.nn.relu(- y_true_norm[:, 2] * y_pred_norm[:, 2])

        # Weighting is handled by sample_weight in data pipeline, not repeated here inside loss
        per_sample = self.beta * loss_cos + self.alpha * loss_mse + self.flip_penalty * flip_penalty
        return per_sample


class DecisionCenterLoss(tf.keras.losses.Loss):
    """Gentle regularization pulling decision output towards 0.5."""
    def __init__(self, name='decision_center_loss'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # Ignore y_true, pull directly towards 0.5
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        target = tf.zeros_like(y_pred) + 0.5
        return tf.square(y_pred - target)


class PointHuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta=POINT_HUBER_DELTA, name='point_huber_loss'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = self.huber(y_true, y_pred)  # (batch,)
        return loss

# Angle Error Metric
class AngleError(tf.keras.metrics.Metric):
    def __init__(self, name='angle_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.angle_error_sum = self.add_weight(name='angle_error_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # For output-bound metrics, this receives corresponding output tensor
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
        # For output-bound metrics, this receives corresponding output tensor
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

# Custom learning rate scheduler
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=1e-5, max_lr=5e-4, step_size=2000):
        super().__init__()
        # Ensure all inputs are float32
        self.base_lr = tf.cast(base_lr, dtype=tf.float32)
        self.max_lr = tf.cast(max_lr, dtype=tf.float32)
        self.step_size = tf.cast(step_size, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
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

# Gradient debugger callback - Fix TensorFlow 2.10 compatibility issues
class GradientDebugger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            try:
                # Check NaN and Inf
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
                    if g is not None:
                        if np.isnan(np.sum(g.numpy())):
                            has_nan = True
                        if np.isinf(np.sum(g.numpy())):
                            has_inf = True

                if has_nan or has_inf:
                    print(f"\nBatch {batch}: Detected {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} gradients!")
            except Exception as e:
                # Print warning if gradient check fails
                print(f"Warning: Error during gradient check: {str(e)}")

# Batch size scheduler
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
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"\nGPU Memory: Used={gpu_info['current'] / 1e9:.2f}GB, Peak={gpu_info['peak'] / 1e9:.2f}GB")
            except:
                pass

# Learning Rate Logger
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_begin(self, epoch, logs=None):
        try:
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
                print("\nCannot directly access current learning rate")
        except Exception as e:
            print(f"\nError getting learning rate: {e}")


# Learning Rate Warmup Wrapper
class WarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_schedule, warmup_steps=3000, init_lr=2e-5):
        super().__init__()
        self.base_schedule = base_schedule
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.init_lr = tf.cast(init_lr, tf.float32)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        base_lr = self.base_schedule(step_f)
        # Linear warmup: from init_lr -> base_lr
        warmup_ratio = tf.clip_by_value(step_f / self.warmup_steps, 0.0, 1.0)
        lr = self.init_lr + (base_lr - self.init_lr) * warmup_ratio
        return lr

    def get_config(self):
        return {"warmup_steps": float(self.warmup_steps), "init_lr": float(self.init_lr)}


# Point Loss Warmup Callback
class PointLossWarmup(tf.keras.callbacks.Callback):
    def __init__(self, total_target=0.01, warmup_epochs=3, ramp_epochs=5):
        super().__init__()
        self.total_target = total_target
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            w = 0.0
        else:
            # Linear ramp up
            progress = min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)
            w = self.total_target * progress
        # Update model loss_weights
        if hasattr(self.model, 'loss_weights') and isinstance(self.model.loss_weights, dict):
            self.model.loss_weights['point_output'] = w
        print(f"[PointLossWarmup] epoch={epoch} point_loss_weight={w:.5f}")

# Hybrid Model Evaluation and Visualization
def evaluate_hybrid_model(model, test_dataset, output_dir='./results_0616'):
    """Evaluate hybrid model and show results as text."""
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
        # Get direction ground truth
        if isinstance(targets, dict):
            dir_true = targets['direction_output']
        elif isinstance(targets, (list, tuple)):
            dir_true = targets[0]
        else:
            dir_true = targets

        # Convert to numpy for compatibility
        p = predictions.numpy() if hasattr(predictions, "numpy") else predictions
        d = decisions.numpy() if hasattr(decisions, "numpy") else decisions
        t = dir_true.numpy() if hasattr(dir_true, "numpy") else dir_true

        all_predictions.append(p)
        all_true_values.append(t)
        all_decisions.append(d)

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

# Evaluate using only the DL branch (no physics fusion)
def evaluate_dl_only_model(model, test_dataset, output_dir='./results_dl_only'):
    """Evaluate pure DL direction output, completely disabling physics fusion."""
    os.makedirs(output_dir, exist_ok=True)

    all_predictions = []
    all_true_values = []
    all_decisions = []  # Optional, record model's decision_output distribution

    for batch in test_dataset:
        # Compatible with (x,y) and (x,y,weights) formats
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, _weights = batch
        else:
            inputs, targets = batch
        # Forward pass directly to get three outputs
        outputs = model(inputs, training=False)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
            direction_pred, decision_pred, _point_pred = outputs[0], outputs[1], outputs[2]
        else:
            # Fallback: if only one output returned, treat as direction
            direction_pred = outputs
            decision_pred = tf.zeros((tf.shape(direction_pred)[0], 1), dtype=tf.float32)

        # Get direction true value
        if isinstance(targets, dict):
            dir_true = targets['direction_output']
        elif isinstance(targets, (list, tuple)):
            dir_true = targets[0]
        else:
            dir_true = targets

        p = direction_pred.numpy() if hasattr(direction_pred, 'numpy') else direction_pred
        t = dir_true.numpy() if hasattr(dir_true, 'numpy') else dir_true
        d = decision_pred.numpy() if hasattr(decision_pred, 'numpy') else decision_pred

        all_predictions.append(p)
        all_true_values.append(t)
        all_decisions.append(d)

    all_predictions = np.vstack(all_predictions)
    all_true_values = np.vstack(all_true_values)
    all_decisions = np.vstack(all_decisions)

    # Calculate angle error and cosine
    y_true_norm = all_true_values / (np.linalg.norm(all_true_values, axis=1, keepdims=True) + 1e-12)
    y_pred_norm = all_predictions / (np.linalg.norm(all_predictions, axis=1, keepdims=True) + 1e-12)
    dot_products = np.sum(y_true_norm * y_pred_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angle_errors = np.arccos(dot_products) * 180.0 / np.pi
    cosine_sim = dot_products

    print("\n===== Pure DL Model Evaluation Results (No Physics Fusion) =====")
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

    # Angle binning (by true incident angle vs Z axis)
    theta_true = np.degrees(np.arccos(np.clip(y_true_norm[:, 2], -1.0, 1.0)))
    bins = [(60, 70), (70, 80), (80, 90)]
    print("\n--- Binned Statistics (By True Incident Angle) ---")
    for lo, hi in bins:
        mask = (theta_true >= lo) & (theta_true < hi)
        if np.any(mask):
            ae = angle_errors[mask]
            print(f"  {lo:02d}–{hi:02d}°: N={ae.size}, Median={np.median(ae):.2f}°, P68={np.percentile(ae, 68):.2f}°, P95={np.percentile(ae, 95):.2f}°")
        else:
            print(f"  {lo:02d}–{hi:02d}°: N=0")

    print("\n--- Model decision_output Distribution (Reference only, not fused) ---")
    print(f"  Mean: {np.mean(all_decisions):.4f}")
    print(f"  Median: {np.median(all_decisions):.4f}")
    print(f"  Std Dev: {np.std(all_decisions):.4f}")
    print(f"  Min: {np.min(all_decisions):.4f}")
    print(f"  Max: {np.max(all_decisions):.4f}")

    np.savez(os.path.join(output_dir, 'evaluation_results_dl_only.npz'),
             predictions=all_predictions,
             true_values=all_true_values,
             predictions_unit=y_pred_norm,
             angle_errors=angle_errors,
             cosine_sim=cosine_sim,
             decisions=all_decisions)

    return {
        'angle_errors': angle_errors,
        'cosine_sim': cosine_sim,
        'decisions': all_decisions
    }

def main():
    print("Initializing Physics-Guided Deep Learning Model Training...")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0

    if use_gpu:
        print(f"Detected {len(gpus)} GPU(s), training on GPU")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except Exception as e:
            print(f"Failed to set GPU memory growth: {e}")
    else:
        print("No GPU detected or GPU disabled, training on CPU")

    # Configuration
    batch_size = 4096 
    epochs = 200      
    train_data_path = './data/train.tfrecord'
    val_data_path = './data/val.tfrecord'

    # Check data paths
    if not os.path.exists(train_data_path):
        print(f"Warning: Training data not found at {train_data_path}")
    if not os.path.exists(val_data_path):
        print(f"Warning: Validation data not found at {val_data_path}")

    # Build data pipeline
    print("Building data pipeline...")
    # Ensure paths exist or handle gracefully in a real scenario
    try:
        train_dataset = build_data_pipeline(train_data_path, batch_size, shuffle=True, include_physics=True)
        val_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False, include_physics=True)
    except Exception as e:
        print(f"Error building data pipeline: {e}")
        return

    history = None

    # Build and train model
    print("Building physics-guided model...")
    try:
        # Create model
        physics_guided_model = build_physics_guided_model()

        # Learning rate schedule + Warmup
        base_lr_schedule = CyclicalLearningRate(base_lr=1e-5, max_lr=8e-4, step_size=2500)
        lr_schedule = WarmupLR(base_schedule=base_lr_schedule, warmup_steps=3000, init_lr=2e-5)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0)

        # Compile model - force eager execution
        physics_guided_model.compile(
            optimizer=optimizer,
            loss={
                'direction_output': DirectionCosLoss(alpha=DIR_ALPHA, beta=DIR_BETA, flip_penalty=DIR_FLIP_PENALTY),
                'decision_output': DecisionCenterLoss(),
                'point_output': PointHuberLoss(delta=POINT_HUBER_DELTA)
            },
            loss_weights={
                'direction_output': 1.0,
                'decision_output': 0.0,  # Temporarily disable decision head loss
                'point_output': 0.01     # Enable point loss, weight 0.01
            },
            metrics={'direction_output': [AngleError(), CosineSimilarity()]},
            run_eagerly=True
        )

        physics_guided_model.summary()

        # Callbacks
        callbacks = [
            LearningRateLogger(),
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
            workers=0,  # Disable multi-threading for data loading
            use_multiprocessing=False
        )

        print("Training complete!")
        # Save final model
        print("Saving final model...")
        try:
            # 1. Save as TensorFlow SavedModel/keras format
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
                physics_guided_model.save_weights('final_physics_guided_model_weights.h5')
                print("Saved model weights as backup due to save error.")
            except Exception as weight_error:
                print(f"Failed to save weights: {weight_error}")

        # Evaluation
        try:
            # Evaluate: pure DL output
            print("Evaluating pure DL output (no physics fusion)...")
            test_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False, include_physics=True)
            evaluation_results = evaluate_dl_only_model(physics_guided_model, test_dataset, output_dir='./results_dl_only')
        except Exception as e:
            print(f"Error during evaluation: {e}")

        # Print history
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