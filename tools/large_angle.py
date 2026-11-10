# -*- coding: utf-8 -*-
# 物理引导的深度学习方法：融合传统物理方法与深度学习
# 用于粒子轨迹重建

# 在文件顶部添加这些环境变量设置，必须在导入TF之前

# 添加这个函数并确保它在导入TF之前被调用
def setup_tf_environment():
    """设置环境变量，必须在导入TensorFlow前调用"""
    import os
    import sys

    # 1. 彻底禁用XLA/JIT编译
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_XLA'] = '1'
    os.environ['TF_DISABLE_JIT'] = '1'
    os.environ['TF_CUDA_ENABLE_TENSOR_CORE_MATH'] = '0'

    # 2. 解决libdevice问题 - 检查并设置CUDA路径
    # 发布版：仅保留通用安装位置；如需自定义，请设置环境变量 CUDA_HOME
    cuda_paths = [
        os.environ.get('CUDA_HOME', ''),
        "/usr/local/cuda",
        "/usr/local/cuda-11.7",
        "/usr/local/cuda-12.4",
    ]

    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            nvvm_path = os.path.join(cuda_path, "nvvm/libdevice")
            if os.path.exists(nvvm_path):
                os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_path}"
                print(f"找到CUDA目录: {cuda_path}")
                break

    # 3. 禁用GPU加速
    # 如果要完全使用CPU模式，可以取消注释下一行
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 4. 设置TF性能优化
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 5. 提前选择使用CPU或GPU
    try:
        use_cpu = "--cpu" in sys.argv or "--CPU" in sys.argv
        if use_cpu:
            print("强制使用CPU模式")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    except:
        pass

    print("TensorFlow环境变量配置完成")


# 在导入TensorFlow前调用此函数
setup_tf_environment()

# 其他导入
import random
import numpy as np
# 现在导入TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
import os

# 添加路径以导入传统物理方法模块
from LinearFit import linear_fit_weighted
from LeiTrackReBuild import layer_centroid, bgo_linear_track_cls, stk_track_bgo_sigma

# 定义常量
BGOL = 7
BGOLB = 22

# ====== 可调节开关与损失超参（便于A/B） ======
# 输入特征通道开关：是否使用按事件归一化通道与log1p通道
USE_PER_EVENT_NORM = True
USE_LOG1P = True
USE_ACTIVATION_MASK = True
USE_SATURATION_MASK = True
USE_SPLIT_ENERGY = True   # 新增：截断通道
N_INPUT_CHANNELS = 5      # 改变为5输入通道: norm, log1p, activation_mask, saturation_mask, low_energy_clip

# 点回归标准化参数（全局均值/方差，可后续用统计结果替换）
PRI_POINT_MEAN = [3.7943913814728782, 2.065510447879724, -307.28289506556894]
PRI_POINT_STD  = [950.2981082528605, 930.3335357602713, 216.40446733015364]

# 点回归Huber损失超参（更严格但稳定）
POINT_HUBER_DELTA = 0.5   # 修改为 0.5 对应±0.5σ内二次

# 方向损失权重（余弦主导 + 轻MSE）
DIR_ALPHA = 0.10  # MSE权重
DIR_BETA  = 0.90  # 余弦项权重
POINT_DELTA = 0.0      # 先完全停止点头对主干的影响，稳定方向收敛

# 方向“翻转”额外惩罚项（基于 z 分量符号一致性）
DIR_FLIP_PENALTY = 0.05

# 针对 60–70 度样本的加权系数（sample_weight）
ENABLE_ANGLE_WEIGHT = False
FOCUS_THETA_LO, FOCUS_THETA_HI = 60.0, 70.0
ANGLE_FOCUS_ALPHA = 1.0  # 权重 = 1 + alpha （位于焦点桶）

# 点目标：是否将监督改为“与 z=Z0 平面的交点”
USE_POINT_PLANE_TARGET = True
POINT_PLANE_Z = 0.0  # 可按实际几何修改
POINT_STANDARDIZE = True   # 启用标准化，避免尺度过大导致点损失失控
POINT_STD_BOUND = 4.0      # 预测输出在标准化空间内用 tanh 限幅（≈±4σ）

# 设置随机种子以确保可重现性
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 应用随机种子
set_seeds(42)

# 提高日志级别以减少警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隐藏非错误信息

# 优化XLA配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # 尽可能使用XLA JIT编译

# 禁用XLA编译以增加确定性
tf.config.optimizer.set_jit(False)


# 环境配置部分需要彻底修改，强制禁用XLA
def set_environment_for_tf210():
    """设置适合TensorFlow 2.10的环境参数，完全禁用XLA和CUDA优化"""
    # 1. 完全禁用XLA - 彻底避免需要libdevice的操作
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_XLA'] = '1'  # 明确禁用XLA

    # 2. 禁用JIT编译和GPU优化
    os.environ['TF_DISABLE_JIT'] = '1'  # 禁用JIT编译
    os.environ['TF_CUDA_ENABLE_TENSOR_CORE_MATH'] = '0'  # 禁用TensorCore加速
    tf.config.optimizer.set_jit(False)  # 在TensorFlow配置中明确禁用JIT

    # 3. 配置GPU内存
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 {len(gpus)} 个 GPU，已配置为按需分配内存")
            # 设置可见设备
            tf.config.set_visible_devices(gpus[0:1], 'GPU')  # 仅使用第一个GPU
            print("已限制仅使用一个GPU")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")

    # 4. 设置常规环境变量
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志级别，隐藏不重要的警告

    # 5. 配置精度和并行度
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print("TensorFlow环境配置完成，已禁用XLA、JIT和CUDA优化")

# 物理特征提取函数
def extract_physics_features(barE_data):
    """
    从能量数据中提取物理特征
    - barE_data: 形状为 [batch_size, 14, 22] 的张量
    - 返回: 形状为 [batch_size, 21+3] 的张量，包含传统物理处理的特征
    """
    # 将TF张量转换为numpy以便处理
    batch_size = tf.shape(barE_data)[0]
    features = tf.TensorArray(tf.float32, size=batch_size)

    def process_single_sample(i, barE):
        # 存储物理特征
        physics_feats = tf.zeros((24,), dtype=tf.float32)

        # 从barE中重建BGO探测器的信息
        BHXXL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHXZL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHXEL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYYL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYZL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)
        BHYEL = tf.zeros((BGOL, BGOLB), dtype=tf.float32)

        # 使用与ReConBGODir2.py一致的几何位置计算
        # 注：实际应用中需要添加对齐参数的读取和应用
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

        # 计算总能量
        BgoEng = tf.reduce_sum(barE)

        # 尝试进行轨迹重建
        ln_x = tf.fill([BGOL], BGOLB)
        ln_y = tf.fill([BGOL], BGOLB)

        # 提取各层能量分布特征
        layer_features = tf.zeros((21,), dtype=tf.float32)
        for layer in range(7):
            # 计算每层的总能量
            layer_energy = tf.reduce_sum(barE[2*layer]) + tf.reduce_sum(barE[2*layer+1])

            # 计算每层的最大能量及其位置
            max_energy_x = tf.reduce_max(barE[2*layer])
            max_energy_y = tf.reduce_max(barE[2*layer+1])
            max_pos_x = tf.cast(tf.argmax(barE[2*layer]), tf.float32)
            max_pos_y = tf.cast(tf.argmax(barE[2*layer+1]), tf.float32)

            # 存储每层的关键特征
            idx = layer * 3
            layer_features = tf.tensor_scatter_nd_update(
                layer_features,
                [[idx], [idx+1], [idx+2]],
                [layer_energy, (max_energy_x + max_energy_y)/2, (max_pos_x + max_pos_y)/2]
            )

        # 计算能量加权的空间分布特征
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0
        total_e = tf.reduce_sum(barE) + 1e-10

        # X方向加权平均
        for i in range(BGOL):
            for j in range(BGOLB):
                pos_x = -288.75 + j * 27.5
                pos_z = 87.50 + i * 58.0
                e = barE[2*i+1, j]
                weighted_x += pos_x * e / total_e
                weighted_z += pos_z * e / total_e

        # Y方向加权平均 
        for i in range(BGOL):
            for j in range(BGOLB):
                pos_y = -288.75 + j * 27.5
                pos_z = 58.50 + i * 58.0
                e = barE[2*i, j]
                weighted_y += pos_y * e / total_e
                weighted_z = (weighted_z + pos_z * e / total_e) / 2  # 平均Z位置

        # 更新物理特征
        physics_feats = tf.concat([layer_features, [weighted_x, weighted_y, weighted_z]], axis=0)
        return physics_feats

    # 对批次中的每个样本调用物理特征提取
    for i in range(batch_size):
        features = features.write(i, process_single_sample(i, barE_data[i]))

    return features.stack()

# TensorFlow函数化版本的物理特征提取器
@tf.function
def physics_features_extractor(barE_data):
    """TensorFlow函数化的物理特征提取器，用于训练期间"""
    return tf.py_function(
        func=lambda x: extract_physics_features(x),
        inp=[barE_data],
        Tout=tf.float32
    )

# 物理方法预测的Python函数封装
# 注意：此函数将在图模式下通过 tf.py_function 调用，因此其内部应尽量自包含
def _calculate_physics_pred(barE_original):
    """
    一个独立的、用于tf.py_function的物理方法预测函数。
    输入: barE_original (Tensor, shape=(14, 22))
    输出: 物理方法预测的方向向量 (Numpy Array, shape=(3,))
    """
    # 将Tensor转换为Numpy
    bgo_energy = barE_original.numpy()

    # 物理方法重建所需的常量和数组初始化
    # 这些值需要与 HybridPredictionModel._apply_physics_method 中的定义保持一致
    BGOL, BGOLB = 14, 22
    BHXXL, BHXZL, BHXEL = np.zeros((BGOL, BGOLB)), np.zeros((BGOL, BGOLB)), np.zeros((BGOL, BGOLB))
    BHYYL, BHYZL, BHYEL = np.zeros((BGOL, BGOLB)), np.zeros((BGOL, BGOLB)), np.zeros((BGOL, BGOLB))
    
    # 几何位置计算 (与ReConBGODir2.py一致)
    for i in range(BGOL):
        for j in range(BGOLB):
            BHXXL[i][j] = -215.55 + 19.6 * j
            BHXZL[i][j] = 300.0 - 25.0 * i
            BHYYL[i][j] = -215.55 + 19.6 * j
            BHYZL[i][j] = 300.0 - 25.0 * i
    
    BHXEL = bgo_energy
    BHYEL = np.transpose(bgo_energy)

    BgoEng = np.sum(bgo_energy)

    # 调用物理方法重建轨迹
    ln_x = [22] * 7
    ln_y = [22] * 7

    try:
        # 假设 ReConBGODir 可以在此作用域内被访问
        # 如果 ReConBGODir 未定义，需要确保它被正确导入或定义
        from ReConBGODir2 import ReConBGODir # 尝试导入
        rec_x, rec_y, rec_z, sx, sy, sz, chi2, itr = ReConBGODir(
            BgoEng, ln_x, ln_y, BHXXL, BHXZL, BHXEL, BHYYL, BHYZL, BHYEL
        )
        # 返回归一化的方向向量
        direction = np.array([sx, sy, sz], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm
        else:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    except Exception as e:
        # 在物理方法失败时返回一个零向量
        # print(f"Warning: _calculate_physics_pred failed with {e}") # 在生产中可关闭
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

# 数据管道
def parse_tfrecord(example_proto):
    feature_description = {
        **{f'barE_{i}': tf.io.FixedLenFeature([22], tf.float32) for i in range(14)},
        'PriDirec_0': tf.io.FixedLenFeature([], tf.float32),
        'PriDirec_1': tf.io.FixedLenFeature([], tf.float32),
        'PriDirec_2': tf.io.FixedLenFeature([], tf.float32),
        # 新增 PriPoint
        'PriPoint_0': tf.io.FixedLenFeature([], tf.float32),
        'PriPoint_1': tf.io.FixedLenFeature([], tf.float32),
        'PriPoint_2': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # 增强数据预处理：按事件归一化 + log1p + 激活掩码 等 5通道
    barE = tf.stack([parsed[f'barE_{i}'] for i in range(14)], axis=0)  # (14,22)
    # 保留原始数据用于后续物理特征/可视化
    barE_original = barE

    # 保证非负（能量沉积）
    barE_pos = tf.nn.relu(barE)

    # per-event 归一化通道
    total_energy = tf.reduce_sum(barE_pos) + 1e-6
    per_event_norm = barE_pos / total_energy  # (14,22)

    # log1p 通道（对能量取对数，保留动态范围）
    log1p_map = tf.math.log1p(barE_pos)  # (14,22)

    # 激活掩码通道
    activation_mask = tf.cast(barE_pos > 0, tf.float32)

    # 饱和掩码通道 (假设 BGO 单根晶体量程上限约 100 GeV = 100000 MeV)
    # 对于 1TeV-10TeV 经常出现饱和，这里设定阈值为 100000
    saturation_threshold = 100000.0
    saturation_mask = tf.cast(barE_pos > saturation_threshold, tf.float32)

    # 低能截断通道 (专门看外围拓扑，削平高能核心，比如限制在 500 MeV 以下)
    low_energy_clip = tf.clip_by_value(barE_pos, 0.0, 500.0) / 500.0

    channel_list = []
    channel_list.append(per_event_norm if USE_PER_EVENT_NORM else tf.zeros_like(barE_pos))
    channel_list.append(log1p_map if USE_LOG1P else tf.zeros_like(barE_pos))
    channel_list.append(activation_mask if USE_ACTIVATION_MASK else tf.zeros_like(barE_pos))
    channel_list.append(saturation_mask if USE_SATURATION_MASK else tf.zeros_like(barE_pos))
    channel_list.append(low_energy_clip if USE_SPLIT_ENERGY else tf.zeros_like(barE_pos))
    
    barE_2ch = tf.stack(channel_list, axis=-1)  # (14, 22, 5) 注意变量名保持barE_2ch以兼容后面

    # 物理特征计算
    _barE = barE_original  # shape (14, 22)
    _total_e = tf.reduce_sum(_barE) + 1e-10
    _y_rows = tf.gather(_barE, [0, 2, 4, 6, 8, 10, 12])   # (7, 22)
    _x_rows = tf.gather(_barE, [1, 3, 5, 7, 9, 11, 13])   # (7, 22)
    _y_layer_e = tf.reduce_sum(_y_rows, axis=1) / _total_e  # 7维
    _x_layer_e = tf.reduce_sum(_x_rows, axis=1) / _total_e  # 7维
    _bar_pos = tf.cast(tf.range(22), tf.float32)
    _y_denom = tf.reduce_sum(_y_rows, axis=1) + 1e-10
    _y_centroids = tf.reduce_sum(_y_rows * _bar_pos[tf.newaxis, :], axis=1) / _y_denom
    _y_centroids = _y_centroids / 21.0 * 2.0 - 1.0   # 归一化至 [-1, 1]
    _log_e = tf.math.log1p(_total_e) / 15.0          
    _active_frac = tf.reduce_sum(tf.cast(_barE > 0, tf.float32)) / (14.0 * 22.0)
    _shower_max = tf.cast(tf.argmax(_y_layer_e + _x_layer_e), tf.float32) / 6.0
    physics_features = tf.concat([
        _y_layer_e,          # 7维
        _x_layer_e,          # 7维
        _y_centroids,        # 7维
        [_log_e],            # 1维
        [_active_frac],      # 1维
        [_shower_max],       # 1维
    ], axis=0)               # 共24维

    # 目标值: 方向 PriDirec_*
    dir_targets = tf.stack([
        parsed['PriDirec_0'],
        parsed['PriDirec_1'],
        parsed['PriDirec_2']
    ])
    dir_targets_norm = tf.norm(dir_targets, axis=0) + 1e-10  # 避免除零
    dir_targets = dir_targets / dir_targets_norm

    # 新增目标值: 点 —— 暂不使用交点转换（减小过拟合风险）
    point_targets_raw = tf.stack([
        parsed['PriPoint_0'],
        parsed['PriPoint_1'],
        parsed['PriPoint_2']
    ])
    # 若未来需要重新启用平面交点，请恢复上述逻辑并设置 USE_POINT_PLANE_TARGET=True

    if POINT_STANDARDIZE:
        # 兼容场景：若 PRI_POINT_MEAN/STD 被外部JSON覆盖为 Python list，需要转为Tensor
        pp_mean = tf.convert_to_tensor(PRI_POINT_MEAN, dtype=tf.float32)
        pp_std  = tf.convert_to_tensor(PRI_POINT_STD, dtype=tf.float32)
        point_targets = (point_targets_raw - pp_mean) / (pp_std + 1e-6)
    else:
        point_targets = point_targets_raw

    # 展平二通道输入，保持与 BGOInput((14,22,2)) 一致的内存布局
    barE_flat = tf.reshape(barE_2ch, [-1])

    # targets 字典，键名与模型输出一致，避免自动打包错误
    targets = {
        'direction_output': dir_targets,     # shape: (3,)
        'point_output': point_targets        # shape: (3,)
    }

    return {
        'barE_input': barE_flat,
        'barE_original': barE_original,  # 用于物理特征提取
        'physics_features': physics_features  # 占位符，后续处理
    }, targets


def process_batch_for_physics(inputs, targets):
    """
    批量处理数据，提取物理特征
    说明：此步骤用于训练/验证期为DL分支提供更丰富的physics_features。
         为避免向DL模型传入多余键，这里不返回 barE_original。
         评估期将设置 include_physics=False，以保留 barE_original 供混合模型使用。
    """
    barE_original = inputs['barE_original']
    physics_features = extract_physics_features(barE_original)

    # 更新输入字典（不包含 barE_original，避免Keras额外键问题）
    updated_inputs = {
        'barE_input': inputs['barE_input'],
        'physics_features': physics_features
    }

    return updated_inputs, targets

def build_data_pipeline(file_pattern, batch_size=2048, shuffle=True, include_physics=True):
    """构建数据管道，从TFRecord文件加载和预处理数据"""
    # 增强数据管道的容错能力和确定性
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    # 如果需要洗牌，设置随机种子以确保可重现性
    if shuffle:
        files = files.shuffle(buffer_size=100, seed=42)

    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024
    )

    # 解析和预处理
    dataset = dataset.map(parse_tfrecord,
                         num_parallel_calls=tf.data.AUTOTUNE)

    # 使用更大的buffer_size进行洗牌，并设置随机种子
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    # 批处理
    dataset = dataset.batch(batch_size)

    # 添加物理特征提取处理
    if include_physics:
        dataset = dataset.map(process_batch_for_physics,
                            num_parallel_calls=tf.data.AUTOTUNE)

    # 角度分桶加权（返回 (x, y, sample_weight) 三元组）
    if ENABLE_ANGLE_WEIGHT:
        def add_angle_weights(inputs, targets):
            # 取出方向真值并计算入射角 θ（与 Z 轴）
            if isinstance(targets, dict):
                dir_true = targets['direction_output']
            else:
                dir_true = targets

            # dir_true 形状：(batch, 3)
            z = tf.clip_by_value(dir_true[:, 2], -1.0, 1.0)
            theta = tf.acos(z) * 180.0 / np.pi
            focus_mask = tf.cast((theta >= FOCUS_THETA_LO) & (theta < FOCUS_THETA_HI), tf.float32)
            w = 1.0 + ANGLE_FOCUS_ALPHA * focus_mask  # (batch,)

            # 为多输出模型提供 dict 权重
            sample_weights = {
                'direction_output': w,
                'point_output': w
            }
            return inputs, targets, sample_weights

        dataset = dataset.map(add_angle_weights, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# 特征工程层
class FeatureEngineering(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch, 14, 22, 1)
        
        # 1. 能量归一化 (按事件总能量)
        total_energy = tf.reduce_sum(inputs, axis=[1, 2], keepdims=True)
        total_energy = tf.maximum(total_energy, 1e-6) # 避免除以零
        energy_normalized = inputs / total_energy

        # 2. 逐层能量占比
        layer_energy = tf.reduce_sum(inputs, axis=2, keepdims=True) # (batch, 14, 1, 1)
        layer_fraction = layer_energy / total_energy
        layer_fraction_map = tf.tile(layer_fraction, [1, 1, 22, 1]) # 扩展为 (batch, 14, 22, 1)

        # 3. 能量重心 (Shower Centroid)
        coords_x = tf.range(22, dtype=tf.float32)
        coords_y = tf.range(14, dtype=tf.float32)
        
        centroid_x = tf.reduce_sum(inputs * coords_x, axis=2) / tf.reduce_sum(inputs, axis=2)
        centroid_y = tf.reduce_sum(inputs * coords_y, axis=1) / tf.reduce_sum(inputs, axis=1)

        centroid_x = tf.maximum(centroid_x, 0.0)
        centroid_y = tf.maximum(centroid_y, 0.0)

        centroid_x_map = tf.tile(tf.reshape(centroid_x, [-1, 14, 1, 1]), [1, 1, 22, 1])
        centroid_y_map = tf.tile(tf.reshape(centroid_y, [-1, 1, 22, 1]), [1, 14, 1, 1])

        # 4. 寻找最大能量沉积层 (Shower Maximum Layer)
        max_layer_idx = tf.argmax(tf.squeeze(layer_energy, axis=[2,3]), axis=1)
        max_layer_map = tf.one_hot(max_layer_idx, depth=14, dtype=tf.float32)
        max_layer_map = tf.reshape(max_layer_map, [-1, 14, 1, 1])
        max_layer_map = tf.tile(max_layer_map, [1, 1, 22, 1])

        # 5. 组合所有新特征
        # 原始能量, 归一化能量, 逐层能量占比, 重心X, 重心Y, 最大能量层
        combined_features = layers.Concatenate(axis=-1)([
            inputs, 
            energy_normalized, 
            layer_fraction_map,
            centroid_x_map,
            centroid_y_map,
            max_layer_map
        ])
        
        return combined_features

# 模型架构
class BGOInput(layers.Layer):
    def __init__(self, **kwargs):
        super(BGOInput, self).__init__(**kwargs)
        # 更稳健的初始化器
        self.reshape = layers.Reshape((14, 22, N_INPUT_CHANNELS))

    def call(self, inputs):
        x = self.reshape(inputs)
        return x

# 增强的注意力机制模块
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

# 增强残差块的数值稳定性并确保可重现性
def residual_block(x, filters, kernel_size=(3,3), expansion=2.0, seed_offset=0, use_attention=True, dropout_rate=0.1):
    shortcut = x
    in_channels = x.shape[-1]

    # 第一个卷积层
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.Activation('swish')(x)

    # 第二个卷积层
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=43+seed_offset))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # 注意力机制
    if use_attention:
        x = CBAM(filters)(x)

    # 如果输入和输出通道数不同，使用1x1卷积进行调整
    if in_channels != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=44+seed_offset))(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)

    # 合并残差连接
    x = layers.add([x, shortcut])
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate, seed=45+seed_offset)(x)

    return x


# 物理引导的多模态模型
def build_physics_guided_model():
    # BGO能量输入
    barE_input = tf.keras.Input(shape=(14 * 22 * N_INPUT_CHANNELS,), name='barE_input')

    # 重塑为图像格式
    x = BGOInput()(barE_input)

    # 提取局部特征
    x = layers.Conv2D(128, (3, 3), padding='same', activation='swish',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # 增加一个额外的初始卷积层
    x = layers.Conv2D(128, (3, 3), padding='same', activation='swish',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=43))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    # 残差块1
    x = residual_block(x, 256, seed_offset=0, dropout_rate=0.1)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 残差块2
    x = residual_block(x, 512, seed_offset=10, dropout_rate=0.15)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 残差块3
    x = residual_block(x, 1024, seed_offset=20, dropout_rate=0.2)

    # 增加一个额外的残差块
    x = residual_block(x, 1024, seed_offset=30, dropout_rate=0.2)

    # 多尺度特征提取
    x_global = layers.GlobalAveragePooling2D()(x)
    x_max = layers.GlobalMaxPooling2D()(x)
    x_combined = layers.Concatenate()([x_global, x_max])

    # 增强的全连接层
    x_global = layers.Dense(512, activation='swish')(x_combined)  # 增加到512
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.3)(x_global)

    x_global = layers.Dense(512, activation='swish')(x_global)  # 添加额外层
    x_global = layers.BatchNormalization()(x_global)
    x_global = layers.Dropout(0.2)(x_global)
    # 增强的物理特征处理分支
    physics_input = tf.keras.Input(shape=(24,), name='physics_features')

    p = layers.Dense(128, activation='swish')(physics_input)  # 增加到128
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # 增加到256
    p = layers.BatchNormalization()(p)
    p = layers.Dense(256, activation='swish')(p)              # 添加额外层
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.2)(p)

    # 增强的融合网络
    combined = layers.Concatenate()([x_global, p])

    # 更深的融合处理网络
    z = layers.Dense(1024, activation='swish')(combined)       # 增加到1024
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Dense(512, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(256, activation='swish')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)

    # 模型的两个输出（论文 Sec. 3.4：方向 + 入射点）：
    # 1. 方向预测（先线性，再单位化）
    direction_raw = layers.Dense(3, name='direction_output_raw')(z)
    direction_output = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=-1), name='direction_output')(direction_raw)

    # 2. 初始点预测 - 回归到直接预测模式
    point_output = layers.Dense(3, name='point_output')(z)

    # 创建模型
    model = tf.keras.Model(
        inputs=[barE_input, physics_input],
        outputs=[direction_output, point_output]
    )

    return model

# 自定义损失函数 - 扩展以兼容点预测
class DirectionCosLoss(tf.keras.losses.Loss):
    """以余弦为主的方向损失，带翻转惩罚，并可选角度加权。"""
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

        # 不在损失内部重复加权，统一由数据管道的 sample_weight 控制
        per_sample = self.beta * loss_cos + self.alpha * loss_mse + self.flip_penalty * flip_penalty
        return per_sample


class DecisionCenterLoss(tf.keras.losses.Loss):
    """已废弃：决策分支已从架构中移除（论文 Sec. 3.4 只保留方向与入射点两个头）。

    仅保留该类，以便仍能反序列化早期训练得到的检查点；训练流程不再使用。
    """
    def __init__(self, name='decision_center_loss'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # 忽略 y_true，直接拉向 0.5
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        target = tf.zeros_like(y_pred) + 0.5
        return tf.square(y_pred - target)


class PointHuberLoss(tf.keras.losses.Loss):
    """
    V3: 内置 tf.Variable weight_var 控制损失权重。
    这样 compile(loss_weights=...) 中只需传入普通浮点数，避免保存模型时 VariableSpec 序列化失败。
    回调 PointWeightAnnealing 通过 loss.weight_var.assign() 动态调整。
    """
    def __init__(self, delta=POINT_HUBER_DELTA, name='point_huber_loss'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.delta = delta  # 保存为实例属性，get_config 使用
        self.huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
        # 权重变量（非 trainable，不会被优化器更新）
        self.weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32,
                                      name='point_loss_weight')

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = self.huber(y_true, y_pred)  # (batch,)
        return self.weight_var * loss       # weight_var=0 时自动静音

    def get_config(self):
        cfg = super().get_config()
        cfg['delta'] = self.delta  # 使用实例属性，不依赖 tf.keras.losses.Huber 的内部属性
        return cfg

# 角度误差指标
class AngleError(tf.keras.metrics.Metric):
    def __init__(self, name='angle_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.angle_error_sum = self.add_weight(name='angle_error_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 对于按输出绑定的指标，这里收到的是对应输出张量
        direction_pred = y_pred

        # 确保维度正确
        if len(tf.shape(y_true)) == 1:
            y_true = tf.expand_dims(y_true, 0)
        if len(tf.shape(direction_pred)) == 1:
            direction_pred = tf.expand_dims(direction_pred, 0)

        # 单位化向量 - 使用axis=-1
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(direction_pred, axis=-1)

        # 计算点积
        dot_product = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

        # 计算角度（弧度）
        angles = tf.math.acos(dot_product)

        # 转换为度
        angles_deg = angles * 180.0 / np.pi

        # 更新状态
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

# 余弦相似度指标
class CosineSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name='cosine_similarity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cos_sim_sum = self.add_weight(name='cos_sim_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 对于按输出绑定的指标，这里收到的是对应输出张量
        direction_pred = y_pred

        # 确保维度正确
        if len(tf.shape(y_true)) == 1:
            y_true = tf.expand_dims(y_true, 0)
        if len(tf.shape(direction_pred)) == 1:
            direction_pred = tf.expand_dims(direction_pred, 0)

        # 单位化向量 - 使用axis=-1
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(direction_pred, axis=-1)

        # 计算余弦相似度
        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

        # 更新状态
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

# 混合预测模型
class HybridPredictionModel(tf.keras.Model):
    def __init__(self, dl_model):
        super().__init__()
        self.dl_model = dl_model
        # 从磁盘加载对齐参数
        self.alignment_params = {
            'BAYYL': np.zeros((BGOL, BGOLB)),
            'BAYZL': np.zeros((BGOL, BGOLB)),
            'BAXXL': np.zeros((BGOL, BGOLB)),
            'BAXZL': np.zeros((BGOL, BGOLB))
        }

    # ...existing code...
        # 加载对齐参数
        try:
            self._load_alignment_params("BGOAlignment.txt")
            print("成功加载BGO对齐参数")
        except Exception as e:
            print(f"加载BGO对齐参数失败: {e}")

    def _load_alignment_params(self, filepath):
        """加载对齐参数"""
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
            print(f"加载对齐参数时出错: {e}")
            # 使用默认值
            pass

    def _apply_physics_method(self, barE_data):
        """应用传统物理方法进行预测"""
        batch_size = tf.shape(barE_data)[0]
        results = tf.TensorArray(tf.float32, size=batch_size)

        for idx in range(batch_size):
            # 还原能量沉积为14x22数组
            bgo_energy = tf.reshape(barE_data[idx], (14, 22))

            # 构造空间坐标和能量数组
            BHXXL = np.zeros((BGOL, BGOLB))
            BHXZL = np.zeros((BGOL, BGOLB))
            BHXEL = np.zeros((BGOL, BGOLB))
            BHYYL = np.zeros((BGOL, BGOLB))
            BHYZL = np.zeros((BGOL, BGOLB))
            BHYEL = np.zeros((BGOL, BGOLB))

            # 使用与ReConBGODir2.py一致的几何位置计算
            for i in range(BGOL):
                for j in range(BGOLB):
                    BHYYL[i][j] = -288.75 + j * 27.5 + self.alignment_params['BAYYL'][i][j]
                    BHYZL[i][j] = 58.50 + i * 58.0 + self.alignment_params['BAYZL'][i][j]
                    BHYEL[i][j] = bgo_energy[2 * i][j].numpy()
                    BHXXL[i][j] = -288.75 + j * 27.5 + self.alignment_params['BAXXL'][i][j]
                    BHXZL[i][j] = 87.50 + i * 58.0 + self.alignment_params['BAXZL'][i][j]
                    BHXEL[i][j] = bgo_energy[2 * i + 1][j].numpy()

            # 计算总能量
            BgoEng = np.sum(bgo_energy)

            # 调用物理方法重建轨迹
            ln_x = [22] * 7  # 每层X方向的条数
            ln_y = [22] * 7  # 每层Y方向的条数

            # 尝试使用物理方法重建
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
                    # 物理方法失败，返回零向量作为标记
                    results = results.write(idx, tf.zeros((3,), dtype=tf.float32))
            except Exception as e:
                # 物理方法出错，返回零向量
                print(f"物理方法出错: {e}")
                results = results.write(idx, tf.zeros((3,), dtype=tf.float32))

        return results.stack()

    def _filter_inputs_for_dl(self, inputs):
        """过滤只保留DL模型需要的键，避免传入额外键"""
        if isinstance(inputs, dict):
            filtered = {}
            if 'barE_input' in inputs:
                filtered['barE_input'] = inputs['barE_input']
            if 'physics_features' in inputs:
                filtered['physics_features'] = inputs['physics_features']
            # 若两键都不在，原样返回
            return filtered if len(filtered) > 0 else inputs
        return inputs

    def _get_barE_original(self, inputs):
        """优先返回inputs中的barE_original；若缺失，则从barE_input前14*22元素兜底重构为[batch,14,22]"""
        # 直接存在时返回
        if isinstance(inputs, dict) and 'barE_original' in inputs:
            return inputs['barE_original']
        # 兜底：从barE_input重构
        source = None
        if isinstance(inputs, dict):
            source = inputs.get('barE_input', None)
        elif isinstance(inputs, (list, tuple)):
            for t in inputs:
                try:
                    if int(t.shape[-1]) == 14*22*N_INPUT_CHANNELS:
                        source = t
                        break
                except Exception:
                    pass
        if source is None:
            raise KeyError('barE_original')
        batch_size = tf.shape(source)[0]
        first_plane = source[:, :14*22]
        return tf.reshape(first_plane, (batch_size, 14, 22))

    def call(self, inputs):
        # DLD 分支直接采用深度网络输出（论文 Sec. 3.4）：
        # 不使用解析重建，也不设门控。
        dl_inputs = self._filter_inputs_for_dl(inputs)
        dl_direction, point_pred = self.dl_model(dl_inputs)
        gate = tf.zeros((tf.shape(dl_direction)[0], 1), dtype=tf.float32)
        return dl_direction, gate, point_pred

    def predict_with_physics(self, inputs):
        """兼容旧接口：DLD 不做物理融合，返回 (方向, 全零门控)。

        保留该接口只为让 evaluate_hybrid_model 等旧调用仍可运行；
        _apply_physics_method 作为 Standard BGO Algorithm 的参考实现保留。
        """
        dl_inputs = self._filter_inputs_for_dl(inputs)
        dl_direction, _point_pred = self.dl_model(dl_inputs)
        gate = tf.zeros((tf.shape(dl_direction)[0], 1), dtype=tf.float32)
        return dl_direction, gate

# 自定义学习率调度器
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=1e-5, max_lr=5e-4, step_size=2000):
        super().__init__()
        # 确保所有输入都是 float32 类型
        self.base_lr = tf.cast(base_lr, dtype=tf.float32)
        self.max_lr = tf.cast(max_lr, dtype=tf.float32)
        self.step_size = tf.cast(step_size, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # 确保步数也是 float32 类型
        cycle = tf.math.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        # 使用 tf.constant 确保 0.0 是 float32 类型
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

class PointWeightAnnealing(tf.keras.callbacks.Callback):
    """
    V3: 在方向头充分收敛后，逐步启用点头的损失权重。
    通过修改 PointHuberLoss 实例内部的 weight_var.assign() 动态调整，
    避免在 compile(loss_weights=...) 中放 tf.Variable 导致保存时序列化报错。

    时间线：
      epoch < warmup_epochs          → point_weight = 0.0（专注方向头）
      warmup_epochs ≤ epoch < 2×warm → 线性升温至 final_weight
      epoch ≥ 2×warmup_epochs        → point_weight = final_weight（稳定）
    """
    def __init__(self, point_loss, warmup_epochs=15, final_weight=0.1):
        super().__init__()
        self.point_loss = point_loss      # PointHuberLoss 实例
        self.warmup_epochs = warmup_epochs
        self.final_weight = final_weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            w = 0.0
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.warmup_epochs)
            w = float(self.final_weight * min(1.0, progress))
        self.point_loss.weight_var.assign(w)
        if epoch >= self.warmup_epochs:
            print(f"\n[PointWeightAnnealing] Epoch {epoch+1}: point_loss_weight = {w:.4f}")

# 梯度调试回调 - 修复TensorFlow 2.10兼容性问题
class GradientDebugger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            try:
                # 检查NaN和Inf - 使用TF 2.10兼容的方法获取梯度
                gradients = []
                variables = self.model.trainable_weights

                # 获取损失函数
                loss_fn = self.model.loss
                if hasattr(self.model, 'compiled_loss'):
                    # TF 2.x 方式
                    with tf.GradientTape() as tape:
                        y_pred = self.model(self.model.inputs, training=True)
                        loss = self.model.compiled_loss(self.model.targets, y_pred)
                    gradients = tape.gradient(loss, variables)

                has_nan = False
                has_inf = False
                for g in gradients:
                    if g is not None:  # 有些层可能没有梯度
                        if np.isnan(np.sum(g.numpy())):
                            has_nan = True
                        if np.isinf(np.sum(g.numpy())):
                            has_inf = True

                if has_nan or has_inf:
                    print(f"\n批次 {batch}: 检测到{'NaN' if has_nan else ''} {'Inf' if has_inf else ''} 梯度！")
            except Exception as e:
                # 如果无法获取梯度，仅打印警告
                print(f"警告：在梯度检查过程中出现错误: {str(e)}")

# 批量大小调度器
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
            # 下降阶段
            step_in_descent = self.current_step - self.steps
            new_batch_size = self.max_size - (self.max_size - self.start_size) * step_in_descent // self.steps

        print(f"\n更新批量大小为: {new_batch_size}")

        # 重建数据集
        if hasattr(self.model, 'train_data'):
            self.model.train_data = build_data_pipeline(self.train_data_path, new_batch_size, shuffle=True)

# GPU监控回调
class GPUMonitor(tf.keras.callbacks.Callback):
    def __init__(self, print_interval=5):
        super().__init__()
        self.print_interval = print_interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.print_interval == 0:
            # 尝试获取GPU使用信息
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"\nGPU内存使用：已用={gpu_info['current'] / 1e9:.2f}GB，峰值={gpu_info['peak'] / 1e9:.2f}GB")
            except:
                # 某些环境可能不支持此功能
                pass

# 学习率记录器
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_begin(self, epoch, logs=None):
        try:
            # 尝试不同方式获取学习率
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr
                if callable(lr):
                    lr = lr()
                if hasattr(lr, 'numpy'):
                    lr = lr.numpy()
                print(f"\n当前学习率: {float(lr):.6f}")
            elif hasattr(self.model.optimizer, 'learning_rate'):
                lr = self.model.optimizer.learning_rate
                if callable(lr):
                    lr = lr()
                if hasattr(lr, 'numpy'):
                    lr = lr.numpy()
                print(f"\n当前学习率: {float(lr):.6f}")
            else:
                # 最后尝试直接访问当前学习率值
                # 这是一个兜底方案
                print("\n无法直接获取当前学习率")
        except Exception as e:
            print(f"\n获取学习率时出错: {e}")


# 学习率 Warmup 包装：在 warmup_steps 内线性插值到基础调度
class WarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_schedule, warmup_steps=3000, init_lr=2e-5):
        super().__init__()
        self.base_schedule = base_schedule
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.init_lr = tf.cast(init_lr, tf.float32)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        base_lr = self.base_schedule(step_f)
        # 线性 warmup: 从 init_lr -> base_lr
        warmup_ratio = tf.clip_by_value(step_f / self.warmup_steps, 0.0, 1.0)
        lr = self.init_lr + (base_lr - self.init_lr) * warmup_ratio
        return lr

    def get_config(self):
        return {"warmup_steps": float(self.warmup_steps), "init_lr": float(self.init_lr)}


# 点损失暖启动回调：前 warmup_epochs 关闭点损失，然后分阶段提升权重
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
            # 线性上升
            progress = min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)
            w = self.total_target * progress
        # 更新模型的 loss_weights
        if hasattr(self.model, 'loss_weights') and isinstance(self.model.loss_weights, dict):
            self.model.loss_weights['point_output'] = w
        print(f"[PointLossWarmup] epoch={epoch} point_loss_weight={w:.5f}")

# 融合模型评估与可视化
def evaluate_hybrid_model(model, test_dataset, output_dir='./results/hybrid'):
    """评估融合模型并以文本形式展示结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 收集预测结果
    all_predictions = []
    all_true_values = []
    all_decisions = []

    # 遍历测试数据集
    for inputs, targets in test_dataset:
        # 使用eager模式预测方法
        predictions, decisions = model.predict_with_physics(inputs)
        # 取出方向真值
        if isinstance(targets, dict):
            dir_true = targets['direction_output']
        elif isinstance(targets, (list, tuple)):
            dir_true = targets[0]
        else:
            dir_true = targets

        # 转numpy，增强兼容性
        p = predictions.numpy() if hasattr(predictions, "numpy") else predictions
        d = decisions.numpy() if hasattr(decisions, "numpy") else decisions
        t = dir_true.numpy() if hasattr(dir_true, "numpy") else dir_true

        all_predictions.append(p)
        all_true_values.append(t)
        all_decisions.append(d)

    # 连接结果
    all_predictions = np.vstack(all_predictions)
    all_true_values = np.vstack(all_true_values)
    all_decisions = np.vstack(all_decisions)

    # 计算角度误差
    y_true_norm = all_true_values / np.linalg.norm(all_true_values, axis=1, keepdims=True)
    y_pred_norm = all_predictions / np.linalg.norm(all_predictions, axis=1, keepdims=True)

    dot_products = np.sum(y_true_norm * y_pred_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angle_errors = np.arccos(dot_products) * 180.0 / np.pi

    # 计算余弦相似度
    cosine_sim = dot_products

    # 打印详细的评估统计数据以替代图表
    print("\n===== 混合模型评估结果 =====")
    print(f"总样本数: {len(angle_errors)}")
    
    print("\n--- 角度误差分布 (度) ---")
    print(f"  平均值: {np.mean(angle_errors):.2f}°")
    print(f"  中位数: {np.median(angle_errors):.2f}°")
    print(f"  标准差: {np.std(angle_errors):.2f}°")
    print(f"  最小值: {np.min(angle_errors):.2f}°")
    print(f"  最大值: {np.max(angle_errors):.2f}°")
    print(f"  25%分位数: {np.percentile(angle_errors, 25):.2f}°")
    print(f"  68%分位数: {np.percentile(angle_errors, 68):.2f}°")
    print(f"  95%分位数: {np.percentile(angle_errors, 95):.2f}°")

    print("\n--- 决策网络输出 (物理方法权重) 分布 ---")
    print(f"  平均值: {np.mean(all_decisions):.4f}")
    print(f"  中位数: {np.median(all_decisions):.4f}")
    print(f"  标准差: {np.std(all_decisions):.4f}")
    print(f"  最小值: {np.min(all_decisions):.4f}")
    print(f"  最大值: {np.max(all_decisions):.4f}")
    print(f"  权重 > 0.5 的样本比例: {np.mean(all_decisions > 0.5):.2%}")
    print(f"  权重 > 0.8 的样本比例: {np.mean(all_decisions > 0.8):.2%}")

    print("\n--- 总体性能 ---")
    print(f"平均余弦相似度: {np.mean(cosine_sim):.4f}")

    # 保存评估结果
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

# 仅使用深度学习分支进行评估（不使用物理方法融合）
def evaluate_dl_only_model(model, test_dataset, output_dir='./results/dl_only'):
    """评估纯DL方向输出，完全取消物理方法融合"""
    os.makedirs(output_dir, exist_ok=True)

    all_predictions = []
    all_true_values = []

    for batch in test_dataset:
        # 兼容 (x,y) 与 (x,y,weights) 两种形式
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, _weights = batch
        else:
            inputs, targets = batch
        # 直接前向：模型输出 [方向, 入射点]
        outputs = model(inputs, training=False)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            direction_pred = outputs[0]
        else:
            direction_pred = outputs

        # 取方向真值
        if isinstance(targets, dict):
            dir_true = targets['direction_output']
        elif isinstance(targets, (list, tuple)):
            dir_true = targets[0]
        else:
            dir_true = targets

        p = direction_pred.numpy() if hasattr(direction_pred, 'numpy') else direction_pred
        t = dir_true.numpy() if hasattr(dir_true, 'numpy') else dir_true

        all_predictions.append(p)
        all_true_values.append(t)

    all_predictions = np.vstack(all_predictions)
    all_true_values = np.vstack(all_true_values)

    # 计算角度误差与余弦
    y_true_norm = all_true_values / (np.linalg.norm(all_true_values, axis=1, keepdims=True) + 1e-12)
    y_pred_norm = all_predictions / (np.linalg.norm(all_predictions, axis=1, keepdims=True) + 1e-12)
    dot_products = np.sum(y_true_norm * y_pred_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angle_errors = np.arccos(dot_products) * 180.0 / np.pi
    cosine_sim = dot_products

    print("\n===== 纯DL模型评估结果（无物理融合） =====")
    print(f"总样本数: {len(angle_errors)}")

    print("\n--- 角度误差分布 (度) ---")
    print(f"  平均值: {np.mean(angle_errors):.2f}°")
    print(f"  中位数: {np.median(angle_errors):.2f}°")
    print(f"  标准差: {np.std(angle_errors):.2f}°")
    print(f"  最小值: {np.min(angle_errors):.2f}°")
    print(f"  最大值: {np.max(angle_errors):.2f}°")
    print(f"  25%分位数: {np.percentile(angle_errors, 25):.2f}°")
    print(f"  68%分位数: {np.percentile(angle_errors, 68):.2f}°")
    print(f"  95%分位数: {np.percentile(angle_errors, 95):.2f}°")

    # 角度分桶（按真实入射角与Z轴夹角）
    theta_true = np.degrees(np.arccos(np.clip(y_true_norm[:, 2], -1.0, 1.0)))
    bins = [(60, 70), (70, 80), (80, 90)]
    print("\n--- 分桶统计（按真实入射角度） ---")
    for lo, hi in bins:
        mask = (theta_true >= lo) & (theta_true < hi)
        if np.any(mask):
            ae = angle_errors[mask]
            print(f"  {lo:02d}–{hi:02d}°: N={ae.size}, 中位={np.median(ae):.2f}°, P68={np.percentile(ae, 68):.2f}°, P95={np.percentile(ae, 95):.2f}°")
        else:
            print(f"  {lo:02d}–{hi:02d}°: N=0")

    np.savez(os.path.join(output_dir, 'evaluation_results_dl_only.npz'),
             predictions=all_predictions,
             true_values=all_true_values,
             predictions_unit=y_pred_norm,
             angle_errors=angle_errors,
             cosine_sim=cosine_sim)

    return {
        'angle_errors': angle_errors,
        'cosine_sim': cosine_sim
    }

# 主函数
def main():
    print("开始初始化物理引导的深度学习模型训练...")

    # 检查是否可用GPU
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0

    if use_gpu:
        print(f"检测到 {len(gpus)} 个 GPU，将使用GPU训练")
        # 设置内存增长 - 在初始化时就设置
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已配置GPU内存增长模式")
        except Exception as e:
            print(f"设置GPU内存增长模式失败: {e}")
    else:
        print("未检测到GPU，或GPU被禁用，将使用CPU训练")

    # 设置参数
    batch_size = 2048 # OOM错误后，从4096减小批量大小
    epochs = 200      # 减少训练轮数
    # 数据路径（发布版占位）：请按本地环境修改，或用环境变量 BGO_DATA_DIR 指定数据目录
    DATA_DIR = os.environ.get('BGO_DATA_DIR', './data')
    train_data_path = os.path.join(DATA_DIR, 'large_angle_train.tfrecord')
    val_data_path = os.path.join(DATA_DIR, 'large_angle_val.tfrecord')

    # 检查数据路径
    if not os.path.exists(train_data_path):
        print(f"Warning: Training data not found at {train_data_path}")
    if not os.path.exists(val_data_path):
        print(f"Warning: Validation data not found at {val_data_path}")

    # 构建数据管道
    print("构建数据管道...")
    train_dataset = build_data_pipeline(train_data_path, batch_size, shuffle=True, include_physics=True)
    val_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False, include_physics=True)

    # 初始化history为None，以处理训练可能失败的情况
    history = None

    # 构建模型并训练
    print("构建物理引导模型...")
    try:
        # 创建模型
        physics_guided_model = build_physics_guided_model()

        # 使用基础学习率调度 + Warmup 包装
        base_lr_schedule = CyclicalLearningRate(base_lr=1e-5, max_lr=8e-4, step_size=2500)
        lr_schedule = WarmupLR(base_schedule=base_lr_schedule, warmup_steps=3000, init_lr=2e-5)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0)

        # 编译模型 - 强制使用eager执行模式
        point_loss = PointHuberLoss(delta=POINT_HUBER_DELTA)

        physics_guided_model.compile(
            optimizer=optimizer,
            loss={
                'direction_output': DirectionCosLoss(alpha=DIR_ALPHA, beta=DIR_BETA, flip_penalty=DIR_FLIP_PENALTY),
                'point_output': point_loss
            },
            loss_weights={
                'direction_output': 1.0,
                'point_output': 1.0     # 固定为1.0，动态权重由 point_loss.weight_var 内部控制
            },
            metrics={'direction_output': [AngleError(), CosineSimilarity()]},
            run_eagerly=True
        )

        # 模型摘要
        physics_guided_model.summary()

        # 修改回调函数，保存完整模型
        callbacks = [
            LearningRateLogger(),
            PointWeightAnnealing(
                point_loss=point_loss,   # 传入同一个 PointHuberLoss 实例
                warmup_epochs=15,        # 前15 epoch专注方向头
                final_weight=0.1         # 最大点权重=0.1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/dld_best',  # 目录名；保存最佳完整模型
                save_best_only=True,
                monitor='val_direction_output_angle_error',
                mode='min',
                save_weights_only=False,  # 保存完整模型而不仅仅是权重
                verbose=1,
                save_format='tf'  # 使用TensorFlow SavedModel格式
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True,
                monitor='val_direction_output_angle_error',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('training_history_large_angle.csv', append=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # 开始训练
        print("开始模型训练...")
        history = physics_guided_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            workers=0,  # 禁用多线程数据加载
            use_multiprocessing=False
        )

        print("训练完成！")
        # 保存最终完整模型 - 使用多种格式保存
        print("保存最终模型...")
        try:
            # 1. 保存为TensorFlow SavedModel/keras 格式
            physics_guided_model.save('dld_large_angle.keras')
            print("✓ 已保存完整模型: dld_large_angle.keras")

            # 3. 额外保存模型架构为JSON文件
            model_json = physics_guided_model.to_json()
            with open('dld_large_angle_architecture.json', 'w') as json_file:
                json_file.write(model_json)
            print("已保存模型架构为JSON格式: dld_large_angle_architecture.json")

            # 4. 保存模型权重（作为备份）
            physics_guided_model.save_weights('dld_large_angle.weights.h5')
            print("已保存模型权重为备份: dld_large_angle.weights.h5")

        except Exception as save_error:
            print(f"保存模型时出现错误: {save_error}")
            # 如果SavedModel格式失败，尝试只保存权重
            try:
                physics_guided_model.save_weights('dld_large_angle.weights.h5')
                print("已保存模型权重作为备份: dld_large_angle.weights.h5")
            except Exception as weight_error:
                print(f"保存权重也失败: {weight_error}")

        # 评估
        try:
            # 评估：纯DL输出（不使用物理融合）
            print("评估纯DL输出（取消混合融合）...")
            # 与训练一致，评估也提供 physics_features（无需 barE_original）
            test_dataset = build_data_pipeline(val_data_path, batch_size, shuffle=False, include_physics=True)
            evaluation_results = evaluate_dl_only_model(physics_guided_model, test_dataset, output_dir='./results/dld_eval')
        except Exception as e:
            print(f"评估过程中出现错误: {e}")

        # 打印训练历史
        if history:
            print("绘制训练历史...")
            try:
                print("\n===== 训练最终结果 =====")
                final_epoch_metrics = {key: value[-1] for key, value in history.history.items()}
                for metric, value in final_epoch_metrics.items():
                    print(f"  - {metric}: {value:.4f}")
                print("训练历史已记录在 training_history_large_model.csv 文件中。")

            except Exception as e:
                print(f"打印训练历史时出错: {e}")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("尝试减小批量大小或简化模型重新运行")

    print("程序执行完成！")

if __name__ == "__main__":
    main()
