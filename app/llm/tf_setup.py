import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] ="1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print("GPU 설정 실패: ", e)

    
print("TF:", tf.__version__)
print("GPUs:", gpus)
        
    