import time
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)

if tf.config.list_physical_devices("GPU"):
  with tf.device("GPU:0"): 
    A = [tf.random.uniform([1600, 1600]) for _ in range(2)]
    B = [tf.random.uniform([100, 100]) for _ in range(20)]

    # Block 1
    C = tf.linalg.matmul(A[0], A[0])
    for i in range(10):
      D = tf.linalg.matmul(B[i], B[i])
    time.sleep(0.001)

    # Block 2
    for i in range(10,20):
      D = tf.linalg.matmul(B[i], B[i])
    C = tf.linalg.matmul(A[1], A[1])
    time.sleep(0.001)
