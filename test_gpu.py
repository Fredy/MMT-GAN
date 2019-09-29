import tensorflow as tf

print('GPU: ', tf.test.gpu_device_name())

print(tf.version.VERSION)
print(tf.keras.__version__)
