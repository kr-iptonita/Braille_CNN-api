import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf  # Importar TensorFlow para aprovechar la GPU

# Configuración para la GPU
gpus = tf.config.list_physical_devices('GPU')
print(gpus)