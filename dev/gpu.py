import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
	import tensorflow as tf
except ImportError as e:
	print("Error al importar TensorFlow:", e)
	print(
		"\nPosible causa: Falta el runtime ROCm (libamdhip64.so.*).\n"
		"Solución rápida en Ubuntu/Debian:\n"
		"  1) Instalar ROCm HIP runtime y herramientas.\n"
		"     sudo apt update && \\n+sudo apt install -y hip-runtime-amd rocminfo rocm-device-libs rocblas miopen-hip\n"
		"  2) Exportar librerías en LD_LIBRARY_PATH (p.ej. /opt/rocm/lib).\n"
		"     export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH\n"
		"  3) Reintentar ejecutar este script.\n"
	)
	raise

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas:", gpus)