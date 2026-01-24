import os
from shutil import copyfile
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT_DIR = "Braille_Dataset"
TARGET_DIR = "train"
CHECKPOINT_FILE = "BrailleNet-amd.keras"
EPOCHS = 666
IMG_SIZE = (28, 28)


def configure_gpu() -> None:
    """Configura la primera GPU disponible (ROCm o CUDA) si existe."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No se detectó GPU, se usará CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], "GPU")
        details = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
        print(f"GPU seleccionada: {details}")
    except RuntimeError as err:
        print(f"No se pudo configurar la GPU: {err}")


def prepare_dataset(root_dir: str, target_dir: str) -> None:
    """Copia el dataset plano a una estructura por clases si aún no existe."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"No se encontró el directorio de datos: {root_dir}")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print(f"Creando estructura de entrenamiento en '{target_dir}'.")
        alpha = "a"
        for _ in range(26):
            os.mkdir(os.path.join(target_dir, alpha))
            alpha = chr(ord(alpha) + 1)

    existing = set(os.listdir(target_dir))
    for file in os.listdir(root_dir):
        letter = file[0]
        if letter not in existing:
            raise ValueError(f"La carpeta para la clase '{letter}' no existe en {target_dir}.")
        src = os.path.join(root_dir, file)
        dst = os.path.join(target_dir, letter, file)
        if not os.path.exists(dst):
            copyfile(src, dst)


def create_generators(base_dir: str):
    datagen = ImageDataGenerator(rotation_range=20, shear_range=10, validation_split=0.2)
    train_gen = datagen.flow_from_directory(base_dir, target_size=IMG_SIZE, subset="training")
    val_gen = datagen.flow_from_directory(base_dir, target_size=IMG_SIZE, subset="validation")
    return train_gen, val_gen


def build_model() -> Model:
    entry = L.Input(shape=(*IMG_SIZE, 3))
    x = L.SeparableConv2D(64, (3, 3), activation="relu")(entry)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.SeparableConv2D(128, (3, 3), activation="relu")(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.SeparableConv2D(256, (3, 3), activation="relu")(x)
    x = L.SeparableConv2D(512, (2, 2), activation="relu")(x)
    x = L.GlobalMaxPooling2D()(x)
    x = L.Dense(512)(x)
    x = L.LeakyReLU()(x)
    x = L.Dense(256, kernel_regularizer=l2(2e-4))(x)
    x = L.LeakyReLU()(x)
    x = L.Dense(26, activation="softmax")(x)

    model = Model(entry, x)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def plot_history(history) -> None:
    plt.plot(history.history["loss"], label="pérdida de entrenamiento")
    plt.plot(history.history["val_loss"], label="pérdida de validación")
    plt.legend()
    plt.show()
    plt.savefig("LossVal_loss")

    plt.plot(history.history["accuracy"], label="precisión de entrenamiento")
    plt.plot(history.history["val_accuracy"], label="precisión de validación")
    plt.legend()
    plt.show()
    plt.savefig("AccVal_acc")


def main() -> None:
    configure_gpu()
    prepare_dataset(ROOT_DIR, TARGET_DIR)

    train_generator, val_generator = create_generators(TARGET_DIR)

    K.clear_session()
    model_ckpt = ModelCheckpoint(CHECKPOINT_FILE, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(patience=8, verbose=0)
    early_stop = EarlyStopping(patience=15, verbose=1)

    model = build_model()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[model_ckpt, reduce_lr, early_stop],
        verbose=1,
    )

    model = load_model(CHECKPOINT_FILE)
    acc = model.evaluate(val_generator)[1]
    print(f"Precisión del modelo: {round(acc, 4)}")

    plot_history(history)


if __name__ == "__main__":
    main()
