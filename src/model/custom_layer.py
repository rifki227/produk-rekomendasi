import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────────────
#  Custom Layer
# ──────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
class ProductInteractionLayer(tf.keras.layers.Layer):
    """
    Custom interaction layer yang menghitung element-wise multiplication
    antara embedding vector user dan produk.

    Digunakan sebagai jalur kedua dalam arsitektur NCF untuk menangkap
    sinyal interaksi non-linear secara eksplisit, berbeda dari jalur
    concatenate yang hanya menggabungkan kedua vektor.

    Input:
        [user_vector, item_vector] — dua tensor dengan shape (batch, embedding_dim)

    Output:
        Tensor shape (batch, embedding_dim) — hasil perkalian elemen per elemen
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: list) -> tf.Tensor:
        user_vector, item_vector = inputs
        return user_vector * item_vector

    def get_config(self) -> dict:
        """Diperlukan agar layer bisa di-serialize dan di-load kembali."""
        return super().get_config()


# ──────────────────────────────────────────────────────────────────────────────
#  Custom Loss Function
# ──────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
def root_mean_squared_error_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom RMSE loss function.

    Menggunakan RMSE alih-alih MSE bawaan Keras agar nilai loss
    berada dalam skala yang sama dengan rating (0–1),
    sehingga lebih mudah diinterpretasikan saat training.

    Args:
        y_true: tensor nilai rating aktual
        y_pred: tensor nilai rating prediksi

    Returns:
        Scalar tensor RMSE
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

class TargetMAEStoppingCallback(tf.keras.callbacks.Callback):
    """
    Early stopping berbasis target MAE.
    Menghentikan training segera ketika MAE pada data training
    mencapai atau melampaui nilai target yang ditentukan.

    Args:
        target_mae: nilai MAE minimum yang menjadi kriteria berhenti (default 0.02)
    """

    def __init__(self, target_mae: float = 0.02):
        super().__init__()
        self.target_mae = target_mae

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        mae = (logs or {}).get("mae")
        if mae is not None and mae <= self.target_mae:
            print(
                f"\n[INFO] Target MAE {self.target_mae} tercapai pada "
                f"epoch ke-{epoch + 1} (MAE={mae:.5f}). Training dihentikan."
            )
            self.model.stop_training = True
