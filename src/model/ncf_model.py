from __future__ import annotations

import logging
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.model.custom_layer import (
    ProductInteractionLayer,
    TargetMAEStoppingCallback,
    root_mean_squared_error_loss,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Arsitektur Model
# ──────────────────────────────────────────────────────────────────────────────

def build_ncf_model(
    num_users: int,
    num_products: int,
    embedding_dim: int = 32,
    dense_units: list[int] | None = None,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
) -> Model:
    """
    Bangun model Neural Collaborative Filtering (NCF).

    Arsitektur:
    - Dua embedding layer terpisah untuk user dan produk
    - Jalur 1: Concatenate → Deep MLP
    - Jalur 2: Element-wise interaction (custom layer)
    - Gabungan kedua jalur → Dense → Output sigmoid [0,1]

    Args:
        num_users:     jumlah user unik (dimensi embedding user)
        num_products:  jumlah produk unik (dimensi embedding produk)
        embedding_dim: ukuran vektor embedding (default 32)
        dense_units:   list ukuran lapisan Dense MLP (default [64, 32, 16])
        dropout_rate:  dropout probability (default 0.5)
        learning_rate: learning rate Adam optimizer (default 0.001)

    Returns:
        Model Keras yang sudah di-compile, siap untuk training
    """
    if dense_units is None:
        dense_units = [64, 32, 16]

    # ── Input ──────────────────────────────────────────────────────────────
    user_id_input    = Input(shape=(1,), name="user_id_input")
    product_id_input = Input(shape=(1,), name="product_id_input")

    # ── Embedding ──────────────────────────────────────────────────────────
    user_embedding = Embedding(
        input_dim=num_users, output_dim=embedding_dim, name="user_embedding"
    )(user_id_input)
    product_embedding = Embedding(
        input_dim=num_products, output_dim=embedding_dim, name="product_embedding"
    )(product_id_input)

    user_flat    = Flatten()(user_embedding)
    product_flat = Flatten()(product_embedding)

    # ── Jalur 1: Concatenate ───────────────────────────────────────────────
    concat_features = Concatenate()([user_flat, product_flat])

    # ── Jalur 2: Custom Interaction Layer ─────────────────────────────────
    interaction_features = ProductInteractionLayer(name="custom_interaction")(
        [user_flat, product_flat]
    )

    # ── Gabungkan kedua jalur ─────────────────────────────────────────────
    combined = Concatenate()([concat_features, interaction_features])

    # ── Deep MLP ──────────────────────────────────────────────────────────
    x = combined
    for i, units in enumerate(dense_units):
        x = Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        if i < len(dense_units) - 1:      # tidak ada dropout setelah lapisan terakhir
            x = Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # ── Output: Sigmoid → rentang [0,1] ───────────────────────────────────
    output = Dense(1, activation="sigmoid", name="output_rating")(x)

    # ── Compile ───────────────────────────────────────────────────────────
    model = Model(inputs=[user_id_input, product_id_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=root_mean_squared_error_loss,
        metrics=["mae"],
    )

    logger.info(f"Model NCF berhasil dibangun: {model.count_params():,} parameter")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────

def train_ncf_model(
    model: Model,
    train_user: "np.ndarray",
    train_product: "np.ndarray",
    y_train: "np.ndarray",
    test_user: "np.ndarray",
    test_product: "np.ndarray",
    y_test: "np.ndarray",
    epochs: int = 25,
    batch_size: int = 128,
    target_mae: float = 0.02,
    log_dir: str | None = None,
) -> tf.keras.callbacks.History:
    """
    Latih model NCF dengan early stopping berbasis MAE target.

    Args:
        model:         model yang sudah di-compile (dari build_ncf_model)
        train_user:    array encoded user ID untuk training
        train_product: array encoded product ID untuk training
        y_train:       array rating training (skala 0–1)
        test_user:     array encoded user ID untuk validasi
        test_product:  array encoded product ID untuk validasi
        y_test:        array rating validasi (skala 0–1)
        epochs:        jumlah epoch maksimum (default 25)
        batch_size:    ukuran batch (default 128)
        target_mae:    MAE target untuk early stopping (default 0.02)
        log_dir:       direktori TensorBoard log (None = tidak dilog)

    Returns:
        History object dari Keras
    """
    import datetime

    callbacks = [TargetMAEStoppingCallback(target_mae=target_mae)]

    if log_dir is not None:
        tb_log = os.path.join(
            log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=tb_log, histogram_freq=1)
        )
        logger.info(f"TensorBoard log disimpan ke: {tb_log}")

    logger.info(
        f"Mulai training: epochs={epochs}, batch={batch_size}, "
        f"target_mae={target_mae}"
    )
    history = model.fit(
        x=[train_user, train_product],
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([test_user, test_product], y_test),
        callbacks=callbacks,
        verbose=1,
    )

    final_mae = history.history["mae"][-1]
    final_val_mae = history.history.get("val_mae", [None])[-1]
    logger.info(
        f"Training selesai: MAE={final_mae:.5f}, val_MAE={final_val_mae}"
    )
    return history



def save_model(model: Model, save_path: str | Path) -> None:
    """
    Simpan model ke file .keras (format Keras v3).

    Args:
        model:     model yang akan disimpan
        save_path: path tujuan (contoh: 'api/saved_model/ncf_product_recommendation.keras')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"Model disimpan ke '{save_path}'")


def load_model(model_path: str | Path) -> Model:
    """
    Muat model dari file .keras.
    Custom components sudah didaftarkan via @register_keras_serializable
    di custom_layer.py, jadi tidak perlu melewatkan custom_objects secara manual.

    Args:
        model_path: path ke file .keras

    Returns:
        Model Keras yang siap dipakai untuk inferensi
    """

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ProductInteractionLayer": ProductInteractionLayer,
            "root_mean_squared_error_loss": root_mean_squared_error_loss,
        },
    )
    logger.info(f"Model dimuat dari '{model_path}'")
    return model


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data.preprocess import run_preprocessing_pipeline

    artifacts = run_preprocessing_pipeline(
        raw_data_path="data/sample_data.csv",
        output_csv_path="data/cleaned_sample_data_scaled.csv",
        scaler_save_path="api/saved_model/rating_scaler.pkl",
    )

    model = build_ncf_model(
        num_users=artifacts["num_users"],
        num_products=artifacts["num_products"],
    )
    model.summary()

    history = train_ncf_model(
        model=model,
        train_user=artifacts["train"]["user_input"],
        train_product=artifacts["train"]["product_input"],
        y_train=artifacts["train"]["y"],
        test_user=artifacts["test"]["user_input"],
        test_product=artifacts["test"]["product_input"],
        y_test=artifacts["test"]["y"],
        log_dir="api/logs",
    )

    save_model(model, "api/saved_model/ncf_product_recommendation.keras")
    print("\n✅ Model berhasil dilatih dan disimpan.")
