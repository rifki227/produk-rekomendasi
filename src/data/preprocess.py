from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Load & Validasi
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Muat data mentah dari CSV.

    Kolom yang diharapkan: user_id, product_id, rating, timestamp
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    logger.info(f"Data dimuat dari '{path}': {len(df):,} baris")

    required_cols = {"user_id", "product_id", "rating", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validasi dan bersihkan data:
    - Hapus baris dengan nilai null
    - Hapus duplikat (user_id, product_id)
    - Pastikan rating dalam rentang 1–5
    """
    before = len(df)

    df = df.dropna(subset=["user_id", "product_id", "rating"])
    df = df.drop_duplicates(subset=["user_id", "product_id"], keep="last")
    df = df[df["rating"].between(1, 5)]

    after = len(df)
    logger.info(f"Validasi: {before:,} → {after:,} baris (dihapus {before - after:,})")

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Normalisasi Rating
# ──────────────────────────────────────────────────────────────────────────────

def scale_ratings(
    df: pd.DataFrame,
    save_scaler_path: str | Path | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalisasi kolom 'rating' dari rentang [1,5] ke [0,1] menggunakan MinMaxScaler.

    Args:
        df: DataFrame dengan kolom 'rating' (skala 1–5)
        save_scaler_path: opsional, path untuk menyimpan scaler (.pkl)

    Returns:
        (df_scaled, scaler) — DataFrame dengan rating ternormalisasi & objek scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.copy()
    df["rating"] = scaler.fit_transform(df[["rating"]])

    if save_scaler_path is not None:
        save_scaler_path = Path(save_scaler_path)
        save_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
        logger.info(f"Scaler disimpan ke '{save_scaler_path}'")

    return df, scaler


def inverse_scale_ratings(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Kembalikan prediksi dari skala [0,1] ke skala asli [1,5].

    Args:
        scaled_values: array 1-D berisi nilai yang akan di-invers
        scaler: objek MinMaxScaler yang sudah di-fit

    Returns:
        array 1-D dalam skala asli
    """
    return scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()


# ──────────────────────────────────────────────────────────────────────────────
#  Encoding ID
# ──────────────────────────────────────────────────────────────────────────────

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode user_id dan product_id dari string ke integer menggunakan LabelEncoder.

    Args:
        df: DataFrame dengan kolom 'user_id' dan 'product_id'

    Returns:
        (df_encoded, user_encoder, product_encoder)
    """
    df = df.copy()

    user_enc    = LabelEncoder()
    product_enc = LabelEncoder()

    df["user_encoded"]    = user_enc.fit_transform(df["user_id"])
    df["product_encoded"] = product_enc.fit_transform(df["product_id"])

    logger.info(
        f"Encoding selesai: {df['user_encoded'].nunique():,} user, "
        f"{df['product_encoded'].nunique():,} produk"
    )
    return df, user_enc, product_enc


# ──────────────────────────────────────────────────────────────────────────────
#  Train/Test Split
# ──────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bagi data menjadi train dan test set.

    Returns:
        train_user, train_product, y_train, test_user, test_product, y_test
    """
    X = df[["user_encoded", "product_encoded"]].values
    y = df["rating"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Split data: train={len(X_train):,}, test={len(X_test):,} "
        f"(ratio {1 - test_size:.0%}/{test_size:.0%})"
    )

    return (
        X_train[:, 0], X_train[:, 1], y_train,
        X_test[:, 0],  X_test[:, 1],  y_test,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline Lengkap
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    raw_data_path: str | Path,
    output_csv_path: str | Path,
    scaler_save_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Jalankan seluruh pipeline preprocessing dari data mentah hingga siap latih.

    Args:
        raw_data_path:   path ke file CSV data mentah
        output_csv_path: path simpan data yang sudah di-scale
        scaler_save_path: path simpan file scaler (.pkl)
        test_size:       proporsi data test (default 0.2)
        random_state:    seed untuk reproduktibilitas

    Returns:
        dict berisi semua artefak yang dibutuhkan untuk training
    """
    # 1. Load & validasi
    df = load_raw_data(raw_data_path)
    df = validate_data(df)

    # 2. Normalisasi rating
    df, scaler = scale_ratings(df, save_scaler_path=scaler_save_path)

    # 3. Simpan CSV hasil scale
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Data scaled disimpan ke '{output_csv_path}'")

    # 4. Encode ID
    df, user_enc, product_enc = encode_ids(df)

    # 5. Split
    train_user, train_prod, y_train, test_user, test_prod, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )

    return {
        "df":              df,
        "scaler":          scaler,
        "user_encoder":    user_enc,
        "product_encoder": product_enc,
        "num_users":       df["user_encoded"].nunique(),
        "num_products":    df["product_encoded"].nunique(),
        "train": {
            "user_input":    train_user,
            "product_input": train_prod,
            "y":             y_train,
        },
        "test": {
            "user_input":    test_user,
            "product_input": test_prod,
            "y":             y_test,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Main (jalankan langsung untuk preprocessing)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    artifacts = run_preprocessing_pipeline(
        raw_data_path="../../data/sample_data.csv",
        output_csv_path="../../data/cleaned_sample_data_scaled.csv",
        scaler_save_path="../api/saved_model/rating_scaler.pkl",
    )

    print("\n=== Ringkasan Preprocessing ===")
    print(f"Total data    : {len(artifacts['df']):,} baris")
    print(f"Jumlah user   : {artifacts['num_users']:,}")
    print(f"Jumlah produk : {artifacts['num_products']:,}")
    print(f"Train samples : {len(artifacts['train']['y']):,}")
    print(f"Test samples  : {len(artifacts['test']['y']):,}")
