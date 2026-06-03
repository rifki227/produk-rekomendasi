import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import keras
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/api_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("ncf-api")


# ─────────────────────────────────────────────
#  Custom Components (wajib didaftarkan sebelum load model)
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class ProductInteractionLayer(tf.keras.layers.Layer):
    """Custom element-wise interaction layer antara embedding user dan produk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        user_vector, item_vector = inputs
        return user_vector * item_vector


@tf.keras.utils.register_keras_serializable()
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# ─────────────────────────────────────────────
#  Path Artefak
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "saved_model", "ncf_product_recommendation.keras")
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "rating_scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "cleaned_sample_data_scaled.csv")

# Kontainer global artefak
_artifacts: dict = {}


def _load_artifacts() -> None:
    """
    Muat model Keras, scaler, data transaksi, dan encoder ke memori.
    Dipanggil sekali saat startup agar tidak reload setiap request.
    """
    logger.info("Memuat model NCF …")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "ProductInteractionLayer": ProductInteractionLayer,
            "root_mean_squared_error_loss": root_mean_squared_error_loss,
        },
    )
    logger.info("Model berhasil dimuat.")

    logger.info("Memuat scaler …")
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler berhasil dimuat.")

    logger.info("Memuat data transaksi …")
    df = pd.read_csv(DATA_PATH)
    logger.info(
        f"Data dimuat: {len(df):,} baris | "
        f"{df['user_id'].nunique():,} user | "
        f"{df['product_id'].nunique():,} produk"
    )

    # Encoder difit dari seluruh data agar konsisten dengan training
    user_enc = LabelEncoder().fit(df["user_id"])
    product_enc = LabelEncoder().fit(df["product_id"])

    _artifacts.update(
        {
            "model": model,
            "scaler": scaler,
            "df": df,
            "user_enc": user_enc,
            "product_enc": product_enc,
            "all_products": df["product_id"].unique().tolist(),
            "all_users": df["user_id"].unique().tolist(),
        }
    )
    logger.info("Semua artefak siap.")


# ─────────────────────────────────────────────
#  Lifespan (pengganti on_event deprecated)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    _load_artifacts()
    yield
    logger.info("API dimatikan, membersihkan sumber daya …")


# ─────────────────────────────────────────────
#  Inisialisasi FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="NCF Product Recommendation API",
    description=(
        "Sistem Rekomendasi Produk E-commerce menggunakan "
        "Neural Collaborative Filtering (NCF). "
        "Capstone Project CC26-PRU466."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Middleware — catat waktu eksekusi
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1_000
    response.headers["X-Response-Time-ms"] = f"{elapsed:.1f}"
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f} ms)")
    return response


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────
def _require_artifacts():
    if not _artifacts:
        raise HTTPException(status_code=503, detail="Model belum siap. Cek log server.")


def _inverse_scale(scaled_value: float) -> float:
    """Kembalikan rating dari skala [0,1] ke skala asli [1,5]."""
    return float(_artifacts["scaler"].inverse_transform([[scaled_value]])[0][0])


def _get_user_or_404(user_id: str) -> str:
    if user_id not in _artifacts["user_enc"].classes_:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' tidak ditemukan dalam dataset.",
        )
    return user_id


def _get_product_or_404(product_id: str) -> str:
    if product_id not in _artifacts["product_enc"].classes_:
        raise HTTPException(
            status_code=404,
            detail=f"Produk '{product_id}' tidak ditemukan dalam dataset.",
        )
    return product_id


# ─────────────────────────────────────────────
#  Schema Pydantic
# ─────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="A1PSUH0U1FPQ6R", description="ID user (string Amazon-style)")
    top_k: int   = Field(10, ge=1, le=100, description="Jumlah rekomendasi (maks 100)")
    include_rated: bool  = Field(False, description="Sertakan produk yang sudah pernah di-rating")
    min_rating: float    = Field(0.0, ge=0.0, le=5.0, description="Filter rating minimum (skala 1–5)")


class BatchRecommendationRequest(BaseModel):
    user_ids: list[str] = Field(..., max_length=50, description="Daftar user_id (maks 50)")
    top_k: int          = Field(10, ge=1, le=50)


class PredictRatingRequest(BaseModel):
    user_id:    str = Field(..., example="A1PSUH0U1FPQ6R")
    product_id: str = Field(..., example="B002QXZPFE")


# ─────────────────────────────────────────────
#  Routes — Info & Health
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "NCF Product Recommendation API",
        "version": "1.0.0",
        "project": "CC26-PRU466",
        "docs": "/docs",
        "endpoints": {
            "health":           "GET  /health",
            "stats":            "GET  /stats",
            "recommend":        "POST /recommend",
            "recommend_batch":  "POST /recommend/batch",
            "predict_rating":   "POST /predict",
            "user_history":     "GET  /users/{user_id}/history",
            "similar_users":    "GET  /users/{user_id}/similar",
            "popular_products": "GET  /products/popular",
            "product_info":     "GET  /products/{product_id}",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    ready = bool(_artifacts)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": "ncf_product_recommendation.keras",
        },
    )


@app.get("/stats", tags=["Info"])
def stats():
    _require_artifacts()
    df     = _artifacts["df"]
    scaler = _artifacts["scaler"]
    rating = df["rating"]
    return {
        "dataset": {
            "total_interactions": int(len(df)),
            "total_users":        int(df["user_id"].nunique()),
            "total_products":     int(df["product_id"].nunique()),
            "date_range":         {"start": df["timestamp"].min(), "end": df["timestamp"].max()},
        },
        "ratings": {
            "mean_scaled":   round(float(rating.mean()), 4),
            "mean_original": round(_inverse_scale(float(rating.mean())), 4),
            "min_original":  round(_inverse_scale(float(rating.min())), 2),
            "max_original":  round(_inverse_scale(float(rating.max())), 2),
        },
        "model": {
            "architecture":   "Neural Collaborative Filtering (NCF)",
            "total_users":    int(_artifacts["user_enc"].classes_.shape[0]),
            "total_products": int(_artifacts["product_enc"].classes_.shape[0]),
            "output_scale":   "0–1 → diinvers ke 1–5",
        },
    }


# ─────────────────────────────────────────────
#  Routes — Rekomendasi
# ─────────────────────────────────────────────
@app.post("/recommend", tags=["Rekomendasi"])
def recommend(req: RecommendationRequest):
    """
    Rekomendasikan produk untuk satu user berdasarkan prediksi model NCF.

    - **user_id**: ID user dalam bentuk string (contoh: `A1PSUH0U1FPQ6R`)
    - **top_k**: jumlah rekomendasi yang dikembalikan (default 10, maks 100)
    - **include_rated**: sertakan produk yang sudah pernah diberi rating
    - **min_rating**: filter produk dengan prediksi di bawah nilai ini (skala 1–5)
    """
    _require_artifacts()
    _get_user_or_404(req.user_id)

    df          = _artifacts["df"]
    model       = _artifacts["model"]
    scaler      = _artifacts["scaler"]
    user_enc    = _artifacts["user_enc"]
    product_enc = _artifacts["product_enc"]
    all_products = _artifacts["all_products"]

    # Kandidat produk
    if req.include_rated:
        candidates = all_products
    else:
        rated = set(df[df["user_id"] == req.user_id]["product_id"])
        candidates = [p for p in all_products if p not in rated]

    if not candidates:
        return {
            "status": "success",
            "user_id": req.user_id,
            "recommendations": [],
            "message": "User telah menilai semua produk yang tersedia.",
        }

    user_encoded  = int(user_enc.transform([req.user_id])[0])
    prod_encoded  = product_enc.transform(candidates).astype(np.int32)
    user_input    = np.full(len(prod_encoded), user_encoded, dtype=np.int32)

    t0 = time.perf_counter()
    preds_scaled   = model.predict([user_input, prod_encoded], batch_size=512, verbose=0).flatten()
    inference_ms   = (time.perf_counter() - t0) * 1_000

    preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    result_df = pd.DataFrame({
        "product_id":             candidates,
        "predicted_rating":       preds_original,
        "predicted_rating_scaled": preds_scaled,
    })

    if req.min_rating > 0:
        result_df = result_df[result_df["predicted_rating"] >= req.min_rating]

    top_df = result_df.nlargest(req.top_k, "predicted_rating")

    recommendations = [
        {
            "rank": i + 1,
            "product_id": row["product_id"],
            "predicted_rating": round(float(row["predicted_rating"]), 4),
            "predicted_rating_scaled": round(float(row["predicted_rating_scaled"]), 6),
        }
        for i, (_, row) in enumerate(top_df.iterrows())
    ]

    return {
        "status": "success",
        "user_id": req.user_id,
        "recommendations": recommendations,
        "meta": {
            "top_k": req.top_k,
            "candidates_evaluated": len(candidates),
            "include_rated": req.include_rated,
            "min_rating_filter": req.min_rating,
            "inference_ms": round(inference_ms, 2),
        },
    }


@app.post("/recommend/batch", tags=["Rekomendasi"])
def recommend_batch(req: BatchRecommendationRequest):
    """
    Rekomendasi untuk beberapa user sekaligus (maks 50 user per request).
    """
    _require_artifacts()

    if len(req.user_ids) == 0:
        raise HTTPException(status_code=400, detail="'user_ids' tidak boleh kosong.")

    df          = _artifacts["df"]
    model       = _artifacts["model"]
    scaler      = _artifacts["scaler"]
    user_enc    = _artifacts["user_enc"]
    product_enc = _artifacts["product_enc"]
    all_products = _artifacts["all_products"]

    results, not_found = [], []

    for uid in req.user_ids:
        if uid not in user_enc.classes_:
            not_found.append(uid)
            continue

        rated      = set(df[df["user_id"] == uid]["product_id"])
        candidates = [p for p in all_products if p not in rated]
        if not candidates:
            results.append({"user_id": uid, "recommendations": []})
            continue

        user_enc_id = int(user_enc.transform([uid])[0])
        prod_enc    = product_enc.transform(candidates).astype(np.int32)
        user_inp    = np.full(len(prod_enc), user_enc_id, dtype=np.int32)

        preds_s = model.predict([user_inp, prod_enc], batch_size=512, verbose=0).flatten()
        preds_o = scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()

        idx_top = np.argsort(preds_o)[::-1][: req.top_k]
        recs = [
            {
                "rank": rank + 1,
                "product_id": candidates[i],
                "predicted_rating": round(float(preds_o[i]), 4),
            }
            for rank, i in enumerate(idx_top)
        ]
        results.append({"user_id": uid, "recommendations": recs})

    return {
        "status": "success",
        "results": results,
        "top_k": req.top_k,
        "processed": len(results),
        "not_found": not_found,
    }


# ─────────────────────────────────────────────
#  Routes — Prediksi Rating
# ─────────────────────────────────────────────
@app.post("/predict", tags=["Prediksi"])
def predict_rating(req: PredictRatingRequest):
    """
    Prediksi rating untuk satu pasang user–produk.
    Jika user sudah pernah memberi rating pada produk tersebut,
    actual_rating akan diisi (berguna untuk evaluasi).
    """
    _require_artifacts()
    _get_user_or_404(req.user_id)
    _get_product_or_404(req.product_id)

    df          = _artifacts["df"]
    model       = _artifacts["model"]
    scaler      = _artifacts["scaler"]
    user_enc    = _artifacts["user_enc"]
    product_enc = _artifacts["product_enc"]

    user_inp    = np.array([int(user_enc.transform([req.user_id])[0])],    dtype=np.int32)
    product_inp = np.array([int(product_enc.transform([req.product_id])[0])], dtype=np.int32)

    pred_scaled   = float(model.predict([user_inp, product_inp], verbose=0)[0][0])
    pred_original = _inverse_scale(pred_scaled)

    # Cek rating aktual (jika ada)
    existing      = df[(df["user_id"] == req.user_id) & (df["product_id"] == req.product_id)]
    actual_rating = None
    if not existing.empty:
        actual_rating = round(_inverse_scale(float(existing["rating"].iloc[0])), 2)

    return {
        "status": "success",
        "user_id":    req.user_id,
        "product_id": req.product_id,
        "predicted_rating":        round(pred_original, 4),
        "predicted_rating_scaled": round(pred_scaled, 6),
        "actual_rating":  actual_rating,
        "already_rated":  actual_rating is not None,
    }


# ─────────────────────────────────────────────
#  Routes — User
# ─────────────────────────────────────────────
@app.get("/users/{user_id}/history", tags=["User"])
def user_history(
    user_id: str,
    page:     int   = Query(1, ge=1),
    per_page: int   = Query(20, ge=1, le=100),
    sort:     str   = Query("date_desc", regex="^(rating_desc|rating_asc|date_desc|date_asc)$"),
):
    """
    Riwayat rating seorang user dengan paginasi dan opsi pengurutan.

    **sort options**: `rating_desc`, `rating_asc`, `date_desc`, `date_asc`
    """
    _require_artifacts()
    _get_user_or_404(user_id)

    df     = _artifacts["df"]
    scaler = _artifacts["scaler"]

    user_df = df[df["user_id"] == user_id].copy()
    user_df["rating_original"] = scaler.inverse_transform(user_df[["rating"]]).flatten()

    sort_map = {
        "rating_desc": ("rating_original", False),
        "rating_asc":  ("rating_original", True),
        "date_desc":   ("timestamp",        False),
        "date_asc":    ("timestamp",        True),
    }
    col, asc = sort_map[sort]
    user_df  = user_df.sort_values(col, ascending=asc)

    total   = len(user_df)
    start   = (page - 1) * per_page
    page_df = user_df.iloc[start : start + per_page]

    history = [
        {
            "product_id":    row["product_id"],
            "rating":        round(float(row["rating_original"]), 2),
            "rating_scaled": round(float(row["rating"]), 6),
            "timestamp":     row["timestamp"],
        }
        for _, row in page_df.iterrows()
    ]

    return {
        "status":  "success",
        "user_id": user_id,
        "history": history,
        "pagination": {
            "page":        page,
            "per_page":    per_page,
            "total":       total,
            "total_pages": (total + per_page - 1) // per_page,
        },
        "summary": {
            "total_rated": total,
            "avg_rating":  round(float(user_df["rating_original"].mean()), 2),
            "min_rating":  round(float(user_df["rating_original"].min()), 2),
            "max_rating":  round(float(user_df["rating_original"].max()), 2),
        },
    }


@app.get("/users/{user_id}/similar", tags=["User"])
def similar_users(
    user_id: str,
    top_n:   int = Query(5, ge=1, le=20),
):
    """
    Temukan user dengan selera serupa berdasarkan Jaccard similarity
    pada produk yang sama-sama diberi rating tinggi (≥ 3.4 bintang).
    """
    _require_artifacts()
    _get_user_or_404(user_id)

    df = _artifacts["df"]

    HIGH_THRESHOLD = 0.6   # ≈ 3.4 bintang pada skala asli
    user_high = set(df[(df["user_id"] == user_id) & (df["rating"] >= HIGH_THRESHOLD)]["product_id"])

    if not user_high:
        return {
            "status":       "success",
            "user_id":      user_id,
            "similar_users": [],
            "message":      "User belum memiliki rating tinggi yang cukup untuk dijadikan acuan.",
        }

    sims = []
    for uid, grp in df[df["user_id"] != user_id].groupby("user_id"):
        other_high = set(grp[grp["rating"] >= HIGH_THRESHOLD]["product_id"])
        if not other_high:
            continue
        inter = len(user_high & other_high)
        if inter == 0:
            continue
        sims.append({
            "user_id":           uid,
            "jaccard_similarity": round(inter / len(user_high | other_high), 4),
            "common_highly_rated": inter,
        })

    sims.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

    return {
        "status":       "success",
        "user_id":      user_id,
        "similar_users": sims[:top_n],
        "meta": {
            "high_rated_products_count": len(user_high),
            "threshold_rating": f"≥ {round(_inverse_scale(HIGH_THRESHOLD), 1)} bintang",
        },
    }


# ─────────────────────────────────────────────
#  Routes — Produk
# ─────────────────────────────────────────────
@app.get("/products/popular", tags=["Produk"])
def popular_products(
    top_n:       int = Query(10, ge=1, le=100),
    min_reviews: int = Query(5, ge=1),
):
    """
    Daftar produk paling populer menggunakan Bayesian average rating
    (menghindari bias produk dengan sedikit review).
    """
    _require_artifacts()
    df     = _artifacts["df"]
    scaler = _artifacts["scaler"]

    agg = (
        df.groupby("product_id")["rating"]
        .agg(count="count", mean="mean")
        .reset_index()
    )
    agg = agg[agg["count"] >= min_reviews]
    if agg.empty:
        raise HTTPException(status_code=404, detail=f"Tidak ada produk dengan minimal {min_reviews} review.")

    agg["rating_original"] = scaler.inverse_transform(agg[["mean"]]).flatten()

    # Bayesian average
    C = agg["count"].mean()
    M = agg["rating_original"].mean()
    agg["bayesian_avg"] = (agg["count"] * agg["rating_original"] + C * M) / (agg["count"] + C)

    top = agg.nlargest(top_n, "bayesian_avg")

    return {
        "status": "success",
        "popular_products": [
            {
                "rank":                i + 1,
                "product_id":          row["product_id"],
                "avg_rating":          round(float(row["rating_original"]), 4),
                "bayesian_avg_rating": round(float(row["bayesian_avg"]), 4),
                "total_reviews":       int(row["count"]),
            }
            for i, (_, row) in enumerate(top.iterrows())
        ],
        "meta": {"top_n": top_n, "min_reviews_filter": min_reviews},
    }


@app.get("/products/{product_id}", tags=["Produk"])
def product_info(product_id: str):
    """Statistik dan distribusi rating untuk satu produk."""
    _require_artifacts()
    _get_product_or_404(product_id)

    df     = _artifacts["df"]
    scaler = _artifacts["scaler"]

    prod_df = df[df["product_id"] == product_id].copy()
    prod_df["rating_original"] = scaler.inverse_transform(prod_df[["rating"]]).flatten()
    ratings = prod_df["rating_original"]

    distribution = (
        ratings.round().astype(int).clip(1, 5)
        .value_counts().sort_index().to_dict()
    )

    return {
        "status":     "success",
        "product_id": product_id,
        "stats": {
            "total_reviews": int(len(prod_df)),
            "avg_rating":    round(float(ratings.mean()), 4),
            "min_rating":    round(float(ratings.min()), 2),
            "max_rating":    round(float(ratings.max()), 2),
            "std_rating":    round(float(ratings.std()), 4),
        },
        "rating_distribution": {str(k): v for k, v in distribution.items()},
        "first_reviewed":      prod_df["timestamp"].min(),
        "last_reviewed":       prod_df["timestamp"].max(),
    }


# ─────────────────────────────────────────────
#  Exception Handler Global
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Terjadi kesalahan internal server."},
    )


# ─────────────────────────────────────────────
#  Entry Point (development)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level="info",
    )
