"""Keras sequence model: a small global LSTM over 24-month windows of log tonnes.

Optional (pip install cargokit[deep]); train it on Colab's free GPU. It is evaluated
on the same backtest folds as the baselines, and is only promoted if it wins there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 24


def make_windows(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, WINDOW, 2), y (n, horizon) and lane index per sample.

    Each lane is scaled by its own window mean so lanes of different size share weights.
    The second input channel is month-of-year as a sine wave.
    """
    xs, ys, lanes = [], [], []
    for i, (_, g) in enumerate(df.sort_values("month").groupby("lane")):
        y = np.log(g["tonnes"].to_numpy().clip(min=1.0))
        season = np.sin(2 * np.pi * g["month"].dt.month.to_numpy() / 12)
        for t in range(WINDOW, len(y) - horizon + 1):
            level = y[t - WINDOW : t].mean()
            xs.append(np.stack([y[t - WINDOW : t] - level, season[t - WINDOW : t]], axis=1))
            ys.append(y[t : t + horizon] - level)
            lanes.append(i)
    return np.array(xs), np.array(ys), np.array(lanes)


def build_model(horizon: int):
    from tensorflow import keras  # imported lazily: TensorFlow is an optional extra

    inputs = keras.Input(shape=(WINDOW, 2))
    x = keras.layers.LSTM(32)(inputs)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(horizon)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
    return model


def fit_predict(train: pd.DataFrame, horizon: int, epochs: int = 60, seed: int = 0) -> pd.DataFrame:
    """Fit on all lanes and forecast `horizon` months ahead for each lane."""
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    X, y, _ = make_windows(train, horizon)
    model = build_model(horizon)
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )
    rows = []
    for lane, g in train.sort_values("month").groupby("lane"):
        y_log = np.log(g["tonnes"].to_numpy().clip(min=1.0))[-WINDOW:]
        season = np.sin(2 * np.pi * g["month"].dt.month.to_numpy()[-WINDOW:] / 12)
        level = y_log.mean()
        pred = model.predict(np.stack([y_log - level, season], axis=1)[None], verbose=0)[0]
        months = pd.date_range(
            g["month"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        rows.append(pd.DataFrame({"month": months, "lane": lane, "forecast": np.exp(pred + level)}))
    return pd.concat(rows, ignore_index=True)
