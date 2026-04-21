from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ONNX_TARGET_OPSET = 15


class ONNXRankerPredictor:
    def __init__(
        self,
        model_path: str | Path,
        feature_names: Iterable[str],
        *,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_names = list(feature_names)
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._input_name: str | None = None

    @property
    def session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError as exc:
                raise RuntimeError("onnxruntime is required to serve ONNX bundles.") from exc
            self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        return self._session

    @property
    def input_name(self) -> str:
        if self._input_name is None:
            self._input_name = self.session.get_inputs()[0].name
        return self._input_name

    def _matrix_from_input(self, features: object) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in features.columns]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(f"ONNX ranker input is missing feature columns: {preview}")
            matrix = features.loc[:, self.feature_names].to_numpy(dtype=np.float32)
        else:
            matrix = np.asarray(features, dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError("ONNX ranker input must be a 2D feature matrix.")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"ONNX ranker expected {len(self.feature_names)} features, "
                f"received {matrix.shape[1]}."
            )
        return matrix.astype(np.float32, copy=False)

    def predict(self, features: object, *args: object, **kwargs: object) -> np.ndarray:
        matrix = self._matrix_from_input(features)
        outputs = self.session.run(None, {self.input_name: matrix})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


def clone_booster_for_onnx(model: object):
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost is required to export ONNX ranker bundles.") from exc

    booster = model.get_booster()
    clone = xgb.Booster()
    clone.load_model(bytearray(booster.save_raw(raw_format="json")))
    clone.feature_names = None
    clone.feature_types = None
    return clone


def export_xgboost_ranker_to_onnx(
    model: object,
    output_path: str | Path,
    *,
    n_features: int,
    target_opset: int = ONNX_TARGET_OPSET,
) -> Path:
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        from onnxmltools.convert.xgboost._parse import WrappedBooster
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxmltools is required to export ONNX ranker bundles.") from exc

    booster = clone_booster_for_onnx(model)
    wrapped = WrappedBooster(booster)
    wrapped.operator_name = "XGBRegressor"
    wrapped.kwargs["objective"] = "rank:ndcg"
    wrapped.kwargs["num_class"] = 0
    wrapped.kwargs["n_targets"] = 1

    onnx_model = convert_xgboost(
        wrapped,
        initial_types=[("input", FloatTensorType([None, int(n_features)]))],
        target_opset=target_opset,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(onnx_model.SerializeToString())
    return path
