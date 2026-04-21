from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("onnxruntime")

from amazon_recsys.ml.onnx import ONNXRankerPredictor, export_xgboost_ranker_to_onnx


@pytest.mark.ranking
def test_xgboost_ranker_exports_to_onnx_with_prediction_parity(workspace_dir) -> None:
    rng = np.random.RandomState(7)
    feature_names = ["retrieval_score", "item_category_idx", "rank"]
    features = pd.DataFrame(rng.randn(12, 3).astype(np.float32), columns=feature_names)
    labels = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32)
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=4,
        max_depth=2,
        tree_method="hist",
        random_state=7,
    )
    model.fit(features, labels, group=[4, 4, 4], verbose=False)

    model_path = export_xgboost_ranker_to_onnx(
        model,
        workspace_dir / "ranker.onnx",
        n_features=len(feature_names),
    )
    predictor = ONNXRankerPredictor(model_path, feature_names)

    shuffled_features = features[["rank", "retrieval_score", "item_category_idx"]]
    actual = predictor.predict(shuffled_features)
    expected = model.predict(features)

    assert model.get_booster().feature_names == feature_names
    assert actual.shape == (len(features),)
    assert not np.isnan(actual).any()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.ranking
def test_onnx_ranker_predictor_validates_input_shape(workspace_dir) -> None:
    rng = np.random.RandomState(11)
    feature_names = ["f0", "f1", "f2"]
    features = pd.DataFrame(rng.randn(12, 3).astype(np.float32), columns=feature_names)
    labels = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32)
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=3,
        max_depth=2,
        tree_method="hist",
        random_state=11,
    )
    model.fit(features, labels, group=[4, 4, 4], verbose=False)
    model_path = export_xgboost_ranker_to_onnx(
        model,
        workspace_dir / "ranker-shape.onnx",
        n_features=len(feature_names),
    )
    predictor = ONNXRankerPredictor(model_path, feature_names)

    assert predictor.predict(features.iloc[[0]]).shape == (1,)
    assert predictor.predict(features.to_numpy(dtype=np.float32)).shape == (len(features),)
    with pytest.raises(ValueError, match="expected 3 features"):
        predictor.predict(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="missing feature columns"):
        predictor.predict(features.drop(columns=["f2"]))
