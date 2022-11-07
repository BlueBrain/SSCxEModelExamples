"""Test the extraction of features required by somatic validation."""

from types import MappingProxyType
import json
from pathlib import Path

import pandas as pd

from ..extract_features import (
    translate_legacy_targets,
    get_files_metadata,
    extract_efeatures,
)


def features_df(features_config: dict, protocol: str) -> pd.DataFrame:
    """Returns the dataframe containing features for the given protocol."""
    df = pd.DataFrame(features_config[protocol]["soma"])
    df["mean"] = df["val"].apply(lambda x: x[0])
    df["variance"] = df["val"].apply(lambda x: x[1])
    df = df.drop(["val", "efeature_name"], axis=1)
    return df


def test_extract_efeatures():
    """Test the feature values."""

    traces_dir = Path("..") / "feature_extraction" / "input-traces"

    cell_ids = (
        "C060109A1-SR-C1",
        "C060109A2-SR-C1",
        "C060109A3-SR-C1",
        "C070109A4-C1",
        "C080501A5-SR-C1",
        "C080501B2-SR-C1",
    )

    with open("experiments.json", "r") as f:
        experiments = MappingProxyType(json.load(f))

    targets = translate_legacy_targets(experiments)

    files_metadata = get_files_metadata(traces_dir, cell_ids, experiments)
    etype = "test_L5PC"
    protocols_rheobase = ["IDthresh", "IDRest"]
    extract_efeatures(
        etype, files_metadata, targets, protocols_rheobase, plot=False, per_cell=False
    )

    with open(f"{etype}/features.json", "r") as json_file:
        results = json.load(json_file)

    with open("tests/data/gt_features_from_all_cells.json", "r") as json_file:
        ground_truth = json.load(json_file)

    for protocol in results:
        results_df = features_df(results, protocol)
        ground_truth_df = features_df(ground_truth, protocol)
        assert results_df.equals(ground_truth_df)
