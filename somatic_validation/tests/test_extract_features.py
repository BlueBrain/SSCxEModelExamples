"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

"""Test the extraction of features required by somatic validation."""

from types import MappingProxyType
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

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
    etype = "test_L5TPC"
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
        assert_frame_equal(results_df, ground_truth_df)
