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

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

"""Feature extraction using bluepyefe."""
import json
from typing import List

import bluepyefe as bpefe


def remove_keys_from_dict(dict_var: dict, keys: List[str]) -> dict:
    """Recursively remove a given key from the dict."""
    for key in keys:
        if key in dict_var:
            del dict_var[key]

    for v in dict_var.values():
        if isinstance(v, dict):
            remove_keys_from_dict(v, keys)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    remove_keys_from_dict(item, keys)

    return dict_var


def test_feature_extraction():
    """Regression test to make sure future changes won't affect the results."""
    etype = "cADpyr"
    test_cell = "C060109A1-SR-C1"

    with open("feature_extraction_config.json", "r") as json_file:
        config = json.load(json_file)

    extractor = bpefe.Extractor(etype, config)
    extractor.create_dataset()

    # does not produce output, stores object attributes
    extractor.extract_features(threshold=-30)
    extractor.mean_features()

    extractor.analyse_threshold()
    extractor.feature_config_cells()

    with open(f"{etype}/{test_cell}/features.json", "r") as json_file:
        results = json.load(json_file)

    with open(f"tests/data/{test_cell}-ground-truth.json", "r") as json_file:
        ground_truth = json.load(json_file)

    # remove the feature id from both dictionaries
    results = remove_keys_from_dict(results, ["fid"])
    ground_truth = remove_keys_from_dict(ground_truth, ["fid"])

    assert results == ground_truth
