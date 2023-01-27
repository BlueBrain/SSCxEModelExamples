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


"""Test the validation results on a single sample."""

import json
from pathlib import Path

import pandas as pd
from pandas import testing as tm

from lib import attenuation


def assert_dataframes_equal(a, b):
    """Makes sure two dataframes are equal."""
    keys = [
        "name",
        "sec_id",
        "x",
        "mid_diam",
        "diam",
        "distance",
        "dend_distance",
        "path_distance",
        "bAP_amp",
        "EPSP_amp_soma",
        "EPSP_amp_dend",
    ]
    assert a.shape == b.shape
    for key in keys:
        tm.assert_series_equal(a[key], b[key])


def test_validation():
    """Validate a single morphology."""
    with open(Path("tests") / "test_att_conf.json") as f:
        conf_dict = json.load(f)

    meg_output = conf_dict["meg_output"]
    morphs_path = conf_dict["morphs_path"]
    apical_points = conf_dict["apical_points"]
    emodels_path = conf_dict["emodels_path"]
    emodel_hash = conf_dict["emodel_hash"]
    emodel_seed = conf_dict["emodel_seed"]
    emodel = conf_dict["emodel"]
    mtype_morph = conf_dict["mtype_morph"]
    only_morph = conf_dict["only_morph"]
    n_morph = conf_dict["n_morph"]
    output = conf_dict["output"]

    attenuation.calculate(
        meg_output,
        morphs_path,
        apical_points,
        emodels_path,
        emodel_hash,
        emodel_seed,
        emodel,
        mtype_morph,
        only_morph,
        n_morph,
        output,
    )

    with open(Path("tests") / "output" / "ground_truth.json") as f:
        gt_dict = json.load(f)

    with open(Path("tests") / "output" / "results.json") as f:
        res_dict = json.load(f)

    gt_apical = pd.DataFrame(gt_dict["0"]["points"]["apical"])
    gt_basal = pd.DataFrame(gt_dict["0"]["points"]["basal"])
    gt_somatic = pd.DataFrame(gt_dict["0"]["points"]["somatic"])

    res_apical = pd.DataFrame(res_dict["0"]["points"]["apical"])
    res_basal = pd.DataFrame(res_dict["0"]["points"]["basal"])
    res_somatic = pd.DataFrame(res_dict["0"]["points"]["somatic"])

    assert_dataframes_equal(gt_apical, res_apical)
    assert_dataframes_equal(gt_basal, res_basal)
    assert_dataframes_equal(gt_somatic, res_somatic)
