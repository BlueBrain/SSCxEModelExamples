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

"""Test to reproduce somatic validation results."""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal


script_dir = Path(__file__).parent

sys.path.append(str(script_dir.parent))

import model
import evaluator


def test_somatic_validation():
    """Test the objective values of somatic validation."""
    protocols_path = Path(".") / "L5TPC" / "protocols.json"
    features_path = Path(".") / "L5TPC" / "features.json"

    opt_model_params_path = (
        Path("..") / "optimization" / "opt_module" / "config" / "params" / "pyr.json"
    )

    opt_pickle_path = (
        Path("..")
        / "optimization"
        / "opt_module"
        / "checkpoints"
        / "run.a6e707a"
        / "cADpyr_L5TPC_1.pkl"
    )

    with open(opt_pickle_path, "rb") as f:
        opt_pickle = pickle.load(f, encoding="latin1")

    nevals = opt_pickle["logbook"].select("nevals")
    opt_pickle = {
        "nevals": np.cumsum(nevals),
        "logbook": opt_pickle["logbook"],
        "hof": opt_pickle["halloffame"],
    }

    morphology_path = str(
        Path("..") / "optimization" / "opt_module" / "morphologies" / "C060114A5.asc"
    )

    fitness_protocols = evaluator.define_protocols(protocols_path)

    evaluator_instance = evaluator.create(
        morphology_path, opt_model_params_path, features_path, protocols_path
    )

    cell = model.create(morphology_path, opt_model_params_path)
    cell_params = [param.name for param in cell.params.values() if not param.frozen]

    responses = evaluator_instance.run_protocols(
        protocols=fitness_protocols.values(),
        param_values=dict(zip(cell_params, opt_pickle["hof"][0])),
    )

    objectives = evaluator_instance.fitness_calculator.calculate_scores(responses)
    objectives = pd.Series(objectives)

    # read gt_objectives Series from json
    gt_objectives = pd.read_json(
        Path(".") / "tests" / "data" / "gt_objectives.json", typ="series"
    )

    ap_amplitude_keys = [k for k in objectives.keys() if "AP_amplitude" in k]
    for k in ap_amplitude_keys:
        assert np.allclose(objectives[k], gt_objectives[k], atol=0.5)

    min_voltage_keys = [k for k in objectives.keys() if "min_voltage" in k]
    for k in min_voltage_keys:
        assert np.allclose(objectives[k], gt_objectives[k], atol=0.5)
