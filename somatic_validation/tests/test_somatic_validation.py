"""Test to reproduce somatic validation results."""

import model
import evaluator
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal


script_dir = Path(__file__).parent

sys.path.append(str(script_dir.parent))


def test_somatic_validation():
    """Test the objective values of somatic validation."""
    protocols_path = Path(".") / "L5TPC" / "protocols.json"
    features_path = Path(".") / "L5TPC" / "features.json"

    opt_model_params_path = (
        Path("..") / "optimisation" / "opt_module" / "config" / "params" / "pyr.json"
    )

    opt_pickle_path = (
        Path("..")
        / "optimisation"
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
        Path("..") / "optimisation" / "opt_module" / "morphologies" / "C060114A5.asc"
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

    # save objectives Series as json
    objectives.to_json("tests/data/gt_objectives.json")

    # read gt_objectives Series from json
    gt_objectives = pd.read_json(
        Path("tests") / "data" / " gt_objectives.json", typ="series"
    )

    assert_series_equal(objectives, gt_objectives)
