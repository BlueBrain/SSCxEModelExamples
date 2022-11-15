"""The somatic validation."""

import json
from pathlib import Path
import pickle

import numpy as np

import evaluator
import model

# get current file's parent directory
script_dir = Path(__file__).parent


def get_model_threshold() -> float:
    """Get model threshold from optimisation output."""
    opt_model_features_path = (
        script_dir.parent
        / "optimisation"
        / "opt_module"
        / "config"
        / "features"
        / "cADpyr_L5TPC.json"
    )

    with open(opt_model_features_path, "r") as f:
        opt_model_features = json.load(f)

    return opt_model_features["Threshold"]["soma.v"][0]["val"][0]


def write_corrected_protocols(model_thresh, thresh_hypamp, prot_path, output_path):
    thresh = thresh_hypamp["Threshold"]["soma.v"][0]["val"]

    with open(prot_path, "r") as file:
        prots = json.load(file)
    prots_th = prots.copy()

    # change holding
    for protocol_name, protocol_definition in prots_th.items():
        for stimulus_definition in protocol_definition["stimuli"]:

            if "APThreshold" in protocol_name:
                stimulus_definition["amp"] = (
                    model_thresh * int(protocol_name[-3:]) / 100
                )

            if "IDhyperpol" in protocol_name:
                stimulus_definition["amp2"] = (
                    model_thresh * int(protocol_name[-3:]) / 100
                )
                print(stimulus_definition["amp2"])
                old_depol = stimulus_definition["amp"]
                stimulus_definition["amp"] = (old_depol / thresh[0]) * model_thresh

            if "sAHP" in protocol_name:
                stimulus_definition["amp2"] = (
                    model_thresh * int(protocol_name[-3:]) / 100
                )
                old_depol = stimulus_definition["amp"]
                stimulus_definition["amp"] = (old_depol / thresh[0]) * model_thresh

    with open(output_path, "w") as fp:
        json.dump(prots_th, fp, indent=4)

opt_model_params_path = (
    script_dir.parent / "optimisation" / "opt_module" / "config" / "params" / "pyr.json"
)

protocols_path = script_dir / "L5TPC" / "protocols.json"

features_path = script_dir / "L5TPC" / "features.json"

# corrected_protocols_path = script_dir / "protocols_corrected.json"

# write_corrected_protocols(


opt_pickle_path = (
    script_dir.parent
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
    script_dir.parent / "optimisation" / "opt_module" / "morphologies" / "C060114A5.asc"
)


# TODO rescale the protocols using corrected_prot function


fitness_protocols = evaluator.define_protocols(protocols_path)

evaluator = evaluator.create(
    morphology_path, opt_model_params_path, features_path, protocols_path
)

cell = model.create(morphology_path, opt_model_params_path)
cell_params = [param.name for param in cell.params.values() if not param.frozen]


responses = evaluator.run_protocols(
    protocols=fitness_protocols.values(),
    param_values=dict(zip(cell_params, opt_pickle["hof"][0])),
)


breakpoint()
