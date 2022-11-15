"""The somatic validation."""

import json
from pathlib import Path
import pickle

import numpy as np

import evaluator
import model

# get current file's parent directory
script_dir = Path(__file__).parent


def get_opt_threshold() -> float:
    """Get threshold from optimisation output."""
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


def get_model_threshold() -> float:
    """Returns the model threshold contained in a static file."""
    threshold_fpath = script_dir / "thresholds" / "models_thresh_all_pyr.json"
    with open(threshold_fpath, "r") as f:
        thresholds = json.load(f)

    return thresholds["cADpyr_L5TPC1"][0]["thresh"]


def write_corrected_protocols(model_thresh, opt_thresh, prot_path, output_path):
    """Rescales the amplitudes of protocols Writes the output to file."""
    with open(prot_path, "r") as file:
        prots = json.load(file)

    for protocol_name, protocol_definition in prots.items():
        if "APThreshold" in protocol_name:
            protocol_definition["step"]["amp"] = (
                model_thresh * int(protocol_name[-3:]) / 100
            )

        if "IDhyperpol" in protocol_name:
            protocol_definition["step"]["long_amp"] = (
                model_thresh * int(protocol_name[-3:]) / 100
            )
            print(protocol_definition["step"]["long_amp"])
            old_depol = protocol_definition["step"]["amp"]
            protocol_definition["step"]["amp"] = (old_depol / opt_thresh) * model_thresh

        if "sAHP" in protocol_name:
            protocol_definition["step"]["long_amp"] = (
                model_thresh * int(protocol_name[-3:]) / 100
            )
            old_depol = protocol_definition["step"]["amp"]
            protocol_definition["step"]["amp"] = (old_depol / opt_thresh) * model_thresh

    with open(output_path, "w") as fp:
        json.dump(prots, fp, indent=4)

opt_model_params_path = (
    script_dir.parent / "optimisation" / "opt_module" / "config" / "params" / "pyr.json"
)

protocols_path = script_dir / "L5TPC" / "protocols.json"

features_path = script_dir / "L5TPC" / "features.json"

model_threshold = get_model_threshold()
opt_threshold = get_opt_threshold()

corrected_protocols_path = script_dir / "protocols_corrected.json"

write_corrected_protocols(model_threshold, opt_threshold, protocols_path, corrected_protocols_path)


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


corrected_protocols_path = str(corrected_protocols_path)

fitness_protocols = evaluator.define_protocols(corrected_protocols_path)

evaluator = evaluator.create(
    morphology_path, opt_model_params_path, features_path, corrected_protocols_path
)

cell = model.create(morphology_path, opt_model_params_path)
cell_params = [param.name for param in cell.params.values() if not param.frozen]


responses = evaluator.run_protocols(
    protocols=fitness_protocols.values(),
    param_values=dict(zip(cell_params, opt_pickle["hof"][0])),
)

objectives = evaluator.fitness_calculator.calculate_scores(responses)
