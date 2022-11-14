import os
import json

from model import create as create_cell
from stimuli import NrnHDPulse

import bluepyopt.ephys as ephys

script_dir = os.path.dirname(__file__)
config_dir = os.path.join(script_dir, "config")


def define_protocols(path_protocols):
    """Define protocols"""

    protocol_definitions = json.load(open(path_protocols))
    protocols = {}

    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
    )

    for protocol_name, protocol_definition in protocol_definitions.items():
        # By default include somatic recording
        somav_recording = ephys.recordings.CompRecording(
            name="%s.soma.v" % protocol_name, location=soma_loc, variable="v"
        )

        recordings = [somav_recording]
        stimuli = []

        step_stimulus_definition = protocol_definition["step"]
        holding_stimulus_definition = protocol_definition["holding"]

        if "IDthresh" in protocol_name:
            stimuli.append(
                ephys.stimuli.NrnSquarePulse(
                    step_amplitude=step_stimulus_definition["amp"],
                    step_delay=step_stimulus_definition["delay"],
                    step_duration=step_stimulus_definition["duration"],
                    location=soma_loc,
                    total_duration=step_stimulus_definition["totduration"],
                )
            )
        elif "APThreshold" in protocol_name:
            # fmt: off
            stimuli.append(ephys.stimuli.NrnRampPulse(
            ramp_amplitude_start=holding_stimulus_definition['amp'],
            ramp_delay=step_stimulus_definition['delay'],
            ramp_amplitude_end=step_stimulus_definition['amp'],
            ramp_duration=step_stimulus_definition['duration'],
            location=soma_loc,
            total_duration=step_stimulus_definition['totduration']))
            # fmt: on
        elif "IDhyperpol" in protocol_name or "sAHP" in protocol_name:
            # fmt: off
            stimuli.append(NrnHDPulse(
            step_amplitude=step_stimulus_definition['long_amp'],
            step_delay=step_stimulus_definition['delay'],
            step_duration=step_stimulus_definition['duration'],
            duration_of_depol1=step_stimulus_definition['tmid']-step_stimulus_definition['delay'],
            duration_of_depol2=step_stimulus_definition['toff']-step_stimulus_definition['tmid2'],
            depol=step_stimulus_definition['amp'],
            location=soma_loc,
            total_duration=step_stimulus_definition['totduration']))
            # fmt: on

        protocols[protocol_name] = ephys.protocols.SweepProtocol(
            protocol_name, stimuli, recordings
        )

    return protocols


def define_fitness_calculator(protocols, path_features):
    """Define fitness calculator"""

    feature_definitions = json.load(open(path_features))

    objectives = []
    for protocol_name, locations in feature_definitions.items():
        for location, features in locations.items():
            for feature in features:
                efel_feature_name = feature["feature"]
                meanstd = feature["val"]
                feature_name = "%s.%s.%s" % (protocol_name, location, efel_feature_name)
                recording_names = {"": "%s.%s.v" % (protocol_name, location)}
                stimulus = protocols[protocol_name].stimuli[0]

                if "APThreshold" in protocol_name:
                    stim_start = stimulus.ramp_delay
                else:
                    stim_start = stimulus.step_delay

                if location == "soma":
                    threshold = -20
                elif "dend" in location:
                    threshold = -55

                if "APThreshold" in protocol_name:
                    stim_end = stimulus.ramp_delay + stimulus.ramp_duration
                else:
                    stim_end = stimulus.step_delay + stimulus.step_duration

                if protocol_name == "bAP":
                    stim_end = stimulus.total_duration
                elif "APThreshold" in protocol_name:
                    stim_end = stimulus.ramp_delay + stimulus.ramp_duration
                else:
                    stim_end = stimulus.step_delay + stimulus.step_duration

                features_sAHP = [
                    "AHP_depth_abs",
                    "sag_amplitude",
                    "sag_ratio1",
                    "steady_state_voltage",
                    "minimum_voltage",
                    "steady_state_voltage_stimend",
                    "AHP_depth_abs_slow",
                ]

                if protocol_name[:-4] == "IDhyperpol" and efel_feature_name in [
                    "sag_amplitude",
                    "sag_ratio1",
                    "minimum_voltage",
                    "steady_state_voltage_stimend",
                ]:
                    feature = ephys.efeatures.eFELFeature(
                        feature_name,
                        efel_feature_name=efel_feature_name,
                        recording_names=recording_names,
                        stim_start=stim_start,
                        stim_end=stim_start + stimulus.duration_of_depol1,
                        exp_mean=meanstd[0],
                        exp_std=meanstd[1],
                        threshold=threshold,
                    )
                elif protocol_name[:-4] == "sAHP":
                    if efel_feature_name in features_sAHP:
                        feature = ephys.efeatures.eFELFeature(
                            feature_name,
                            efel_feature_name=efel_feature_name,
                            recording_names=recording_names,
                            stim_start=stim_end - stimulus.duration_of_depol2,
                            stim_end=stim_end,
                            exp_mean=meanstd[0],
                            exp_std=meanstd[1],
                            threshold=threshold,
                        )
                    else:
                        feature = ephys.efeatures.eFELFeature(
                            feature_name,
                            efel_feature_name=efel_feature_name,
                            recording_names=recording_names,
                            stim_start=stim_start + stimulus.duration_of_depol1,
                            stim_end=stim_end - stimulus.duration_of_depol2,
                            exp_mean=meanstd[0],
                            exp_std=meanstd[1],
                            threshold=threshold,
                        )
                else:
                    feature = ephys.efeatures.eFELFeature(
                        feature_name,
                        efel_feature_name=efel_feature_name,
                        recording_names=recording_names,
                        stim_start=stim_start,
                        stim_end=stim_end,
                        exp_mean=meanstd[0],
                        exp_std=meanstd[1],
                        threshold=threshold,
                    )
                objective = ephys.objectives.SingletonObjective(feature_name, feature)
                # print(objective)
                objectives.append(objective)

    fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
    return fitcalc


def create(morpho_path, path_params, path_features, path_protocols):
    """Setup"""

    cell = create_cell(morpho_path, path_params)
    print(morpho_path)
    fitness_protocols = define_protocols(path_protocols)
    fitness_calculator = define_fitness_calculator(fitness_protocols, path_features)
    param_names = [param.name for param in cell.params.values() if not param.frozen]

    sim = ephys.simulators.NrnSimulator()

    return ephys.evaluators.CellEvaluator(
        cell_model=cell,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=sim,
        timeout=500,
    )
