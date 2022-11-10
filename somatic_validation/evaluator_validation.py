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

        for stimulus_definition in protocol_definition["stimuli"]:

            if protocol_name in ["IDthresh_200", "IDRest_200", "IV_-100.0"]:
                stimuli.append(
                    ephys.stimuli.NrnSquarePulse(
                        step_amplitude=stimulus_definition["amp"],
                        step_delay=stimulus_definition["delay"],
                        step_duration=stimulus_definition["duration"],
                        location=soma_loc,
                        total_duration=stimulus_definition["totduration"],
                    )
                )
            elif protocol_name in [
                "APThreshold_200",
                "APThreshold_170",
                "APThreshold_220",
                "APThreshold_250",
                "APThreshold_300",
                "APThreshold_150",
                "APThreshold_400",
                "APThreshold_430",
                "APThreshold_330",
                "APThreshold_390",
            ]:

                # fmt: off
                stimuli.append(ephys.stimuli.NrnRampPulse(
                ramp_amplitude_start=stimulus_definition['hypamp'],
                ramp_delay=stimulus_definition['ton'],
                ramp_amplitude_end=stimulus_definition['amp'],
                ramp_duration=stimulus_definition['toff']-stimulus_definition['ton'],
                location=soma_loc,
                total_duration=stimulus_definition['tend']))
                # fmt: on
            else:
                # fmt: off
                stimuli.append(NrnHDPulse(
                step_amplitude=stimulus_definition['amp2'],
                step_delay=stimulus_definition['ton'],
                step_duration=stimulus_definition['toff']-stimulus_definition['ton'],
                duration_of_depol1=stimulus_definition['tmid']-stimulus_definition['ton'],
                duration_of_depol2=stimulus_definition['toff']-stimulus_definition['tmid2'],
                depol=stimulus_definition['amp'],
                location=soma_loc,
                total_duration=stimulus_definition['tend']))
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
            for efel_feature_name, meanstd in features.items():
                feature_name = "%s.%s.%s" % (protocol_name, location, efel_feature_name)
                recording_names = {"": "%s.%s.v" % (protocol_name, location)}
                stimulus = protocols[protocol_name].stimuli[0]

                if protocol_name in [
                    "APThreshold_100",
                    "APThreshold_170",
                    "APThreshold_220",
                    "APThreshold_250",
                    "APThreshold_200",
                    "APThreshold_300",
                    "APThreshold_150",
                    "APThreshold_400",
                    "APThreshold_330",
                    "APThreshold_430",
                    "APThreshold_390",
                ]:
                    stim_start = stimulus.ramp_delay
                else:
                    stim_start = stimulus.step_delay

                if location == "soma":
                    threshold = -20
                elif "dend" in location:
                    threshold = -55

                if protocol_name in [
                    "APThreshold_100",
                    "APThreshold_170",
                    "APThreshold_220",
                    "APThreshold_250",
                    "APThreshold_200",
                    "APThreshold_300",
                    "APThreshold_150",
                    "APThreshold_400",
                    "APThreshold_430",
                    "APThreshold_330",
                    "APThreshold_390",
                ]:
                    stim_end = stimulus.ramp_delay + stimulus.ramp_duration
                else:
                    stim_end = stimulus.step_delay + stimulus.step_duration

                if protocol_name == "bAP":
                    stim_end = stimulus.total_duration
                elif protocol_name in [
                    "APThreshold_100",
                    "APThreshold_170",
                    "APThreshold_200",
                    "APThreshold_250",
                    "APThreshold_220",
                    "APThreshold_300",
                    "APThreshold_150",
                    "APThreshold_400",
                    "APThreshold_430",
                    "APThreshold_330",
                    "APThreshold_390",
                ]:
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
