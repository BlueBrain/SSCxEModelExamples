"""Run simple cell optimization"""

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

import json

import bluepyopt.ephys as ephys

from . import template
from . import protocols

import logging

logger = logging.getLogger(__name__)
import os
import bluepyopt as bpopt

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)

import numpy


def read_step_protocol(
    protocol_name, protocol_definition, recordings, stochkv_det=None
):
    """Read step protocol from definition"""

    step_definitions = protocol_definition["stimuli"]["step"]
    if isinstance(step_definitions, dict):
        step_definitions = [step_definitions]

    step_stimuli = []
    for step_definition in step_definitions:
        step_stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=step_definition["amp"],
            step_delay=step_definition["delay"],
            step_duration=step_definition["duration"],
            location=soma_loc,
            total_duration=step_definition["totduration"],
        )
        step_stimuli.append(step_stim)

    if "holding" in protocol_definition["stimuli"]:
        holding_definition = protocol_definition["stimuli"]["holding"]
        holding_stimulus = ephys.stimuli.NrnSquarePulse(
            step_amplitude=holding_definition["amp"],
            step_delay=holding_definition["delay"],
            step_duration=holding_definition["duration"],
            location=soma_loc,
            total_duration=holding_definition["totduration"],
        )
    else:
        holding_stimulus = None

    if stochkv_det is None:
        stochkv_det = (
            step_definition["stochkv_det"] if "stochkv_det" in step_definition else None
        )

    return protocols.StepProtocol(
        name=protocol_name,
        step_stimuli=step_stimuli,
        holding_stimulus=holding_stimulus,
        recordings=recordings,
        stochkv_det=stochkv_det,
    )


def read_step_threshold_protocol(
    protocol_name, protocol_definition, recordings, stochkv_det=None
):
    """Read step protocol from definition"""

    step_definitions = protocol_definition["stimuli"]["step"]
    if isinstance(step_definitions, dict):
        step_definitions = [step_definitions]

    step_stimuli = []
    for step_definition in step_definitions:
        step_stim = ephys.stimuli.NrnSquarePulse(
            step_delay=step_definition["delay"],
            step_duration=step_definition["duration"],
            location=soma_loc,
            total_duration=step_definition["totduration"],
        )
        step_stimuli.append(step_stim)

    holding_stimulus = ephys.stimuli.NrnSquarePulse(
        step_delay=0.0,
        step_duration=step_definition["totduration"],
        location=soma_loc,
        total_duration=step_definition["totduration"],
    )

    if stochkv_det is None:
        stochkv_det = (
            step_definition["stochkv_det"] if "stochkv_det" in step_definition else None
        )

    return protocols.StepThresholdProtocol(
        name=protocol_name,
        step_stimuli=step_stimuli,
        holding_stimulus=holding_stimulus,
        thresh_perc=step_definition["thresh_perc"],
        recordings=recordings,
        stochkv_det=stochkv_det,
    )


class NrnSomaDistanceCompLocation(ephys.locations.NrnSomaDistanceCompLocation):
    def __init__(
        self,
        name,
        soma_distance=None,
        seclist_name=None,
        comment="",
        do_simplify_morph=False,
    ):
        super(NrnSomaDistanceCompLocation, self).__init__(
            name, soma_distance, seclist_name, comment
        )
        self.do_simplify_morph = do_simplify_morph

    def instantiate(self, sim=None, icell=None):
        """Find the instantiate compartment"""

        soma = icell.soma[0]

        sim.neuron.h.distance(0, 0.5, sec=soma)

        iseclist = getattr(icell, self.seclist_name)

        icomp = None

        for isec in iseclist:
            start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
            end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

            min_distance = min(start_distance, end_distance)
            max_distance = max(start_distance, end_distance)

            if min_distance <= self.soma_distance <= end_distance:
                comp_x = float(self.soma_distance - min_distance) / (
                    max_distance - min_distance
                )

                if self.do_simplify_morph:
                    isec.nseg = 1 + 2 * int(isec.L / 40.0)
                icomp = isec(comp_x)
                seccomp = isec
                break

        if icomp is None:
            raise ephys.locations.EPhysLocInstantiateException(
                "No comp found at %s distance from soma" % self.soma_distance
            )

        print(
            (
                "Using %s at distance %f, nseg %f, length %f"
                % (
                    icomp,
                    sim.neuron.h.distance(1, comp_x, sec=seccomp),
                    seccomp.nseg,
                    end_distance - start_distance,
                )
            )
        )

        return icomp


class NrnSomaDistanceCompLocationApical(ephys.locations.NrnSomaDistanceCompLocation):
    def __init__(
        self,
        name,
        soma_distance=None,
        seclist_name=None,
        comment="",
        apical_point_isec=None,
        do_simplify_morph=False,
    ):
        super(NrnSomaDistanceCompLocationApical, self).__init__(
            name, soma_distance, seclist_name, comment
        )
        self.apical_point_isec = apical_point_isec
        self.do_simplify_morph = do_simplify_morph

    def instantiate(self, sim=None, icell=None):
        """Find the instantiate compartment"""

        if self.do_simplify_morph:
            soma = icell.soma[0]

            sim.neuron.h.distance(0, 0.5, sec=soma)

            iseclist = getattr(icell, self.seclist_name)

            icomp = None

            for isec in iseclist:
                start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
                end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

                min_distance = min(start_distance, end_distance)
                max_distance = max(start_distance, end_distance)

                if min_distance <= self.soma_distance <= end_distance:
                    comp_x = float(self.soma_distance - min_distance) / (
                        max_distance - min_distance
                    )

                    isec.nseg = 1 + 2 * int(isec.L / 100.0)
                    icomp = isec(comp_x)
                    seccomp = isec
                    break

            if icomp is None:
                raise ephys.locations.EPhysLocInstantiateException(
                    "No comp found at %s distance from soma" % self.soma_distance
                )

            print(
                (
                    "Using %s at distance %f, nseg %f, length %f"
                    % (
                        icomp,
                        sim.neuron.h.distance(1, comp_x, sec=seccomp),
                        seccomp.nseg,
                        end_distance - start_distance,
                    )
                )
            )

        else:
            if self.apical_point_isec is None:
                raise ephys.locations.EPhysLocInstantiateException(
                    "No apical point was given"
                )

            apical_branch = []
            section = icell.apic[self.apical_point_isec]
            while True:
                name = str(section.name()).split(".")[-1]
                if name == "soma[0]":
                    break
                apical_branch.append(section)

                if sim.neuron.h.SectionRef(sec=section).has_parent():
                    section = sim.neuron.h.SectionRef(sec=section).parent
                else:
                    raise ephys.locations.EPhysLocInstantiateException(
                        "soma[0] was not reached from apical point"
                    )

            soma = icell.soma[0]

            sim.neuron.h.distance(0, 0.5, sec=soma)

            icomp = None

            for isec in apical_branch:
                start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
                end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

                min_distance = min(start_distance, end_distance)
                max_distance = max(start_distance, end_distance)

                if min_distance <= self.soma_distance <= end_distance:
                    comp_x = float(self.soma_distance - min_distance) / (
                        max_distance - min_distance
                    )

                    icomp = isec(comp_x)
                    seccomp = isec

            if icomp is None:
                raise ephys.locations.EPhysLocInstantiateException(
                    "No comp found at %s distance from soma" % self.soma_distance
                )

            print(
                (
                    "Using %s at distance %f"
                    % (icomp, sim.neuron.h.distance(1, comp_x, sec=seccomp))
                )
            )

        return icomp


def define_protocols(
    protocols_filename,
    stochkv_det=None,
    runopt=False,
    prefix="",
    apical_point_isec=None,
    stage=None,
    do_simplify_morph=False,
):
    """Define protocols"""

    with open(
        os.path.join(os.path.dirname(__file__), "..", protocols_filename)
    ) as protocol_file:
        protocol_definitions = json.load(protocol_file)

    if "__comment" in protocol_definitions:
        del protocol_definitions["__comment"]

    protocols_dict = {}

    for protocol_name, protocol_definition in protocol_definitions.items():
        if ("stage" in protocol_definition) and (stage is not None) and (stage > 0):
            if stage not in protocol_definition["stage"]:
                continue  # protocol not used in this stage

        if protocol_name not in ["Main", "RinHoldcurrent"]:
            # By default include somatic recording
            somav_recording = ephys.recordings.CompRecording(
                name="%s.%s.soma.v" % (prefix, protocol_name),
                location=soma_loc,
                variable="v",
            )

            recordings = [somav_recording]

            if "extra_recordings" in protocol_definition:
                for recording_definition in protocol_definition["extra_recordings"]:
                    if recording_definition["type"] == "somadistance":
                        location = NrnSomaDistanceCompLocation(
                            name=recording_definition["name"],
                            soma_distance=recording_definition["somadistance"],
                            seclist_name=recording_definition["seclist_name"],
                            do_simplify_morph=do_simplify_morph,
                        )

                    elif recording_definition["type"] == "somadistanceapic":
                        location = NrnSomaDistanceCompLocationApical(
                            name=recording_definition["name"],
                            soma_distance=recording_definition["somadistance"],
                            seclist_name=recording_definition["seclist_name"],
                            apical_point_isec=apical_point_isec,
                            do_simplify_morph=do_simplify_morph,
                        )

                    elif recording_definition["type"] == "nrnseclistcomp":
                        location = ephys.locations.NrnSeclistCompLocation(
                            name=recording_definition["name"],
                            comp_x=recording_definition["comp_x"],
                            sec_index=recording_definition["sec_index"],
                            seclist_name=recording_definition["seclist_name"],
                        )

                    else:
                        raise Exception(
                            "Recording type %s not supported"
                            % recording_definition["type"]
                        )

                    var = recording_definition["var"]
                    recording = ephys.recordings.CompRecording(
                        name="%s.%s.%s.%s"
                        % (prefix, protocol_name, location.name, var),
                        location=location,
                        variable=recording_definition["var"],
                    )
                    recordings.append(recording)

            if (
                "type" in protocol_definition
                and protocol_definition["type"] == "StepProtocol"
            ):
                protocols_dict[protocol_name] = read_step_protocol(
                    protocol_name, protocol_definition, recordings, stochkv_det
                )
            elif (
                "type" in protocol_definition
                and protocol_definition["type"] == "StepThresholdProtocol"
            ):
                protocols_dict[protocol_name] = read_step_threshold_protocol(
                    protocol_name, protocol_definition, recordings, stochkv_det
                )
            elif (
                "type" in protocol_definition
                and protocol_definition["type"] == "RatSSCxThresholdDetectionProtocol"
            ):
                protocols_dict["ThresholdDetection"] = (
                    protocols.RatSSCxThresholdDetectionProtocol(
                        "IDRest",
                        step_protocol_template=read_step_protocol(
                            "Threshold",
                            protocol_definition["step_template"],
                            recordings,
                        ),
                        prefix=prefix,
                    )
                )
            else:
                stimuli = []
                for stimulus_definition in protocol_definition["stimuli"]:
                    stimuli.append(
                        ephys.stimuli.NrnSquarePulse(
                            step_amplitude=stimulus_definition["amp"],
                            step_delay=stimulus_definition["delay"],
                            step_duration=stimulus_definition["duration"],
                            location=soma_loc,
                            total_duration=stimulus_definition["totduration"],
                        )
                    )

                protocols_dict[protocol_name] = ephys.protocols.SweepProtocol(
                    name=protocol_name, stimuli=stimuli, recordings=recordings
                )

    if "Main" in list(protocol_definitions.keys()):
        protocols_dict["RinHoldcurrent"] = protocols.RatSSCxRinHoldcurrentProtocol(
            "RinHoldCurrent",
            rin_protocol_template=protocols_dict["Rin"],
            holdi_precision=protocol_definitions["RinHoldcurrent"]["holdi_precision"],
            holdi_max_depth=protocol_definitions["RinHoldcurrent"]["holdi_max_depth"],
            prefix=prefix,
        )

        other_protocols = []

        for protocol_name in protocol_definitions["Main"]["other_protocols"]:
            if protocol_name in protocols_dict:
                other_protocols.append(protocols_dict[protocol_name])

        pre_protocols = []
        preprot_score_threshold = 1

        if "pre_protocols" in protocol_definitions["Main"]:
            for protocol_name in protocol_definitions["Main"]["pre_protocols"]:
                pre_protocols.append(protocols_dict[protocol_name])
            preprot_score_threshold = protocol_definitions["Main"][
                "preprot_score_threshold"
            ]

        protocols_dict["Main"] = protocols.RatSSCxMainProtocol(
            "Main",
            rmp_protocol=protocols_dict["RMP"],
            rmp_score_threshold=protocol_definitions["Main"]["rmp_score_threshold"],
            rinhold_protocol=protocols_dict["RinHoldcurrent"],
            rin_score_threshold=protocol_definitions["Main"]["rin_score_threshold"],
            thdetect_protocol=protocols_dict["ThresholdDetection"],
            other_protocols=other_protocols,
            pre_protocols=pre_protocols,
            preprot_score_threshold=preprot_score_threshold,
            use_rmp_rin_thresholds=runopt,
        )

    return protocols_dict


from bluepyopt.ephys.efeatures import eFELFeature


# Limit the score to 1, prevent optimizing on scores that are already good
class eFELFeatureLimit(eFELFeature):
    def calculate_score(self, responses, trace_check=False):
        """Limit the score"""
        score = max(
            super(eFELFeatureLimit, self).calculate_score(responses, trace_check), 1.0
        )
        logger.debug("Limiting score for %s: %f", self.name, score)

        return score


class eFELFeatureExtra(eFELFeature):
    """eFEL feature extra"""

    SERIALIZED_FIELDS = (
        "name",
        "efel_feature_name",
        "recording_names",
        "stim_start",
        "stim_end",
        "exp_mean",
        "exp_std",
        "threshold",
        "comment",
    )

    def __init__(
        self,
        name,
        efel_feature_name=None,
        recording_names=None,
        stim_start=None,
        stim_end=None,
        exp_mean=None,
        exp_std=None,
        exp_vals=None,
        threshold=None,
        stimulus_current=None,
        comment="",
        interp_step=None,
        double_settings=None,
        int_settings=None,
        prefix="",
        use_powertransform=False,
    ):
        """Constructor

        Args:
            name (str): name of the eFELFeature object
            efel_feature_name (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            recording_names (dict): eFEL features can accept several recordings
                as input
            stim_start (float): stimulation start time (ms)
            stim_end (float): stimulation end time (ms)
            exp_mean (float): experimental mean of this eFeature
            exp_std(float): experimental standard deviation of this eFeature
            threshold(float): spike detection threshold (mV)
            comment (str): comment
        """

        super(eFELFeatureExtra, self).__init__(
            name,
            efel_feature_name,
            recording_names,
            stim_start,
            stim_end,
            exp_mean,
            exp_std,
            threshold,
            stimulus_current,
            comment,
            interp_step,
            double_settings,
            int_settings,
        )

        extra_features = [
            "spikerate_tau_jj_skip",
            "spikerate_drop_skip",
            "spikerate_tau_log_skip",
            "spikerate_tau_fit_skip",
        ]

        if self.efel_feature_name in extra_features:
            self.extra_feature_name = self.efel_feature_name
            self.efel_feature_name = "peak_time"
        else:
            self.extra_feature_name = None

        self.prefix = prefix
        self.exp_vals = exp_vals
        self.use_powertransform = use_powertransform

    def get_bpo_feature(self, responses):
        """Return internal feature which is directly passed as a response"""

        if f"{self.prefix}.{self.efel_feature_name}" not in responses:
            return None
        else:
            return responses[f"{self.prefix}.{self.efel_feature_name}"]

    def get_bpo_score(self, responses):
        """Return internal score which is directly passed as a response."""
        feature_value = self.get_bpo_feature(responses)
        if feature_value is None:
            score = 250.0
        else:
            score = abs(feature_value - self.exp_mean) / self.exp_std
        return score

    def calculate_features(self, responses, raise_warnings=False):
        """Calculate feature value"""

        if self.efel_feature_name.startswith("bpo_"):  # check if internal feature
            feature_values = numpy.array(self.get_bpo_feature(responses))
        else:
            efel_trace = self._construct_efel_trace(responses)

            if efel_trace is None:
                feature_values = None
            else:
                self._setup_efel()

                import efel

                values = efel.getFeatureValues(
                    [efel_trace],
                    [self.efel_feature_name],
                    raise_warnings=raise_warnings,
                )

                feature_values = values[0][self.efel_feature_name]

                efel.reset()

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))

        return feature_values

    def calculate_score(self, responses, trace_check=False):
        """Calculate the score"""

        if self.efel_feature_name.startswith("bpo_"):  # check if internal feature
            score = self.get_bpo_score(responses)

        elif self.exp_mean is None:
            score = 0

        else:
            feature_values = self.calculate_features(responses)
            if (feature_values is None) or (len(feature_values) == 0):
                score = 250.0
            else:
                if (len(self.exp_vals) == 2) or (self.use_powertransform == False):
                    # assume gaussian, use no conversion
                    score = (
                        numpy.sum(numpy.fabs(feature_values - self.exp_mean))
                        / self.exp_std
                        / len(feature_values)
                    )
                    logger.debug("Calculated score for %s: %f", self.name, score)

        return score


from bluepyopt.ephys.objectives import EFeatureObjective


class SingletonWeightObjective(EFeatureObjective):
    """Single EPhys feature"""

    def __init__(self, name, feature, weight):
        """Constructor

        Args:
            name (str): name of this object
            features (EFeature): single eFeature inside this objective
        """

        super(SingletonWeightObjective, self).__init__(name, [feature])
        self.weight = weight

    def calculate_score(self, responses):
        """Objective score"""

        return self.calculate_feature_scores(responses)[0] * self.weight

    def __str__(self):
        """String representation"""

        return "( %s ), weight:%f" % (self.features[0], self.weight)


def define_fitness_calculator(
    main_protocol, features_filename, prefix="", stage=None, use_powertransform=False
):
    """Define fitness calculator"""

    with open(
        os.path.join(os.path.dirname(__file__), "..", features_filename)
    ) as features_file:
        feature_definitions = json.load(features_file)

    if "__comment" in feature_definitions:
        del feature_definitions["__comment"]

    objectives = []
    efeatures = {}
    features = []

    for protocol_name, locations in feature_definitions.items():
        for recording_name, feature_configs in locations.items():
            for feature_config in feature_configs:
                if ("stage" in feature_config) and (stage is not None) and (stage > 0):
                    if stage not in feature_config["stage"]:
                        continue  # feature not used in this stage

                efel_feature_name = feature_config["feature"]
                meanstd = feature_config["val"]

                if hasattr(main_protocol, "subprotocols"):
                    protocol = main_protocol.subprotocols()[protocol_name]
                else:
                    protocol = main_protocol[protocol_name]

                feature_name = "%s.%s.%s.%s" % (
                    prefix,
                    protocol_name,
                    recording_name,
                    efel_feature_name,
                )
                recording_names = {
                    "": "%s.%s.%s" % (prefix, protocol_name, recording_name)
                }

                if "weight" in feature_config:
                    weight = feature_config["weight"]
                else:
                    weight = 1

                if "strict_stim" in feature_config:
                    strict_stim = feature_config["strict_stim"]
                else:
                    strict_stim = True

                if hasattr(protocol, "stim_start"):
                    stim_start = protocol.stim_start

                    if "threshold" in feature_config:
                        threshold = feature_config["threshold"]
                    else:
                        threshold = -30

                    if "bAP" in protocol_name:
                        # bAP response can be after stimulus
                        stim_end = protocol.total_duration
                    elif "H40S8" in protocol_name:
                        stim_end = protocol.stim_last_start
                    else:
                        stim_end = protocol.stim_end

                    stimulus_current = protocol.step_amplitude

                else:
                    stim_start = None
                    stim_end = None
                    stimulus_current = None
                    threshold = None

                feature = eFELFeatureExtra(
                    feature_name,
                    efel_feature_name=efel_feature_name,
                    recording_names=recording_names,
                    stim_start=stim_start,
                    stim_end=stim_end,
                    exp_mean=meanstd[0],
                    exp_std=meanstd[1],
                    exp_vals=meanstd,
                    stimulus_current=stimulus_current,
                    threshold=threshold,
                    prefix=prefix,
                    int_settings={"strict_stiminterval": strict_stim},
                    use_powertransform=use_powertransform,
                )
                efeatures[feature_name] = feature
                features.append(feature)
                objective = SingletonWeightObjective(feature_name, feature, weight)
                objectives.append(objective)

    # objectives.append(MaxObjective('global_maximum', features))
    fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

    return fitcalc, efeatures


class MultiEvaluator(bpopt.evaluators.Evaluator):
    """Multiple cell evaluator"""

    def __init__(
        self,
        evaluators=None,
        sim=None,
    ):
        """Constructor

        Args:
            evaluators (list): list of CellModel evaluators
        """

        self.sim = sim
        self.evaluators = evaluators
        objectives = []
        # loop objectives for all evaluators, rename based on evaluators
        for evaluator in self.evaluators:
            for objective in evaluator.objectives:
                objectives.append(objective)

        # these are identical for all models. Better solution available?
        self.param_names = self.evaluators[0].param_names
        params = self.evaluators[0].cell_model.params_by_names(self.param_names)

        super(MultiEvaluator, self).__init__(objectives, params)

    def param_dict(self, param_array):
        """Convert param_array in param_dict"""
        return dict(zip(self.param_names, param_array))

    def objective_dict(self, objective_array):
        """Convert objective_array in objective_dict"""
        objective_names = [objective.name for objective in self.objectives]
        if len(objective_names) != len(objective_array):
            raise Exception(
                "MultiEvaluator: list given to objective_dict() "
                "has wrong number of objectives"
            )

        return dict(zip(objective_names, objective_array))

    def objective_list(self, objective_dict):
        """Convert objective_dict in objective_list"""
        objective_names = [objective.name for objective in self.objectives]
        return [objective_dict[objective_name] for objective_name in objective_names]

    def evaluate_with_dicts(self, param_dict=None):
        """Run evaluation with dict as input and output"""

        scores = {}
        for evaluator in self.evaluators:
            score = evaluator.evaluate_with_dicts(param_dict=param_dict)
            scores.update(score)

        return scores

    def evaluate_with_lists(self, param_list=None):
        """Run evaluation with lists as input and outputs"""

        param_dict = self.param_dict(param_list)

        obj_dict = self.evaluate_with_dicts(param_dict=param_dict)

        return self.objective_list(obj_dict)

    def evaluate(self, param_list=None):
        """Run evaluation with lists as input and outputs"""

        return self.evaluate_with_lists(param_list)

    def __str__(self):
        content = "multi cell evaluator:\n"

        content += "  evaluators:\n"
        for evaluator in self.evaluators:
            content += "    %s\n" % str(evaluator)

        return content


def create(
    etype,
    stochkv_det=None,
    usethreshold=False,
    runopt=False,
    altmorph=None,
    stage=None,
    past_params=None,
    do_simplify_morph=False,
):
    """Setup"""

    if past_params is None:
        past_params = []
    cell_evals = []

    with open(
        os.path.join(os.path.dirname(__file__), "..", "config/recipes/recipes.json")
    ) as f:
        recipes = json.load(f)

    recipe = recipes[etype]

    if usethreshold:
        if "mm_test_recipe" in recipe:
            etype = recipe["mm_test_recipe"]
        else:
            etype = etype.replace("_legacy", "")
            etype = etype.replace("_combined", "")

    if "use_powertransform" in recipe:
        use_powertransform = recipe["use_powertransform"]
    else:
        use_powertransform = False

    prot_path = recipe["protocol"]

    if stochkv_det is False:
        nrn_sim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    else:
        nrn_sim = ephys.simulators.NrnSimulator()

    if altmorph is None:
        # get morphologies, convert to list if not given as list
        morphs = recipe["morphology"]
        if not isinstance(morphs, (list)):
            morphs = [["_", morphs]]
    elif not isinstance(altmorph, (list)):
        # use directly, either given as absolute or relative
        morphs = [["alt", altmorph]]
    else:
        morphs = altmorph

    for morphval in morphs:
        if len(morphval) == 3:
            morphname, morph, apical_point_isec0 = morphval
        else:
            morphname, morph = morphval
            apical_point_isec0 = None

        # modify the morph path if a directory is given!
        if (altmorph is None) and ("morph_path" in recipe):
            morph_path = "morphologies/"

            basename = os.path.basename(morph)
            filename = os.path.splitext(basename)[0]
            morph = os.path.join(morph_path, basename)
            apsec_file = os.path.join(morph_path, "apical_points_isec.json")
            apical_points_isecs = json.load(open(apsec_file))
            logger.debug("Reading %s", apsec_file)
            if filename in apical_points_isecs:
                apical_point_isec = int(apical_points_isecs[filename])
            else:
                apical_point_isec = None
        else:
            apical_point_isec = None

        if do_simplify_morph:
            apical_point_isec = None
        else:
            # check if apical point section should be overridden
            if apical_point_isec0 is not None:
                apical_point_isec = apical_point_isec0
                logger.debug("Apical point override with %d", apical_point_isec)

            if apical_point_isec is not None:
                logger.debug("Apical point at apical[%d]", apical_point_isec)

        if stage is None:
            stage_ = stage
        else:
            stage_ = abs(stage)
        cell = template.create(
            recipe, etype, morph, stage_, past_params, do_simplify_morph
        )

        protocols_dict = define_protocols(
            prot_path,
            stochkv_det,
            runopt,
            morphname,
            apical_point_isec,
            stage,
            do_simplify_morph,
        )

        if "Main" in list(protocols_dict.keys()):
            fitness_calculator, efeatures = define_fitness_calculator(
                protocols_dict["Main"],
                recipe["features"],
                morphname,
                stage,
                use_powertransform,
            )

            protocols_dict["Main"].fitness_calculator = fitness_calculator

            protocols_dict["Main"].rmp_efeature = efeatures[
                f"{morphname}.RMP.soma.v.voltage_base"
            ]

            protocols_dict["Main"].rin_efeature = efeatures[
                f"{morphname}.Rin.soma.v.ohmic_input_resistance_vb_ssse"
            ]

            protocols_dict["Main"].rin_efeature.stimulus_current = protocols_dict[
                "Main"
            ].rinhold_protocol.rin_protocol_template.step_amplitude

            protocols_dict["RinHoldcurrent"].voltagebase_efeature = efeatures[
                f"{morphname}.Rin.soma.v.voltage_base"
            ]

            protocols_dict["ThresholdDetection"].holding_voltage = efeatures[
                f"{morphname}.Rin.soma.v.voltage_base"
            ].exp_mean

            fitness_protocols = {"main_protocol": protocols_dict["Main"]}

        else:
            fitness_calculator, efeatures = define_fitness_calculator(
                protocols_dict, recipe["features"], morphname, stage
            )

            fitness_protocols = protocols_dict

        param_names = [
            param.name for param in list(cell.params.values()) if not param.frozen
        ]

        cell_eval = ephys.evaluators.CellEvaluator(
            cell_model=cell,
            param_names=param_names,
            fitness_protocols=fitness_protocols,
            fitness_calculator=fitness_calculator,
            sim=nrn_sim,
            use_params_for_seed=True,
        )
        cell_eval.prefix = morphname
        cell_evals.append(cell_eval)

    multi_eval = MultiEvaluator(evaluators=cell_evals, sim=nrn_sim)

    return multi_eval
