"""Protocols involved in fitting of RatSSCx neuron ephys parameters"""

import numpy

import collections
import copy

import bluepyopt.ephys as ephys

import logging

logger = logging.getLogger(__name__)


class RatSSCxMainProtocol(ephys.protocols.Protocol):

    """Main protocol to fit RatSSCx neuron ephys parameters.

    Pseudo code:
        Find resting membrane potential
        Find input resistance
        If both of these scores are within bounds, run other protocols:
            Find holding current
            Find rheobase
            Run IDRest
            Possibly run other protocols (based on constructor arguments)
            Return all the responses
        Otherwise return return Rin and RMP protocol responses

    """

    def __init__(
        self,
        name,
        rmp_protocol=None,
        rmp_efeature=None,
        rmp_score_threshold=None,
        rinhold_protocol=None,
        rin_efeature=None,
        rin_score_threshold=None,
        thdetect_protocol=None,
        other_protocols=None,
        pre_protocols=None,
        preprot_score_threshold=None,
        fitness_calculator=None,
        use_rmp_rin_thresholds=False,
    ):
        """Constructor"""

        super(RatSSCxMainProtocol, self).__init__(name=name)

        self.rmp_protocol = rmp_protocol
        self.rmp_efeature = rmp_efeature
        self.rmp_score_threshold = rmp_score_threshold

        self.rinhold_protocol = rinhold_protocol
        self.rin_efeature = rin_efeature
        self.rin_score_threshold = rin_score_threshold

        self.thdetect_protocol = thdetect_protocol
        self.other_protocols = other_protocols

        self.pre_protocols = pre_protocols
        self.preprot_score_threshold = preprot_score_threshold

        self.use_rmp_rin_thresholds = use_rmp_rin_thresholds
        self.fitness_calculator = fitness_calculator

    def subprotocols(self):
        """Return all the subprotocols contained in this protocol, is recursive"""

        subprotocols = collections.OrderedDict({self.name: self})
        subprotocols.update(self.rmp_protocol.subprotocols())
        subprotocols.update(self.rinhold_protocol.subprotocols())
        subprotocols.update(self.thdetect_protocol.subprotocols())
        for other_protocol in self.other_protocols:
            subprotocols.update(other_protocol.subprotocols())
        for pre_protocol in self.pre_protocols:
            subprotocols.update(pre_protocol.subprotocols())

        return subprotocols

    @property
    def rin_efeature(self):
        """Get in_efeature"""

        return self.rinhold_protocol.rin_efeature

    @rin_efeature.setter
    def rin_efeature(self, value):
        """Set rin_efeature"""

        self.rinhold_protocol.rin_efeature = value


class RatSSCxRinHoldcurrentProtocol(ephys.protocols.Protocol):

    """IDRest protocol to fit RatSSCx neuron ephys parameters"""

    def __init__(
        self,
        name,
        rin_protocol_template=None,
        voltagebase_efeature=None,
        voltagebase_score_threshold=None,
        holdi_estimate_multiplier=2,
        holdi_precision=0.1,
        holdi_max_depth=5,
        prefix=None,
    ):
        """Constructor"""

        super(RatSSCxRinHoldcurrentProtocol, self).__init__(name=name)
        self.rin_protocol_template = rin_protocol_template
        self.voltagebase_efeature = voltagebase_efeature
        self.voltagebase_score_threshold = voltagebase_score_threshold
        self.holdi_estimate_multiplier = holdi_estimate_multiplier
        self.holdi_precision = holdi_precision
        self.holdi_max_depth = holdi_max_depth

        self.prefix = "" if prefix is None else f"{prefix}."
        # This will be set after the run()
        self.rin_protocol = None

    def subprotocols(self):
        """Return subprotocols"""

        subprotocols = collections.OrderedDict({self.name: self})

        subprotocols.update(
            {self.rin_protocol_template.name: self.rin_protocol_template}
        )

        return subprotocols


class RatSSCxThresholdDetectionProtocol(ephys.protocols.Protocol):

    """IDRest protocol to fit RatSSCx neuron ephys parameters"""

    def __init__(
        self,
        name,
        step_protocol_template=None,
        max_threshold_voltage=-40,
        holding_voltage=None,
        prefix=None,
    ):
        """Constructor"""

        super(RatSSCxThresholdDetectionProtocol, self).__init__(name=name)

        self.step_protocol_template = step_protocol_template
        self.max_threshold_voltage = max_threshold_voltage

        self.short_perc = 0.1
        self.short_steps = 20
        self.holding_voltage = holding_voltage

        self.prefix = "" if prefix is None else f"{prefix}."

    def subprotocols(self):
        """Return subprotocols"""

        subprotocols = collections.OrderedDict({self.name: self})

        subprotocols.update(self.step_protocol_template.subprotocols())

        return subprotocols

    def run(self, cell_model, param_values, sim, holdi, rin, rmp):
        """Run protocol"""

        responses = collections.OrderedDict()

        # Calculate max threshold current
        max_threshold_current = self.search_max_threshold_current(rin=rin, rmp=rmp)

        # Calculate spike threshold
        threshold_current = self.search_spike_threshold(
            cell_model,
            {},
            holdi=holdi,
            lower_bound=-holdi,
            upper_bound=max_threshold_current,
            sim=sim,
        )

        cell_model.threshold_current = threshold_current

        responses[f"{self.prefix}bpo_threshold_current"] = threshold_current

        return responses

    def search_max_threshold_current(self, rin=None, rmp=None):
        """Find the current necessary to get to max_threshold_voltage"""

        max_threshold_current = (
            float(self.max_threshold_voltage - self.holding_voltage) / rin
        )

        print("Max threshold current: %.6g" % max_threshold_current)

        return max_threshold_current

    def create_step_protocol(self, holdi=0.0, step_current=0.0):
        """Create threshold protocol"""

        threshold_protocol = copy.deepcopy(self.step_protocol_template)
        threshold_protocol.name = "Threshold"
        for recording in threshold_protocol.recordings:
            recording.name = recording.name.replace(
                self.step_protocol_template.name, threshold_protocol.name
            )

        if threshold_protocol.holding_stimulus is not None:
            threshold_protocol.holding_stimulus.step_amplitude = holdi

        for step_stim in threshold_protocol.step_stimuli:
            step_stim.step_amplitude = step_current

        return threshold_protocol

    def detect_spike(
        self,
        cell_model,
        param_values,
        sim=None,
        step_current=None,
        holdi=None,
        short=False,
    ):
        """Detect if spike is present at current level"""

        # Only run short pulse if percentage set smaller than 100%
        if short and self.short_perc < 1.0:
            protocol = self.create_short_threshold_protocol(
                holdi=holdi, step_current=step_current
            )
        else:
            protocol = self.create_step_protocol(holdi=holdi, step_current=step_current)

        response = protocol.run(cell_model, param_values, sim=sim)

        feature = ephys.efeatures.eFELFeature(
            name="ThresholdDetection.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"{self.prefix}ThresholdDetection.soma.v"},
            stim_start=protocol.stim_start,
            stim_end=protocol.stim_end,
            exp_mean=1,
            exp_std=0.1,
        )

        spike_count = feature.calculate_feature(response)

        return spike_count >= 1

    def binsearch_spike_threshold(
        self,
        cell_model,
        param_values,
        sim=None,
        holdi=None,
        lower_bound=None,
        upper_bound=None,
        precision=0.01,
        max_depth=5,
        depth=1,
    ):
        """
        Do binary search to find spike threshold

        Assumption is that lower_bound has no spike, upper_bound has
        """

        if depth > max_depth or abs(upper_bound - lower_bound) < precision:
            return upper_bound
        else:
            middle_bound = upper_bound - abs(upper_bound - lower_bound) / 2
            spike_detected = self.detect_spike(
                cell_model,
                param_values,
                sim=sim,
                holdi=holdi,
                step_current=middle_bound,
                short=False,
            )
            if spike_detected:
                return self.binsearch_spike_threshold(
                    cell_model,
                    param_values,
                    sim=sim,
                    holdi=holdi,
                    lower_bound=lower_bound,
                    upper_bound=middle_bound,
                    depth=depth + 1,
                )
            else:
                return self.binsearch_spike_threshold(
                    cell_model,
                    param_values,
                    sim=sim,
                    holdi=holdi,
                    lower_bound=middle_bound,
                    upper_bound=upper_bound,
                    depth=depth + 1,
                )

    def search_spike_threshold(
        self,
        cell_model,
        param_values,
        sim=None,
        holdi=None,
        lower_bound=None,
        upper_bound=None,
    ):
        """Find the current step spiking threshold"""
        step_currents = numpy.linspace(lower_bound, upper_bound, num=self.short_steps)

        if len(step_currents) == 0:
            return None

        for step_current in step_currents:
            spike_detected = self.detect_spike(
                cell_model,
                param_values,
                sim=sim,
                holdi=holdi,
                step_current=step_current,
                short=True,
            )

            if spike_detected:
                upper_bound = step_current
                break

        # if upper bound didn't have spike with short stimulus
        # check if there is one with longer stimulus
        if not spike_detected:
            spike_detected = self.detect_spike(
                cell_model,
                param_values,
                sim=sim,
                holdi=holdi,
                step_current=step_current,
                short=False,
            )

            if spike_detected:
                upper_bound = step_current
            else:
                return None

        threshold_current = self.binsearch_spike_threshold(
            cell_model,
            {},
            sim=sim,
            holdi=holdi,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return threshold_current


class StepProtocol(ephys.protocols.SweepProtocol):

    """Protocol consisting of step and holding current"""

    def __init__(
        self,
        name=None,
        step_stimuli=None,
        holding_stimulus=None,
        recordings=None,
        cvode_active=None,
        stochkv_det=None,
    ):
        """Constructor

        Args:
            name (str): name of this object
            step_stimuli (list of Stimuli): List of Stimulus objects used in protocol
            recordings (list of Recordings): Recording objects used in the
                protocol
            cvode_active (bool): whether to use variable time step
        """

        super(StepProtocol, self).__init__(
            name,
            stimuli=step_stimuli + [holding_stimulus]
            if holding_stimulus is not None
            else step_stimuli,
            recordings=recordings,
            cvode_active=cvode_active,
        )

        self.step_stimuli = step_stimuli
        self.holding_stimulus = holding_stimulus
        self.stochkv_det = stochkv_det

    def instantiate(self, sim=None, icell=None):
        """Instantiate"""

        for stimulus in self.stimuli:
            stimulus.instantiate(sim=sim, icell=icell)

        for recording in self.recordings:
            try:
                recording.instantiate(sim=sim, icell=icell)
            except ephys.locations.EPhysLocInstantiateException as e:
                logger.debug(
                    "SweepProtocol: Instantiating recording generated location "
                    "exception: %s, will return empty response for this recording" % e
                )

    def run(self, cell_model, param_values, sim=None, isolate=None):
        """Run protocol"""

        responses = {}
        if self.stochkv_det is not None and not self.stochkv_det:
            for mechanism in cell_model.mechanisms:
                if "Stoch" in mechanism.prefix:
                    mechanism.deterministic = False
            self.cvode_active = False

        responses.update(
            super(StepProtocol, self).run(
                cell_model, param_values, sim=sim, isolate=isolate
            )
        )

        if self.stochkv_det is not None and not self.stochkv_det:
            for mechanism in cell_model.mechanisms:
                if "Stoch" in mechanism.prefix:
                    mechanism.deterministic = True
            self.cvode_active = True

        return responses

    @property
    def stim_start(self):
        """Time stimulus starts"""
        return self.step_stimuli[0].step_delay

    @property
    def stim_end(self):
        """Time stimulus starts"""
        return self.step_stimuli[-1].step_delay + self.step_stimuli[-1].step_duration

    @property
    def step_amplitude(self):
        """Time stimulus starts"""
        amplitudes = [step_stim.step_amplitude for step_stim in self.step_stimuli]
        return None if None in amplitudes else numpy.mean(amplitudes)


class StepThresholdProtocol(StepProtocol):

    """Step protocol based on threshold"""

    def __init__(
        self,
        name,
        thresh_perc=None,
        step_stimuli=None,
        holding_stimulus=None,
        recordings=None,
        cvode_active=None,
        stochkv_det=None,
    ):
        """Constructor"""

        super(StepThresholdProtocol, self).__init__(
            name,
            step_stimuli=step_stimuli,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochkv_det=stochkv_det,
        )

        self.thresh_perc = thresh_perc
