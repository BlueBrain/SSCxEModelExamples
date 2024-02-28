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

import sys
import numpy
import os
import neuron

import re
import bluepyopt.ephys as ephys
import multiprocessing
import multiprocessing.pool
import ipyparallel

from lib import tools
import traceback
import json
import pickle
import pandas as pd


def get_filename(etype, seed, stage=0):
    if stage > 1:
        return "%s_%d_s%d" % (etype, seed, stage)
    else:
        return "%s_%d" % (etype, seed)


def run_emodel_morph_isolated(args):
    """Run emodel in isolated environment"""
    (
        uid,
        emodels_path,
        emodel_hash,
        emodel_seed,
        emodel,
        morph_path,
        apical_point_isec,
    ) = args

    return_dict = {"uid": uid, "exception": None, "morph_path": morph_path}
    pool = NestedPool(1, maxtasksperchild=1)

    try:
        return_dict["points"] = pool.apply(
            run_emodel_morph,
            (
                emodels_path,
                emodel_hash,
                emodel_seed,
                emodel,
                morph_path,
                apical_point_isec,
            ),
        )
    except Exception:
        return_dict["points"] = None
        return_dict["exception"] = "".join(traceback.format_exception(*sys.exc_info()))

    pool.terminate()
    pool.join()
    del pool

    return return_dict


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestedPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestedPool, self).__init__(*args, **kwargs)

    @staticmethod
    def Process(ctx, *args, **kwds):
        return ctx.Process(*args, **kwds)


def calculate(
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
    use_ipyp=None,
    ipyp_profile=None,
):
    """Calculate"""

    if os.path.splitext(meg_output)[1] == ".tsv":
        df = pd.read_csv(meg_output, sep="\t")
    else:
        df = pd.read_csv(meg_output)

    df = df[df["fullmtype"].str.contains(str(mtype_morph))]
    if only_morph is not None:
        df = df[df["morph_name"] == str(only_morph)]
    numpy.random.seed(1)
    df = df.reindex(numpy.random.permutation(df.index))
    # morphs = list(df['morph_name'])[0:n_morph]
    morphs = df["morph_name"].sample(n=n_morph).tolist()

    apical_points_isecs = json.load(open(apical_points))

    print("Creating argument list for parallelisation")

    arg_list = []
    for uid, morph in enumerate(morphs):
        if morph in apical_points_isecs:
            apical_point_isec = int(apical_points_isecs[morph])
        else:
            dend_morph = morph[:]
            if dend_morph.startswith("dend-"):
                dend_morph = dend_morph[5:]
            if "_axon-" in dend_morph:
                dend_morph = dend_morph.split("_axon-")[0]
            dend_morph = dend_morph.split("_-_")[0]

            if dend_morph in apical_points_isecs:
                apical_point_isec = int(apical_points_isecs[dend_morph])
            else:
                apical_point_isec = None

        morph_path = os.path.join(morphs_path, morph + ".asc")
        args = (
            uid,
            emodels_path,
            emodel_hash,
            emodel_seed,
            emodel,
            morph_path,
            apical_point_isec,
        )
        arg_list.append(args)

    print("Parallelising attenuation evaluation of %d morphologies" % len(arg_list))

    if use_ipyp:
        # use ipyparallel
        client = ipyparallel.Client(profile=ipyp_profile)
        lview = client.load_balanced_view()
        results = lview.imap(run_emodel_morph_isolated, arg_list, ordered=False)
    else:
        # use multiprocessing
        pool = NestedPool()
        results = pool.imap_unordered(run_emodel_morph_isolated, arg_list)
        # run_emodel_morph(emodels_path, emodel_hash, emodel_seed, emodel, morph_path, apical_point_isec) # for testing

    # Keep track of the number of received results
    uids_received = 0
    results_full = {}

    # Every time a result comes in, save it
    for result in results:
        uid = result["uid"]
        results_full[uid] = result
        uids_received += 1

        indent = 2
        s = json.dumps(results_full, indent=indent)
        s = tools.collapse_json(s, indent=4 * indent)
        with open(output, "w") as f:
            f.write(s)

        print(
            "Saved scores for uid %s (%d out of %d)"
            % (uid, uids_received, len(arg_list))
        )
        sys.stdout.flush()


def run_emodel_morph(
    emodels_path, emodel_hash, emodel_seed, emodel, morph_path, apical_point_isec
):
    """Run emodel morph combo"""
    path = emodels_path
    # switch to run dir and load modules
    # sys.path.append(path)

    print(f"Changing path to {path}")
    import neuron

    neuron.h.nrn_load_dll("../x86_64/.libs/libnrnmech.so")

    from lib import setup

    print(f"Loading modules from {setup.__file__}")

    with tools.cd(path):
        checkpoints_dir = f"checkpoints/run.{emodel_hash}"
        filename = get_filename(emodel, emodel_seed)
        cp_filename = os.path.join(checkpoints_dir, filename + ".pkl")

        cp = pickle.load(open(cp_filename, "rb"), encoding="latin1")
        hof = cp["halloffame"]

    evaluator = setup.evaluator.create(
        etype=f"{emodel}", altmorph=[["alt", morph_path, apical_point_isec]]
    )
    evaluator = evaluator.evaluators[0]  # only one evaluator

    emodel_params = evaluator.param_dict(hof[0])
    evaluator.cell_model.freeze(emodel_params)

    points = all_att(evaluator, emodels_path, apical_point_isec)

    return points


def interpolated_neuron_coordinates(hsection):
    """Computes the interpolated coordinates of a NEURON hsection."""
    # get the data for the section
    n_segments = int(neuron.h.n3d(sec=hsection))
    n_comps = float(hsection.nseg)

    xs = numpy.zeros(n_segments)
    ys = numpy.zeros(n_segments)
    zs = numpy.zeros(n_segments)
    lengths = numpy.zeros(n_segments)

    for index in range(n_segments):
        xs[index] = neuron.h.x3d(index, sec=hsection)
        ys[index] = neuron.h.y3d(index, sec=hsection)
        zs[index] = neuron.h.z3d(index, sec=hsection)
        lengths[index] = neuron.h.arc3d(index, sec=hsection)

    # to use Vector class's .interpolate()
    # must first scale the independent variable
    # i.e. normalize length along centroid
    lengths /= lengths[-1]

    # initialize the destination "independent" vector
    # range = numpy.array(n_comps+2)
    comp_range = numpy.arange(0, n_comps + 2) / n_comps - 1.0 / (2 * n_comps)
    comp_range[0] = 0
    comp_range[-1] = 1

    # length contains the normalized distances of the pt3d points
    # along the centroid of the section. These are spaced at
    # irregular intervals.
    # range contains the normalized distances of the nodes along the
    # centroid of the section.  These are spaced at regular intervals.
    # Ready to interpolate.

    xs_interp = numpy.interp(comp_range, lengths, xs)
    ys_interp = numpy.interp(comp_range, lengths, ys)
    zs_interp = numpy.interp(comp_range, lengths, zs)

    return xs_interp, ys_interp, zs_interp


def xyz(section, location=0.5):
    x, y, z = interpolated_neuron_coordinates(section)
    d = diams(section)

    xm = x[int(numpy.floor((len(x) - 1) * location))]
    ym = y[int(numpy.floor((len(y) - 1) * location))]
    zm = z[int(numpy.floor((len(z) - 1) * location))]
    dm = d[int(numpy.floor((len(d) - 1) * location))]

    return xm, ym, zm, dm


def diam(section, location=0.5):
    d = diams(section)
    return d[int(numpy.floor((len(d) - 1) * location))]


def diams(section):
    d = [seg.diam for seg in section.allseg()]
    return numpy.array(d)


def path_distance(section_a, section_b, location=0.5, location_b=None):
    neuron.h.distance(0, location, sec=section_a)
    return neuron.h.distance(1, location_b, sec=section_b)


def euclid_section_distance(
    section_a, section_b, location=0.5, location_b=None, dimensions="xy"
):
    """
    Calculate euclidean distance between section_a and section_b
    :param section_a:
    :param section_b:
    :param location: normalized position within section
    :param dimensions: dimensions to use, default assume expermenters ignored z dimension
    :return: distance
    """
    if location_b is None:
        location_b = location

    xam, yam, zam, _ = xyz(section_a, location=location)
    xbm, ybm, zbm, _ = xyz(section_b, location=location_b)

    d2 = 0
    if "x" in dimensions:
        d2 += (xam - xbm) ** 2.0
    if "y" in dimensions:
        d2 += (yam - ybm) ** 2.0
    if "z" in dimensions:
        d2 += (zam - zbm) ** 2.0

    return numpy.sqrt(d2)


def get_points_random(evaluator, apical_point_isec):
    max_dist_apical = 900.0
    div_max = 100.0  # recording points per div um from soma to apical point
    n_basal = 10  # number of points on basal dendrites to record

    evaluator.cell_model.instantiate(sim=evaluator.sim)
    icell = evaluator.cell_model.icell
    morph = evaluator.cell_model.morphology.morphology_path

    points = {}
    morphname = str(os.path.basename(morph)).split(".")[0]

    print("Getting points for morphology %s" % morphname)

    secnames = [
        ("apical", "apic"),
        ("basal", "dend"),
        ("somatic", "soma"),
    ]

    for secname in secnames:
        seclist_name = secname[0]
        points[seclist_name] = {}
        points[seclist_name]["name"] = []
        points[seclist_name]["sec_id"] = []
        points[seclist_name]["x"] = []
        points[seclist_name]["mid_diam"] = []
        points[seclist_name]["diam"] = []
        points[seclist_name]["distance"] = []
        points[seclist_name]["dend_distance"] = []
        points[seclist_name]["path_distance"] = []

    # apical
    seclist_name = "apical"

    div = div_max
    curr_length = div_max
    section = icell.apic[apical_point_isec]
    while True:
        name = str(section.name()).split(".")[-1]

        if "soma[0]" == name:
            break

        sec_id = int(re.search("\[(\d*)\]", name).group(1))

        nseg = float(section.nseg)
        comp_range = numpy.arange(0, nseg + 2) / nseg - 1.0 / (2 * nseg)
        comp_range[0] = 0
        comp_range[-1] = 1
        comp_range = comp_range[1:-1][::-1]
        length = section.L / nseg

        for x in comp_range:
            pathdistance = path_distance(
                icell.soma[0], section, location=0.5, location_b=x
            )
            curr_length += length
            # only use if not too far away from soma and
            # if sufficiently far away from last recording poing
            if (pathdistance <= max_dist_apical) and (curr_length >= div):
                print(
                    "Using %s(%f) fromn last point %f and pathdistance from soma %f"
                    % (name, x, curr_length, pathdistance)
                )
                curr_length = 0.0  # reset counter
                div = numpy.random.uniform(0, div_max)

                points[seclist_name]["path_distance"].append(pathdistance)

                points[seclist_name]["name"].append(
                    name + "_" + re.sub("[.]", "", str(x))
                )
                points[seclist_name]["sec_id"].append(sec_id)
                points[seclist_name]["x"].append(x)
                points[seclist_name]["mid_diam"].append(None)

                distance = euclid_section_distance(
                    icell.soma[0], section, location=0.5, location_b=x
                )
                points[seclist_name]["distance"].append(distance)

                d = diam(section, location=0.5)
                points[seclist_name]["diam"].append(d)

                denddistance = path_distance(
                    icell.apic[0], section, location=0.0, location_b=x
                )
                points[seclist_name]["dend_distance"].append(denddistance)

        if neuron.h.SectionRef(sec=section).has_parent():
            section = neuron.h.SectionRef(sec=section).parent
        else:
            print("- soma[0] was not reached")
            break

    # basal
    seclist_name = "basal"

    ibasals = range(len(icell.dend))
    numpy.random.seed(11)
    ibasals = numpy.random.permutation(ibasals)

    for isec in ibasals:
        section = icell.dend[isec]
        name = str(section.name()).split(".")[-1]
        sec_id = int(re.search("\[(\d*)\]", name).group(1))

        x = numpy.random.uniform(0, 1.0)

        pathdistance = path_distance(icell.soma[0], section, location=0.5, location_b=x)

        if pathdistance <= max_dist_apical:
            points[seclist_name]["path_distance"].append(pathdistance)

            points[seclist_name]["name"].append(name + "_" + re.sub("[.]", "", str(x)))
            points[seclist_name]["sec_id"].append(sec_id)
            points[seclist_name]["x"].append(x)
            points[seclist_name]["mid_diam"].append(None)

            distance = euclid_section_distance(
                icell.soma[0], section, location=0.5, location_b=x
            )
            points[seclist_name]["distance"].append(distance)

            d = diam(section, location=0.5)
            points[seclist_name]["diam"].append(d)

            # find last dendrite section
            parent = section
            while True:
                parent_name = str(parent.name()).split(".")[-1]
                if "soma[0]" == parent_name:
                    break

                if neuron.h.SectionRef(sec=parent).has_parent():
                    parent = neuron.h.SectionRef(sec=parent).parent
                else:
                    print("- soma[0] was not reached")
                    break

            denddistance = path_distance(
                icell.apic[0], section, location=0.0, location_b=x
            )
            points[seclist_name]["dend_distance"].append(denddistance)

        if len(points[seclist_name]["name"]) >= n_basal:
            break

    # soma
    seclist_name = "somatic"
    section = icell.soma[0]
    x = 0.5
    name = str(section.name()).split(".")[-1]
    sec_id = int(re.search("\[(\d*)\]", name).group(1))

    pathdistance = path_distance(icell.soma[0], section, location=0.5, location_b=x)

    points[seclist_name]["path_distance"].append(pathdistance)

    points[seclist_name]["name"].append(name + "_" + re.sub("[.]", "", str(x)))
    points[seclist_name]["sec_id"].append(sec_id)
    points[seclist_name]["x"].append(x)
    points[seclist_name]["mid_diam"].append(None)

    distance = euclid_section_distance(
        icell.soma[0], section, location=0.5, location_b=x
    )
    points[seclist_name]["distance"].append(distance)

    d = diam(section, location=0.5)
    points[seclist_name]["diam"].append(d)
    points[seclist_name]["dend_distance"].append(pathdistance)

    evaluator.cell_model.destroy(sim=evaluator.sim)
    icell = None

    for _ in range(100):
        neuron.h.pop_section()

    return points


def all_att(evaluator, emodels_path, apical_point_isec):
    points = get_points_random(evaluator, apical_point_isec)

    print("All points gathered")
    # points = get_points(evaluator)

    if points is not None:
        print("Running bAP")
        bAP(evaluator, emodels_path, points)
        print("Running EPSP")
        EPSP(evaluator, emodels_path, points)

    return points


def bAP(evaluator, emodels_path, points):
    protocol_name = "bAP"

    cell_model = evaluator.cell_model
    sim = evaluator.sim

    # Set recordings
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
    )

    somav_recording = ephys.recordings.CompRecording(
        name="RMP.soma.v", location=soma_loc, variable="v"
    )
    recordings = [somav_recording]

    # Estimate RMP
    RMP = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0,
        step_delay=700,
        step_duration=100,
        location=soma_loc,
        total_duration=800,
    )

    rmp_feature = ephys.efeatures.eFELFeature(
        "RMP.soma.v.voltage_base",
        efel_feature_name="voltage_base",
        recording_names={"": "RMP.soma.v"},
        stim_start=700.0,
        stim_end=1000.0,
        exp_mean=0,
        exp_std=1,
        threshold=-20,
    )

    from lib.setup import protocols

    rmp_protocol = protocols.StepProtocol(
        name="RMP", step_stimuli=[RMP], holding_stimulus=None, recordings=recordings
    )

    rmp_response = rmp_protocol.run(cell_model, {}, sim=sim)
    rmp = rmp_feature.calculate_feature(rmp_response)

    # Estimate Rin
    somav_recording = ephys.recordings.CompRecording(
        name="Rin.soma.v", location=soma_loc, variable="v"
    )
    recordings = [somav_recording]

    Rin = ephys.stimuli.NrnSquarePulse(
        step_amplitude=-0.01,
        step_delay=700,
        step_duration=500,
        location=soma_loc,
        total_duration=1300.0,
    )

    rin_feature = ephys.efeatures.eFELFeature(
        "Rin.soma.v.ohmic_input_resistance_vb_ssse",
        efel_feature_name="ohmic_input_resistance_vb_ssse",
        recording_names={"": "Rin.soma.v"},
        stim_start=700.0,
        stim_end=1200.0,
        stimulus_current=Rin.step_amplitude,
        exp_mean=0,
        exp_std=1,
        threshold=-20,
    )

    rin_protocol = protocols.StepProtocol(
        name="Rin", step_stimuli=[Rin], holding_stimulus=None, recordings=recordings
    )

    rin_response = rin_protocol.run(cell_model, {}, sim=sim)
    rin = rin_feature.calculate_feature(rin_response)

    # Threshold
    somav_recording = ephys.recordings.CompRecording(
        name="ThresholdDetection.soma.v", location=soma_loc, variable="v"
    )
    recordings = [somav_recording]

    step_stimulus = ephys.stimuli.NrnSquarePulse(
        step_amplitude=None,
        step_delay=700,
        step_duration=5,
        location=soma_loc,
        total_duration=1000,
    )

    step_protocol = protocols.StepProtocol(
        name="Threshold",
        step_stimuli=[step_stimulus],
        holding_stimulus=None,
        recordings=recordings,
    )

    protocol = protocols.RatSSCxThresholdDetectionProtocol(
        "IDRest",
        step_protocol_template=step_protocol,
        max_threshold_voltage=100,
        holding_voltage=rmp,
    )

    protocol.short_perc = 1.0
    protocol.run(cell_model, {}, sim=sim, rmp=rmp, rin=rin, holdi=0.0)

    print("Found threshold: %f" % cell_model.threshold_current)

    features = {}
    somav_recording = ephys.recordings.CompRecording(
        name="%s.soma.v" % protocol_name, location=soma_loc, variable="v"
    )
    recordings = [somav_recording]

    # Stimulate soma only
    step_stimulus = ephys.stimuli.NrnSquarePulse(
        step_amplitude=cell_model.threshold_current * 1.1,
        step_delay=700.0,
        step_duration=5.0,
        location=soma_loc,
        total_duration=1000.0,
    )

    # extra recordings on dendrites for bAP
    for seclist_name, secdict in points.items():
        features[seclist_name] = []

        for i, name in enumerate(secdict["name"]):
            location = ephys.locations.NrnSeclistCompLocation(
                name=name,
                comp_x=secdict["x"][i],
                sec_index=secdict["sec_id"][i],
                seclist_name=seclist_name,
            )

            var = "v"
            recording_name = "%s.%s.%s" % (protocol_name, location.name, var)
            recording = ephys.recordings.CompRecording(
                name=recording_name, location=location, variable=var
            )
            recordings.append(recording)

            efel_feature_name = "maximum_voltage_from_voltagebase"
            feature_name = "%s.%s" % (recording_name, efel_feature_name)

            recording_names = {"": "%s" % (recording_name)}

            # also set feature definition
            feature = ephys.efeatures.eFELFeature(
                feature_name,
                efel_feature_name=efel_feature_name,
                recording_names=recording_names,
                stim_start=700.0,
                stim_end=1000.0,
                exp_mean=0,
                exp_std=1,
                threshold=-20,
            )

            features[seclist_name].append(feature)

    protocol = protocols.StepProtocol(
        name=protocol_name, step_stimuli=[step_stimulus], recordings=recordings
    )

    print("Running bAP protocol")

    response = evaluator.run_protocol(
        protocol, cell_model=cell_model, param_values={}, sim=sim
    )

    print("Finished bAP protocol")

    for seclist_name, secdict in points.items():
        if "bAP_amplitude" not in secdict:
            secdict["bAP_amp"] = []
        for feature in features[seclist_name]:
            secdict["bAP_amp"].append(feature.calculate_feature(response))

    return points


def EPSP(evaluator, emodels_path, points):
    protocol_name = "EPSP"

    cell_model = evaluator.cell_model
    sim = evaluator.sim

    for seclist_name, secdict in points.items():
        if seclist_name in {"apical", "basal", "somatic"}:
            syn_weight = 1.13

        for i, name in enumerate(secdict["name"]):
            # Soma recordings
            soma_loc = ephys.locations.NrnSeclistCompLocation(
                name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
            )

            recording_name = "%s.soma.v" % protocol_name
            somav_recording = ephys.recordings.CompRecording(
                name=recording_name, location=soma_loc, variable="v"
            )
            recordings = [somav_recording]

            # Soma feature
            efel_feature_name = "maximum_voltage_from_voltagebase"
            feature_name = "%s.%s" % (recording_name, efel_feature_name)

            recording_names = {"": "%s" % (recording_name)}

            feature_soma = ephys.efeatures.eFELFeature(
                feature_name,
                efel_feature_name=efel_feature_name,
                recording_names=recording_names,
                stim_start=700.0,
                stim_end=1000.0,
                exp_mean=0,
                exp_std=1,
                threshold=-20,
            )

            # Dend recording and stimulation
            location = ephys.locations.NrnSeclistCompLocation(
                name=name,
                comp_x=secdict["x"][i],
                sec_index=secdict["sec_id"][i],
                seclist_name=seclist_name,
            )

            var = "v"
            recording_name = "%s.%s.%s" % (protocol_name, location.name, var)
            recording = ephys.recordings.CompRecording(
                name=recording_name, location=location, variable=var
            )
            recordings.append(recording)

            # Synaptic stimulation of point only
            syn_stimulus = NrnSynStim(
                syn_weight=syn_weight,
                syn_delay=700.0,
                location=location,
                total_duration=1000.0,
            )

            protocol = ephys.protocols.SweepProtocol(
                name=protocol_name,
                stimuli=[syn_stimulus],
                recordings=recordings,
                cvode_active=False,
            )

            # Dend feature
            efel_feature_name = "maximum_voltage_from_voltagebase"
            feature_name = "%s.%s" % (recording_name, efel_feature_name)

            recording_names = {"": "%s" % (recording_name)}

            # also set feature definition
            feature_dend = ephys.efeatures.eFELFeature(
                feature_name,
                efel_feature_name=efel_feature_name,
                recording_names=recording_names,
                stim_start=700.0,
                stim_end=1000.0,
                exp_mean=0,
                exp_std=1,
                threshold=-20,
            )

            response = evaluator.run_protocol(
                protocol, cell_model=cell_model, param_values={}, sim=sim
            )

            if "EPSP_amp_soma" not in secdict:
                secdict["EPSP_amp_soma"] = []
            if "EPSP_amp_dend" not in secdict:
                secdict["EPSP_amp_dend"] = []
            secdict["EPSP_amp_soma"].append(feature_soma.calculate_feature(response))
            secdict["EPSP_amp_dend"].append(feature_dend.calculate_feature(response))

    return points


class NrnSynStim(ephys.stimuli.Stimulus):
    """Square pulse current clamp injection"""

    def __init__(
        self, syn_weight=None, syn_delay=None, total_duration=None, location=None
    ):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            total_duration (float): total duration (ms)
            location (Location): stimulus Location
        """

        super(NrnSynStim, self).__init__()
        self.syn_weight = syn_weight
        self.syn_delay = syn_delay
        self.location = location
        self.total_duration = total_duration
        self.synapse = None
        self.netstim = None
        self.netcon = None

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)
        print(
            "Adding synaptic excitatory stimulus to %s with delay %f, "
            "and weight %f" % (str(self.location), self.syn_delay, self.syn_weight)
        )

        self.synapse = sim.neuron.h.ProbAMPANMDA_EMS(icomp.x, sec=icomp.sec)
        self.synapse.Use = 1.0

        self.netstim = neuron.h.NetStim(sec=icomp.sec)
        stimfreq = 70
        self.netstim.interval = 1000 / stimfreq
        self.netstim.number = 1
        self.netstim.start = self.syn_delay
        self.netstim.noise = 0
        self.netcon = sim.neuron.h.NetCon(
            self.netstim, self.synapse, 10, 0, 700, sec=icomp.sec
        )
        self.netcon.weight[0] = self.syn_weight

    def destroy(self, sim=None):
        """Destroy stimulus"""

        self.syn = None
        self.netstim = None
        self.netcon = None
