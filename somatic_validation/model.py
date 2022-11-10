import os
import collections
import math
import numpy
import json
import random

import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(__file__)
config_dir = os.path.join(script_dir, 'config')

def multi_locations(sectionlist):
    """Define mechanisms"""

    if sectionlist == "alldend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal")
        ]
    elif sectionlist == "somadend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation(
                "apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation(
                "basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation(
                "somatic", seclist_name="somatic")
        ]
    elif sectionlist == "somaxon":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation(
                "axonal", seclist_name="axonal"),
            ephys.locations.NrnSeclistLocation(
                "somatic", seclist_name="somatic")
        ]
    elif sectionlist == "allact":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation(
                "apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation(
                "basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation(
                "somatic", seclist_name="somatic"),
            ephys.locations.NrnSeclistLocation(
                "axonal", seclist_name="axonal")
        ]
    else:
        seclist_locs = [ephys.locations.NrnSeclistLocation(
            sectionlist,
            seclist_name=sectionlist)]

    return seclist_locs


def define_mechanisms(path_params):
    """Define mechanisms"""
    
    with open(path_params) as params_file:
        mech_definitions = json.load(params_file)["mechanisms"]
    
    mechanisms_list = []
    for sectionlist, channels in mech_definitions.items():
        seclist_locs = multi_locations(sectionlist)
        for channel in channels["mech"]:
            mechanisms_list.append(ephys.mechanisms.NrnMODMechanism(
                name='%s.%s' % (channel, sectionlist),
                mod_path=None,
                prefix=channel,
                locations=seclist_locs,
                preloaded=True))
    return mechanisms_list


def define_parameters(path_params):
    """Define parameters"""
    
    stage = None
    past_params = []

    parameters = []
    # Fixed section parameters
    with open(path_params) as params_file:
        definitions = json.load(params_file, object_pairs_hook=collections.OrderedDict)

    # set distributions
    distributions = collections.OrderedDict()
    distributions["uniform"] = ephys.parameterscalers.NrnSegmentLinearScaler()

    distributions_definitions = definitions["distributions"]
    for distribution, definition in distributions_definitions.items():

        if "parameters" in definition:
            dist_param_names = definition["parameters"]
        else:
            dist_param_names = None
        distributions[distribution] = \
            ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                name=distribution ,distribution=definition["fun"],
                dist_param_names=dist_param_names)

    params_definitions = definitions["parameters"]

    if "__comment" in params_definitions:
        del params_definitions["__comment"]

    for sectionlist, params in params_definitions.items():
        if sectionlist == 'global':
            seclist_locs = None
            is_global = True
            is_dist = False
        elif 'distribution_' in sectionlist:
            is_dist = True
            seclist_locs = None
            is_global = False
            dist_name = sectionlist.split('distribution_')[1]
            dist = distributions[dist_name]
        else:
            seclist_locs = multi_locations(sectionlist)
            is_global = False
            is_dist = False

        bounds = None
        value = None
        for param_config in params:
            param_name = param_config["name"]

            if isinstance(param_config["val"], (list, tuple)):

                full_name = '%s.%s' % (param_name, sectionlist)

                # check and define stage of this parameter,
                # if not given, stage is 1
                if "stage" in param_config:
                    this_stage = param_config["stage"]
                else:
                    this_stage = [1]

                if (stage is None):
                    # used for analysis, use previous values or leave as bounds
                    if full_name in past_params:
                        is_frozen = True
                        value = past_params[full_name]
                        bounds = None
                        logger.debug(
                            'Param %s, use value %f from past_params' % (full_name, value))
                    else: # optimize
                        is_frozen = False
                        bounds = param_config["val"]
                        value = None

                elif (stage in this_stage):
                    # use for optimization here
                    is_frozen = False
                    bounds = param_config["val"]
                    value = None

                elif stage > max(this_stage):
                    # optimization was done in an earlier stage
                    # use previous values
                    is_frozen = True
                    value = past_params[full_name]
                    bounds = None
                    logger.debug(
                        'Param %s, use value %f from stage %s, stage now %s' % (full_name, value, this_stage, stage))

                elif stage < min(this_stage):
                    # not yet fitted yet, set to 0
                    is_frozen = True
                    value = 0
                    bounds = None

                    logger.debug(
                        'Param %s, not yet used, set to 0, use in %s, stage now %s' % (full_name, this_stage, stage))

            else:
                is_frozen = True
                value = param_config["val"]
                bounds = None

            if is_global:
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_name,
                        param_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value))
            elif is_dist:
                parameters.append(
                    ephys.parameters.MetaParameter(
                        name='%s.%s' % (param_name, sectionlist),
                        obj=dist,
                        attr_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value))

            else:
                if "dist" in param_config:
                    dist = distributions[param_config["dist"]]
                    use_range = True
                else:
                    dist = distributions["uniform"]
                    use_range = False

                if use_range:
                    parameters.append(ephys.parameters.NrnRangeParameter(
                        name='%s.%s' % (param_name, sectionlist),
                        param_name=param_name,
                        value_scaler=dist,
                        value=value,
                        bounds=bounds,
                        frozen=is_frozen,
                        locations=seclist_locs))
                else:
                    parameters.append(ephys.parameters.NrnSectionParameter(
                        name='%s.%s' % (param_name, sectionlist),
                        param_name=param_name,
                        value_scaler=dist,
                        value=value,
                        bounds=bounds,
                        frozen=is_frozen,
                        locations=seclist_locs))

    return parameters


from bluepyopt.ephys.morphologies import NrnFileMorphology

class NrnFileMorphologyCustom(NrnFileMorphology):

    def __init__(
            self,
            morphology_path,
            do_replace_axon=False,
            do_set_nseg=True,
            comment='',
            replace_axon_hoc=None):

        name = os.path.basename(morphology_path)
        super(NrnFileMorphologyCustom, self).__init__(morphology_path, do_replace_axon,
                                        do_set_nseg, comment, replace_axon_hoc)


    def instantiate(self, sim=None, icell=None):
        """Load morphology"""

        super(NrnFileMorphologyCustom, self).instantiate(sim, icell)

    def section_area(self, sim, section):
        """Section area"""

        return sum(sim.neuron.h.area(seg.x, sec=section) for seg in section)

    def cell_area(self, sim):
        """Cell area"""

        total_area = 0
        for section in sim.neuron.h.allsec():
            total_area += self.section_area(sim, section)

        return total_area


    #@@staticmethod
    def set_nseg(self, icell):
        """Set the nseg of every section"""

        if self.do_set_nseg:
            if self.do_set_nseg == True:
                div = 40
            else:
                div = self.do_set_nseg

            logger.debug(
                'Using set_nseg divider %f' % div)

        for section in icell.all:
            section.nseg = 1 + 2 * int(section.L / div)


    def replace_axon(self, sim=None, icell=None):
        """Replace axon"""

        L_target = 60  # length of stub axon
        nseg0 = 5  # number of segments for each of the two axon sections

        nseg_total = nseg0 * 2
        chunkSize = L_target / nseg_total

        diams = []
        lens = []

        count = 0
        for section in icell.axonal:
            L = section.L
            nseg = 1 + int(L / chunkSize / 2.) * 2  # nseg to get diameter
            section.nseg = nseg

            for seg in section:
                count = count + 1
                diams.append(seg.diam)
                lens.append(L / nseg)
                if count == nseg_total:
                    break
            if count == nseg_total:
                break

        for section in icell.axonal:
            sim.neuron.h.delete_section(sec=section)

        #  new axon array
        sim.neuron.h.execute('create axon[2]', icell)

        L_real = 0
        count = 0

        for index, section in enumerate(icell.axon):
            section.nseg = int(nseg_total / 2)
            section.L = int(L_target / 2)

            for seg in section:
                seg.diam = diams[count]
                L_real = L_real + lens[count]
                count = count + 1

            icell.axonal.append(sec=section)
            icell.all.append(sec=section)

        #childsec.connect(parentsec, parentx, childx)
        icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
        icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

        sim.neuron.h.execute('create myelin[1]', icell)
        icell.myelinated.append(sec=icell.myelin[0])
        icell.all.append(sec=icell.myelin[0])
        icell.myelin[0].nseg = 5
        icell.myelin[0].L = 1000
        icell.myelin[0].diam = diams[count-1]
        icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)

        logger.debug(
            'Replace axon with tapered AIS of length %f, target length was %f, diameters are %s' %
            (L_real, L_target, diams))

replace_axon_hoc = """
    proc replace_axon(){ local nSec, L_chunk, dist, i1, i2, count, L_target, chunkSize, L_real localobj diams, lens

        L_target = 60  // length of stub axon
        nseg0 = 5  // number of segments for each of the two axon sections

        nseg_total = nseg0 * 2
        chunkSize = L_target/nseg_total

        nSec = 0
        forsec axonal{nSec = nSec + 1}

        // Try to grab info from original axon
        if(nSec < 3){ //At least two axon sections have to be present!

            execerror("Less than three axon sections are present! This emodel can't be run with such a morphology!")

        } else {

            diams = new Vector()
            lens = new Vector()

            access axon[0]
            axon[0] i1 = v(0.0001) // used when serializing sections prior to sim start
            axon[1] i2 = v(0.0001) // used when serializing sections prior to sim start
            axon[2] i3 = v(0.0001) // used when serializing sections prior to sim start

            count = 0
            forsec axonal{ // loop through all axon sections

                nseg = 1 + int(L/chunkSize/2.)*2  //nseg to get diameter

                for (x) {
                    if (x > 0 && x < 1) {
                        count = count + 1
                        diams.resize(count)
                        diams.x[count-1] = diam(x)
                        lens.resize(count)
                        lens.x[count-1] = L/nseg
                        if( count == nseg_total ){
                            break
                        }
                    }
                }
                if( count == nseg_total ){
                    break
                }
            }

            // get rid of the old axon
            forsec axonal{delete_section()}
            execute1("create axon[2]", CellRef)

            L_real = 0
            count = 0

            // new axon dependant on old diameters
            for i=0,1{
                access axon[i]
                L =  L_target/2
                nseg = nseg_total/2

                for (x) {
                    if (x > 0 && x < 1) {
                        diam(x) = diams.x[count]
                        L_real = L_real+lens.x[count]
                        count = count + 1
                    }
                }

                all.append()
                axonal.append()

                if (i == 0) {
                    v(0.0001) = i1
                } else {
                    v(0.0001) = i2
                }
            }

            nSecAxonal = 2
            soma[0] connect axon[0](0), 1
            axon[0] connect axon[1](0), 1

            create myelin[1]
            access myelin{
                    L = 1000
                    diam = diams.x[count-1]
                    nseg = 5
                    v(0.0001) = i3
                    all.append()
                    myelinated.append()
            }
            connect myelin(0), axon[1](1)
        }
    }
"""


def define_morphology(path):
    """Define morphology"""
    return NrnFileMorphologyCustom(path,
                                   do_replace_axon=True,
                                   replace_axon_hoc=replace_axon_hoc,
                                   do_set_nseg=40.)


def create(path,path_params=False):
    """Create cell template"""

    cell = ephys.models.CellModel(
            "l6pc",
            morph=define_morphology(path),
            mechs=define_mechanisms(path_params),
            params=define_parameters(path_params))

    return cell
