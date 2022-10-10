#!/usr/bin/env python

"""Run simple cell optimisation"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import sys
import bluepyopt.ephys as ephys

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
logger = logging.getLogger(__name__)

import json
json.encoder.FLOAT_REPR = lambda x: format(x, '.17g')

from collections import OrderedDict
import pickle
import time
from . import plot


def makedirs(filename): # also accepts filename
    try:
        os.makedirs(os.path.dirname(filename))
    except:
        pass

def makedir(directory):
    try:
        os.makedirs(directory)
    except:
        pass

def get_filename(etype, seed, stage):
    if stage > 1:
        return '%s_%d_s%d' % (etype, int(seed), int(abs(stage)))
    else:
        return '%s_%d' % (etype, int(seed))


class Analyse(object):

    # build cell
    def __init__(self, rundir, githash,
            seed="1", rank=0,
            etype=None, # use this to evaluate model
            etypetest=None, # also a test model can be given to evaluate against the model given in etype
            hoc=False, oldhoc=False,
            main_path=None,
            recipes_path='config/recipes/recipes.json',
            grouping=['etype', 'githash', 'seed', 'rank', 'altmorph'],
            figpath='figures',
            altmorph=None,
            stage=None,
            parameters=False,
            apical_points_file="morphologies/apical_points_isec.json"
            ):

        self.githash = githash
        self.seed = seed
        self.rank = rank
        self.etypetest = etypetest
        self.etype = etype

        # options that can be used later:
        self.usethreshold = False
        self.stochdet = None
        self.simpmorph = False

        self.grouping = grouping
        self.notes = ""
        self.parameters = parameters

        self.hoc = hoc
        self.oldhoc = oldhoc

        self.altmorph = altmorph
        self.stage = stage

        self.apical_points_file = apical_points_file

        # switch to run dir and load modules
        path = os.path.join(main_path, rundir)

        sys.path.insert(0, path)
        os.chdir(path)
        logger.info('Path %s', path)

        from .. import setup
        self.setup = setup

        import inspect
        argspec = inspect.getargspec(self.setup.evaluator.create)
        self.has_etest = ('etypetest' in argspec.args)

        logger.info('Loading %s modules from %s', etype, setup.__file__)
        #self.currentdir = currentdir

        self.path_final = os.path.join(main_path, "final.json")
        self.main_path = main_path

        self.figpath = os.path.join(self.main_path, figpath)
        makedir(self.figpath)

        self.recipes = json.load(open(recipes_path))
        self.hof_fitness_sum = False

        # load from checkpoint, get parameters
        if self.githash is not None:

            self.checkpoints_dir = 'checkpoints/run.%s' % self.githash

            cp_filename = os.path.join(
                self.checkpoints_dir, f'{self.etype}_{int(self.seed)}.pkl')

            if os.path.isfile(cp_filename): # check if old checkpoint exists
                pass
            else: # checkpoint based on etype and stage
                filename = get_filename(self.etype, self.seed, self.stage)
                cp_filename = os.path.join(self.checkpoints_dir, filename + '.pkl')
                if os.path.isfile(cp_filename) is False:
                    print('No checkpoint file available, run optimization first')
                    exit(1)

            with open(cp_filename, "rb") as file_handle:
                cp = pickle.load(file_handle, encoding="latin1")

            hof = cp['halloffame']

        self.set_pasts()
        self.set_evaluator()

        if self.githash is not None:
            if self.parameters is False:
                self.parameters = self.evaluator.param_dict(hof[rank])
            self.cp = cp
            self.hof_fitness_sum = sum(hof[rank].fitness.values)
            #self.hof_objectives = evaluator.objective_dict(hof[rank].fitness.values)

        print (self.parameters)

    def set_pasts(self):

        if (self.stage is not None) and (self.githash is not None): # just go on as before
            # get params from previous optimizations
            pasts = int(abs(self.stage))-1
            self.past_params = OrderedDict()

            if pasts > 0:
                rank = 0 # hardcoded, use best model

                for past in range(1,pasts+1): # loop over all previous stages
                    past_filename = get_filename(self.etype, self.seed, past)
                    past_path = os.path.join(self.checkpoints_dir, past_filename + '_hof.json')
                    past_param = json.load(open(past_path),
                                    object_pairs_hook=OrderedDict)[rank]
                    self.past_params.update(past_param) # values from later stages overwrite previous ones

            if self.stage < 0:
                self.parameters = self.past_params
                self.stage = None # Now set to None


    def set_evaluator(self):

        if (self.stage is None) or (self.githash is None): # just go on as before
            if self.has_etest:
                self.evaluator = self.setup.evaluator.create(etype=self.etype,
                                        altmorph=self.altmorph,
                                        etypetest=self.etypetest,
                                        stochkv_det=self.stochdet,
                                        usethreshold=self.usethreshold,
                                        do_simplify_morph=self.simpmorph)
            else:
                self.evaluator = self.setup.evaluator.create(etype=self.etype,
                                        altmorph=self.altmorph,
                                        do_simplify_morph=self.simpmorph)

        else:
            if self.has_etest:
                self.evaluator = self.setup.evaluator.create(etype=self.etype,
                                        apical_points_file=self.apical_points_file,
                                        altmorph=self.altmorph,
                                        etypetest=self.etypetest,
                                        stochkv_det=self.stochdet,
                                        usethreshold=self.usethreshold,
                                        stage=self.stage, past_params=self.past_params,
                                        do_simplify_morph=self.simpmorph)
            else:
                self.evaluator = self.setup.evaluator.create(etype=self.etype,
                                        apical_points_file=self.apical_points_file,
                                        altmorph=self.altmorph,
                                        stochkv_det=self.stochdet,
                                        usethreshold=self.usethreshold,
                                        stage=self.stage, past_params=self.past_params,
                                        do_simplify_morph=self.simpmorph
                                        )

    def get_name(self):

        altmorph = self.altmorph
        if altmorph is not None:
            if isinstance(altmorph, (list)):
                altmorph = altmorph[0][1]
            altmorph = str(os.path.basename(altmorph)).split(".")[0]

        # define output label
        if self.etypetest is not None:
            reportname = 'etype:%s-test ' % self.etypetest
            label = reportname
            report_elem = OrderedDict()
            if altmorph is not None:
                report_elem['altmorph'] = altmorph
        else:
            reportname = ''
            label = ''
            if altmorph is None:
                report_elem = OrderedDict([('etype',self.etype), ('githash',self.githash)
                                        ,('seed',self.seed), ('rank',self.rank)
                                        ,('stage',self.stage)])
            else:
                report_elem = OrderedDict([('etype',self.etype), ('githash',self.githash)
                                        ,('altmorph',altmorph)
                                        ,('seed',self.seed), ('rank',self.rank)
                                        ,('stage',self.stage)])

        # stimulation options
        if self.usethreshold:
            report_elem['usethreshold'] = 'True'

        if self.stochdet is not None:
            report_elem['stochdet'] = str(self.stochdet)

        for elem_name, elem_val in report_elem.items():
            if elem_name in self.grouping:
                reportname += '%s:%s ' % (elem_name,elem_val)
            else:
                label += '%s:%s ' % (elem_name,elem_val)

        return reportname.rstrip(), label.rstrip()


    def plot_evolution(self, figs, color='b'):

        if hasattr(self, 'cp'):
            reportname, label = self.get_name()
            evol_fig = plot.evolution(self.cp['logbook'], figs=figs,
                                color=color, reportname=reportname)
            plt.show(block=False)


    def plot_diversity(self, figs, color='b'):

        if hasattr(self, 'cp'):
            reportname, label = self.get_name()

            evol_fig = plot.diversity(self.cp, evaluator=self.evaluator,
                                color=color, figs=figs,
                                reportname=reportname)
            plt.show(block=False)


    def do_model_export(self):

        if hasattr(self, 'cp') and (self.altmorph is None):
            # generate hoc

            # also make it compatible to old single-morph evaluators
            if hasattr(self.evaluator, 'evaluators'):
                evaluators = self.evaluator.evaluators
            else:
                evaluators = [self.evaluator]

            for i, evl in enumerate(evaluators):
                hoccode = evl.cell_model.create_hoc(param_values=self.parameters)
                hoc_path = os.path.join(self.checkpoints_dir,
                            "%s_%s_%s_%s_s%s.hoc" % (i, self.etype, self.githash,
                                                    self.seed, self.stage))
                with open(hoc_path, "w") as f:
                    f.write(hoccode)

            #generating parameter definition for this individual
            params_path = self.recipes[self.etype]['params']
            with open(params_path) as params_file:
                definitions = json.load(
                    params_file,
                    object_pairs_hook=OrderedDict)
            params_definitions = definitions["parameters"]

            for param_name, param_value in self.parameters.items():
                name = param_name.split(".")[0]
                location = param_name.split(".")[1]
                for param in params_definitions[location]:
                    if name == param["name"]:
                        param["val"] = param_value
                        if "test" in param:
                            del param["test"]

            path = os.path.join(self.checkpoints_dir,
                    "%s_%s_%s_s%s.json" % (self.etype, self.githash,
                                            self.seed, self.stage))

            s = json.dumps(definitions, indent=2)
            s = self.collapse_json(s, indent=6)
            with open(path, "w") as f:
                f.write(s)


    def create_cell_model(self, evaluator):

        self.sim = evaluator.sim

        # only use alternative mode, e.g. hoc if etypetest is given
        if self.etypetest is not None:

            # generate parameters
            parameters = {}

            params_path = self.recipes[self.etypetest]['params']
            with open(params_path) as params_file:
                definitions = json.load(
                    params_file,
                    object_pairs_hook=OrderedDict)
            params_definitions = definitions["parameters"]

            if "__comment" in params_definitions:
                del params_definitions["__comment"]

            for sectionlist, params in params_definitions.items():
                for param_config in params:
                    param_name = param_config["name"]
                    if isinstance(param_config["val"], (list, tuple)):
                        test = param_config["test"]
                        parameters['%s.%s' % (param_name, sectionlist)] = test

            self.parameters = parameters

            for evl in self.evaluators:

                morph = evl.cell_model.morphology.morphology_path
                if self.hoc:
                    evl.sim.neuron.h.celsius = 34
                    evl.sim.neuron.h.v_init = -80
                    evl.cell_model = ephys.models.HocCellModel(
                                        'hoc', morph, self.hoc)

                elif self.oldhoc:
                    raise NotImplementedError(
                        "old hoc model is not used in the reproducibility notebook.")


    def sim_objectives(self):

        # also make it compatible to old single-morph evaluators
        if hasattr(self.evaluator, 'evaluators'):
            self.evaluators = self.evaluator.evaluators
        else:
            self.evaluators = [self.evaluator]

        self.create_cell_model(self.evaluator)

        start_time_all = time.time()

        features = []
        responses = {}
        objectives = {}

        for evl in self.evaluators:

            fitness_protocols = evl.fitness_protocols

            for protocol in fitness_protocols.values():
                start_time = time.time()
                response = evl.run_protocol(protocol,
                    param_values=self.parameters)
                responses.update(response)
                logger.info(" Ran protocol in %f seconds",
                                time.time() - start_time)

            for obj in evl.fitness_calculator.objectives:
                for feature in obj.features:
                    features.append(feature.calculate_feature(responses))

            score = evl.fitness_calculator.calculate_scores(responses)
            objectives.update(score)

        self.features = features
        self.responses = responses
        self.objectives = objectives

        logger.info(" Full evaluation took %f seconds",
                        time.time() - start_time_all)


    def plot_obj(self, figs, color='b', split_sim=1):

        reportname, label = self.get_name()
        responses_fig = plot.responses(self.responses, figs=figs,
                        color=color, cols=split_sim,
                        reportname=reportname,
                        label=label)
        plt.show(block=False)

        objectives_fig = plot.objectives(self.objectives, figs=figs,
                            color=color, reportname=reportname,
                            label=label)

        if self.hof_fitness_sum:
            objsum = sum(self.objectives.values())
            msg = ("Sum of scores from hof: %s, from simulation: %s" %
                            (self.hof_fitness_sum, objsum) )
            logger.info(" " + msg)

            if (self.usethreshold is False) and (self.stochdet is None):
                # check if equal to optimization
                if abs(objsum-self.hof_fitness_sum) > 0.5 * 10**(-6):
                    color = 'red'
                    prefix = 'ERROR:'
                else:
                    color = 'green'
                    prefix = 'GOOD:'

                msg = (prefix + " Sum of scores " +
                        "from HOF: %0.2f, from simulation: %0.2f diff: %0.4f" %
                        (self.hof_fitness_sum, objsum, abs(objsum-self.hof_fitness_sum)) )

                objectives_fig.text(0.05, 0.5, msg,
                         fontsize=15, color=color,
                         ha='center', va='center',
                         rotation=90)

            plt.show(block=False)

        return reportname


    def sim_plot_obj(self, figs={}, color='b', split_sim=1):

        self.sim_objectives()
        self.plot_obj(figs=figs, color=color, split_sim=split_sim)


    def save_pdf(self, reportname, figs, subdir='figures'):
        reportfile = reportname.rstrip().replace (" ", "_").replace (":", "_")
        report_path = os.path.join(self.figpath, reportfile + '.pdf')
        pdf_pages = PdfPages(report_path)
        for figname, fig in figs.items():
            pdf_pages.savefig(fig['fig'])
        pdf_pages.close()

