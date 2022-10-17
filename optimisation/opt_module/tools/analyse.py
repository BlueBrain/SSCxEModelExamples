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

import os


import logging
logger = logging.getLogger(__name__)

import json
json.encoder.FLOAT_REPR = lambda x: format(x, '.17g')

from collections import OrderedDict
import pickle

from .. import setup


def get_filename(etype, seed, stage):
    if stage > 1:
        return '%s_%d_s%d' % (etype, int(seed), int(abs(stage)))
    else:
        return '%s_%d' % (etype, int(seed))


class Analyse(object):

    # build cell
    def __init__(self, githash,
            seed="1", rank=0,
            etype=None, # use this to evaluate model
            hoc=False, oldhoc=False,
            main_path=None,
            recipes_path='config/recipes/recipes.json',
            grouping=['etype', 'githash', 'seed', 'rank', 'altmorph'],
            altmorph=None,
            stage=None,
            parameters=False,
            ):

        self.githash = githash
        self.seed = seed
        self.rank = rank
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

        self.path_final = os.path.join(main_path, "final.json")
        self.main_path = main_path

        self.recipes = json.load(open(recipes_path))
        self.hof_fitness_sum = False

        # load from checkpoint, get parameters
        if self.githash is not None:

            self.checkpoints_dir = 'checkpoints/run.%s' % self.githash

            cp_filename = os.path.join(
                self.checkpoints_dir, f'{self.etype}_{int(self.seed)}.pkl')

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
            self.evaluator = setup.evaluator.create(etype=self.etype,
                                        altmorph=self.altmorph,
                                        do_simplify_morph=self.simpmorph)
        else:
            self.evaluator = setup.evaluator.create(etype=self.etype,
                                    altmorph=self.altmorph,
                                    stochkv_det=self.stochdet,
                                    usethreshold=self.usethreshold,
                                    stage=self.stage, past_params=self.past_params,
                                    do_simplify_morph=self.simpmorph
                                    )
