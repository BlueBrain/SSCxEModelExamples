#!/usr/bin/env python

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

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

import os


import logging

logger = logging.getLogger(__name__)

import json

json.encoder.FLOAT_REPR = lambda x: format(x, ".17g")

from collections import OrderedDict
import pickle

from .. import setup


def get_filename(etype, seed, stage):
    if stage > 1:
        return "%s_%d_s%d" % (etype, int(seed), int(abs(stage)))
    else:
        return "%s_%d" % (etype, int(seed))


class Analyse(object):

    # build cell
    def __init__(
        self,
        githash,
        seed="1",
        rank=0,
        etype=None,
        hoc=False,
        oldhoc=False,
        main_path=None,
        recipes_path="config/recipes/recipes.json",
        grouping=None,
        altmorph=None,
        stage=None,
        parameters=False,
    ):

        if grouping is None:
            grouping = ["etype", "githash", "seed", "rank", "altmorph"]
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

            self.checkpoints_dir = "checkpoints/run.%s" % self.githash

            cp_filename = os.path.join(
                self.checkpoints_dir, f"{self.etype}_{int(self.seed)}.pkl"
            )

            with open(cp_filename, "rb") as file_handle:
                cp = pickle.load(file_handle, encoding="latin1")

            hof = cp["halloffame"]

        self.set_pasts()
        self.set_evaluator()

        if self.githash is not None:
            if self.parameters is False:
                self.parameters = self.evaluator.param_dict(hof[rank])
            self.cp = cp
            self.hof_fitness_sum = sum(hof[rank].fitness.values)
            # self.hof_objectives = evaluator.objective_dict(hof[rank].fitness.values)

        print(self.parameters)

    def set_pasts(self):

        if (self.stage is not None) and (
            self.githash is not None
        ):  # just go on as before
            # get params from previous optimizations
            pasts = int(abs(self.stage)) - 1
            self.past_params = OrderedDict()

            if pasts > 0:
                rank = 0  # hardcoded, use best model

                for past in range(1, pasts + 1):  # loop over all previous stages
                    past_filename = get_filename(self.etype, self.seed, past)
                    past_path = os.path.join(
                        self.checkpoints_dir, f"{past_filename}_hof.json"
                    )
                    past_param = json.load(
                        open(past_path), object_pairs_hook=OrderedDict
                    )[rank]
                    self.past_params.update(
                        past_param
                    )  # values from later stages overwrite previous ones

            if self.stage < 0:
                self.parameters = self.past_params
                self.stage = None  # Now set to None

    def set_evaluator(self):
        if (self.stage is None) or (self.githash is None):  # just go on as before
            self.evaluator = setup.evaluator.create(
                etype=self.etype,
                altmorph=self.altmorph,
                do_simplify_morph=self.simpmorph,
            )
        else:
            self.evaluator = setup.evaluator.create(
                etype=self.etype,
                altmorph=self.altmorph,
                stochkv_det=self.stochdet,
                usethreshold=self.usethreshold,
                stage=self.stage,
                past_params=self.past_params,
                do_simplify_morph=self.simpmorph,
            )
