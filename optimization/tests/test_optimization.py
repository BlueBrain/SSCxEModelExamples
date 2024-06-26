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

"""Test covering the optimisation notebook's functionality."""

from contextlib import contextmanager
import os
from pathlib import Path
import pickle
import subprocess
import json

import efel
import numpy as np
import pytest

from opt_module.tools.plot import diversity
from opt_module.setup.evaluator import create as create_evaluator
from opt_module.tools.analyse import Analyse


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def get_responses(cell_evaluator, top_individual):
    """Get voltage, calcium recording, threshold and holding currents from the individual."""
    individual_dict = cell_evaluator.param_dict(top_individual)
    responses = cell_evaluator.run_protocols(
        cell_evaluator.fitness_protocols.values(), param_values=individual_dict
    )

    return responses


class TestOptimizationNotebook:
    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        checkpoint_file = (
            Path(__file__).parent.parent
            / "opt_module"
            / "checkpoints"
            / "run.a6e707a"
            / "cADpyr_L5TPC_1.pkl"
        )
        cls.metype = "cADpyr_L5TPC"

        apical_points_file = os.path.join(
            ".", "opt_module", "morphologies", "apical_points_isec.json"
        )
        cls.evaluator = create_evaluator(cls.metype, apical_points_file)

        with open(checkpoint_file, "rb") as file_handle:
            cls.cp = pickle.load(file_handle, encoding="latin1")

    def test_checkpoint_keys(self):
        assert self.cp.keys() == {
            "history",
            "parents",
            "generation",
            "logbook",
            "rndstate",
            "halloffame",
            "population",
        }

    def test_checkpoint_logbook(self):
        logbook = self.cp["logbook"]
        gen_numbers = logbook.select("gen")
        assert gen_numbers == [x for x in range(1, 101)]

        mean = np.array(logbook.select("avg"))
        mean_gt = np.loadtxt("tests/logbook_mean.dat")
        assert np.array_equal(mean, mean_gt)

        std = np.array(logbook.select("std"))
        std_gt = np.loadtxt("tests/logbook_std.dat")
        assert np.array_equal(std, std_gt)

        minimum = np.array(logbook.select("min"))
        minimum_gt = np.loadtxt("tests/logbook_min.dat")
        assert np.array_equal(minimum, minimum_gt)
        minmin = min(minimum)
        assert minmin == 61.593137593478396

    def test_evaluator_params(self):
        evaluator_params = {param.name: param.bounds for param in self.evaluator.params}
        with open("tests/evaluator_params.json") as f_handle:
            evaluator_params_gt = json.load(f_handle)

        assert evaluator_params == evaluator_params_gt

    def test_hof_objectives(self):
        hof = self.cp["halloffame"]
        rank = 0
        hof_objectives = self.evaluator.objective_dict(hof[rank].fitness.values)
        with open("tests/hof_objectives.json") as f_handle:
            hof_objectives_gt = json.load(f_handle)

        assert hof_objectives == hof_objectives_gt

        sum_of_objectives = sum(hof_objectives.values())
        sum_of_objectives_on_notebook = 61.59313759347834
        assert sum_of_objectives == pytest.approx(sum_of_objectives_on_notebook)

    def test_diversity_data(self):
        param_names = self.evaluator.param_names
        n_params = len(param_names)
        assert n_params == 31

        hof = self.cp["halloffame"]
        fitness_cut_off = 2.0 * sum(hof[0].fitness.values)
        fitness_cut_off_on_notebook = 123.18627518695679
        assert fitness_cut_off == pytest.approx(fitness_cut_off_on_notebook)

    def test_render_diversity_plot(self):
        """To test if diversity fig is generated without errors."""
        diversity_fig = diversity(self.cp, self.evaluator)

    def test_responses(self):
        maindir = os.path.join(".", "opt_module")

        rank = 0
        seed = 1
        stage = 2
        etype = self.metype

        compilation_output = subprocess.run(
            ["nrnivmodl", "opt_module/mechanisms"], capture_output=True, check=True
        )

        with cwd("opt_module"):
            analysis_obj = Analyse(
                githash="a6e707a",
                seed=seed,
                rank=rank,
                etype=etype,
                main_path=maindir,
                recipes_path="./config/recipes/recipes.json",
                stage=stage,
            )

            hof = self.cp["halloffame"]
            top_individual = hof[0]
            analysis_obj.set_evaluator()

            responses = get_responses(
                analysis_obj.evaluator.evaluators[0], top_individual
            )

        assert responses.keys() == {
            "L5TPCa.RMP.soma.v",
            "L5TPCa.Rin.soma.v",
            "L5TPCa.bpo_holding_current",
            "L5TPCa.bpo_threshold_current",
            "L5TPCa.bAP.soma.v",
            "L5TPCa.bAP.dend1.v",
            "L5TPCa.bAP.dend2.v",
            "L5TPCa.bAP.ca_prox_apic.cai",
            "L5TPCa.bAP.ca_prox_basal.cai",
            "L5TPCa.bAP.ca_soma.cai",
            "L5TPCa.bAP.ca_ais.cai",
            "L5TPCa.Step_150.soma.v",
            "L5TPCa.Step_200.soma.v",
            "L5TPCa.Step_280.soma.v",
            "L5TPCa.APWaveform_320.soma.v",
            "L5TPCa.IV_-100.soma.v",
            "L5TPCa.SpikeRec_600.soma.v",
        }

        currents = {k: v for k, v in responses.items() if k.endswith("current")}

        assert currents["L5TPCa.bpo_holding_current"] == pytest.approx(
            -0.14824624653402146
        )
        assert currents["L5TPCa.bpo_threshold_current"] == pytest.approx(
            0.4762841614252752
        )

        voltage_responses = {k: v for k, v in responses.items() if k.endswith(".v")}

        # use the step voltage to make sure there are spikes, thus features can be extracted
        step_voltage_responses = {
            k: v for k, v in voltage_responses.items() if k.startswith("L5TPCa.Step")
        }

        for v_key, v_value in step_voltage_responses.items():
            result = v_value.response
            gt = np.loadtxt(f"tests/voltage_responses/{v_key}.dat")
            # use eFEL features to compare voltages since NEURON sampling rate varies
            result_trace = {}
            result_trace["T"] = result.time.values
            result_trace["V"] = result.voltage.values
            result_trace["stim_start"] = [0]
            result_trace["stim_end"] = [len(result) - 1]

            gt_trace = {}
            gt_trace["T"] = gt[:, 0]
            gt_trace["V"] = gt[:, 1]
            gt_trace["stim_start"] = [0]
            gt_trace["stim_end"] = [len(gt) - 1]

            traces = [result_trace, gt_trace]

            traces_results = efel.getFeatureValues(
                traces,
                [
                    "minimum_voltage",
                    "maximum_voltage",
                    "Spikecount",
                    "min_voltage_between_spikes",
                    "AP_duration",
                    "peak_indices",
                    "peak_time",
                    "peak_voltage",
                ],
            )

            assert np.allclose(
                traces_results[0]["minimum_voltage"],
                traces_results[1]["minimum_voltage"],
                atol=1e-2,
            )
            assert np.allclose(
                traces_results[0]["maximum_voltage"],
                traces_results[1]["maximum_voltage"],
            )
            assert np.allclose(
                traces_results[0]["Spikecount"], traces_results[1]["Spikecount"]
            )
            assert np.allclose(
                traces_results[0]["min_voltage_between_spikes"],
                traces_results[1]["min_voltage_between_spikes"],
                atol=1e-1,
            )
            assert np.allclose(
                traces_results[0]["AP_duration"],
                traces_results[1]["AP_duration"],
                atol=1e-1,
            )
            assert np.allclose(
                traces_results[0]["peak_indices"],
                traces_results[1]["peak_indices"],
                atol=5,
            )
            assert np.allclose(
                traces_results[0]["peak_time"], traces_results[1]["peak_time"], atol=1
            )
            assert np.allclose(
                traces_results[0]["peak_voltage"],
                traces_results[1]["peak_voltage"],
                atol=2,
            )

        ca_responses = {k: v for k, v in responses.items() if k.endswith(".cai")}
        for c_key, c_value in ca_responses.items():
            calcium_response = c_value.response
            calcium_response_gt = np.loadtxt(f"tests/calcium_responses/{c_key}.dat")
            assert np.allclose(calcium_response, calcium_response_gt, atol=1e-2)
