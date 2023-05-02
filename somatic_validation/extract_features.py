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

from __future__ import annotations
from itertools import product
import json
from pathlib import Path
from pprint import pprint
import re
from typing import Mapping
from types import MappingProxyType

from bluepyefe.extract import (
    _read_extract,
    compute_rheobase,
    group_efeatures,
    create_feature_protocol_files,
    extract_efeatures_per_cell,
)
from bluepyefe.plotting import plot_all_recordings_efeatures


def get_protocol_timing_information() -> MappingProxyType:
    """Return the timing info for protocols."""
    timings = MappingProxyType(
        {
            "IDhyperpol": {
                "dt": 0.00025,
                "ton": 100,
                "tmid": 700,
                "tmid2": 2700.0,
                "toff": 2900.0,
                "tend": 3000.0,
            },
            "APThreshold": {"ton": 10.0, "toff": 2000},
            "IDthresh": {"ton": 700.0, "toff": 2700},
            "sAHP": {
                "dt": 0.0005,
                "ton": 20.0,
                "tmid": 520,
                "tmid2": 720,
                "toff": 2720,
                "tend": 2740,
            },
            "IV": {"dt": 0.00025, "ton": 20.0, "toff": 1020, "tend": 1200},
        }
    )
    return timings


def translate_legacy_targets(experiments: Mapping) -> list:
    """Translates the targets from legacy format to bluepyefe2 format."""
    idhyperpol_features_tmid = frozenset(
        [
            "sag_amplitude",
            "sag_ratio1",
            "minimum_voltage",
            "steady_state_voltage_stimend",
        ]
    )
    sahp_features_tmid = frozenset(
        [
            "Spikecount",
            "AP_amplitude",
            "inv_first_ISI",
            "AP_height",
            "inv_time_to_first_spike",
            "decay_time_constant_after_stim",
            "AHP_depth_abs",
        ]
    )

    targets = []
    timings = get_protocol_timing_information()

    for protocol in experiments:
        for efeature in experiments[protocol]["efeatures"]:
            for amp, tol in product(
                experiments[protocol]["targets"], experiments[protocol]["tolerances"]
            ):
                targets.append(
                    {
                        "efeature": efeature,
                        "protocol": protocol,
                        "amplitude": amp,
                        "tolerance": tol,
                    }
                )

                if protocol == "IDhyperpol" and efeature in idhyperpol_features_tmid:
                    targets[-1].update(
                        {
                            "efel_settings": {
                                "strict_stiminterval": True,
                                "stim_start": timings[protocol]["ton"],
                                "stim_end": timings[protocol]["tmid"],
                            }
                        }
                    )
                elif protocol == "sAHP" and efeature in sahp_features_tmid:
                    targets[-1].update(
                        {
                            "efel_settings": {
                                "strict_stiminterval": True,
                                "stim_start": timings[protocol]["tmid"],
                                "stim_end": timings[protocol]["tmid2"],
                            }
                        }
                    )

    return targets


def get_files_metadata(
    traces_dir: str | Path, cell_ids: tuple, experiments: Mapping
) -> Mapping:
    """Return the metadata for the files to be extracted."""
    if isinstance(traces_dir, str):
        traces_dir = Path(traces_dir)

    timings = get_protocol_timing_information()
    files_metadata = {}
    for cell in cell_ids:
        for p in (traces_dir / cell).glob("*.ibw"):
            p = str(p)
            if "A_" in p or "AA_" in p or "B_" in p or "temp" in p:
                continue

            ch = p.split("/")[-1][:-4]
            if "Ch" not in ch and "ch" not in ch:
                continue

            ch = re.findall("ch\d{1,2}", ch.lower())[0]
            if int(ch[2:]) % 2 == 0:
                continue

            i_path = p.replace(ch.lower(), "ch{}".format(0))
            v_path = p

            for prot in experiments:
                if prot not in p:
                    continue

                tracedata = {
                    "i_file": i_path,
                    "v_file": v_path,
                    "ljp": 14.0,
                    "t_unit": "s",
                    "i_unit": "A",
                    "v_unit": "V",
                }

                if prot in timings:
                    tracedata.update(timings[prot])

                if cell not in files_metadata:
                    files_metadata[cell] = {}
                if prot not in files_metadata[cell]:
                    files_metadata[cell][prot] = []
                files_metadata[cell][prot].append(tracedata)

    print(f"Cells used {len(files_metadata)}/{len(cell_ids)}")

    return files_metadata


def extract_efeatures(
    etype, files_metadata, targets, protocols_rheobase, plot=True, per_cell=False
) -> None:
    """Extract efeatures from the files."""

    efel_settings = {
        "Threshold": -30.0,
        "interp_step": 0.1,
        "strict_stiminterval": True,
    }

    cells = _read_extract(
        files_metadata=files_metadata,
        recording_reader=None,
        map_function=map,
        targets=targets,
        efel_settings=efel_settings,
    )

    compute_rheobase(
        cells,
        protocols_rheobase=protocols_rheobase,
        rheobase_strategy="absolute",
        rheobase_settings={"spike_threshold": 1},
    )

    protocols = group_efeatures(
        cells,
        targets,
        use_global_rheobase=True,
        protocol_mode="mean",
        efel_settings=efel_settings,
    )

    # stimuli of a minimum duration
    for i, prot in enumerate(protocols):
        if prot.name == "IDhyperpol":
            protocols[i].mode = "min"

    _, _, _ = create_feature_protocol_files(
        cells,
        protocols,
        output_directory=f"./{etype}",
        threshold_nvalue_save=3,
        write_files=True,
    )

    if per_cell:
        print("extracting features for per cell...")
        extract_efeatures_per_cell(
            files_metadata=files_metadata,
            cells=cells,
            output_directory=f"./{etype}",
            targets=targets,
            protocol_mode="mean",
            threshold_nvalue_save=3,
            write_files=True,
        )
    if plot:
        plot_all_recordings_efeatures(cells, protocols, output_dir=f"./{etype}")


def main():
    traces_dir = Path("..") / "feature_extraction" / "input-traces"

    cell_ids = (
        "C060109A1-SR-C1",
        "C060109A2-SR-C1",
        "C060109A3-SR-C1",
        "C070109A4-C1",
        "C080501A5-SR-C1",
        "C080501B2-SR-C1",
    )

    with open("experiments.json", "r") as f:
        experiments = MappingProxyType(json.load(f))

    print("Input experiments:")
    pprint(experiments)
    targets = translate_legacy_targets(experiments)
    print("Translated targets:")
    pprint(targets)

    files_metadata = get_files_metadata(traces_dir, cell_ids, experiments)
    etype = "L5TPC"
    protocols_rheobase = ["IDthresh", "IDRest"]
    extract_efeatures(
        etype, files_metadata, targets, protocols_rheobase, plot=False, per_cell=False
    )


if __name__ == "__main__":
    main()
