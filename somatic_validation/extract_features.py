from itertools import product
from pathlib import Path
import re
import json


IDhyperpol_features_tmid = [
    "sag_amplitude",
    "sag_ratio1",
    "minimum_voltage",
    "steady_state_voltage_stimend",
]
sAHP_features_tmid = [
    "Spikecount",
    "AP_amplitude",
    "inv_first_ISI",
    "AP_height",
    "inv_time_to_first_spike",
    "decay_time_constant_after_stim",
    "AHP_depth_abs",
]

timings = {
    "IDhyperpol": {
        "dt": 0.00025,
        "ton": 100,
        "tmid": 700,
        "tmid2": 2700.0,
        "toff": 2900.0,
        "tend": 3000.0,
    },
    "APThreshold": {"ton": 10.0, "toff": 2000},
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


def translate_legacy_targets(experiments):

    targets = []

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

                if protocol == "IDhyperpol" and efeature in IDhyperpol_features_tmid:
                    targets[-1].update(
                        {
                            "efel_settings": {
                                "strict_stiminterval": True,
                                "stim_start": timings[protocol]["ton"],
                                "stim_end": timings[protocol]["tmid"],
                            }
                        }
                    )
                elif protocol == "sAHP" and efeature in sAHP_features_tmid:
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


def get_config(cells_dir, cell_ids, experiments):

    if isinstance(cells_dir, str):
        cells_dir = Path(cells_dir)

    files_metadata = {}
    for cell in cell_ids:
        for p in (cells_dir / cell).glob("*.ibw"):
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

    import pprint

    pprint.pprint(experiments)

    targets = translate_legacy_targets(experiments)
    pprint.pprint(targets)

    return files_metadata, targets


def extract_efeatures(etype, files_metadata, targets, protocols_rheobase):

    from bluepyefe.extract import (
        _read_extract,
        compute_rheobase,
        group_efeatures,
        create_feature_protocol_files,
        extract_efeatures_per_cell,
    )
    from bluepyefe.plotting import plot_all_recordings_efeatures

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

    efeatures, protocol_definitions, current = create_feature_protocol_files(
        cells,
        protocols,
        output_directory=f"./{etype}",
        threshold_nvalue_save=3,
        write_files=True,
    )

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

    plot_all_recordings_efeatures(cells, protocols, output_dir=f"./{etype}")


def main():

    cells_dir = Path("..") / "feature_extraction" / "input-traces"

    cell_ids = [
        "C060109A1-SR-C1",
        "C060109A2-SR-C1",
        "C060109A3-SR-C1",
        "C070109A4-C1",
        "C080501A5-SR-C1",
        "C080501B2-SR-C1",
    ]

    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    files_metadata, targets = get_config(cells_dir, cell_ids, experiments)
    etype = "L5PC"
    protocols_rheobase = ["IDthresh", "IDRest"]
    extract_efeatures(etype, files_metadata, targets, protocols_rheobase)


if __name__ == "__main__":
    main()
