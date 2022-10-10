"""Python Model Management"""

import json

import argparse

from lib import attenuation


def main():
    """Main"""

    print("\n###################################")
    print("# Starting Attenuation Validation #")
    print("###################################\n")
    args = parse_args()
    run(args)


def parse_args(arg_list=None):
    """Parse the arguments"""

    parser = argparse.ArgumentParser(description="Blue Brain Model Management")
    parser.add_argument("conf_filename")
    parser.add_argument("--ipyp", action="store_true")
    parser.add_argument("--ipyp_profile")

    return parser.parse_args(arg_list)


def run(args):
    """Run the program"""
    print("Reading configuration at %s" % args.conf_filename)

    # Read configuration
    conf_dict = json.loads(open(args.conf_filename).read())

    meg_output = conf_dict["meg_output"]
    morphs_path = conf_dict["morphs_path"]
    apical_points = conf_dict["apical_points"]
    emodels_path = conf_dict["emodels_path"]
    emodel_hash = conf_dict["emodel_hash"]
    emodel_seed = conf_dict["emodel_seed"]
    emodel = conf_dict["emodel"]
    mtype_morph = conf_dict["mtype_morph"]
    only_morph = conf_dict["only_morph"]
    n_morph = conf_dict["n_morph"]
    output = conf_dict["output"]

    print("Calculating attenuation")
    # Calculate scores for combinations in sqlite3 db
    attenuation.calculate(
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
        use_ipyp=args.ipyp,
        ipyp_profile=args.ipyp_profile,
    )

    print("Attenuation validation finished\n")


if __name__ == "__main__":
    main()
