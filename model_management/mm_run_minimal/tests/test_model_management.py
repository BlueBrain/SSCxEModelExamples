"""Test to validate the model management output."""

from pathlib import Path
import shutil

import bluepymm
from bluepymm import prepare_combos, run_combos, select_combos
import pandas as pd
import pytest

test_dir = "tests"


@pytest.fixture(autouse=True)
def setup_tests():
    """Deletes the tmp directory used in prepare step."""
    tmp_path = Path(test_dir) / "tmp"
    if tmp_path.exists() and tmp_path.is_dir():
        shutil.rmtree(tmp_path)


def test_select_output():
    """Runs all 3 steps and checks the output of select step."""
    prepare_config = "prepare.json"
    with bluepymm.tools.cd(test_dir):
        prepare_combos.prepare_combos(conf_filename=prepare_config, continu=False)

    run_config = "run.json"
    with bluepymm.tools.cd(test_dir):
        run_combos.run_combos(conf_filename=run_config)

    select_config = "select.json"
    with bluepymm.tools.cd(test_dir):
        select_combos.select_combos(conf_filename=select_config, n_processes=None)

    gt_df = pd.read_csv("tests/ground_truth/mecombo_emodel.tsv", sep="\t")
    res_df = pd.read_csv("tests/output_select/mecombo_emodel.tsv", sep="\t")

    pd.testing.assert_frame_equal(res_df, gt_df)
