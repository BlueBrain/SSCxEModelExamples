from morph_tool.morphdb import MorphDB
import shutil
from pathlib import Path

if __name__ == "__main__":
    path = Path(
        "/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/06_RepairUnravel-asc/"
    )
    n_morphs = 5
    local_morph_path = Path("morphologies")
    local_morph_path.mkdir(exist_ok=True)

    df = MorphDB.from_neurondb(path / "neuronDB.xml", morphology_folder=path).df
    cells = (
        df[df.mtype == "L5_TPC:A"].sample(n_morphs, random_state=42)["path"].to_list()
    )
    for cell in cells:
        shutil.copy(cell, local_morph_path / Path(cell).name)
