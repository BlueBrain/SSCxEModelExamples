import os
import shutil
import json

runpath = "./run"

subdirs = [d for d in os.listdir(runpath) if os.path.isdir(os.path.join(runpath, d))]

for subdir in subdirs:
    path = os.path.join(runpath, subdir)
    results_path = os.path.join(path, "output/results.json")

    if os.path.isfile(results_path) is False:
        shutil.rmtree(path)
    else:
        results = json.load(open(results_path))
        noinfo = True
        for uid, result in results.items():
            point = result["points"]
            if point is not None:
                noinfo = False
        if noinfo:
            print(f"{results_path} has no results, deleting dir")
            shutil.rmtree(path)
