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
