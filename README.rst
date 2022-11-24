An example of the creation, validation and generalization of L5PC canonical e-model. 
-----------------------

|build| |black|

Introduction
---------

The detailed neuronal model is one of the central tools in neuroscience. Here we present a demo of the single neuron e-model creation, validation, and generalization described in "A universal workflow for creation, validation, and generalization of detailed neuronal models" (Reva, Rossert et al) paper. The demo is built on the example of L5PC of the SSCx in juvenile rat. 

Single cell e-model pipeline
---------
1. E-feature extraction
2. E-model optimization
3. E-model validation
4. Generalization

E-feature extraction
________

``/feature_extraction folder`` contains data and codes that allow the extraction of electrical features (e-features) from the voltage traces. E-features are extracted for six L5PCs, their traces are located in ``feature_extraction/input-traces``.
The `requirements.txt` at the main directory needs to be installed to run the feature-extraction.ipynb. It can be done via `pip install -r requirements.txt`.

E-model optimization
________
``/optimization contains`` tools and codes necessary to run and display the result of the canonical e-model optimization. Install `requirements.txt` at the main directory and run ``Minimal_cADpyr_L5TPC_Optimization.ipynb`` to visualize the results of the optimization. Prior to launching the notebook, one needs to compile e-model' mechanisms by running ``compile_mechanisms.sh``.
 ``/optimization/opt_module`` contains tools and data necessary for the optimization such as morphology, mechanisms, checkpoints, and config files.

E-model validations
_________
Two types of validation were performed for the optimized L5PC e-model.

- The visualization of the bAP/EPSP validations can be found in ``validation.ipynb`` in the ``/validation`` folder. The morphologies for these validations are located in ``input/morphologies``. To run bAP/EPSP validations use ``python main.py att_conf.json``. The requirements.txt at the main directory need to be installed and the mechanisms need to be compiled with `nrnivmodl mechanisms`.
- Somatic validations are located in ``/somatic_validation`` folder.
This is the only step that does not use the `requirements.txt` in the main directory.
`somatic-val-requirements.txt` needs to be installed and the mechanisms need to be compiled with `nrnivmodl mechanisms` prior to running the notebooks.
 First, e-feature for the validations have to be extracted from the chosen patch clamp protocol. To extract e-features use ``feature-extraction.ipynb``, the results of this extraction can be found in ``/somatic_validation/L5TPC``. To run and visualize results of the somatic validation run ``somatic-validation.ipynb``.

Generalization
______________

The requirements.txt at the main directory needs to be installed.
The generalization of the canonical L5PC e-model to a number of morphologies is done with the model-management tool. To run a generalization use ``model-management.ipynb`` in ``/model_management/mm_run_minimal``. The morphologies used in the step can be found in ``/model_management/mm_run_minimal/morphologies/``.


Testing
---------

Each step is of the pipeline contains tests.
Before running the notebooks, we recommend running the tests to make sure you will get the expected results.
To run the tests, in addition to the `requirements.txt` you need to install `test-requirements.txt` and then you can run the tests using `pytest`.

The github workflow located at `.github/workflows/python-app.yml` contains the complete sequence of commands needed to run the tests.


Requirements
---------

The requirements.txt at the main directory should be used for all steps except for the somatic validations.
Install `somatic_validation/somatic-val-requirements.txt` before running the somatic validation notebooks or tests.


Reference
---------

.. |build| image:: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml/badge.svg
                :target: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml
                :alt: Build Status
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
