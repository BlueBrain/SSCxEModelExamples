Creation, validation and generalization of a canonical electrical model
=======================================================================

|build| |black|

Introduction
---------

The biophysically detailed electrical neuron model (e-model) is one of the central tools in neuroscience. Here we present a demo of the single neuron e-model creation, validation, and generalization described in "A universal workflow for creation, validation, and generalization of detailed neuronal models" (Reva, Rossert et al). This demo is built on the example of L5PC of the SSCx in juvenile rat. 

Pipeline
---------

1. E-feature extraction
~~~~~~~~~~~~~~~~~~~~~~~
The `feature_extraction <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction>`_ folder contains data and codes that allow the extraction of electrical features (e-features) from the voltage traces.

E-features are extracted for six L5PCs, their traces are located in the `feature_extraction/input-traces <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces>`_ folder.

The `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ at the main directory needs to be installed to run the `feature-extraction.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/feature_extraction/feature-extraction.ipynb>`_.

Run the following command to install the dependencies::

    pip install -r requirements.txt

2. Optimization
~~~~~~~~~~~~~~~

The `optimization <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/optimization>`_ folder contains tools and codes necessary to run and display the result of the canonical e-model optimization.

Install the `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ at the main directory and run `Minimal_cADpyr_L5TPC_Optimization.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/optimization/Minimal_cADpyr_L5TPC_Optimization.ipynb>`_ to visualize the results of the optimization.

Prior to launching the notebook, one needs to compile e-model' mechanisms by running::

    sh compile_mechanisms.sh

The `optimization/opt_module <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/optimization/opt_module>`_ folder contains tools and data necessary for the optimization such as morphology, mechanisms, checkpoints, and config files.

3. Validation
~~~~~~~~~~~~~
Two types of validation were performed for the optimized L5PC e-model.

The visualization of the bAP/EPSP validations can be found in `validation.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/validation/validation.ipynb>`_ notebook.

The morphologies for these validations are located in the `input/morphologies <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/validation/input/morphologies>`_ folder.

To run bAP/EPSP validations use:: 

  python main.py att_conf.json

The `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ at the main directory needs to be installed and the mechanisms need to be compiled with::

  nrnivmodl mechanisms

Somatic validations are located in the `somatic_validation <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/somatic_validation>`_ folder.

Note that this is the only step that does not use the ``requirements.txt`` in the main directory.

`somatic-val-requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/somatic_validation/somatic-val-requirements.txt>`_ needs to be installed and the mechanisms need to be compiled with the following command before running the notebooks:: 

  nrnivmodl mechanisms 
  
First, e-features for the validations have to be extracted from the chosen patch clamp protocol. To extract e-features use `feature-extraction.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/somatic_validation/feature-extraction.ipynb>`_, the results of this extraction can be found in the `somatic_validation/L5TPC <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/somatic_validation/L5TPC>`_ folder. To run and visualize results of the somatic validation run `somatic-validation.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/somatic_validation/somatic-validation.ipynb>`_.

4. Generalization
~~~~~~~~~~~~~~

Once again the `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ at the main directory needs to be installed.

The generalization of the canonical L5PC e-model to a number of morphologies is done with `BluePyMM <https://github.com/BlueBrain/BluePyMM>`_.

To run a generalization use the `model-management.ipynb <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/model_management/mm_run_minimal/model-management.ipynb>`_ notebook in `/model_management/mm_run_minimal <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/model_management/mm_run_minimal>`_ directory.

The morphologies used in the step can be found in the `/model_management/mm_run_minimal/morphologies/ <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/model_management/mm_run_minimal/morphologies>`_ folder.


Testing
---------

Each step is of the pipeline contains tests.
Before running the notebooks, we recommend running the tests to make sure you will get the expected results.

To run the tests, in addition to the `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ you need to install `test-requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/test-requirements.txt>`_ and then you can run the tests using ``pytest``.

The github workflow located at `.github/workflows/python-app.yml <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/.github/workflows/python-app.yml>`_ contains the complete sequence of commands needed to run the tests.


Requirements
---------

The `requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/requirements.txt>`_ at the main directory should be used for all steps except for the somatic validations.
Install `somatic-val-requirements.txt <https://github.com/BlueBrain/SSCxEModelExamples/blob/main/somatic_validation/somatic-val-requirements.txt>`_ before running the somatic validation notebooks or tests.


Reference
---------

.. |build| image:: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml/badge.svg
                :target: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml
                :alt: Build Status
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
