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

E-model optimization
________
``/optimization contains`` tools and codes necessary to run and display the result of the canonical e-model optimization. Run ``Minimal_cADpyr_L5TPC_Optimization.ipynb`` to visualize the results of the optimization. Prior to launching the notebook, one needs to compile e-model' mechanisms by running ``compile_mechanisms.sh``. ``/optimization/opt_module`` contains tools and data necessary for the optimization such as morphology, mechanisms, checkpoints, and config files. 

E-model validations
_________
Two types of validation were performed for the optimized L5PC e-model.

- The visualization of the bAP/EPSP validations can be found in ``validation.ipynb`` in the ``/validation`` folder. The morphologies for these validations are located in ``input/morphologies``. To run bAP/EPSP validations use ``python main.py att_conf.json``. 
- Somatic validations are located in ``/somatic_validation`` folder. First, e-feature for the validations have to be extracted from the chosen patch clamp protocol. To extract e-features use ``feature-extraction.ipynb``, the results of this extraction can be found in ``/somatic_validation/L5TPC``. To run and visualize results of the somatic validation run ``somatic-validation.ipynb``.

Generalization
______________

The generalization of the canonical L5PC e-model to a number of morphologies is done with the model-management tool. To run a generalization use ``model-management.ipynb`` in ``/model_management/mm_run_minimal``. The morphologies used in the step can be found in ``/model_management/mm_run_minimal/morphologies/``.


Usage
---------

Requirements
---------

Reference
---------

.. |build| image:: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml/badge.svg
                :target: https://github.com/BlueBrain/SSCxEModelExamples/actions/workflows/python-app.yml
                :alt: Build Status
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
