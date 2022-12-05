# turn .github/workflows/python-app.yml into a make file

tests: feature-extraction optimization model-management validation somatic-validation

install-requirements:
	pip install -r requirements.txt
	pip install -r test-requirements.txt

install-somatic-validation-requirements:
	pip install -r test-requirements.txt
	cd somatic_validation; pip install -r somatic-val-requirements.txt

feature-extraction-tests:
	cd feature_extraction; pytest -sx tests

optimization-tests:
	cd optimization; nrnivmodl opt_module/mechanisms
	cd optimization; pytest -sx tests

# optimization modules need to be compiled before running model management tests
model-management-tests:
	cd model_management/mm_run_minimal; pytest -sx tests

validation-tests:
	cd validation; nrnivmodl mechanisms
	cd validation; pytest -sx tests

somatic-validation-tests:
	cd somatic_validation; nrnivmodl mechanisms
	cd somatic_validation; pytest -sx tests

feature-extraction: install-requirements feature-extraction-tests
optimization: install-requirements optimization-tests
model-management: install-requirements model-management-tests
validation: install-requirements validation-tests
somatic-validation: install-somatic-validation-requirements somatic-validation-tests

check-code-style:
	pip install black
	black --check .

# for somatic_validation mechanisms it's ok to use requirements.txt since we only need NEURON
run-compile-all-mechanisms:
	cd optimization; nrnivmodl opt_module/mechanisms
	cd validation; nrnivmodl mechanisms
	cd somatic_validation; nrnivmodl mechanisms

compile-all-mechanisms: install-requirements run-compile-all-mechanisms
