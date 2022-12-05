# turn .github/workflows/python-app.yml into a make file

tests: test-feature-extraction test-optimization test-model-management test-validation test-somatic-validation

install-requirements:
	pip install -r requirements.txt
	pip install -r test-requirements.txt

install-somatic-validation-requirements:
	pip install -r test-requirements.txt
	cd somatic_validation; pip install -r somatic-val-requirements.txt

run-feature-extraction-tests:
	cd feature_extraction; pytest -sx tests

run-optimization-tests:
	cd optimization; nrnivmodl opt_module/mechanisms
	cd optimization; pytest -sx tests

# optimization modules need to be compiled before running model management tests
run-model-management-tests:
	cd model_management/mm_run_minimal; pytest -sx tests

run-validation-tests:
	cd validation; nrnivmodl mechanisms
	cd validation; pytest -sx tests

run-somatic-validation-tests:
	cd somatic_validation; nrnivmodl mechanisms
	cd somatic_validation; pytest -sx tests

test-feature-extraction: install-requirements run-feature-extraction-tests
test-optimization: install-requirements run-optimization-tests
test-model-management: install-requirements run-model-management-tests
test-validation: install-requirements run-validation-tests
test-somatic-validation: install-somatic-validation-requirements run-somatic-validation-tests

check-code-style:
	pip install black
	black --check .

# for somatic_validation mechanisms it's ok to use requirements.txt since we only need NEURON
run-compile-all-mechanisms:
	cd optimization; nrnivmodl opt_module/mechanisms
	cd validation; nrnivmodl mechanisms
	cd somatic_validation; nrnivmodl mechanisms

compile-all-mechanisms: install-requirements run-compile-all-mechanisms
