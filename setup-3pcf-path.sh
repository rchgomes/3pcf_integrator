#!/bin/bash
# Setup path of the project

# Common setup for every environment
here=$(dirname "$(pwd)/${BASH_SOURCE[0]}")
# path to this project directory, used in the ini file.
export COSMOSIS_3PCF_INTEGRATOR=$here
# path to the python module.
export PYTHONPATH=$here/python:$PYTHONPATH

# environment dependent setups
# path to cosmosis-standard-library, used in the ini file.
export COSMOSIS_STD_LIB=${here}/../cosmosis-standard-library/