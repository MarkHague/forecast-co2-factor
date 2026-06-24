# Forecast CO2 Factor - Netherlands Energy Mix 
A repo for forecasting the hourly CO2 factor for the Nederlands national grid, based on the mix of renewables and non-renewables. 

## Installation
To install locally, it is recommended to use `poetry` for dependency management. Alternatively, you can simply install the dependencies with `pip install -r requirements.txt`. With poetry, these steps will produce a clean install:
1. After cloning, set the local python version to 3.12.12 (e.g. `pyenv install 3.12.12`, then `pyenv local 3.12.12`).
2. Run `pip install poetry`. This links poetry to the local version of python.
3. Run `poetry install` - installs the dependencies, creates the virtual env. 

Optional:
4. Activate the virtual env with `poetry shell` (this allows you to run scripts without prefixing `poetry run`). You will need to first install the shell plugin with `pip install poetry-plugin-shell`.

## Project Structure