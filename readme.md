# Process Simulation - A Graph-Based Process Simulation Algorithm

## Table of Contents
- [Introduction](#introduction)
- [Simulation Model](#simulation-model)
- [Raw Data](#raw-data)
- [Analysis Scripts](#analysis-scripts)
- [License](#license)
- [Contact](#contact)

## Introduction
This repository contains the source code, raw data, and analysis scripts for the study "The Impact of Process Automation on Process Change Dynamics - A Simulation Study" by Lennart Ebert and Jan Mendling.

We extend the process simulation model by Pentland et al. (2020), introducing automation and exceptions:
Pentland, B. T., Liu, P., Kremser, W., and Haerem, T. 2020. “The Dynamics of Drift in Digitized Processes,” MIS Quarterly (44:1), pp. 19–47. [DOI: 10.25300/MISQ/2020/14458](https://doi.org/10.25300/MISQ/2020/14458).

This repository enables future research to reproduce our study's results, validate the simulation model, and implement further extensions.

## Simulation Model
The simulation model is defined in the [`process_simulation.py`](process_simulation.py) module, class `ProcessSimulationModel`.

### Instructions to Run the Simulation Model
1. Install and activate the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    conda activate process_simulation
    ```
2. Set the simulation settings in the `simulation_settings.yml`. All parameter combinations will be run for the specified number of `simulation_runs`. `sensitivity_fixed_params` and `sensitivity_varying_params` are only used if the `sensitivity_analysis` flag is set when running the simulation model.
3. Run the simulation model:
    1. **Option 1: Run from command line**
        ```
        usage: process_simulation.py [-h] --settings SETTINGS [--output OUTPUT] [--random_order] [--sensitivity_analysis] [--fill]

        Run simulations in parallel and save results to a CSV file.

        options:
        -h, --help            show this help message and exit
        --settings SETTINGS   Path to a YAML file containing the simulation settings.
        --output OUTPUT       Output CSV file name (default: results.csv).
        --random_order        If set, performs the simulation in random instead of sequential order.
        --sensitivity_analysis If set, performs a sensitivity analysis.
        --fill                If set, first reads the results file at output path and then completes missing simulations.
        ```
        Example:
        ```sh
        python process_simulation.py --settings simulation_settings.yml --output experiment_results/experiment_output.csv --random_order --fill
        ```
    2. **Option 2: Run by executing the `Run mass simulation.ipynb` notebook.**
    3. **Option 3: Run from your own script by importing the `process_simulation` module.** Refer to [`Impact of Process Automation on Process Change Dynamics.ipynb`](Impact%20of%20Process%20Automation%20on%20Process%20Change%20Dynamics.ipynb) for how to use the module.

## Raw Data
See the folder [`reported_results/raw_data`](reported_results/raw_data) for the raw data for each reported simulation run.

## Analysis Scripts
The notebook [`Impact of Process Automation on Process Change Dynamics.ipynb`](Impact%20of%20Process%20Automation%20on%20Process%20Change%20Dynamics.ipynb) contains the analysis script for the single-run experiments, 500-run experiments, and sensitivity analysis.

## License
(c) by Lennart Ebert

This project is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.

You should have received a copy of the license along with this project. If not, see <https://creativecommons.org/licenses/by-sa/4.0/>.

## Contact
Feel free to reach out via email in case of any questions, feedback, or suggestions: lennart.ebert@hu-berlin.de