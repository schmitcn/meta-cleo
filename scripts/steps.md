# Steps to generate the repository

### Data preparation
#### Datasets
- From Auto-Sklearn's selected tasks, remove the ones with NaN values;
- From the non-NaN datasets, select randomly 100;
- Selected tasks with ids `75115, 75121, 75125, 75126, 189859, 189878, 167204, 75156, 190158, 168791, 146597, 167203, 167085, 190154, 126030, 146594, 189864, 189863, 189858, 75236, 211720, 75108, 146592, 166866, 146576, 75154, 146574, 75180, 166951, 189828, 3049, 75139, 167100, 75146, 126031, 288, 189899, 75133, 75092, 75100, 75239, 146596, 189840, 189779, 75199, 75235, 75233, 75159, 189875, 75187, 167086, 167089, 75089, 166905, 189845, 75219, 126021, 75185, 211721, 75147, 189843, 75142, 75112, 167094, 167101, 75149, 146593, 166950, 167096, 279, 166915, 75176, 75141, 75171, 2121, 75134, 166872, 166932, 167103, 2122, 75223, 3048, 75148, 3053, 126024, 167105, 75131, 166931, 75210, 146586, 166956, 166958, 189902, 189887, 189890, 167099, 189881, 189882, 189894, 189846`

### Running locally
#### Local machine setup
Using pyenv is highly recommended for running the scripts.
1. Install python requirements at `requirements.txt` in the repository root
2. Install python requirements at `scripts/requirements.txt`

#### Steps
1. Create the commands that will generate the ensembles for the given tasks and configurations
2. Run the generated commands by
   1. Re-naming the commands file using `mv scripts/commands.txt scripts/commands.sh`
   2. Allow executing the file using `chmod +x scripts/commands.sh`
   3. Run the commands from the repository root using `sh scripts/commands.sh`
      1. It is possible that an error indicating that it could not find automlopen is raised. 
      If that is the case, run `export PYTHONPATH=~/{path-to-autml-ensembles-repo-root}:$` to fix it.
   - The command will generate:
      - in a successful scenario
        - a pickle file with the automl-open ensemble and a csv file containing performance and diversity metrics
        - a csv file indicating some run configs and metrics, as well as a path to the pickled ensemble learner
      - in a failure scenario
        - a text file with the exception stack trace
3. Aggregate resulting csv files into one (using `3-joincsvs.py` script)
4. Extract meta-features
   1. Cleanse resulting file by manually
      1. Removing the header rows (except for the first one)
      2. Fixing rows generated from failed calculations (essentially column mismatch, the solution was to fill with more commas)
      3. Adding a failed row for 189878 as it had failed due to unrecoverable memory consumption 
      4. Removing empty anonymous column (first column) with no added value (all zeroes)
