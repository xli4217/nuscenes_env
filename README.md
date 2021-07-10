# Download dataset
Download the nuscenes dataset (the mini version is sufficient for now). This is required for rollout visualization, but not for training
1. Go to https://www.nuscenes.org/login
2. Register if don't already have an account
3. Go to the download page and follow instructions

# Set up conda environment
Navigate to the project root directory, In terminal, run 
```
conda env create -f env.yml
conda activate nuscenes_env
```

To setup pythonpaths, in the project root directory, run `source setup_pythonpath.sh local`

# Process Data
Dataset processing is divided into 2 stage

1. The first stage extracts all relevant information from nuscenes and form a pandas dataframe. Run `python process_data.py --mode=raw`. 
2. The second stage takes the first dataframe as input and runs it through user defined filters to add high-level interaction labels / maneuvers. Run `python process_data.py --mode=filter`.

The filters functions reside in <project_root>/create_dataset/vehicle_behavior_filter/filters


