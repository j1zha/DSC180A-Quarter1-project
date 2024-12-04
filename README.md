Lag-Llama: Reproducing Experiments 
This repository demonstrates how to reproduce experiments from the paper Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting using the codebase provided at Lag-Llama GitHub Repository. 
Due to the sudden change of research project topic on November 25th and the time limitation, I only had time to reproduce the model’s performance on the Weather dataset. I used the pretraining script provided in the original repository and restricted our experiment to the Weather dataset.

Getting Started

1. Clone the Repository 
First, clone the original Lag-Llama repository:\
git clone https://github.com/time-series-foundation-models/lag-llama.git\
cd lag-llama

2. Install Dependencies
Set up a new Python environment and install the required dependencies:\
conda create -n lag-llama python=3.10.8\
conda activate lag-llama\
pip install -r requirements.txt

4. Download Datasets
Download the necessary datasets from here, and extract them into the datasets/ folder:\
tar -xvzf nonmonash_datasets.tar.gz -C datasets

Adjustments and Challenges:\
Modified Epochs:To save time and computational resources, the number of training epochs was reduced from the original 1000 to 5. While this impacts the model's final performance, it was sufficient for validating the reproducibility of the paper's methodology.

Manual Checkpoint Path Adjustment:One of the key challenges arose during the fine-tuning stage. The script attempted to locate a pretraining checkpoint file automatically for initialization. However, the default script setup could not find the desired checkpoint file, and as a result, the process failed.

To address this:
I manually identified the path of the checkpoint file generated during the pretraining stage.\
Moved the file to the appropriate fine-tuning directory.\
Updated the script to correctly load the checkpoint for fine-tuning.

6. Run Pretraining
To replicate the pretraining process, execute the pretraining script:
python run.py \
    -e pretraining_lag_llama -d "datasets" --seed 42 \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --all_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" ... \
    --num_workers 2 --args_from_dict_path configs/lag_llama.json --lr 0.0001
   
7. Run Fine-tuning
Once pretraining is complete, fine-tune the model on specific datasets like Weather:
python run.py \
    -e pretraining_lag_llama_finetune_on_weather -d "datasets" --seed 42 \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --single_dataset "weather" \
    --get_ckpt_path_from_experiment_name pretraining_lag_llama \
    --lr 0.00001 --num_validation_windows 1 --single_dataset_last_k_percentage 100
   
Reproducibility\
After running the scripts, you should see the following CRPS results in the log file:\
- Weather: 0.1428 (Reproduced result from finetuning script)
- Weather: 0.2037 (Reproduced result from pretraining script)
  
Clone the original repository.
Follow the dataset preparation and dependency installation steps.\
Run the provided run.py scripts for both pretraining and fine-tuning.\
Acknowledgements\
This work reproduces experiments using the official codebase provided at Lag-Llama GitHub Repository.
