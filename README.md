Lag-Llama: Reproducing Experiments 
This repository demonstrates how to reproduce experiments from the paper Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting using the codebase provided at Lag-Llama GitHub Repository. 

Getting Started

1. Clone the Repository 
First, clone the original Lag-Llama repository:\
git clone https://github.com/time-series-foundation-models/lag-llama.git\
cd lag-llama

2. Install Dependencies
Set up a new Python environment and install the required dependencies:\
conda create -n lag-llama python=3.10.8\
conda activate lag-llama\
pip install -r requirements.txt\

4. Download Datasets
Download the necessary datasets from here, and extract them into the datasets/ folder:\
tar -xvzf nonmonash_datasets.tar.gz -C datasets\

5. Run Pretraining
To replicate the pretraining process, execute the pretraining script:\
python run.py \
    -e pretraining_lag_llama -d "datasets" --seed 42 \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --all_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" ... \
    --num_workers 2 --args_from_dict_path configs/lag_llama.json --lr 0.0001\
   
6. Run Fine-tuning
Once pretraining is complete, fine-tune the model on specific datasets like Weather:\
python run.py \
    -e pretraining_lag_llama_finetune_on_weather -d "datasets" --seed 42 \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --single_dataset "weather" \
    --get_ckpt_path_from_experiment_name pretraining_lag_llama \
    --lr 0.00001 --num_validation_windows 1 --single_dataset_last_k_percentage 100
   
Reproducibility\
After running the fine-tuning script, you should see the following CRPS results in the log file:\
- Weather: 0.1428 (Reproduced result)\
  
Clone the original repository.\
Follow the dataset preparation and dependency installation steps.\
Run the provided run.py scripts for both pretraining and fine-tuning.\
Acknowledgements\
This work reproduces experiments using the official codebase provided at Lag-Llama GitHub Repository.\
