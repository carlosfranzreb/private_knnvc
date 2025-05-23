# Private kNN-VC

Here are the relevant resources for our submission. They include samples of anonymized speech, the full results of the VPC evaluation, as well as code and instructions to run the model.

## Samples

You can listen to some samples [here](https://carlosfranzreb.github.io/2025/05/23/private-knnvc.html).

You can also download the samples, which can be found in the released ZIP file called "samples.zip". Folders are named by the model configuration: The file names consist of the source file and target speaker, combined with underscores.

We have used four target speakers, whose WavLM features you can find in "target_feats.zip". The four speakers are named by their indices, and correspond to the following LibriTTS speakers:

- `0.pt`-> 7585 (female)
- `1.pt`-> 6967 (female)
- `2.pt`-> 6077 (male)
- `3.pt`-> 517 (male)

## How to run the model

We have developed a self-contained version of the model as a demo. Run the notebook `demo.ipynb` to see how it works.
We provide all model weights as a release in this repository.
You can also find the code for training the phone and duration predictors in the folder "train_predictors".

## How to reproduce our experiments

We have summarized the most important results in the CSV file called "vpc_results.csv".

The model is implemented to work with the [SpAnE framework](https://github.com/carlosfranzreb/spkanon_eval). All datasets required by the VPC evaluation are anonymized with SpAnE. Then, the datafiles for the VPC 2024 evaluation are created and the evaluation is run.

### 1. Install SpAnE

Open a terminal and change directory to the location where you want to install the software.

```bash
git clone https://github.com/carlosfranzreb/spkanon_eval
cd spkanon_eval
git checkout better_batches
pip install .
```

### 2. Add the model

Now, to add the code of private kNN-VC to SpAnE, place the folder `private_knnvc` there. It should be placed at the root directory of SpAnE for all the paths set in the config to work.

### 3. Download the model checkpoints

The model checkpoints are all zipped together in the released `checkpoints.zip` file. Unzip them at the same level as `private_knnvc`. The structure should be like follows:

```bash
spkanon_eval/       # root directory
    spkanon_eval/   # this is the source code of the framework
    private_knnvc/  # this is the model code
    checkpoints/    # here are the model checkpoints
```

### 4. Run the inference

The configuration file to run the inference is called `vpc2024/spane_cfg.yaml`. There you will find comments explaining the most relevant parameters.

#### Datasets

One parameter which you will definitely need to change is `data.config.root_folder`, where you have to write the location of the directory that contains all the datasets. To run the VPC24 evaluation, you will need:

1. Their LibriSpeech test and dev sets, which they have converted from FLAC to WAV. See the [VPC repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024).
2. the original LibriSpeech train-clean-360, which you can download from [OpenSLR](https://www.openslr.org/12).
3. The IEMOCAP datasets, for which you have to [request access from USC](https://sail.usc.edu/iemocap/iemocap_release.htm).

For the target speakers, we use a subset from LibriTTS train-other-500. You can download it from [OpenSLR](https://openslr.org/60/).

Then you can run the inference where all data is anonymized with the following command from the root, where the configuration file is expected to be:

```bash
python spkanon_eval/run.py --config ./vpc2024/spane_cfg.yaml
```

### 5. Run the VPC evaluation

To run the VPC evaluation, you have to install it following their instructions, which you can find in their [GitHub repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024). You should install it beside SpAnE.

Then, you need to create the necessary datafiles with the Python script `vpc2024/create_datafiles_vpc.py`, which you can download from the resources. The script takes two arguments:

1. `suffix`: how the experiment is called. The VPC folders with the anonymized data will be appended this suffix, e.g. "iemocap_experiment1".
2. `data_dir`: the directory where the anonymized data is found. Check the parameter `log_dir` from the SpAnE inference configuration.

Once the datafiles have been created, you can run the VPC evaluation with the following bash script:

```bash
SUFFIX=your_suffix_here
cd /path/to/Voice-Privacy-Challenge-2024
pip install -r ./requirements.txt
apt update && apt install -y zip
python run_evaluation.py --config configs/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"_$SUFFIX\"}" --force_compute True
python run_evaluation.py --config configs/eval_post.yaml --overwrite "{\"anon_data_suffix\": \"_$SUFFIX\"}" --force_compute True
results_summary_path_orig=exp/results_summary/eval_orig${SUFFIX}/results_orig.txt # the same value as $results_summary_path in configs/eval_pre.yaml
results_summary_path_anon=exp/results_summary/eval_anon${SUFFIX}/results_anon.txt # the same value as $results_summary_path in configs/eval_post.yaml
results_exp=exp/results_summary
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${SUFFIX}"
zip ${results_exp}/result_for_submission${SUFFIX}.zip -r exp/asr/*${SUFFIX} exp/asr/*${SUFFIX}.csv exp/ser/*${SUFFIX}.csv exp/results_summary/*${SUFFIX}* exp/asv_orig/*${SUFFIX} exp/asv_orig/*${SUFFIX}.csv exp/asv_anon${SUFFIX}
```
