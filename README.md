# UNIKUD: Hebrew nikud with transformers

If you are accessing this repo via GitHub, please see the [project page on DAGSHub](https://dagshub.com/morrisalp/unikud) for data files, pipelines and more.

# Requirements

First install:

* Conda
* Rust compiler:
  * `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  * Reopen shell or run `source $HOME/.cargo/env`

Then create and activate the UNIKUD environment with:

* `conda env create -f environment.yml`
* `conda activate unikud`

You may then download the required data files using DVC:

* `dvc remote add origin https://dagshub.com/morrisalp/unikud.dvc`
* `dvc pull -r origin`

# Data

Sources of data:

* Public-domain works from the [Ben-Yehuda Project](https://benyehuda.org/)
* Wikimedia sources:
  * [Hebrew Wikipedia](https://he.wikipedia.org/)
  * [Hebrew Wikisource](https://he.wikisource.org/) (ויקיטקסט)
  * [Hebrew Wiktionary](https://he.wiktionary.org/) (ויקימילון)

To preprocess data, run:

# Training

To reproduce the training pipeline, perform the following steps:

1. Preprocess data:
  * `dvc repro preprocessing`
2. Train ktiv male model:
  * `dvc repro train-ktiv-male`
3. Aadd ktiv male to data file:
  * `dvc repro add-ktiv-male`

Training steps will automatically log to MLflow (via the Huggingface Trainer object) if the following environment variables are set: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.

Scripts will automatically use GPU when available. If you want to run on CPU, set the environment variable `CUDA_VISIBLE_DEVICES` to be empty (`export CUDA_VISIBLE_DEVICES=`).