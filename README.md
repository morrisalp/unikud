# UNIKUD: Hebrew nikud with transformers

If you are accessing this repo via GitHub, please see the [project page on DAGSHub](https://dagshub.com/morrisalp/unikud) for data files, pipelines and more.

# Description

We provide a short description of UNIKUD here. For more information, please see the article: [UNIKUD: Adding Vowels to Hebrew Text with Deep Learning](https://towardsdatascience.com/unikud-adding-vowels-to-hebrew-text-with-deep-learning-powered-by-dagshub-56d238e22d3f).

UNIKUD is an open-source tool for adding vowel signs (*nikud*) to Hebrew text with deep learning, using absolutely no rule-based logic.  UNIKUD uses Google's CANINE transformer model as its backbone, and treats text vocalization as a character token multilabel classification problem.

<p align="center">
<img src="img/training.png" width="70%" height="70%" alt="How data is used to train UNIKUD">
</p>

*How Hebrew text with vowels is used to train UNIKUD. The text with vowels removed is used as the model's input, and the original text with vowels is used as the target (what we are trying to predict).*

UNIKUD's training data requires preprocessing, because texts in Hebrew without vowel marks may be written using "full spelling" (כתיב מלא) where extra letters are occasionally added to words:

<p align="center">
<img src="img/ktiv-male.png" width="30%" height="30%" alt="Illustration of full spellings in Hebrew">
</p>

*"Ktiv male" (full spelling): The red letter is only used without vowels.*

The core UNIKUD model uses a multilabel classification head as shown below:

<p align="center">
<img src="img/ohe.png" width="70%" height="70%" alt="Illustration of label-encoded target">
</p>

*Hebrew vocalization as multilabel classification: Each Hebrew letter may be decorated with multiple nikud, which can be represented as a binary vector. UNIKUD uses this label encoding as its target. The figure is condensed for clarity but UNIKUD's binary targets actually contain 15 entries.*

See the "Experiments" tab on the UNIKUD DagsHub repository page for training and evaluation metrics.

# Requirements

## Inference only

Install the UNIKUD framework PyPI package via pip:

    pip install unikud

## For training

First install:

* Rust compiler:
  * `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  * Reopen shell or run `source $HOME/.cargo/env`

Then install requirements for UNIKUD and activate its environment with either of:

* Conda (recommended):
  * `conda env create -f environment.yml`
  * `conda activate unikud`
* Pip:
  * `pip install -r requirements.txt`

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

# Training

To reproduce the training pipeline, perform the following steps:

1. Preprocess data:
  * `dvc repro preprocessing`
2. Train ktiv male model:
  * `dvc repro train-ktiv-male`
3. Add ktiv male to data file:
  * `dvc repro add-ktiv-male`
4. Train UNIKUD model:
  * `dvc repro train-unikud`

Training steps will automatically log to MLflow (via the Huggingface Trainer object) if the following environment variables are set: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.

Scripts will automatically use GPU when available. If you want to run on CPU, set the environment variable `CUDA_VISIBLE_DEVICES` to be empty (`export CUDA_VISIBLE_DEVICES=`).

# Inference

If you installed UNIKUD via pip, you may add nikud to Hebrew text as follows:

    from unikud.framework import Unikud

    u = Unikud() # installs required files

    print(u('שלום חברים'))

Note: `Unikud()` takes optional keyword argument `device=` for CPU/GPU inference. `Unikud.__call__` takes optional keyword arguments to adjust decoding hyperparameters.

# Contributing

Please file an issue on this project's DagsHub or GitHub repo pages to report bugs or suggest improvements.

# Other Links
* [HF Hub model page: malper/unikud](https://huggingface.co/malper/unikud)
* [HF Spaces deployment](https://huggingface.co/spaces/malper/unikud)