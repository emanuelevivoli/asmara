asmara
==============================

ASMARA is a land mine detection project from MICC/UNIFI/NATO

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Install Miniconda

To install Miniconda, run:

```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

## Conda environment

To create the conda environment, from `requirements.txt`, run:

```bash
$ conda create --name asmara --file requirements.txt -y
```

## Install

To install the package, run:

```bash
$ pip install -e .
```

## Install Matlab

As we use some matlab code (if you want to trabnslate MATLAB code into puython, ploease submitt an PR), we need to install MATLAB.
Please refer to the [MATLAB README](matlab/README.md) to know how to install MATLAB.

In case is not possible to install it, you can download the precomputed hologram numpy arrays from [here]().

## Prepare dataset

### case 1: MATLAB installed
To prepare the dataset, we first need to check if the dataset is already downloaded. If not, we need to download it. 
Please, go to the [data README](data/README.md) to know how to download the dataset.

Then, we need to extract the holograms from the raw data. To do that, we need to run the following command:
    
```bash
$ python src/data/extract_holograms.py
```

### case 2: MATLAB not installed
In case MATLAB is not installed, you can download the precomputed hologram numpy arrays from [here]().
Then, you need to extract the holograms from the raw data. To do that, we need to run the following command:
    
```bash
$ python src/data/extract_holograms.py --precomputed
```

### Options
When extracting the holograms we have the options to save the holograms in a different format.
The format are `npy`, `png`, and `inv` (which is a 3D numpy array from the angular spectrum). There is another param, which is `meta`, which is a json file with the metadata of the holograms.

To save the holograms in a different format, you can run the following command:

```bash
$ python src/data/extract_holograms.py --format npy,png,inv,meta
```

Note: when using the `precomputed` format, the `npy` format must be present.

### Mix the dataset

To mix the dataset, run:

```bash
python src/data/make_dataset.py
```

This will create a `data/processed` folder with the mixed dataset. The folder is composed by the following files:
- `holograms`: the dataset with the mixed holograms (npy, 2D matrix)
- `images`: the dataset with the mixed holograms images (png)
- `inversions`: the dataset with the mixed inversion of the holograms (npy, 3D tensor)
- `meta`: the dataset with the mixed metadata of the holograms (csv)

### Split the dataset

To split the dataset, run:

```bash
python src/data/split_dataset.py
```

This will create a `data/splits` folder with the splitted dataset. The folder is composed by the following files:
- `train.csv`: the dataset indexes of the train set
- `val.csv`: the dataset indexes of the validation set
- `test.csv`: the dataset indexes of the test set

## Run

To run the project, run:

```bash
$ python src/main.py
```

## Train 

To train the project, using the train_model.py script, run:

```bash
$ python src/models/train_model.py
```



## Test

To test the project, run:

```bash
$ pytest
```
