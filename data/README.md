# DATA

Here you can locate all the data about Holographic scans.

The steps for downloading the dataset and creating the Holographic images, inversions, and other type of datya is reported in the following.

## Download data

In Drive link [here](https://drive.google.com/file/d/1Ehs_LeYQLsvmc9blAIPOvhsWpZPM54q7/view?usp=share_link) you can find the `raw` zipped file. In that, there are the `logs` and `csv` files for the scans. The first one is produced by FESTO (mechanical system that moves the radar), while the second one is produced by PLUTO (holographic radar produced by UNIFI). You just need to download the file, locate it inside the `/data` folder and unzip it.

You'll have something like this:

```
└── data/
    └── raw/            <- Folder with all raw data (FESTO and PLUTO scans)
       ├── indoor/              <- folder with all csvs and logs for indoor scans
       ├── outdoor/             <- folder with all csvs and logs for outdoor scans
       ├── indoor_objects.csv   <- names and types of indoor objects
       ├── indoor_objects.png   <- images of indoor objects
       └── sources.txt          <- source from which we took names and types of indoor objects
```

## Extract holograms

To create the holograms, we need to use some matlab code. As we like python, we just `wrapped` matlab code in python, so we can stay in jupyter notebooks and python scripts and still using matlab :) yey!

First step is to run:
```bash
$ 
```