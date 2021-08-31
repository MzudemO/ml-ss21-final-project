# Machine Learning Summer Semester 2021 - Final Project

## General

This project attempted to perform a genre classification task on album cover images.
Data was gathered with the [last.fm](https://www.last.fm/) API.

To reproduce the results:
```console
$ pip install -r requirements.txt
```
- Get the dataset
- For training from scratch: `from_scratch.py`
- for finetuning: `fine_tuning.py`

Hyperparam search is not included because it was done manually and not very thoroughly.

## Dataset

- [Full Dataset 80/20 Split (Google Drive)](https://drive.google.com/file/d/1sYaW_o7A9k5yrs_wB54cXJYMycodd3vL/view?usp=sharing)
- [4 Class Dataset 80/20 Split (Google Drive)](https://drive.google.com/file/d/1-55FKk9DZAuzAymxaxLPbOEnZT0e07_7/view?usp=sharing)

Generating the dataset:
1. Get an api key from last.fm, set environment variables as specified in `api.py`.
2. Download the data using `api.py`. Experiments were done with the tags listed there and 1000 albums per tag.
3. Run `cleanup.py` and `verify_images.py` to remove duplicate filenames and invalid images.
4. Run `splits.py` to split into train/test data. Experiment was done with the default 0.8.

## Improvements 

Areas for dataset improvement:
- Deduplicate images using hash or similar (image-based)
- Collect larger sampleset
- Filter incorrect albums (e.g. [Plague Mass](https://www.last.fm/music/Diamanda+Gal%C3%A1s/Plague+Mass/+tags))

Generally, the data quality from the source is not great. The API is great to use though.

Labeling is highly ambiguous for many of the albums.

Areas for training improvement:
- Use less classes and more data
- Rigorous hyper-parameter search
- Pretrain on OMACIR (unlabeled data, e.g. with Autoencoder)
- Try more powerful network

## Personal Note

Feel free to pick up this project, it was fun to work on but you can do a lot better than me.