# GIZ NLP Agricultural Keyword Spotter! #3 on the private leaderbord

## Winning submission info:
    ID: ymQAZUgG
    Filename: eff5_20f_eff6_20f_eff7_20f_dnet201_20f.csv
    Comment: E5 * 0.5 + E6 * 0.2 + E7 * 0.2 + DNet201 * 0.1
    Submitted: 29 November 15:27

### TeamName: zindi-giz

#### 1.Data preprocessing
- first we download all data from compitation
- extract all .zip files to input folder
- convert all train & test .wav files to 32k sample_rate using librosa and soundfile packages
- then we upload preprocessed data into kaggle dataset
- because it easy to download every time we run new experement and new notebooks
- you can find preprocessed kaggle dataset hear : https://www.kaggle.com/gopidurgaprasad/giz-nlp-agricultural-keyword-spotter
- we made this dataset public after compitation end
- you can check those steps in `00-DataDownloadPreprocess.ipynb` notebook

#### 2.Packages used
- we are used google colab pro high ram and 16gb GPU for model traing
- we are using Pytorch deeplearing librarie for models training
- soundfile <--- fast to read .wav files
- timm <--- for pretrain imagenet models
- albumentations <--- for image augmentations
- audiomentations <--- for audio augmentation
- torchlibrosa <--- extract features from raw audio files faster the librosa
- catalyst <--- for upsampling and downsampling
- transformers <--- for linear_warmup shedulers
- pytorch-gradual-warmup-lr <--- for waramup shedulers

#### 3.Model Pipeline
1. Download preprocessed data from kaggle dataset using kaggle api 
2. Create train and test dataframes
3. Create folds for training using StratifiedKFold
4. Create Codes give one label for every target label sort by name, total 193 targets 0 to 102 labels
5. save those codes in `Codes.py`
6. Create Datasets for pytroch dataloaders it takes raw audio file and its true label you can find in `Datasets.py`
7. Create Audio augmentations for raw audio file using Audiomentation package ypu can find in `Augmentation.py`
8. Create Model pipeline using timm image net pretrain models package
9. Utils for Metrics
10. train & valid & test function for model train and validation and test predictions
11. main function takes fold number to run that fold
12. args class takes model arguments for traing

#### 4.Models training
- we trined 4 models Efficientnet-5, Efficientnet-6, Efficientnet-7 and DenseNet201.
- each model we trined for 20 epochs with different seeds
- model_param = {
        'encoder' : 'tf_efficientnet_b5_ns',
        'sample_rate': 32000,
        'window_size' : 1024,
        'hop_size' : 320,
        'mel_bins' : 64,
        'fmin' : 50,
        'fmax' : 14000,
        'classes_num' : 193 
    }
- optimizer: AdamW
- scheduler: get_linear_schedule_with_warmup

#### 5.Model save paths
- for each and every notebook you need to give where the trained models save 
- we five that parameter in `Run.py` file `args` class `output_dir`

#### 6.Training pipeline
- you can find all traing notebooks in `pipeline notebooks` folder
- run those notebooks one by one by sequence order
- 00-DataDownloadPreprocess.ipynb
- 01-Eff5_20fold_base part1.ipynb
- 02-Eff5_20fold_base part2.ipynb
- 03-Eff6_20fold_base.ipynb
- 04-Eff7_20fold_base.ipynb
- 05-DNet201_20fold_base.ipynb
- 06-Final Submission Ensemble.ipynb

#### 7.Trined Model Weights
- we are sharing google drive folder link for all model pipelines
- FOLDER LINK : https://drive.google.com/drive/folders/1-JBpZyXWy7Zi8NJFg5eBxaXjXoompBUb?usp=sharing
- form this link you add `ZINDI GIZ NLP Agricultural Keyword Spotter #3 place solution` folder to your drive and run all notebooks you finlly get `eff5_20f_eff6_20f_eff7_20f_dnet201_20f.csv` final submission file
- make sure you must have kaggle api to download preprocessed data from kaggle
- we have all ready trined models weights in `trained_weights` folder



##### Thank you
- if you have any doubts `gopidurgapasad762@gmail.com` please mail us.
