# Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models

## Setup 🔧

<ins>NB:</ins> the Python version used is 3.10.12.

<br>

Create a virtual environment and activate it:

```bash
# Example using the virtualenv package on linux
python3 -m pip install --user virtualenv
python3 -m virtualenv RS-TransCLIP-venv
source RS-TransCLIP-venv/bin/activate.csh
```

Install Pytorch:
```bash
pip3 install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
```

Clone GitHub and move to the appropriate directory:

```bash
git clone https://github.com/elkhouryk/RS-TransCLIP
cd RS-TransCLIP
```

Install the remaining Python packages requirements:
```bash
pip3 install -r requirements.txt
```

<br>

You are ready to start! 🎉

---



## Datasets 🗂️



10 Remote Sensing Scene Classification datasets are already available for evaluation: 

* The whurs19 dataset is already uploaded to the repository for reference and can be used directly.


### Step 1 Dataset Download
  
* The following 6 datasets (eurosat, optimal31, patternnet, resisc45, rsc11, rsicb256) will be automatically downloaded and formatted from Hugging Face using the _run_dataset_download.py_ script.

```bash
# <dataset_name> can take the following values: eurosat, optimal31, patternnet, resisc45, rsc11, rsicb256
python3 run_dataset_download.py --dataset_name <dataset_name> 
```

  
* You must download the aid, mlrsnet and rsicb128 datasets manually from Kaggle and place them in '/datasets/' directory. You can format them manually to follow the dataset directory structure listed above and use them for evaluation **OR** you can use the _run_dataset_download_2.py_ script by placing the .zip files from Kaggle in the '/datasets/' directory.


```bash
# <dataset_name> can take the following values: aid, mlrsnet, rsicb128
python3 run_dataset_download_2.py --dataset_name <dataset_name> 
```

* Download links: [aid](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) | [rsicb128](https://www.kaggle.com/datasets/noamaanabdulazeem/myfile) | [mlrsnet](https://www.kaggle.com/datasets/fes2255/mlrsnet) --- <ins>NB:</ins> On the Kaggle website, click on the download **Arrow** in the center of the page instead of the **Download** button to preserve the data structure needed to use the run_dataset_formatting.py_ script (check figure bellow).


<p align="center">
  <img src="github_data/arrow.png" alt="arrow" style="width: 50%; max-width: 500px; height: auto;">
  <br>
</p>


<br>


After these steps your datasets directory structure should be as follows:
```
$datasets/
└── <dataset_name>/
  └── classes.txt
  └── class_changes.txt
  └── images/
    └── <classname>_<id>.jpg
    └── ...
```

### Step 2 Dataset Partitioning

* Create train/val/test

```bash
# <dataset_name> can take the following values: aid, eurosat, mlrsnet, optimal31, patternnet, resisc45, rsc11, rsicb128, rsicb256
python3 run_dataset_partition.py --dataset_name <dataset_name> 
```

After this step your datasets directory structure should be as follows:

```
$datasets/
└── <dataset_name>/
  └── classes.txt
  └── class_changes.txt
  └── images/
    └── train/
      └── <classname>_<id>.jpg
    └── val/
      └── <classname>_<id>.jpg
    └── test/
      └── <classname>_<id>.jpg
```

### Step 3 Dataset Formatting

* Setup the final dataset format following the format from [CoOp]([https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets](https://github.com/KaiyangZhou/CoOp)).

```bash
# <dataset_name> can take the following values: aid, eurosat, mlrsnet, optimal31, patternnet, resisc45, rsc11, rsicb128, rsicb256
python3 run_dataset_formatting.py --dataset_name <dataset_name> 
```

After this step your final datasets directory structure should be as follows:

```
$datasets/
└── <dataset_name>/
  └── 2750/
    └── <classname>/
      └── <classname>_<id>.jpg
  └── split.json
```
  
