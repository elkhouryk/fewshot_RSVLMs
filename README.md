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

* The WHURS19 dataset is already uploaded to the repository for reference and can be used directly.
  
* The following 6 datasets (EuroSAT, OPTIMAL31, PatternNet, RESISC45, RSC11, RSICB256) will be automatically downloaded and formatted from Hugging Face using the _run_dataset_download.py_ script.

```bash
# <dataset_name> can take the following values: EuroSAT, OPTIMAL31, PatternNet, RESISC45, RSC11, RSICB256
python3 run_dataset_download.py --dataset_name <dataset_name> 
```

Dataset directory structure should be as follows:
```
$datasets/
└── <dataset_name>/
  └── classes.txt
  └── class_changes.txt
  └── images/
    └── <classname>_<id>.jpg
    └── ...
```

  
* You must download the AID, MLRSNet and RSICB128 datasets manually from Kaggle and place them in '/datasets/' directory. You can format them manually to follow the dataset directory structure listed above and use them for evaluation **OR** you can use the _run_dataset_formatting.py_ script by placing the .zip files from Kaggle in the '/datasets/' directory.


```bash
# <dataset_name> can take the following values: AID, MLRSNet, RSICB128
python3 run_dataset_formatting.py --dataset_name <dataset_name> 
```

* Download links: [AID](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) | [RSICB128](https://www.kaggle.com/datasets/noamaanabdulazeem/myfile) | [MLRSNet](https://www.kaggle.com/datasets/fes2255/mlrsnet) --- <ins>NB:</ins> On the Kaggle website, click on the download **Arrow** in the center of the page instead of the **Download** button to preserve the data structure needed to use the run_dataset_formatting.py_ script (check figure bellow).


<p align="center">
  <img src="github_data/arrow.png" alt="arrow" style="width: 50%; max-width: 500px; height: auto;">
  <br>
</p>


<br>




><ins>Notes:</ins>
>* The class_changes.txt file inserts a space between combined class names. For example, the class name "railwaystation" becomes "railway station." This change is applied consistently across all datasets.
>* The WHURS19 dataset is already uploaded to the repository for reference.




  
