# [TMLR] Distributionally Robust Alignment for Medical Federated Vision-Language Pre-training Under Data Heterogeneity

This is the official codebase for FedDRA: Distributionally Robust Alignment for Medical Federated Vision-Language Pre-training Under Data Heterogeneity. We built our repository based on [MGCA](https://github.com/HKU-MedAI/MGCA).

# Dataset downloading

Datasets we used are as follows:

### Chest X-Ray Domain

- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

- **RSNA**: We used the stage 2 of RSNA dataset in [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data). 

- **COVIDx**: We used the version 6 of COVIDx dataset in [Kaggle](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2).

### Retinal Domain

- **BRSET**: We used the version 1.0.0 of BRSET dataset in [Physionet](https://physionet.org/content/brazilian-ophthalmological/1.0.0/)).

- **MBrset**: We used the version 1.0 of MBrset dataset in [Physionet](https://physionet.org/content/mbrset/1.0/).

- **MESSIDOR**: We downloaded the MESSIDOR dataset on Aug 2024 in [MESSIDOR](https://www.adcis.net/en/third-party/messidor/).
  

### Data Preprocessing

Our data preprocessing for Chest X-Ray domain is built on: [MGCA](https://github.com/HKU-MedAI/MGCA).

Our data preprocessing for Retinal domain is built on: [BRSET](https://github.com/luisnakayama/BRSET).


# Dependencies

To install all packages in this codebase along with their dependencies, run
```
/home/ztshuai/.conda/envs/ldm/bin/pip install -r requirements.txt
```

# Citation

Please consider citing this paper if you find the code useful

```bibtex
@article{shuai2024distributionally,
  title={Distributionally Robust Alignment for Medical Federated Vision-Language Pre-training Under Data Heterogeneity},
  author={Shuai, Zitao and Wu, Chenwei and Tang, Zhengxu and Shen, Liyue},
  journal={arXiv preprint arXiv:2404.03854},
  year={2024}
}
```
