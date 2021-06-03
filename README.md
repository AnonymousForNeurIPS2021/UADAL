# Unknown-Aware Domain Adversarial Learning for Open-Set Domain Adaptation

This repository is the official implementation of Unknown-Aware Domain Adversarial Learning for Open-Set Domain Adaptation. 

## Requirements

Dependency: python3.6

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

Download [Office-31][office31link], [Office-Home][officehomelink], [VisDA][visdalink]

[office31link]: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code "Go "

[officehomelink]: https://www.hemanthdv.org/officeHomeDataset.html "Go "

[visdalink]: http://ai.bu.edu/visda-2017/#download "Go "

Datasets should have the following structures:

```bash
├── data
│   ├── office
│   │   ├── amazon
│   │   │   ├── images
│   │   ├── dslr
│   │   │   ├── images
│   │   ├── webcam
│   │   │   ├── images
│   ├── visda
│   │   ├── train
│   │   ├── validation
│   ├── officehome
│   │   ├── ...
```

Please run python3.6 utils/list_office.py

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Results

Our model achieves the following performance on Open-Set Domain Adaptation Setting :

### [Performances]

| Model name         | Office-31 (ResNet-50) | Office-31 (VGGNet) | Office-Home (ResNet-50) | VisDA (ResNet-50) |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| UADAL  |     85%         |      95%       | 85%         |      95%       |

- The metric is HOS.

