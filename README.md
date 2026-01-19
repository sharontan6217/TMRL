This is for "Unsupervised Method of Transformer-based Multimodal Representation Learning for Acoustic Event Classification and Blind Acoustic Scene Classification".

This code is based on "[Masked Modeling Duo (M2D) & M2D-CLAP](https://github.com/nttcslab/m2d)" and "[MPNet](https://github.com/microsoft/MPNet)".

Data:
1. ["TUT Rare sound events, Development dataset](https://doi.org/10.5281/zenodo.603106)".
2. ["TUT Rare sound events, Evaluation dataset](https://doi.org/10.5281/zenodo.1160454)".

These publicly available implementations and open data for the experiments are sincerely appreciated.

Installation:
Please download the pretrained weights from the list below, and run 'pip -r install requirements.txt' with Python 3.10.

| Weight        | Recommendation  | Description | Fur-PT Ready | AS2M mAP |
|:--------------|:----------------|:------------|:------:|:--------:|
| [m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip) | Recommended for this implementation | M2D-CLAP fine-tuned on AS2M | N/A | 0.485 |
| [m2d_as_vit_base-80x1001p16x16-240213_AS-FT_enconly](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16-240213_AS-FT_enconly.zip) | Recommended for this implementation | M2D-AS fine-tuned on AS2M | N/A | 0.485 |
| [m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip) | 3rd best for AT/SED. (Encoder only) | M2D/0.7 fine-tuned on AS2M | N/A | 0.479 |
| [m2d_vit_base-80x200p16x4-230529](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x200p16x4-230529.zip) | General-purpose transfer learning and further pre-training w/ finer time frame. | M2D/0.7 (t.f. 40ms) | ✅ | - |
| [m2d_clap_vit_base-80x608p16x16-240128](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_clap_vit_base-80x608p16x16-240128.zip) | General-purpose transfer learning and further pre-training, especially when application data is closer to the AudioSet ontology. | M2D-CLAP | ✅ | - |
| [m2d_as_vit_base-80x608p16x16-240213](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_as_vit_base-80x608p16x16-240213.zip) | General-purpose transfer learning and further pre-training, especially when application data is closer to the AudioSet ontology. | M2D-AS | ✅ | - |
| [m2d_vit_base-80x608p16x16-221006-mr7](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr7.zip) | Recommended for the visual analysis in this implementation. | M2D/0.7 | ✅ | - |
| [m2d_vit_base-80x608p16x16-221006-mr6](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr6.zip) | General-purpose transfer learning and further pre-training. | M2D/0.6 | ✅ | - |
| [m2d_vit_base-80x608p16x16-221006-mr7_enconly](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr7_enconly.zip) | General-purpose transfer learning. (Encoder only) | M2D/0.7 | N/A | - |
| [m2d_vit_base-80x608p16x16-220930-mr7_enconly](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-220930-mr7_enconly.zip) | General-purpose transfer learning. (Encoder only) | M2D/0.7 | N/A | - |


Experiment:
To generate meta file :
```shell
python ./sound_datasets/rare_sound_event/meta/yaml2csv.py 
```

To generate results of AEC task :
```shell
python generate_labels_event.py 
```
or

To generate results of ASC task :
```shell
python generate_labels_env.py 
```

For visualization:
```shell
python visualize.py
```

- `data_dir`: type=str,default='./source_datasets/', 'directory of the original data.'  
- `graph_dir`,type=str,default='./graph/', 'directory of graphs'
- `event_dir`,type=str,default='./event/', 'directory of outputs for AEC task.'
- `env_dir`,type=str,default='./env/', 'directory of outputs for ASC task.'
- `log_dir`,type=str,default='./log/', 'directory of the transaction logs.'

## References
- M2D: "[Masked Modeling Duo: Towards a Universal Audio Pre-training Framework](https://ieeexplore.ieee.org/document/10502167)"
- MPNet: "[MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/pdf/2004.09297)"