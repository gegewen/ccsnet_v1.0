# CCSNet: a deep learning modeling suite for CO2 storage

Gege Wen, Catherine Hay, Sally M. Benson, CCSNet: a deep learning modeling suite for CO2 storage, Advances in Water Resources(2021), doi:https://doi.org/10.1016/j.advwatres.2021.104009

![abstract](https://user-images.githubusercontent.com/34537648/127579929-acdcca8c-0123-4375-aa8b-42eaf1252022.jpg)


## Data sets

The data set is available at https://drive.google.com/drive/folders/1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7?usp=sharing 

*Note: variable name is same as file name.*

#### Train:
- Input: `train_x.h5`
- Gas saturation: `trian_y_SG.h5` 
- Reservoir pressure: `train_y_BPR.h5`
- Initial pressure: `train_y_P_init.h5`
- xCO2 molar fraction: `train_y_BXMF.h5`
- yCO2 molar fraction: `train_y_BYMF.h5`
- Liquid phase density: `train_y_BDENW.h5`
- Gas phase density: `train_y_BDENG.h5`

#### Test:
- Input: `test_x.h5`
- Gas saturation: `test_y_SG.h5` 
- Reservoir pressure: `test_y_BPR.h5`
- Initial pressure: `test_y_P_init.h5`
- xCO2 molar fraction: `test_y_BXMF.h5`
- yCO2 molar fraction: `test_y_BYMF.h5`
- Liquid phase density: `test_y_BDENW.h5`
- Gas phase density: `test_y_BDENG.h5`

## Pre-trained models

The data set is available at https://drive.google.com/drive/folders/1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7?usp=sharing 

- Gas saturation: `trained_models/SG_v1.h5` 
- Pressure buildup: `trained_models/dP_v1.h5`
- xCO2 molar fraction: `trained_models/bxmf_v1.h5`
- yCO2 molar fraction: `trained_models/bymf_v1.h5`
- Liquid phase density: `trained_models/bdenw_v1.h5`
- Gas phase density: `trained_models/bdeng_v1.h5`

## Requirements
- `tensorflow==1.15`
- `1.19.0`

## Web application

The pre-trained models are hosted at [ccsnet.ai](http://ccsnet.ai). 

Please refer to https://youtu.be/5bIlfjyo6Jkfor a video demonstration.
