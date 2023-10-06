# Requirements

1. ubuntu
2. miniconda3/anaconda3

# Installation

`conda create -n similar python=3.8`

`conda activate similar`

`pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`

`pip install -r requirements.txt`

# Scripts
## parse official coco's torch pt to torchscipt(if `ckpts/coco_yolov5m.torchscript` not exists, do it)
```
python export.py --weights yolov5m.pt --include torchscript
rm -rf yolov5m.pt
mv yolov5m.torchscript ckpts/coco_yolov5m.torchscript
```

## find similar van
`python find_similary_van.py`

# TODO

1. finish embedding_extract function
2. get&save avg embeddings of both cars and amazon deliveries

