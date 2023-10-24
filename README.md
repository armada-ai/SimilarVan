# Requirements

1. ubuntu
2. miniconda3/anaconda3

# Installation

```
conda create -n similar python=3.8
conda activate similar
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

To run open world object detection, run following commands to finish installation

```
pip install -r requirements.txt
cd models/ops
sh make.sh
```


# Scripts

## parse official coco's torch pt to torchscipt(if `ckpts/coco_yolov5x.torchscript` does not exist, run following commands)

```
cd ckpts
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
cd ..
python export.py --weights ckpts/yolov5x.pt --include torchscript
rm -rf ckpts/yolov5x.pt
```

## find similar objs pipeline

### step 1. cache dataset embeddings

```
python make_dataset_embeddings.py
```

#### How to use

1. gen cropped imgs
2. clean up cropped imgs
3. extract embeddings

#### model selection

`det_model_name` ------select detection model, support `["yolov5_coco", "yolov5_amazon", "mdef", "grid"]`

`encoding_model_name`------select encoding model, support `["cvnet", "vit"]`

`img_root`------the root directory of the dataset you want to extract

`mode`------which way to summarize embeddings, only support `["mean", "seperate"]`

`cleaned_img_root`------the root directory of the cleaned cropped images

### step 2. inference

`python find_similar_objs.py`

#### model selection

`det_model_name` ------select detection model, only support `["yolov5_coco", "yolov5_amazon", "mdef", "grid"]`

`encoding_model_name`------select encoding model, only support `["cvnet", "vit"]`

`dataset_name`------the name of the dataset
`infer_vid_path`------the video's path

`mode`------the ways to summarize embeddings, only support `["mean", "seperate"]`

`configs.json`------store all models' config



```
configs.json's format
{
	det_args: {
		det_model_name: {
			init_params: {} # use to create a detector
			det_params: {} # use to run detection
		}
	}
	
	encoding_args: {
		encoding_model_name: {
			init_params: {} # use to create an encoder
		}
	}
}
```

`det_args` is the settings of the detection models, only support `["yolov5_coco", "yolov5_amazon", "mdef", "grid"]`.

`init_params` is the initial parameters to create a detector, please do not modify them.

`det_params` is the parameters to run detection. If you want to print dets, please change `verbose: false` to  `verbose: true`.

`encoding_args` is the settings of the encoding models, only support `["cvnet", "vit"]`

`init_params` is the initial parameters to create a detector, please do not modify them.


## run trajectory prediction
`python infer_tutr.py`

the results will be in `fig/sdd`



# Reference

[https://github.com/sungonce/CVNet](https://github.com/sungonce/CVNet)

[https://github.com/mmaaz60/mvits_for_class_agnostic_od](https://github.com/mmaaz60/mvits_for_class_agnostic_od)

[https://github.com/lssiair/TUTR](https://github.com/lssiair/TUTR)