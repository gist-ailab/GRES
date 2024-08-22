

## Environment Setting

- CUDA 11.8, Pytorch 1.11.0 and Detectron2 0.6
```
<!-- conda create -n GRES python=3.8
conda activate GRES

## set cuda version to 11.3

## install torch 1.11.0
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -->

conda create -n G-RES --clone CARIS
conda activate G-RES

## install Detectron2 from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

##
cd gres_model/modeling/pixel_decoder/ops
sh make.sh

## 
pip install -r requirements.txt

## to use google translate API
pip install googletrans==4.0.0-rc1

```

## Dataset Setting
- RefCOCO
```
cd datasets
ln -s /ailab_mat/dataset/refCOCO/images
ln -s /ailab_mat/dataset/RIS/refcoco
ln -s /ailab_mat/dataset/RIS/gRefCOCO grefcoco
```

## Model Setting
- Download models from https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EqyL6nftLjdIihQG2rYirPoB1Sm3HBJwuZgtPII8WcevQw?e=pI1rrg

```
mkdir ckpt
unzip GRES.zip -d ckpt/
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/referring_swin_base.yaml --num-gpus 2 --dist-url auto --eval-only MODEL.WEIGHTS ckpt/gres_swin_base.pth OUTPUT_DIR output/
```

```
from transformers import BertTokenizer
from googletrans import Translator

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lang_tokens = inputs[0]['lang_tokens']
decoded_text = tokenizer.decode(lang_tokens[0], skip_special_tokens=True)
print(decoded_text)

translator = Translator()
translated = translator.translate(decoded_text, src='en', dest='ko')
print(translated.text)
```


```
def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

```


## Train
- get backbone weights
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl


CUDA_VISIBLE_DEVICES=0,1,3,5 python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 4 --dist-url auto \
    MODEL.WEIGHTS ckpt/swin_base_patch4_window12_384_22k.pth \
    OUTPUT_DIR work_dirs/base
```
