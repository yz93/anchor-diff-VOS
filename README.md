# Anchor diffusion VOS
<img align="right" src="http://www.robots.ox.ac.uk/~yz/img/gif.gif" width="250px" />

This repository contains code for the paper

**Anchor Diffusion for Unsupervised Video Object Segmentation** <br />
Zhao Yang\*, [Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)\*, [Luca Bertinetto](http://www.robots.ox.ac.uk/~luca), [Weiming Hu](https://scholar.google.com/citations?user=Wl4tl4QAAAAJ&hl=en), [Song Bai](http://songbai.site/), [Philip H.S. Torr](http://www.robots.ox.ac.uk/~tvg/) <br />
**ICCV 2019** | **[PDF](https://arxiv.org/abs/1910.10895)** | **[BibTex](bib)** <br />

## Setup
Code tested for Ubuntu 16.04, Python 3.7, PyTorch 0.4.1, and CUDA 9.2.

* Clone the repository and change to the new directory.
```
git clone https://github.com/yz93/anchor-diff-VOS-internal.git && cd anchor-diff-VOS
```
* Save the working directory to an environment variable for reference.
```shell
export AnchorDiff=$PWD
```
* Set up a new conda environment.
    * For installing PyTorch 0.4.1 with different versions of CUDA, see [here](https://pytorch.org/get-started/previous-versions/#via-conda). 
```
conda create -n anchordiff python=3.7 pytorch=0.4.1 cuda92 -c pytorch
source activate anchordiff
pip install -r requirements.txt
```


## Data preparation
- Download the data set
```shell
cd $AnchorDiff
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip -d data
```
* Download pre-trained weights.
```shell
cd $AnchorDiff
wget www.robots.ox.ac.uk/~yz/snapshots.zip
unzip snapshots.zip -d snapshots
```
* (If you do not intend to apply instance pruning described in the paper, feel free to skip this.) Download the detection results that we have computed using [ExtremeNet](https://github.com/xingyizhou/ExtremeNet),
and generate the pruning masks.
```shell
cd $AnchorDiff
wget www.robots.ox.ac.uk/~yz/detection.zip
unzip detection.zip
python detection_filter.py
```

## Evaluation on [DAVIS 2016](https://davischallenge.org/davis2016/code.html)
* Examples for evaluating mean IoU on the validation set with options,
    * *save-mask* (default 'True') for saving the predicted masks,
    * *ms-mirror* (default 'False') for multiple-scale and mirrored input (slow),
    * *inst-prune* (default 'False') for instance pruning,
    * *model* (default 'ad') specifying models in Table 1 of the paper,
    * *eval-sal* (default 'False') for computing saliency measures, MAE and F-score.
```shell
cd $AnchorDiff
python eval.py
python eval.py --ms-mirror True --inst-prune True --eval-sal True
```
* Use the [benchmark tool](https://github.com/davisvideochallenge/davis-matlab) to evaluate the saved masks under more metrics.
* [Pre-computed results](https://www.robots.ox.ac.uk/~yz/val_results.zip)

## License
The [MIT License](https://choosealicense.com/licenses/mit/).
