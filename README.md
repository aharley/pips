# Persistent Independent Particles (PIPs)

This is the official code release for our ECCV 2022 paper, "Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories". **[[Paper](https://arxiv.org/abs/2204.04153)] [[Project Page](https://particle-video-revisited.github.io/)]**

<img src='https://particle-video-revisited.github.io/images/fig1.jpg'>

### Update 09/18/23:

[PIPs++](https://github.com/aharley/pips2) is now available. This is the upgrade of PIPs presented in our ICCV 2023 paper, "PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking". **[[Paper](https://arxiv.org/abs/2307.15055)] [[Project Page](https://pointodyssey.com/)]**


### Update 12/15/22:

- Added new reference model, trained on a slightly harder version of FlyingThings++ and larger batch size.
- Updated `train.py` and `flyingthings.py` with these settings. Get the new reference model (as before) with `./get_reference_model.sh` or from [HuggingFace](https://huggingface.co/aharley/pips).
- New results are better than the paper. Here they are: 

  BADJA:
  ```
  bear: 76.1
  camel: 91.6
  cows: 87.7
  dog-agility: 31.0
  dog: 45.4
  horsejump-high: 60.9
  horsejump-low: 58.1
  avg: 64.4
  ```

  CroHD:
  ```
  vis: 4.57
  occ: 7.71
  ```

  FlyingThings:
  ```
  pips: ate_vis = 6.03, ate_occ = 19.56
  raft: ate_vis = 16.65, ate_occ = 43.22
  dino: ate_vis = 42.98, ate_occ = 76.78
  ```

## Requirements

The lines below should set up a fresh environment with everything you need: 
```
conda create --name pips
source activate pips
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
conda install pip
pip install -r requirements.txt
```

## Demo

To download our reference model, download the weights from [Hugging Face. ![](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/aharley/pips)

Alternatively, you can run this:

```
sh get_reference_model.sh
```

To run this model on a sample video, run this:
```
python demo.py
```

This will run the model on a sequence included in `demo_images/`.

For each 8-frame subsequence, the model will return `trajs_e`. This is estimated trajectory data for the particles, shaped `B,S,N,2`, where `S` is the sequence length and `N` is the number of particles, and `2` is the `x` and `y` coordinates. The script will also produce tensorboard logs with visualizations, which go into `logs_demo/`, as well as a few gifs in `./*.gif`. 

In the tensorboard for `logs_demo/` you should be able to find visualizations like this: 
<img src='https://particle-video-revisited.github.io/images/puppy_wide.gif'>

To track points across arbitrarily-long videos, run this:
```
python chain_demo.py
```
In the tensorboard for `logs_chain_demo/` you should be able to find visualizations like this:
<img src='https://particle-video-revisited.github.io/images/pup_long_compressed.gif'>

This type of tracking is much more challenging, so you can expect to see more failures here. In particular, here we are using our visibility-aware chaining method, so mistakes tend to propagate into the future. 

The original video is `https://www.youtube.com/watch?v=LaqYt0EZIkQ`. The file `demo_images/extract_frames.sh` shows the ffmpeg command we used to export frames from the mp4.


## Model implementation

To inspect our PIPs model implementation, the main file to look at is `nets/pips.py`

## FlyingThings++ dataset

To download our exact FlyingThings++ dataset, try [this link](https://drive.google.com/drive/folders/1zzWkGGFgJPyHpVaSA19zpYlux1Mf6wGC?usp=share_link). If the link doesn't work, contact me for a secondary link, or create the data from the original FlyingThings, as described next. 

To create our FlyingThings++ dataset, first [download FlyingThings](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The data should look like:

```
../flyingthings/optical_flow/
../flyingthings/object_index/
../flyingthings/frames_cleanpass_webp/
```

Once you have the flows and masks, you can run `python make_trajs.py`. This will put 80G of trajectory data into:
```
../flyingthings/trajs_ad/
```

In parallel, you can run `python make_occlusions.py`. This will put 537M of occlusion data into:

```
../flyingthings/occluders_al
```

This data will be loaded and joined with corresponding rgb by the `FlyingThingsDataset` class in `flyingthingsdataset.py`, when training and testing.

(The suffixes "ad" and "al" are version counters.)

Once loaded by the dataloader (`flyingthingsdataset.py`), the RGB will look like this:
<img src='https://particle-video-revisited.github.io/images/flt_rgbs.gif'>

The corresponding trajectories will look like this:
<img src='https://particle-video-revisited.github.io/images/flt_trajs.gif'>


## Training

To train a model on the flyingthings++ dataset:

```
python train.py
```

First it should print some diagnostic information about the model and data. Then, it should print a message for each training step, indicating the model name, progress, read time, iteration time, and loss. 

```
model_name 1_8_128_I6_3e-4_A_tb89_21:34:46
loading FlyingThingsDataset [...] found 13085 samples in ../flyingthings (dset=TRAIN, subset=all, version=ad)
loading occluders [...] found 7856 occluders in ../flyingthings (dset=TRAIN, subset=all, version=al)
not using augs in val
loading FlyingThingsDataset [...] found 2542 samples in ../flyingthings (dset=TEST, subset=all, version=ad)
loading occluders...found 1631 occluders in ../flyingthings (dset=TEST, subset=all, version=al)
warning: updated load_fails (on this worker): 1/13085...
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000001/100000; rtime 9.79; itime 20.24; loss = 40.30593
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000002/100000; rtime 0.01; itime 0.37; loss = 43.12448
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000003/100000; rtime 0.01; itime 0.36; loss = 36.60324
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000004/100000; rtime 0.01; itime 0.38; loss = 40.91223
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000005/100000; rtime 0.01; itime 0.35; loss = 79.32227
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000006/100000; rtime 0.01; itime 0.53; loss = 22.14469
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000007/100000; rtime 0.04; itime 0.46; loss = 24.75386
[...]
```
Occasional `load_fails` warnings are typically harmless. They indicate when the dataloader fails to get `N` trajectories for a given video, which simply causes a retry. If you greatly increase `N` (the number of trajectories), or reduce the crop size, you can expect this warning to occur more frequently, since these constraints make it more difficult to find viable samples. As long as your `rtime` (read time) is small, then things are basically OK. 

To reproduce the result in the paper, you should train with 4 gpus, with horizontal and vertical flips, with a command like this:
```
python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1,2,3]
```

***
## Testing

We provide evaluation scripts for all of the datasets reported in the paper. By default, these scripts will evaluate a PIPs model, with a checkpoint folder specified by the `--init_dir` argument.

You can also try a baseline, with `--modeltype='raft'` or `--modeltype='dino'`. To do this, you will also want to download a [RAFT](https://github.com/princeton-vl/RAFT) model. The DINO model should download itself, since torch makes this easy.

### CroHD

The CroHD head tracking data comes from the "Get all data" link on the [Head Tracking 21 MOT Challenge](https://motchallenge.net/data/Head_Tracking_21/) page. Downloading and unzipping that should give you the folders HT21 and HT21Labels, which our dataloader relies on. After downloading the data (and potentially editing the `dataset_location` in `crohddataset.py`), you can evaluate the model in CroHD with: `python test_on_crohd.py`

### FlyingThings++

To evaluate the model in Flyingthings++, first set up the data as described in the earlier section of this readme, then: `python test_on_flt.py`

### DAVIS

The DAVIS dataset comes from the "TrainVal - Images and Annotations - Full-Resolution" link, on the [DAVIS Challenge](https://davischallenge.org/davis2017/code.html) page. After downloading the data (and potentially editing the `data_path` in `test_on_davis.py`), you can visualize the model's outputs in DAVIS with: `
python test_on_davis.py`

### BADJA

To evaluate the model in BAJDA, first follow the instructions at the [BADJA repo](https://github.com/benjiebob/BADJA). This will involve downloading DAVIS trainval full-resolution data. After downloading the data (and potentially editing `BADJA_PATH` in `badjadataset.py`), you can evaluate the model in BADJA with: `python test_on_badja.py`



## Citation

If you use this code for your research, please cite:

**Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories**.
[Adam W. Harley](https://adamharley.com/),
[Zhaoyuan Fang](https://zfang399.github.io/),
[Katerina Fragkiadaki](http://cs.cmu.edu/~katef/). In ECCV 2022.

Bibtex:
```
@inproceedings{harley2022particle,
  title={Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories},
  author={Adam W Harley and Zhaoyuan Fang and Katerina Fragkiadaki},
  booktitle={ECCV},
  year={2022}
}
```

