# Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories

This is the official code release for our ECCV22 paper on tracking particles through occlusions, presenting our new "particle video" method, **Persistent Independent Particles (PIPs)**. 

**[[Paper](https://arxiv.org/abs/2204.04153)] [[Project Page](https://particle-video-revisited.github.io/)]**

<img src='https://particle-video-revisited.github.io/images/fig1.jpg'>

This repo will be updated soon with more content and instructions.

## Requirements



## Demo


To download our reference model, run this script:

```
sh get_reference_model.sh
```

To run this model on a sample video, run this script:
```
python demo.py
```

This will run the model on a sequence included in `demo_images/`.

For each 8-frame subsequence, the model will return `trajs_e`. This is estimated trajectory data for the particles, shaped `B,S,N,2`, where `S` is the sequence length and `N` is the number of particles, and `2` is the `x` and `y` coordinates. The script will also produce tensorboard logs with visualizations, which go into `logs_demo/`, as well as a few gifs in `./*.gif`. 

In the tensorboard you should be able to find visualizations like this: 
<img src='https://particle-video-revisited.github.io/images/puppy_wide.gif'>

The original video is `https://www.youtube.com/watch?v=LaqYt0EZIkQ`. The file `demo_images/extract_frames.sh` shows the ffmpeg command we used to export frames from the mp4.


## 



## FlyingThings++ dataset

To create our FlyingThings++ dataset, first [download FlyingThings](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The data should look like:

```
../flyingthings/optical_flow/
../flyingthings/object_index/
../flyingthings/frames_cleanpass_webp/
```

Once you have the flows and masks, you can run `python link_flt3d_traj.py`. This will put 80G of trajectory data into:
```
../flyingthings/trajs_ad/
```

In parallel, you can run `python make_occlusions.py`. This will put 537M of occlusion data into:

```
../flyingthings/occluders_al
```

This data will be loaded and joined with corresponding rgb by the `FlyingThingsDataset` class in `flyingthingsdataset.py`, when training and testing.

(The suffixes "ad" and "al" are version counters.)


## Training

To train a model on the flyingthings++ dataset:

```
python train.py
```

It should print some diagnostic information about the model and data, then print a message for each iteration.

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


## Testing


```
../badja_data/DAVIS
```





### Citation

If you use this code for your research, please cite:

**Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories**.
[Adam W. Harley](https://cs.cmu.edu/~aharley),
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

