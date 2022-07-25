# Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories

This is the official code release for our ECCV22 paper on tracking particles through occlusions, presenting our new "particle video" method, **Persistent Independent Particles (PIPs)**. 

**[[Paper](https://arxiv.org/abs/2204.04153)] [[Project Page](https://particle-video-revisited.github.io/)]**

<img src='https://particle-video-revisited.github.io/images/fig1.png'>

This repo will be updated soon with more content and instructions.

## Requirements


## Demo

For a quick demo, try: 
```
python demo.py
```

This will run the model on a sequence included in `demo_images/`.

For each 8-frame subsequence, the model will return `trajs_e`. This is estimated trajectory data for the particles, shaped `B,S,N,2`, where `S` is the sequence length and `N` is the number of particles, and `2` is the `x` and `y` coordinates. The script will also produce tensorboard logs with visualizations, which go into `logs_demo/`, as well as a few gifs in `./*.gif`. 

In the tensorboard you should be able to find visualizations like this: 
<img src='https://particle-video-revisited.github.io/images/puppy_wide.gif'>

The original video is `https://www.youtube.com/watch?v=LaqYt0EZIkQ`. The file `demo_images/extract_frames.sh` shows the ffmpeg command we used to export frames from the mp4.


## 





## Training


## Testing


```
../badja_data/DAVIS
```



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

