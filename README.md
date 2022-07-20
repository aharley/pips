# Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories

This is the official code release for our ECCV22 paper on tracking particles through occlusions. 

**[[Paper](https://arxiv.org/abs/2204.04153)] [[Project Page](https://particle-video-revisited.github.io/)]**

<img src='https://particle-video-revisited.github.io/images/fig1.png'>

This repo will be updated soon with more content.

## FlyingThings++ dataset

To create our FlyingThings++ dataset, first [download FlyingThings](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).

Once you have the flows (`optical_flow/`) and masks (`object_index/`), you can run `python link_flt3d_traj.py`. This will create an additional 80G of trajectory data.

Finally, run `python make_occlusions.py`. This will create 537M of occlusion data.

This data will be loaded and joined with corresponding rgb (`frames_cleanpass_webp/`) by `flyingthingsdataset.py` for training and testing.


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

