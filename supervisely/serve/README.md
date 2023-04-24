<div align="center" markdown>
<img src="xxx" />
  
# PIPs object tracking

state-of-the art interactive tracking using point trajectories integrated into Supervisely Videos Labeling tool

<img src='https://particle-video-revisited.github.io/images/fig1.jpg'>

<p align="center">
  <a href="#Overview">Overview</a> •w
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Controls">Controls</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/pips)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/pips)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/pips)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/pips)](https://supervise.ly)


</div>

# Overview

This app is an integration of PIPs model, which is a NN-assisted object tracking model. The PIPs model can track point trajectories. It is used to implement the tracking of polygons and rectangles that have multiple points on videos. The app could only be applied to videos.

# How to Run

1. Run the application from Ecosystem

2. Open Video Labeling interface

3. Configure tracking settings

4. Press `Track` button

<img src="" />

# Controls

| Key                                                           | Description                               |
| ------------------------------------------------------------- | ------------------------------------------|
| <kbd>5</kbd>                                       | Rectangle Tool                |
| <kbd>Ctrl + Space</kbd>                                       | Complete Annotating Object                |
| <kbd>Space</kbd>                                              | Complete Annotating Figure                |
| <kbd>Shift + T</kbd>                                          | Track Selected     |
| <kbd>Shift + Enter</kbd>                                      | Play Segment     |

# Acknowledgement

This app is based on the great work `PIPs` 
- [GitHub](https://github.com/aharley/pips) ![GitHub Org's stars](https://img.shields.io/github/stars/aharley/pips?style=social) 
- [Website](https://particle-video-revisited.github.io/)
- [Paper](https://arxiv.org/abs/2204.04153) 






