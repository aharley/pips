<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/233959102-9c48949f-b353-4a4b-ab7d-c1da99dfd914.jpg" />
  
# PIPs object tracking

state-of-the art interactive tracking using point trajectories integrated into Supervisely Videos Labeling tool

<img src='https://particle-video-revisited.github.io/images/fig1.jpg'>

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-tracking-keypoints-of-a-bird-using-pips">Example: tracking keypoints of a bird using PIPs</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/pips)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/pips)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/pips/supervisely/serve)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/pips/supervisely/serve)](https://supervise.ly)


</div>

# Overview

This app is an integration of PIPs model, which is a NN-assisted interactive object tracking model. The PIPs model can track point trajectories. It is used to implement the tracking of polygons, points and rectangles on videos.

# How to Run

0. Run the application from Ecosystem

1. Open Video Labeling interface

2. Configure tracking settings

3. Press `Track` button

https://user-images.githubusercontent.com/115161827/234020558-86403646-0cc4-4832-a0d7-3eef5834a35e.mp4

4. After finishing working with the app, stop the app manually in the `App sessions` tab

You can also use this app to track keypoints. This app can track keypoints graph of any shape and number of points. Here is result of tracking cheetah keypoints:

https://user-images.githubusercontent.com/91027877/238157074-bbc06d9c-aa54-4b18-a777-8c41625ffdb6.mp4

# Example: tracking keypoints of a bird using PIPs

1. Open your video project, select suitable frame and click on "Screenshot" button in the upper right corner:

https://user-images.githubusercontent.com/91027877/238152827-1a6fcc7b-7d68-4168-86af-7406d6255d9c.mp4

2. Create keypoints class based on your screenshot:

https://user-images.githubusercontent.com/91027877/238153794-43870be8-37bd-434a-bdf7-536da5267602.mp4

3. Go back to video, set your recently created keypoints graph on target object, select number of frames to be tracked and click on "Track" button:

https://user-images.githubusercontent.com/91027877/238156937-5a61bde3-19fb-4ce6-8d7a-3fd022813710.mp4

You can change visualization settings of your keypoints graph in right sidebar:

https://user-images.githubusercontent.com/91027877/238154341-ed9acea5-2693-421d-a673-a6f4ab8f515a.mp4

# Acknowledgment

This app is based on the great work `PIPs` 
- [GitHub](https://github.com/aharley/pips) ![GitHub Org's stars](https://img.shields.io/github/stars/aharley/pips?style=social) Code in this repository is distributed under the MIT license.
- [Website](https://particle-video-revisited.github.io/)
- [Paper](https://arxiv.org/abs/2204.04153) 






