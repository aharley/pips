#!/bin/bash

ffmpeg  -loglevel panic -i black_french_bulldog.mp4 -q:v 1 -vf fps=24 ./%06d.jpg
