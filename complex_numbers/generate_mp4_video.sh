#! /bin/bash

mogrify -format jpg -quality 80 -resize 1920x1080 *.png
# create a video out of pictures
ffmpeg -framerate 10 -i f_%03d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
# ffmpeg -framerate 10 -i f_mod_%03d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
rm *.jpg
