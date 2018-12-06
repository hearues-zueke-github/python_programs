#! /bin/bash

# TODO: need to change the directory before applying these functions!

# mogrify -format jpg *.png
# mogrify -resize 125x *.jpg
# convert -delay 4 -loop 0 *.jpg myimage.gif

# or

mogrify -format jpg -quality 80 -resize 480x270 *.png
convert -delay 10 -loop 0 *.jpg myimage.gif
rm *.jpg

# extract pictures from gif: convert -coalesce angle_orig.gif out%05d.png
