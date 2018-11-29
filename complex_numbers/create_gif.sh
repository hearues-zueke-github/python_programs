#! /bin/bash

# TODO: need to change the directory before applying these functions!

# mogrify -format jpg *.png
# mogrify -resize 125x *.jpg
# convert -delay 4 -loop 0 *.jpg myimage.gif

# or

mogrify -format jpg -quality 80 -resize x250 *.png
convert -delay 4 -loop 0 *.jpg myimage.gif
