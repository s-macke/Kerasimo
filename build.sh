##!/bin/bash

function SVG2GIF {

for f in $1*.svg; do

echo "Processing $f file..";
filename=$(basename "$f")
filename="${filename%.*}"
rsvg-convert -a -w $2 $f -o $filename.png

done

convert -delay 100 -loop 0 -trim +repage $1*.png images/$1.gif

}

set -e

python models/or.py
SVG2GIF or 300

#python models/and.py
#SVG2GIF and 300

#python models/xor.py
#SVG2GIF xor 300

#python models/xor_wrong.py
#SVG2GIF xor_wrong 300

#python models/digits.py
#SVG2GIF digits 400

#python models/encode.py 3 0
#SVG2GIF encode30 400

## 6000 epochs
#python models/encode.py 2 0
#SVG2GIF encode20 400

#python models/encode.py 1 0
#SVG2GIF encode10 400

#python models/encode.py 1 8
#SVG2GIF encode18 400

#python models/digits_cnn.py
#SVG2GIF digits_cnn 800

#python models/mnist_cnn.py
#SVG2GIF mnist_cnn 1600

#set image_data_format first
#python models/snake.py
#SVG2GIF snake 1600

#python models/mnist_acgan.py
#convert plot_epoch*.png images/mnist_acgan.gif
