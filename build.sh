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

python models/or.py
SVG2GIF or 300

python models/and.py
SVG2GIF and 300

python models/xor.py
SVG2GIF xor 300

python models/digits.py
SVG2GIF digits 400

python models/compress.py
SVG2GIF compress 400

python models/digits_cnn.py
SVG2GIF digits_cnn 800

python models/mnist_cnn.py
SVG2GIF mnist_cnn 1600
