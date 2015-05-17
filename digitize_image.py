#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import sys, os
import ipdb

def usage():
    print("USAGE: {0} directory outdirectory\
    \ndirectory and outdirectory must be different".format(sys.argv[0]))

def digitize_image(filepath):
    im = Image.open(filepath)
    if im.mode != "RGB":
        raise RuntimeError("Image {0} is not RGB".format(filepath))
    out = Image.new("L", im.size)
    w, h = im.size
    npixels = w*h
    out_data = []

    data = list(im.getdata())
    average_lum = 0
    for pixel in data:
        r,g,b = pixel
        lum = 0.2989*r + 0.5870*g + 0.1140*b
        out_data.append(lum)
        average_lum += lum
    average_lum *= 1/npixels
    for i in range(len(out_data)):
        greyvalue = out_data[i]
        if greyvalue > average_lum:
            out_data[i] = 0
        else:
            out_data[i] = 255
    out.putdata(out_data)
    return(out)

def main(dir,outdir):
    os.mkdir(outdir)
    for root,dirs,files in os.walk(dir):
        for d in dirs:
            new_dir = outdir+'/'+root.lstrip(dir).lstrip('/') + '/' +d
            os.mkdir(new_dir)
        if len(files) > 0:
            for f in files:
                if f[-4:] == '.jpg':
                    original_image = root.rstrip('/')+'/'+f
                    #digitized_image = outdir+('/'+root.lstrip(dir).strip('/')).strip('/') + '/' + f
                    digitized_image = outdir+'/'+root.lstrip(dir).lstrip('/') + '/' + f
                    try:
                        out = digitize_image(original_image)
                        out.save(digitized_image)
                    except:
                        pass

if __name__ == '__main__':
    if len(sys.argv) != 3  or sys.argv[1] == sys.argv[2]:
        usage()
        exit(1)
    else:
        main(sys.argv[1].rstrip('/'),sys.argv[2].rstrip('/'))

