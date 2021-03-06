#!/usr/bin/env python3
from PIL import Image
import numpy as np
from pandas import Series
import sys, os, shutil
import ipdb

def usage():
    print("USAGE: {0} command args\
    \nwhere \"command args\" is one of the following:\
    \n\ndigitize_dir dir outdir - recursively digitize images in dir and save them in outdir, preserving the directory tree structure\
    \n\ndistribution dir patternsize outfile - saves to outfile the distribution of patternsize x patternsize patterns in dir (recursive)")

def pad_zeros(bit_string, length):
    while(len(bit_string)<length):
        bit_string = '0' + bit_string
    return(bit_string)

def string2pattern(string,shape):
    string = pad_zeros(string, shape[0]*shape[1])
    h,w = shape
    pattern = np.zeros((h,w),dtype='int')
    for row in range(h):
        for col in range(w):
            pattern[(row,col)] = string[w*row + col]
    return(pattern)

def pattern2string(pattern):
    string=''
    h,w = pattern.shape
    for row in range(h):
        for col in range(w):
            string += str(pattern[(row,col)])
    return(string)

def img2pattern(img,maxv):
    w,h = img.size
    pattern = np.zeros((w,h),dtype='int')
    data = np.array(img)
    for row in range(int(h)):
        for col in range(int(w)):
            #v = data[w*row + col]
            v = data[row,col]
            if v > maxv/2:
                pattern[(row,col)] = 1
    return(pattern)
    
def possible_patterns(shape):
    h,w = shape
    patterns = []
    for i in range(2**(h*w)):
        string = pad_zeros(bin(i)[2:],h*w)
        pattern = string2pattern(string)
        patterns.append(pattern)
    return(patterns)

def digitize_image(img_path):
    im = Image.open(img_path)
    if im.mode != "RGB":
        raise RuntimeError("Image {0} is not RGB".format(img_path))
    out = Image.new("L", im.size)
    w, h = im.size
    npixels = w*h
    lum = []
    out_data = []
    
    data = list(im.getdata())
    average_lum = 0
    for pixel in data:
        r,g,b = pixel
        #pixel_lum = 0.2989*r + 0.5870*g + 0.1140*b
        pixel_lum = r + g + b
        lum.append(pixel_lum)
        average_lum += pixel_lum
    average_lum *= 1/npixels
    for i in range(len(lum)):
        greyvalue = lum[i]
        if greyvalue > average_lum:
            out_data.append(0)
        else:
            out_data.append(255)
    out.putdata(out_data)
    return(out)

def digitize_directory_tree(dir_tree,outdir):
    if os.path.isdir(outdir):
        ans = input("{0} exists, overwrite? y/[n]".format(outdir))
        if ans == "y":
            shutil.rmtree(outdir)
        else:
            print("Exiting")
            exit(1)
            
    os.mkdir(outdir)
    for root,dirs,files in os.walk(dir_tree):
        for d in dirs:
            new_dir = outdir+'/'+root.lstrip(dir_tree).lstrip('/') + '/' +d
            os.mkdir(new_dir)
        if len(files) > 0:
            for f in files:
                if f[-4:] == '.jpg':
                    original_image = root.rstrip('/')+'/'+f
                    digitized_image = outdir+'/'+root.lstrip(dir_tree).lstrip('/') + '/' + f
                    out = digitize_image(original_image)
                    out.save(digitized_image)

                    

def distribution(dir_tree,psize,maxfiles=None):
    #w,h = pattern_shape
    if psize % 2 == 0:
        raise RuntimeError("Pattern size must be odd")
    patterns = {}
    n_samples = 0
    n_images = 0
    for root,dirs,files in os.walk(dir_tree):
        for f in files:
            if f[-4:] == '.jpg':
                n_images += 1
    counter = 1
    for root,dirs,files in os.walk(dir_tree):
        for f in files:
            if f[-4:] == '.jpg' and (maxfiles == None or counter <= maxfiles):
                imname = root+'/'+f
                im = Image.open(imname)
                if im.mode != 'L':
                    print("Image must be in binary format")
                    sys.exit(1)
                print("Analyzing: [{0}/{1}] {2}".format(counter,n_images,imname))
                width,height = im.size
                #WITHOUT OVERLAP:
                #hi = int((height - (height % h))/h)
                #wi = int((width - (width % w))/w)
                #for i in range(hi*wi):
                #    row = int((i - (i % wi))/wi)
                #    col = i%wi
                #    rect = (w*col,h*row,w*(col + 1), h*(row + 1))
                #    pattern = im.crop(rect)
                #    string=pattern2string(img2pattern(pattern))
                #    if string in patterns.keys():
                #        patterns[string] += 1
                #    else:
                #        patterns[string] = 1
                #    n_samples += 1

                #WITH OVERLAP:
                maxv = np.array(im).max()
                offset = int((psize-1)/2)
                center = (offset,offset)
                for i in range(int((width - 2*offset)*(height - 2*offset))):
                    row = int(offset + int(i/(width - 2*offset)))
                    col = int(offset + (i%(width - 2*offset)))
                    box = (col - offset,row - offset, col + offset + 1, row + offset + 1)
                    #if counter == 1:
                    #    print(box)
                    #ipdb.set_trace()
                    crop_img = im.crop(box)
                    string=pattern2string(img2pattern(crop_img,maxv))
                    if string in patterns.keys():
                        patterns[string] += 1
                    else:
                        patterns[string] = 1
                    n_samples += 1
                counter += 1

    for k in patterns.keys():
        v = patterns[k]
        patterns[k] = v/(n_samples)

    return(Series(patterns))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit(1)
    else:
        cmd = sys.argv[1]
        if cmd == "digitize_dir":
            digitize_directory_tree(sys.argv[2].rstrip('/'),sys.argv[3].rstrip('/'))
        elif cmd == "distribution" and len(sys.argv) == 5:
            dir, psize, outfile = sys.argv[2:]
            out = open(outfile,'w')
            dist = distribution(dir, int(psize), 3)
            #sorted_dist = sorted(dist, key=dist.get, reverse=True)
            #for k in sorted_dist:
            for k in dist:
                out.write(k + "," + str(dist[k]) + "\n")
            out.close()
        else:
            usage()
            exit(1)
