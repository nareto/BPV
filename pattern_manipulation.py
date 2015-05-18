from PIL import Image
import numpy as np
import sys, os
import ipdb


def pad_zeros(bit_string, length):
    while(len(bit_string)<length):
        bit_string = '0' + bit_string
    return(bit_string)

def string2pattern(string,shape):
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

def img2pattern(img):
    w,h = img.size
    pattern = np.zeros((w,h),dtype='int')
    data = img.getdata()
    for row in range(h):
        for col in range(w):
            v = data[w*row + col]
            if int(v) == 255:
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

def digitize_directory_tree(dir_tree,outdir):
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
                    try:
                        out = digitize_image(original_image)
                        out.save(digitized_image)
                        #print(digitized_image)
                    except:
                        pass


def distribution(dir_tree,pattern_shape):
    w,h = pattern_shape
    patterns = {}
    n_samples = 0
    for root,dirs,files in os.walk(dir_tree):
        for f in files:
            if f[-4:] == '.jpg':
                im = Image.open(root+'/'+f)
                width,height = im.size
                hi = int((height - (height % h))/h)
                wi = int((width - (width % w))/w)
                for i in range(hi*wi):
                    row = int((i - (i % wi))/wi)
                    col = i%wi
                    rect = (w*col,h*row,w*(col + 1), h*(row + 1))
                    pattern = im.crop(rect)
                    string=pattern2string(img2pattern(pattern))
                    if string in patterns.keys():
                        patterns[string] += 1
                    else:
                        patterns[string] = 1
                    n_samples += 1

    for k in patterns.keys():
        v = patterns[k]
        patterns[k] = v/(n_samples)

    return(patterns)
