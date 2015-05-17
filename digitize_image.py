import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

plot_histograms = 0
im = Image.open("img/stock1.jpg")
#im = Image.open("img/rgbspirals.jpg")
w, h = im.size
npixels = w*h
out = Image.new("L", im.size)
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
        
print(average_lum, average_lum/256)

out.putdata(out_data)
out.save("img/out.jpg")
if plot_histograms:
    hist = im.histogram()
    r = np.array(hist[:256])
    g = np.array(hist[256:256*2])
    b = np.array(hist[256*2:256*3])
    lumarray = 0.2989*r + 0.5870*g + 0.1140*b

    index = np.arange(0, 256, 1)
    plt.plot(index, r, 'r', index, g, 'g', index, b, 'b', index, lumarray, 'k')
    plt.show()
