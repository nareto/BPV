#!/usr/bin/env python3

import sys,os
import numpy as np
from PIL import Image
import pattern_manipulation as pm

def usage():
    print("USAGE: {0} image csv_outfile".format(sys.argv[0]))


def main(image, outfile):
    n = 3
    patterns = pe.possible_patterns((n,n))
    im = Image.open(image)
    width,height = im.size
    w = (width - (width % n))/n
    h = (height - (height % n))/n
    for i in range(w*h):
        pass
    
if __name__ == '__main__':
    if len(sys.argv) != 3  or sys.argv[1] == sys.argv[2]:
        usage()
        exit(1)
    else:
        main(sys.argv[1].rstrip('/'),sys.argv[2])

