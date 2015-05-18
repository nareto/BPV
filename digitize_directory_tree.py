#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys, os
import ipdb
import pattern_manipulaiton as pm

def usage():
    print("USAGE: {0} directory outdirectory\
    \ndirectory and outdirectory must be different".format(sys.argv[0]))

if __name__ == '__main__':
    if len(sys.argv) != 3  or sys.argv[1] == sys.argv[2]:
        usage()
        exit(1)
    else:
        dir_tree = sys.argv[1].rstrip('/')
        outdir = sys.argv[2].rstrip('/')
        pm.digitize_directory_tree(dir_tree,outdir)

