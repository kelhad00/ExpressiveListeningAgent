#!/bin/python
# python2.7

import os

print("Baseline optimised for arousal:")
delay      = 2.2
modalities = "0 0 1"  # text only
os.system("python run_baseline.py " + str(delay) + " " + modalities)
print(" ")

print("Baseline optimised for valence:")
delay      = 1.2
modalities = "1 1 1"  # multimodal
os.system("python run_baseline.py " + str(delay) + " " + modalities)
print(" ")

print("Baseline optimised for liking:")
delay      = 2.0
modalities = "0 0 1"  # text only
os.system("python run_baseline.py " + str(delay) + " " + modalities)
