#!/bin/python
# python2.7

import os

print("Baseline optimised for arousal:")
delay      = 2.2
# text only
os.system("python ./emotion_baseline_scripts/run_baseline.py --arousal --valence --liking -delay " + str(delay) + " --text")
print(" ")

print("Baseline optimised for valence:")
delay      = 1.2
# multimodal
os.system("python ./emotion_baseline_scripts/run_baseline.py --arousal --valence --liking -delay " + str(delay) + " --audio --video --text")
print(" ")

print("Baseline optimised for liking:")
delay      = 2.0
# text only
os.system("python ./emotion_baseline_scripts/run_baseline.py --arousal --valence --liking -delay " + str(delay) + " --text")
