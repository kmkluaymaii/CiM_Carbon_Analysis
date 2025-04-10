#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:39:58 2024

@author: rawisara
"""

eyeriss_thp_alexnet = 1/(25.9*1e-3)
eyeriss_en_alexnet = 7.19 * 1e-3 # J

eyeriss_thp_vgg_16 = 1/(1252*1e-3)
eyeriss_en_vgg_16 = 295.4 * 1e-3 # J

print("EYERISS ENERGY:")
print("alexnet,{:.2e}".format(eyeriss_en_alexnet))
print("vgg-16,{:.2e}".format(eyeriss_en_vgg_16))
print("")
print("EYERISS THROUGHPUT:")
print("alexnet,{:.2e}".format(eyeriss_thp_alexnet))
print("vgg-16,{:.2e}".format(eyeriss_thp_vgg_16))

print("--- "*5)

unpu_thp_alexnet = 1/(2.89*1e-3)
unpu_en_alexnet = 0.84 * 1e-3 # J

unpu_thp_vgg_16 = 1/(54.6*1e-3)
unpu_en_vgg_16 = 16.2 * 1e-3 # J

print("UNPU ENERGY:")
print("alexnet,{:.2e}".format(unpu_en_alexnet))
print("vgg-16,{:.2e}".format(unpu_en_vgg_16))
print("")
print("UNPU THROUGHPUT:")
print("alexnet,{:.2e}".format(unpu_thp_alexnet))
print("vgg-16,{:.2e}".format(unpu_thp_vgg_16))


print("--- "*5)

envision_thp_alexnet = 1/(21.3*1e-3)
envision_en_alexnet = 0.94 * 1e-3 # J

envision_thp_vgg_16 = 1/(598.8*1e-3)
envision_en_vgg_16 = 15.6 * 1e-3 # J

print("ENVISION ENERGY:")
print("alexnet,{:.2e}".format(envision_en_alexnet))
print("vgg-16,{:.2e}".format(envision_en_vgg_16))
print("")
print("ENVISION THROUGHPUT:")
print("alexnet,{:.2e}".format(envision_thp_alexnet))
print("vgg-16,{:.2e}".format(envision_thp_vgg_16))