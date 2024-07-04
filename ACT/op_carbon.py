#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:40:25 2024

@author: rawisara
"""
import json
import sys


class OP_Carbon():
    def __init__(self, carbon_intensity= "loc_taiwan", 
                 power= 6, ips=1000, no_int= 1000000000):

        if "loc" in carbon_intensity:
                with open("carbon_intensity/location.json", 'r') as f:
                    loc_configs = json.load(f)
    
                    loc = carbon_intensity.replace("loc_", "")
    
                    assert loc in loc_configs.keys()
    
                    fab_ci = loc_configs[loc]
    
        elif "src" in carbon_intensity:
                with open("carbon_intensity/source.json", 'r') as f:
                    src_configs = json.load(f)
    
                    src = carbon_intensity.replace("src_", "")
    
                    assert src in src_configs.keys()
    
                    fab_ci = src_configs[src]
    
        else:
                print("Error: Carbon intensity must either be loc | src dependent")
                sys.exit()
        
        self.latency = no_int/ips
        self.ci = fab_ci/(60*60*1000)
        self.opcarbon = (self.ci)*power*(self.latency)
    
    def get_carbon(self,):
        return self.opcarbon
    
    def get_latency(self,):
        return self.latency





# opcarbon = (300*2*9.2)/3600000000 
#     print("Operational Carbon: ",format(opcarbon*1000000, ".2f"), "ug")