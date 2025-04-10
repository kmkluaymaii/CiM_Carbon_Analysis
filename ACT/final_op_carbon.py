#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:23:03 2024

@author: rawisara
"""

import json
import sys


class Final_OP_Carbon():
    def __init__(self, carbon_intensity= "loc_taiwan", 
                 energy = 10, ips = 10, no_int= 1000000000):

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
        self.opcarbon = (self.ci)*energy
    
    def get_carbon(self,):
        return self.opcarbon
    
    def get_latency(self,):
        return self.latency