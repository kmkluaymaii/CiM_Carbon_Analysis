#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:44:19 2024

@author: rawisara
"""

import json
import sys


class OP_Carbon():
    def __init__(self, carbon_intensity, power, latency)

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
        
        self.opcarbon = carbon_intensity*(power/latency)*1000
    
    def get_carbon(self, ):
        return self.opcarbon




