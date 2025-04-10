
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
import math
import pandas as pd

from dram_model  import Fab_DRAM
from ssd_model   import Fab_SSD
from logic_model import Fab_Logic
from op_carbon import OP_Carbon
from final_op_carbon import Final_OP_Carbon

def main():
    Fab_DRAM(config="ddr4_10nm")
    Fab_SSD(config="nand_10nm")

    data = pd.read_csv('SRAM_CSV.csv')

    i = 0
    area = [0]*4
    energy = [0]*4
    ips = [0]*4

    for i in range(4):
        area[i] = data.loc[i, 'Area']
        energy[i] = data.loc[i, 'Energy']
        ips[i] = data.loc[i, 'Throughput']

    # F3_Logic = Fab_Logic(gpa="95", carbon_intensity = "src_coal", 
    #                      debug = "True", process_node=28, fab_yield=0.875)
    # print("Carbon per Area: ", F3_Logic.get_cpa())
    # F3_Logic.set_area(area=46.4/100)
    # packaging = 150
    # print("Total Carbon: ", format((F3_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")

    # Dell_Logic = Fab_Logic(gpa="95", carbon_intensity = "src_coal", 
    #                       debug = "True", process_node=28, fab_yield=0.875)
    # print("Carbon per Area: ", Dell_Logic.get_cpa())
    # Dell_Logic.set_area(area=698/100)
    # packaging = 150
    # print("Total Carbon: ", format(2*(Dell_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")
    
    # # Resnet50
    # print("RN50")
    
    # de_yield = math.exp (- (681.37/100) * 0.1)
    # RN50_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
    #                       debug = "True", process_node=20, fab_yield=de_yield)
    # print("Carbon per Area: ", RN50_Logic.get_cpa())
    # RN50_Logic.set_area(area=681.37/100)
    # packaging = 150
    # print("Embodied Carbon: ", format((RN50_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")
    # OP_CF = OP_Carbon(carbon_intensity = "src_wind", power=109.8, ips=37577, no_int=1000000)
    # print("Operational Carbon: ", format(OP_CF.get_carbon()*1000, ".2f"), "mg")
    # runtime = OP_CF.get_latency()/(60*60*24*365)
    # print("Total Carbon: ", format( (OP_CF.get_carbon()+((RN50_Logic.get_carbon() 
    #                                 + packaging)*(runtime/5)))*1000,
    #                                 ".2f"), "mg")
    
    # # RNN-T
    # print("RNN-T")
    # RNN_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
    #                       debug = "True", process_node=20, fab_yield=de_yield)
    # print("Carbon per Area: ", RNN_Logic.get_cpa())
    # RNN_Logic.set_area(area=681.37/100)
    # packaging = 150
    # print("Embodied Carbon: ", format((RNN_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")
    # OP_CF_RNN = OP_Carbon(carbon_intensity = "src_wind", power=74, ips=9720, no_int=1000000)
    # print("Operational Carbon: ", format(OP_CF_RNN.get_carbon()*1000, ".2f"), "mg")
    # runtime_RNN = OP_CF_RNN.get_latency()/(60*60*24*365)
    # print("Total Carbon: ", format( (OP_CF_RNN.get_carbon()+((RNN_Logic.get_carbon() 
    #                                 + packaging)*(runtime_RNN/5)))*1000,
    #                                 ".2f"), "mg")
    
    # Bert-large
    # print("Bert-large")
    # BL_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
    #                       debug = "True", process_node=20, fab_yield=de_yield)
    # print("Carbon per Area: ", BL_Logic.get_cpa())
    # BL_Logic.set_area(area=681.37/100)
    # packaging = 150
    # print("Embodied Carbon: ", format((BL_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")
    # OP_CF_BL = OP_Carbon(carbon_intensity = "src_wind", power=61.7, ips=3507, no_int=1000000)
    # print("Operational Carbon: ", format(OP_CF_BL.get_carbon()*1000, ".2f"), "mg")
    # runtime_BL = OP_CF_BL.get_latency()/(60*60*24*365)
    # print("Total Carbon: ", format( (OP_CF_BL.get_carbon()+((BL_Logic.get_carbon() 
    #                                 + packaging)*(runtime_BL/5)))*1000,
    #                                 ".2f"), "mg")
    
    # # VGG-16
    # print("vgg-16, RRAM")
    # for i in range(4):
    #     print("\n")
    #     print(f'{32 * (2**i)}x{32 * (2**i)}')
    #     de_yield = math.exp (- (area[i]/100) * 0.1)
    #     VGG_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
    #                           debug = "True", process_node=20, fab_yield=de_yield)
    #     # print("Carbon per Area: ", VGG_Logic.get_cpa())
    #     VGG_Logic.set_area(area=area[i]/100)
    #     packaging = 150
    #     print("Embodied Carbon: ", format((VGG_Logic.get_carbon() 
    #                                     + packaging)/1000, ".2f") , "kg")
    #     OP_CF = Final_OP_Carbon(carbon_intensity = "src_wind", energy = energy[i], 
    #                             ips = ips[i], no_int=1000000)
    #     print("Operational Carbon: ", format(OP_CF.get_carbon()*1000, ".2f"), "mg")
    #     runtime = OP_CF.get_latency()/(60*60*24*365)
    
    #     print("Total Carbon: ", format( (OP_CF.get_carbon()+((VGG_Logic.get_carbon() 
    #                                     + packaging)*(runtime/5)))*1000,
    #                                     ".2f"), "mg")
    
    # VGG-16
    print("vgg-16, SRAM")
    for i in range(4):
        print("\n")
        print(f'{32 * (2**i)}x{32 * (2**i)}')
        de_yield = math.exp (- (area[i]/100) * 0.1)
        VGG_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
                              debug = "True", process_node=20, fab_yield=de_yield)
        # print("Carbon per Area: ", VGG_Logic.get_cpa())
        VGG_Logic.set_area(area=area[i]/100)
        packaging = 150
        print("Embodied Carbon: ", format((VGG_Logic.get_carbon() 
                                        + packaging)/1000, ".3f") , "kg")
        OP_CF = Final_OP_Carbon(carbon_intensity = "src_solar", energy = energy[i], 
                                ips = ips[i], no_int=1)
        print("Operational Carbon: ", format(OP_CF.get_carbon()*1000000, ".3f"), "ug")
        runtime = OP_CF.get_latency()/(60*60*24*365)
    
        print("Total Carbon: ", format( (OP_CF.get_carbon()+((VGG_Logic.get_carbon() 
                                        + packaging)*(runtime/3)))*1000000,
                                        ".3f"), "ug")
  
    # print("Eyeriss")
    # de_yield = math.exp (- 0.1225 * 0.1)
    # BL_Logic = Fab_Logic(gpa="95", carbon_intensity = "loc_taiwan", 
    #                       debug = "True", process_node=65, fab_yield=de_yield)
    # print("Carbon per Area: ", BL_Logic.get_cpa())
    # BL_Logic.set_area(area=0.1225)
    # packaging = 150
    # print("Embodied Carbon: ", format((BL_Logic.get_carbon() 
    #                                 + packaging)/1000, ".2f") , "kg")
    # OP_CF_BL = Final_OP_Carbon(carbon_intensity = "src_solar", energy = 0.2954, ips = 1/1.252, no_int=1)
    # print("Operational Carbon: ", format(OP_CF_BL.get_carbon()*1000000, ".3f"), "ug")
    # runtime_BL = OP_CF_BL.get_latency()/(60*60*24*365)
    # print("Total Carbon: ", format( (OP_CF_BL.get_carbon()+((BL_Logic.get_carbon() 
    #                                 + packaging)*(runtime_BL/3)))*1000000,
    #                                 ".3f"), "ug")

    
    # print(OP_CF.get_ci())
    # opcarbon = (300*2*9.2)/3600000000 
    # print("Operational Carbon: ",format(opcarbon*1000000, ".2f"), "ug")

if __name__=="__main__":
    main()
