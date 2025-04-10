# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
import math
import pandas as pd

from dram_model import Fab_DRAM
from ssd_model import Fab_SSD
from logic_model import Fab_Logic
from op_carbon import OP_Carbon
from final_op_carbon import Final_OP_Carbon

def main():
    # Initialize DRAM and SSD fabrication models
    Fab_DRAM(config="ddr4_10nm")
    Fab_SSD(config="nand_10nm")

    # Read the combined CSV file
    data = pd.read_csv('combined_models.csv')  # update filename as needed

    # Iterate through each row/model/memory combo
    for index, row in data.iterrows():
        model = row['Model']
        mem_type = row['MemType']
        energy = float(row['Energy'])
        ips = float(row['Throughput'])
        area = float(row['Area'])

        print(f"\nModel: {model}, MemType: {mem_type}")
        de_yield = math.exp(- (area / 100) * 0.1)
        
        VGG_Logic = Fab_Logic(
            gpa="95", 
            carbon_intensity="loc_taiwan", 
            debug="True", 
            process_node=28, 
            fab_yield=de_yield
        )
        VGG_Logic.set_area(area=area / 100)
        packaging = 150

        embodied_carbon = (VGG_Logic.get_carbon() + packaging) / 1000  # kg
        print("Embodied Carbon: ", format(embodied_carbon, ".3f"), "kg")

        OP_CF = Final_OP_Carbon(
            carbon_intensity="src_solar", 
            energy=energy, 
            ips=ips, 
            no_int=1
        )
        operational_carbon = OP_CF.get_carbon() * 1e6  # ¼g
        print("Operational Carbon: ", format(operational_carbon, ".6f"), "ug")

        runtime = OP_CF.get_latency() / (60 * 60 * 24 * 365)
        total_carbon = (
            OP_CF.get_carbon() + ((VGG_Logic.get_carbon() + packaging) * (runtime / 3))
        ) * 1e6  # ¼g
        print("Total Carbon: ", format(total_carbon, ".3f"), "ug")

if __name__ == "__main__":
    main()
