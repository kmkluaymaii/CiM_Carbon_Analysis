import glob
import os

#models = ["vgg16", "gpt2"]
# models = ["myalexnet2", "alexnet", "resnet-50", "vgg-16", "googlenet", "efficientnet-b7"]
# arch = "eyeriss_like"

# models = ["alexnet"]
# arch = "sys_array"


def extract_outputs(model):
    
    output_dir = "../outputs/" + model + "/" 
    print(output_dir)
    

    print ("Model \t Energy/inference (J) \t Area (mm2) \t Throughput (inf/s)")
    #for model in models:    
    energy = 0
    area = 0
    cycles = 0
    
    
    for folder in os.listdir(output_dir):
        if model in folder:
            print(folder)
            if os.path.isdir(output_dir + folder):
                stat_file = output_dir + folder + "/timeloop-mapper.stats.txt"
        
                with open(stat_file, 'r') as f:
                    energy_stat = [line for line in f if line.startswith("Energy:")][0]
                    energy_uj = energy_stat.split(' ')[1] # uJ
                    energy += float(energy_uj)
                
                with open(stat_file, 'r') as f:
                    cycle_stat = [line for line in f if line.startswith("Cycles:")][0]                  
                    cycle = cycle_stat.split(' ')[1] 
                    cycles += float(cycle)
                    
                # with open(stat_file, 'r') as f:
                #     dram_parts = f.read().split("=== DRAM ===")[1]
                #     input_parts = dram_parts.split("Inputs:")[1]
                    
                #     input_dram_energy = [line for line in input_parts.split('\n') 
                #                          if "Energy (total)" in line][0]
                #     input_dram_energy = input_dram_energy.split(' ')[-2] # pJ
                    
                #     print(input_dram_energy)
                
                if area==0:
                    art_file = output_dir + folder + "/timeloop-mapper.ART.yaml"
                    with open(art_file, 'r') as f:
                        lines = f.readlines()
                        names = [l for l in lines if 'name' in l]
                        areas = [l for l in lines if 'area' in l]
                        
                        for i in range (len(names)):
                            #print(names[i], areas[i])
                            
                            count = float(names[i].split('..')[1].replace(']', ''))
                            area_individual_um = float(areas[i].split(': ')[1].replace('\n', ''))
                            #print(area_individual_um, count)
                            
                            area_um = area_individual_um * count
                            area += area_um
                        
                        
                        
    energy *= 1e-6 # conver to J
    area = area * 1e-6 # mm2
    throughput = 1 / (cycles * 1e-9) # s
    
    #print("Total energy/inference: {:.2e} J".format(energy))
    print ("{} \t {:.2e} \t {:.2f} \t {:.2e}".format(model, energy, area, throughput))
    
    return energy, area

extract_outputs("bert-large", "sys_array_adept")