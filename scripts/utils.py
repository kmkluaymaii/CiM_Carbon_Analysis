import shutil
import time
from typing import Callable, Union, Iterable, List
import os
import threading
import joblib
import timeloopfe.v4 as tl
import sys
import importlib.util
import sys
import glob
from tqdm import tqdm
import yaml

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)
from processors import ArrayProcessor
from tl_output_parsing import parse_timeloop_output, MacroOutputStats, MacroOutputStatsList

from plots import *
# fmt: on

from joblib import delayed as delayed


def single_test(result) -> MacroOutputStatsList:
    return MacroOutputStatsList([result])


def parallel_test(
    delayed_calls: List[Callable], n_jobs: int = 32
) -> MacroOutputStatsList:
    if not isinstance(delayed_calls, Iterable):
        delayed_calls = [delayed_calls]

    delayed_calls = list(delayed_calls)
    return MacroOutputStatsList(
        tqdm(
            joblib.Parallel(return_as="generator", n_jobs=n_jobs)(delayed_calls),
            total=len(delayed_calls),
        )
    )


def path_from_model_dir(*args):
    return os.path.abspath(os.path.join(THIS_SCRIPT_DIR, "..", "models", *args))


def get_run_dir():
    out_dir = os.path.join(
        THIS_SCRIPT_DIR,
        "..",
        "outputs",
        f"{os.getpid()}.{threading.current_thread().ident}",
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_spec(
    macro: str,
    tile: str = None,
    chip: str = None,
    system: str = "ws_dummy_buffer_one_macro",
    iso: str = None,
    dnn: str = None,
    layer: str = None,
    max_utilization: bool = False,
    extra_print: str = "",
    jinja_parse_data: dict = None,
) -> tl.Specification:
    paths = [
        os.path.abspath(
            os.path.join(THIS_SCRIPT_DIR, "..", "models", "top.yaml.jinja2")
        )
    ]

    jinja_parse_data = {
        **(jinja_parse_data or {}),
        "macro": macro,
        "tile": tile,
        "chip": chip,
        "system": system,
        "iso": iso if iso else macro,
        "dnn": dnn,
        "layer": layer,
    }
    jinja_parse_data = {k: v for k, v in jinja_parse_data.items() if v is not None}

    paths2print = [p for p in paths]
    while any(paths2print):
        if all(paths2print[0][0] == p[0] for p in paths2print):
            paths2print = [p[1:] for p in paths2print]
        else:
            break
    paths2print = ", ".join(paths2print)

    if not extra_print:
        extra_print = f"{os.getpid()}.{threading.current_thread().ident}"

    spec = tl.Specification.from_yaml_files(
        *paths, processors=[ArrayProcessor], jinja_parse_data=jinja_parse_data
    )
    if max_utilization:
        spec.variables["MAX_UTILIZATION"] = True

    return spec

# def create_layer_directory(DNN: str, layer: int) -> str:
#     layer_dir = f"{DNN}_layer_{layer}"
#     if not os.path.exists(layer_dir):
#         os.makedirs(layer_dir)
#     return layer_dir

def create_layer_directory(base_output_dir: str, DNN: str, layer: int) -> str:
    layer_dir = os.path.join(base_output_dir, f"{DNN}_layer_{layer}")
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    return layer_dir

num_layers = 1

def run_mapper(
    spec: tl.Specification,
    accelergy_verbose: bool = False,
) -> dict:
    output_dir = get_run_dir()
    base_output_dir = os.path.dirname(output_dir)
    base_output_dir = os.path.join(base_output_dir,"vgg16_256_RRAM_de") // Change this everytime according to the type of cim
    print(base_output_dir)
    # results = {}
    # num_layers = len(spec.layer) 
    global num_layers

    if num_layers < 17:
        # Create a directory for the current layer
        layer_dir = create_layer_directory(base_output_dir,"vgg16_256_RRAM_de", num_layers) // Change this accordingly, has to be the same as above 
        print(f"Layer directory: {layer_dir}")
        num_layers += 1

        run_prefix = f"{layer_dir}/timeloop-mapper"
        mapper_result = tl.call_mapper(
            specification=spec,
            output_dir=layer_dir,
            log_to=os.path.join(output_dir, f"{run_prefix}.log"),
        )
        if accelergy_verbose:
            tl.call_accelergy_verbose(
                specification=spec,
                output_dir=layer_dir,
                log_to=os.path.join(output_dir, "accelergy.log"),
            )

    # results[layer_number] = MacroOutputStats.from_output_stats(mapper_result)

    # return results

    # output_dir = get_run_dir()
    # run_prefix = f"{output_dir}/timeloop-mapper"
    
    # mapper_result = tl.call_mapper(
    #     specification=spec,
    #     output_dir=output_dir,
    #     log_to=os.path.join(output_dir, f"{run_prefix}.log"),
    # )
    # if accelergy_verbose:
    #     tl.call_accelergy_verbose(
    #         specification=spec,
    #         output_dir=output_dir,
    #         log_to=os.path.join(output_dir, "accelergy.log"),
    #     )
    return MacroOutputStats.from_output_stats(mapper_result)
    
def quick_run(
    macro: str,
    variables: dict = None,
    accelergy_verbose: bool = False,
    **kwargs,
):
    spec = get_spec(
        macro=macro,
        system="ws_dummy_buffer_one_macro",
        max_utilization=True,
        **kwargs,
    )
    variables = variables or {}
    spec.variables.update(variables)
    for k in list(spec.variables.keys()):
        if k not in variables:
            spec.variables[k] = spec.variables.pop(k)

    return run_mapper(spec, accelergy_verbose=accelergy_verbose)


def get_diagram(
    macro: str,
    container_names: Union[str, List[str]] = (),
    ignore: List[str] = (),
    variables: dict = None,
    **kwargs,
):
    spec = get_spec(
        macro=macro,
        system="ws_dummy_buffer_one_macro",
        max_utilization=True,
        **kwargs,
    )
    spec.variables.update(variables or {})
    return spec.to_diagram(container_names, ignore)


def get_test(
    macro: str,
    function_name: str,
):
    # Python path is macro path + _tests.py
    path = os.path.abspath(
        os.path.join(
            THIS_SCRIPT_DIR, "..", "models", "arch", "1_macro", macro, "_tests.py"
        )
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No test file found at {path}")
    modspec = importlib.util.spec_from_file_location("modname", path)
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    return getattr(module, function_name)


def run_layer(
    macro: str,
    layer: str,
    variables: dict = None,
    callfunc: Callable = None,
    iso: str = None,
    tile=None,
    chip=None,
    system="ws_dummy_buffer_many_macro",
):
    spec = get_spec(
        macro=macro, iso=iso, layer=layer, tile=tile, chip=chip, system=system
    )
    spec.architecture.name2leaf("macro").attributes["has_power_gating"] = True

    variables = variables or {}

    spec.variables.update(variables)
    for k in list(spec.variables.keys()):
        if k not in variables:
            spec.variables[k] = spec.variables.pop(k)

    if callfunc is not None:
        callfunc(spec)

    try:
        return run_mapper(spec=spec)
    except Exception as e:
        print(f"Error processing spec with {macro}, {iso}, {layer}, {variables}")
        raise e

import re

def extract_range(line):
    match = re.search(r'\[(\d+):(\d+)(,\d+)?\)', line)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return end - start
    return None

def read_and_compute_multiplication(filename, i):
    filename = f"../outputs/{filename}/{filename}_layer_{i+1}/timeloop-mapper.map.txt"
    with open(filename, 'r') as file:
        lines = file.readlines()

    start_searching = False
    ranges = []

    for line in lines:
        # Start searching when 'inter_macro_in_system_spatial' is found
        if "inter_macro_in_system_spatial" in line:
            start_searching = True
        
        # Stop searching if we leave the 'inter_macro_in_system_spatial' section
        elif start_searching and "inter_" in line and "inter_macro_in_system_spatial" not in line:
            break

        # If we are in the 'inter_macro_in_system_spatial' section, look for the ranges
        if start_searching:
            if "for" in line:
                range_size = extract_range(line)
                if range_size is not None:
                    ranges.append(range_size)
            elif line.strip() == "":  # Exit the section on an empty line
                break

    # Compute the multiplication of all ranges, or return 1 if no ranges were found
    if not ranges:
        print(f"All macro: 1")
        return 1

    result = 1
    for size in ranges:
        result *= size
        print(f"Size: {size}")
    print(f"All macro: {result}")
    return result


def extract_layer_number(folder_name):
    match = re.search(r'layer_(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')

def extract_outputs(model):
    
    output_dir = "../outputs/" + model + "/" 
    print(output_dir)

    print ("Model \t Energy/inference (J) \t Area (mm2) \t Throughput (inf/s)")
    #for model in models:    
    energy = 0
    area = 0
    cycles = 0
    folders = sorted([folder for folder in os.listdir(output_dir) if model in folder], key=extract_layer_number)
    area_per_many_macro = 0
    area_layer = 0
    j = 0
    total_sum = 0
    total_adc_area = 0  # Total area for ADC across all layers
    total_other_areas = {}  # Dictionary to store total areas for other components
    total_col_area = 0
    total_row_area = 0
    total_cim_unit_area = 0
    total_area = 0
    total_macro = 0
    
    for folder in folders:
        total = 0
        if model in folder:
            print(f"Processing layer: {folder}")
            # print(folder)
            if os.path.isdir(output_dir + folder):
                stat_file = output_dir + folder + "/timeloop-mapper.stats.txt"
                art_file = output_dir + folder + "/timeloop-mapper.ART.yaml"
        
                with open(stat_file, 'r') as f:
                    energy_stat = [line for line in f if line.startswith("Energy:")][0]
                    energy_uj = energy_stat.split(' ')[1] # uJ
                    energy += float(energy_uj)
                    # print(f"Energy for this layer: {float(energy_uj)} uj")
                
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
                
                # if area==0:
                # print(area)
                # art_file = output_dir + folder + "/timeloop-mapper.ART.yaml"
                # with open(art_file, 'r') as f:
                #     lines = f.readlines()
                #     names = [l for l in lines if 'name' in l]
                #     areas = [l for l in lines if 'area' in l]

                #     for i in range (len(names)):
                #         #print(names[i], areas[i])
                        
                #         count = float(names[i].split('..')[1].replace(']', ''))
                #         area_individual_um = float(areas[i].split(': ')[1].replace('\n', ''))
                #         #print(area_individual_um, count)
                #         area_um = area_individual_um * count
                #         area_per_many_macro += area_um
                        
                #     print(f"Area before: {area_per_many_macro}")
                #     macro = read_and_compute_multiplication(model, j)
                #     area_layer = (area_per_many_macro / 4096) * macro
                #     print(f"Area after with {macro} macros: {area_layer}")
                #     area += area_layer 
                #     print(f"Total Area: {area}")
                #     area_per_many_macro = 0
                #     # print(j)
                #     j += 1    
                #     print("\n")
            if j < 16:
                with open(art_file, 'r') as file:
                    data = yaml.safe_load(file)
                
                    # Extract and print the name, area, number of entries, and ratio for non-zero areas
                for table in data['ART']['tables']:
                    name = table.get('name')
                    area = table.get('area')
                    if area != 0:
                        # Simplify the name for the output
                        if 'system_top_level.' in name:
                            name = name.split('system_top_level.')[-1]
                        
                        # Extract the number of entries from the name
                        if '[' in name:
                            name_part, entries_part = name.split('[')
                            number_of_entries = entries_part.split('..')[1].replace(']', '')
                            name = name_part.strip()
                        else:
                            number_of_entries = 'Unknown'
                        
                        macro = read_and_compute_multiplication(model, j)
                        # Calculate total ratio
                        if number_of_entries != 'Unknown' and number_of_entries.isdigit():
                            total_ratio = (area / 4096) * macro * int(number_of_entries)
                        else:
                            total_ratio = 0
                        
                        total += total_ratio
                           
                        # Accumulate area for ADC or other components
                        if 'adc' in name.lower():
                            total_adc_area += total_ratio
                        else:
                            if name not in total_other_areas:
                                total_other_areas[name] = 0
                            total_other_areas[name] += total_ratio
            
                        print(f"{name.replace('_', ' ')} area: {area}, number of {name}: {number_of_entries}, \ntotal {name.replace('_', ' ')} area: ({area}*{number_of_entries}/4096) * macro = {total_ratio}\n")
                j += 1 
                total_sum += total
                total_macro += macro
                print(f"Total Area for layer {j}: {total}\n")
                print(f"Total Macro for layer {j}: {total_macro}\n")
            print(f"Total Area for all layers: {total_sum}")
            print(f"Total Area for ADC across all layers: {total_adc_area}")
    
    # Print total area for other components
    for component, area in total_other_areas.items():
        print(f"Total Area for {component.replace('_', ' ')} across all layers: {area}")
        # print(component.replace('_', ' '))
        if component.replace('_', ' ') ==  "column drivers":
            total_col_area = area
        elif component.replace('_', ' ') ==  "row drivers":
            total_row_area = area
        else:
            total_cim_unit_area = area
                                
    energy *= 1e-6 # conver to J
    total_area = total_sum * 1e-6 # mm2
    throughput = 1 / (cycles * 1e-7) # s

    total_adc_area = total_adc_area * 1e-6 # mm2
    total_col_area = total_col_area * 1e-6 # mm2
    total_row_area = total_row_area * 1e-6 # mm2
    total_cim_unit_area = total_cim_unit_area * 1e-6 # mm2
    print("\n")
    print(f"Total Area for ADC across all layers: {total_adc_area}")
    print(f"Total Area for column driver across all layers: {total_col_area}")
    print(f"Total Area for row driver across all layers: {total_row_area}")
    print(f"Total Area for cim unit across all layers: {total_cim_unit_area}")
    print(f"Total Macro across all layers: {total_macro}")
    #print("Total energy/inference: {:.2e} J".format(energy))
    print ("{} \t {:.2e} \t {:.2f} \t {:.2e}".format(model, energy, total_area, throughput))
    
    return total_area, total_adc_area, total_col_area, total_row_area, total_cim_unit_area, total_macro, energy, throughput


# def extract_outputs2(model):
    
#     output_dir = "../outputs/" + model + "/"
#     print(output_dir)

#     print("Model \t Energy/inference (J) \t Area (mm2) \t Throughput (inf/s)")
    
#     energy = 0
#     cycles = 0
#     total_adc_area = 0
#     total_other_areas = {}
#     total_sum = 0
#     folders = sorted([folder for folder in os.listdir(output_dir) if model in folder], key=extract_layer_number)
    
#     j = 0
    
#     for folder in folders:
#         if model in folder:
#             print(f"Processing layer: {folder}")
#             if os.path.isdir(output_dir + folder):
#                 stat_file = output_dir + folder + "/timeloop-mapper.stats.txt"
                
#                 with open(stat_file, 'r') as f:
#                     energy_stat = [line for line in f if line.startswith("Energy:")][0]
#                     energy_uj = energy_stat.split(' ')[1]  # uJ
#                     energy += float(energy_uj)
                    
#                     cycle_stat = [line for line in f if line.startswith("Cycles:")][0]
#                     cycles += float(cycle_stat.split(' ')[1])

#                 with open(stat_file, 'r') as f:
#                     data = yaml.safe_load(f)
                    
#                     # Extract and print the name, area, number of entries, and ratio for non-zero areas
#                 total = 0
#                 for table in data['ART']['tables']:
#                     name = table.get('name')
#                     area = table.get('area', 0)
                    
#                     if area != 0:
#                         # Simplify the name for the output
#                         if 'system_top_level.' in name:
#                             name = name.split('system_top_level.')[-1]
                        
#                         # Extract the number of entries from the name
#                         if '[' in name:
#                             name_part, entries_part = name.split('[')
#                             number_of_entries = entries_part.split('..')[1].replace(']', '')
#                             name = name_part.strip()
#                         else:
#                             number_of_entries = 'Unknown'
                        
#                         macro = read_and_compute_multiplication(model, j)
#                         # Calculate total ratio
#                         if number_of_entries != 'Unknown' and number_of_entries.isdigit():
#                             total_ratio = (area / 4096) * macro * int(number_of_entries)
#                         else:
#                             total_ratio = 0
                        
#                         total += total_ratio
#                         j += 1    
#                         # Accumulate area for ADC or other components
#                         if 'adc' in name.lower():
#                             total_adc_area += total_ratio
#                         else:
#                             if name not in total_other_areas:
#                                 total_other_areas[name] = 0
#                             total_other_areas[name] += total_ratio
            
#                         print(f"{name.replace('_', ' ')} area: {area}, number of {name}: {number_of_entries}, \ntotal {name.replace('_', ' ')} area: ({area}*{number_of_entries}/4096) * macro = {total_ratio}\n")
                
#                 total_sum += total
#                 print(f"Total Area for layer {folder}: {total}\n")
            
# print(f"Total Area for all layers: {total_sum}")
# print(f"Total Area for ADC across all layers: {total_adc_area}")
    
# # Print total area for other components
# for component, area in total_other_areas.items():
#     print(f"Total Area for {component.replace('_', ' ')} across all layers: {area}")

# energy *= 1e-6 # conver to J
# area = area * 1e-6 # mm2
# throughput = 1 / (cycles * 1e-9) # s

#     #print("Total energy/inference: {:.2e} J".format(energy))
#     print ("{} \t {:.2e} \t {:.2f} \t {:.2e}".format(model, energy, area, throughput))
    
#     return energy, area, throughput
