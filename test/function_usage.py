#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:05:06 2025

@author: fawaz
"""

#IMPORTS
from count_common_mac import count_common_mac_occurrences
from amp_phase_fft_plot import parse_csi_amp_phase_fft_plot
from csv_merge import combine_csv_files
from mac_id_counter import count_mac_occurrences
from get_n_csv_filepaths import get_n_csv_filepaths

#%% count_common_mac_occurences()

list_of_file_paths = ["/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_13_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_15_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_2.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_3.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv" \
                          ]
count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids=10, minimum_number_of_samples=None)

#%% parse_csi_amp_phase_fft_plot

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_18_feb_1.csv"

'''
target_macs = ["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"70:DB:98:9E:3A:A0", \
"70:DB:98:9E:3A:A1", \
"70:DB:98:9E:3A:A2", \
"FE:19:28:38:54:40", \
"70:0F:6A:DE:ED:20", \
"70:0F:6A:DE:ED:21", \
"70:0F:6A:DE:ED:22"
]
'''

'''
target_macs = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01"
]
'''

'''
target_macs = ["34:5F:45:A8:3C:19", \
"20:43:A8:64:3A:C1", \
"34:5F:45:A9:A4:19", \
"3C:8A:1F:90:E3:31", \
"8C:4F:00:3C:BF:4D"
]
'''


target_macs = ["34:5F:45:A8:3C:19", \
"20:43:A8:64:3A:C1"
]

    
subcarriers = [40]
cutoff_freq=0.1
start_time=1000
end_time=1100
plot_raw=True
plot_fil=True
plot_phase=True

parse_csi_amp_phase_fft_plot(file_path = file_path, target_macs=target_macs, subcarriers=subcarriers, cutoff_freq=cutoff_freq, start_time=start_time, \
                             end_time=end_time, plot_raw=plot_raw, plot_fil=plot_fil, plot_phase=plot_phase)



#%% combine_csv_files()

file_paths = ["/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_13_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_15_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_2.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_3.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv", \
                      "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv" \
                          ]

output_file =  "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/merged_files/merged.csv"   
combine_csv_files(file_paths=file_paths, output_file=output_file)

#%% count_mac_occurrences

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/data_collection/systematic_collection/24_mar_25_p2_04_00_07_00.csv"
n = 5

count_mac_occurrences(file_path, n=n)

#%% get_n_csv_filepaths()

folder_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025"
n = 5  # Number of files you want to retrieve

get_n_csv_filepaths(folder_path, n)