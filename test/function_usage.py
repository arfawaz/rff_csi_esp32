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
from csv_merge_4_cols import combine_csv_files_4_cols
from mac_id_counter import count_mac_occurrences
from get_n_csv_filepaths import get_n_csv_filepaths
from copy_csv_with_serial import copy_csv_with_serial
from common_macids_in_list_of_macids import common_macids_in_list_of_macids
from count_target_macid_occurrences import count_target_macid_occurrences

#%% count_common_mac_occurences()

list_of_file_paths = [
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\17_27_feb_25_p8_07_10_09_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\13_27_feb_25_p5_01_10_01_45.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\10_26_feb_25_p4_05_00_07_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\15_27_feb_25_p6_03_15_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\16_27_feb_25_p7_05_00_07_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\11_26_feb_25_p5_07_00_.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\12_27_feb_25_p4_11_50_01_05.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\14_27_feb_25_p5_02_50_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\18_28_feb_25_p7_04_40_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\19_28_feb_25_p8_05_15_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\1_24_feb_25_p3_04_31_05_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\20_28_feb_25_p9_07_00_9_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\21_01_mar_25_p6_12_35_01_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\22_01_mar_25_p7_01_15_02_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\23_01_mar_25_p8_02_30_03_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\24_01_mar_25_p9_03_40_04_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\25_01_mar_25_p10_04_45_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\26_02_mar_25_p7_01_00_02_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\27_02_mar_25_p8_02_00_03_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\28_02_mar_25_p9_03_00_04_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\29_02_mar_25_p10_04_15_05_15.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\2_24_feb_25_p4_05_17_06_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\30_02_mar_25_p1_05_15_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\31_03_mar_25_p10_04_00_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\32_03_mar_25_p1_05_00_06_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\33_04_mar_25_p9_02_00_03_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\34_04_mar_25_p10_03_00_04_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\35_04_mar_25_p1_04_00_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\36_04_mar_25_p2_05_00_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\37_08_mar_25_p3_05_15_06_15.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\38_08_mar_25_p4_06_20_07_20.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\39_08_mar_25_p5_07_20_08_25.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\3_24_feb_25_p5_07_00_09_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\40_08_mar_25_p6_08_25_09_25.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\41_08_mar_25_p7_09_25_10_25.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\42_09_mar_25_p4_01_50_02_20.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\43_09_mar_25_p5_02_20_02_50.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\44_09_mar_25_p6_02_50_03_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\45_09_mar_25_p7_03_20_03_50.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\46_09_mar_25_p8_03_50_04_20.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\47_10_mar_25_p5_05_45_06_45.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\48_10_mar_25_p6_06_45_07_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\49_11_mar_25_p3_05_40_06_40.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\4_25_feb_25_p6_1_00_01_15.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\50_11_mar_25_p6_06_45_08_10.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\51_11_mar_25_p7_08_15_09_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\52_11_mar_25_p8_09_15_10_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\53_11_mar_25_p9_10_15_11_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\54_12_mar_25_p7_04_55_06_10.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\55_12_mar_25_p8_06_10_06_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\56_12_mar_25_p9_06_55_07_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\57_12_mar_25_p10_07_55_08_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\58_12_mar_25_p1_08_55_09_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\59_13_mar_25_p7_04_30_05_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\5_25_feb_25_p7_01_15_03_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\60_13_mar_25_p8_05_30_06_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\61_13_mar_25_p9_06_30_07_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\62_13_mar_25_p10_07_30_08_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\63_13_mar_25_p2_10_50_11_50.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\64_18_mar_25_p2_01_00_04_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\65_18_mar_25_p3_04_00_05_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\66_19_mar_25_p2_02_00_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\67_19_mar_25_p3_05_00_05_50.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\68_20_mar_25_p6_03_00_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\69_20_mar_25_p10_05_00_05_45.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\6_25_feb_25_p4_03_05_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\70_22_mar_25_p2_03_10_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\71_24_mar_25_p2_04_00_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\72_24_mar_25_p3_07_00_10_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\73_24_mar_25_p4_10_00_12_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\74_25_mar_25_p4_04_00_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\75_26_mar_25_p8_05_44_08_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\76_26_mar_25_p9_08_45_11_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\77_27_mar_25_p2_12_30_03_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\78_27_mar_25_p10_03_40_06_40.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\79_31_mar_25_p5_04_50_08_10.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\7_25_feb_25_p5_05_05_07_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\80_31_mar_25_p6_08_10_10_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\81_01_apr_25_p3_12_15_03_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\82_01_apr_25_p7_03_15_06_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\83_01_apr_25_p10_06_50_09_50.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\84_02_apr_25_p10_04_50_08_10.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\85_02_apr_25_p10_08_00_10_20.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\86_03_apr_25_p5_01_00_04_20.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\87_03_apr_25_p6_04_45_07_50.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\88_03_apr_25_p4_07_55_10_55.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\89_05_apr_25_p4_02_30_05_30.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\8_25_feb_25_p6_07_05_09_00.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\90_05_apr_25_p5_05_30_8_40.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\91_05_apr_25_p6_08_40_11_40.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\92_06_apr_25_p4_02_05_05_05.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\93_06_apr_25_p5_05_05_08_05.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\94_07_apr_25_p6_04_20_07_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_numbered\systematic_collection_numbered\9_26_feb_25_p3_01_00_01_50.csv"
]
                          
count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids=12, minimum_number_of_samples=None)

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

#%% combine_csv_files_4_cols()

file_paths = [
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\13_27_feb_25_p5_01_10_01_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\10_26_feb_25_p4_05_00_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\15_27_feb_25_p6_03_15_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\11_26_feb_25_p5_07_00_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\12_27_feb_25_p4_11_50_01_05.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\14_27_feb_25_p5_02_50_.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\21_01_mar_25_p6_12_35_01_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\2_24_feb_25_p4_05_17_06_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\38_08_mar_25_p4_06_20_07_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\39_08_mar_25_p5_07_20_08_25.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\3_24_feb_25_p5_07_00_09_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\40_08_mar_25_p6_08_25_09_25.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\42_09_mar_25_p4_01_50_02_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\43_09_mar_25_p5_02_20_02_50.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\44_09_mar_25_p6_02_50_03_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\47_10_mar_25_p5_05_45_06_45.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\48_10_mar_25_p6_06_45_07_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\4_25_feb_25_p6_1_00_01_15.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\50_11_mar_25_p6_06_45_08_10.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\68_20_mar_25_p6_03_00_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\6_25_feb_25_p4_03_05_05_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\73_24_mar_25_p4_10_00_12_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\74_25_mar_25_p4_04_00_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\79_31_mar_25_p5_04_50_08_10.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\7_25_feb_25_p5_05_05_07_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\80_31_mar_25_p6_08_10_10_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\86_03_apr_25_p5_01_00_04_20.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\87_03_apr_25_p6_04_45_07_50.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\88_03_apr_25_p4_07_55_10_55.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\89_05_apr_25_p4_02_30_05_30.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\8_25_feb_25_p6_07_05_09_00.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\90_05_apr_25_p5_05_30_8_40.csv", \
    # r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\91_05_apr_25_p6_08_40_11_40.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\92_06_apr_25_p4_02_05_05_05.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\93_06_apr_25_p5_05_05_08_05.csv", \
     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\94_07_apr_25_p6_04_20_07_20.csv"
]


output_file =  r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_p5_p6_last_merged\p4_p5_p6_last_merged.csv"
combine_csv_files_4_cols(file_paths=file_paths, output_file=output_file)

#%% count_mac_occurrences

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/merged_files/merged.csv"
n = 5

count_mac_occurrences(file_path, n=n)

#%% get_n_csv_filepaths()

folder_path = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6"
n = 1000  # Number of files you want to retrieve

get_n_csv_filepaths(folder_path, n)

#%% copy_csv_with_serial

source_folder = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/data_collection/systematic_collection'
destination_folder = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/data_collection/systematic_collection_numbered' 
copy_csv_with_serial(source_folder, destination_folder)

#%% count_target_macid_occurrences

target_macids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]  # List of MAC IDs to track
    
csv_file = r'C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv'  # Replace with the path to your CSV file

# Run the function to count occurrences
count_target_macid_occurrences(csv_file, target_macids)

#%% build_macid_position_table

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
count_dict, adjusted_counts = build_macid_position_table(mac_ids, file_paths)