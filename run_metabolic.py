import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

from metabolic_2_phasors_imaging_script import MetabolicPhasors

def main():
    """
    To run the demo example:
    imaging_file_path = "./demo/metabolic_2_1736248883_imaging.json"
    phasors_file_path = "./demo/metabolic_2_1736248883_phasor_ch1_h1.json"

    Otherwise create a folder "data" and put the files in there.
    The "data" folder is ignored by git.

    """

    root = tk.Tk()
    root.withdraw()

    # print(f"\n\nSELECT FILE IMAGING\n\n")
    # imaging_file_path = filedialog.askopenfilename(title="Select File Imaging")

    # print(f"\n\nSELECT FILE PHASORS\n\n")
    # phasors_file_path = filedialog.askopenfilename(title="Select File Phasors")

    print(f"\n\nSELECT FILES\n\n")
    file_paths = filedialog.askopenfilenames(title="Select Files")
    imaging_file_path = "./demo/metabolic_2_1736248883_imaging.json"
    phasors_file_path = "./demo/metabolic_2_1736248883_phasor_ch1_h1.json"
    for file_path in file_paths:
        if file_path.find('imaging') != -1:
            imaging_file_path = file_path
        elif file_path.find('phasor') != -1:
            phasors_file_path = file_path

    metabolic_phasors = MetabolicPhasors(
        imaging_file_path,
        phasors_file_path,
        0,
        0
    )
    metabolic_phasors.update_phasors_file_to_read()
    metabolic_phasors.read_phasors_file()
    metabolic_phasors.read_phasors_metadata()
    metabolic_phasors.read_imaging_file()
    metabolic_phasors.read_image_data()
    metabolic_phasors.fig = plt.figure(figsize=(18, 12))
    metabolic_phasors.gs = metabolic_phasors.fig.add_gridspec(
        1, # rows
        1, # columns
        width_ratios=[1],
        height_ratios=[1]
    )
    metabolic_phasors.plot_imaging()
    plt.show()

    # imaging_file_path = "./data/metabolic_2_1736248883_imaging.json"
    # phasors_file_path = "./data/metabolic_2_1736248883_phasor_ch1_h1.json"
    # threshold = 0
    # median_filter_iterations = 1

    threshold = input('Insert the threshold: ')
    median_filter_iterations = input('Insert the median filter iterations: ')
    threshold = int(threshold)
    median_filter_iterations = int(median_filter_iterations)

    metabolic_phasors = MetabolicPhasors(
        imaging_file_path,
        phasors_file_path,
        threshold,
        median_filter_iterations
    )
    metabolic_phasors.analyze_imaging_and_phasors_files()

    return None

if __name__ == "__main__":
    main()