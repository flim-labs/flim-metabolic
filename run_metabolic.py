import tkinter as tk
from tkinter import filedialog

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

    print(f"\n\nSELECT FILE IMAGING\n\n")
    imaging_file_path = filedialog.askopenfilename(title="Select File Imaging")

    print(f"\n\nSELECT FILE PHASORS\n\n")
    phasors_file_path = filedialog.askopenfilename(title="Select File Phasors")

    # imaging_file_path = "./data/metabolic_2_1736248883_imaging.json"
    # phasors_file_path = "./data/metabolic_2_1736248883_phasor_ch1_h1.json"
    threshold = 0
    median_filter_iterations = 1

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