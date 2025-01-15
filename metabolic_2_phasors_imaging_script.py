
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import matplotlib
import matplotlib.colors as mcolors

"""
CONFIG SECTION
In this section you can configure your settings for the analysis.
"""

class MetabolicPhasors:

    __slots__ = [
        'taus',
        'imaging_file_path',
        'phasors_file_path',
        'active_channels',
        'active_channels_list',
        'channel',
        'harmonic',
        'colormap',
        'phasors_path',
        'phasors_data',
        'phasors_data_header',
        'laser_period_ns',
        'image_width',
        'image_height',
        'imaging_data',
        'enabled_channels',
        'image_data',
        'decay_curve_data',
        'df',
        'threshold',
        'median_filter_iterations',
        'median_filter_window',
        'fig',
        'gs',
        'metabolic_index_mean',
        'metabolic_color_map'
    ]

    def __init__(self, imaging_file_path, phasors_file_path, threshold, median_filter_iterations):
        """
        Here we define some parameters.

        imaging_file_path: Path to the target imaging acquisition .json file.
                            You can change this variable.

        phasors_file_path: Path to the target phasors .json file. 
                            You can change this variable.

        active_channels: You can change this variable.

        harmonic: Phasor harmonic to analyze. You can customize this value, 
                    depending on the total number of harmonics for the acquisition.
                    Default is 1.

        colormap: Colormap to apply to the image reconstruction. 
                    You can customize this value. Default is 'hot'
                    To correctly set the colormap name refer to this documentation:
                    https://matplotlib.org/stable/gallery/color/colormap_reference.html

        phasors_path: taking the phasors_file_path updates the file name based on
                        channel and harmonic

        phasors_data: what is inside phasors_path

        phasors_data_header: useful to print information

        image_data: it is a dictionary labeled by the channel number, so as
                    {0: array([[ 7,  0,  0, ...,  7, 13,  8],
                        [12,  0,  0, ..., 12,  8, 12],
                        [ 9,  0,  0, ...,  9, 10, 11],
                        ...,
                        [ 7,  0,  0, ...,  7, 13,  8],
                        [12,  0,  0, ..., 12,  8, 12],
                        [ 9,  0,  0, ...,  9, 10, 11]], shape=(256, 256), dtype=uint32)
                    }
        """

        self.taus = {
            'phase': {
                'ox': 3.4e-9, # in s
                'glyco': 0.4e-9
            },
            'modulation': {
                'ox': 3.4e-9, # in s
                'glyco': 0.4e-9
            }
        }
        
        self.imaging_file_path = imaging_file_path
        self.phasors_file_path = phasors_file_path
        self.active_channels = "0"
        self.active_channels_list = list(map(int, self.active_channels.split(',')))
        self.channel = self.active_channels_list[0]
        self.harmonic = 1
        self.colormap = 'grey'
        self.threshold = threshold
        self.median_filter_iterations = median_filter_iterations
        self.median_filter_window = 3

        self.df = pd.DataFrame()

        #### cool
        colors = [
            "white",
            "yellow",
            "green",
            "lightblue",
            "blue",
            "violet",
            "red"
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_colormap",
            colors,
            N=256
        )
        self.metabolic_color_map = cmap

        return None

    def read_imaging_file(self):
        with open(os.path.join(self.imaging_file_path), "r") as f:
            data = json.load(f)
            if "header" not in data:
                print("Invalid data file. Missing 'header' field.")
                exit(0)
            if "data" not in data:
                print("Invalid data file. Missing 'data' field.")
                exit(0)

            # file_id must be IMG1
            # 'IMG1' is an identifier for imaging .json files
            magic_bytes = bytes(data["header"]["file_id"])
            magic_bytes_string = magic_bytes.decode("ascii")
            if magic_bytes_string != "IMG1":
                print(
                    "Invalid data file. Selected file is not an Imaging .json file"
                )
                exit(0)
            # Extract file metadata
            metadata = data["header"]
            self.laser_period_ns = metadata["laser_period_ns"]
            self.image_width = metadata["image_width"]
            self.image_height = metadata["image_height"]
            self.imaging_data = data["data"]
            self.enabled_channels = [index for index, value in enumerate(metadata["channels"]) if value]

            return None
        
    def read_phasors_file(self):
        with open(self.phasors_path, "r") as f:
            data = json.load(f)
            print('###############################################################')
            print('#### Phasor file')
            for label in data:
                print(label)
            print('###############################################################')
            if "header" not in data:
                print("Invalid data file. Missing 'header' field.")
                exit(0)
            if "data" not in data:
                print("Invalid data file. Missing 'data' field.")
                exit(0)

            # file_id must be IPG1
            # 'IPG1' is an identifier for phasors imaging .json files
            magic_bytes = bytes(data["header"]["file_id"])
            magic_bytes_string = magic_bytes.decode("ascii")
            if magic_bytes_string != "IPG1":
                print(
                    "Invalid data file. Selected file is not a Phasors Imaging .json file"
                )
                exit(0)

            self.phasors_data = data["data"]
            self.phasors_data_header = data["header"]

            for index in self.phasors_data:
                print(index)

            print('----------------------')
            print(self.phasors_data['frame']) # 2
            print(self.phasors_data['channel']) # 1
            print(self.phasors_data['harmonic']) # 1
            print(np.shape(self.phasors_data["g_data"]))
            print(np.shape(self.phasors_data["s_data"]))

            g_data = np.array(self.phasors_data["g_data"]).ravel()
            s_data = np.array(self.phasors_data["s_data"]).ravel()
            # mask = (np.abs(g_data) < 1e9) & (np.abs(s_data) < 1e9) & ~((g_data == 0) & (s_data == 0))
            # g_data = g_data[mask]
            # s_data = s_data[mask]
            # condition = (np.abs(g_data) < 1e9) & (np.abs(s_data) < 1e9) & ~((g_data == 0) & (s_data == 0))
            condition1 = (np.abs(g_data) < 2) & (np.abs(s_data) < 1)
            condition2 = (g_data > 0) & (s_data > 0)
            condition3 = ~((g_data == 0) & (s_data == 0))
            g_data = np.where(condition1 & condition2 & condition3, g_data, np.nan)
            s_data = np.where(condition1 & condition2 & condition3, s_data, np.nan)

            #### DEBUG
            self.df["g_data"] = g_data
            self.df["s_data"] = s_data

            return None
             
    def read_phasors_metadata(self):
        # Active channels
        if "channels" in self.phasors_data_header:
            active_channels = [index for index, ch in enumerate(self.phasors_data_header["channels"]) if ch]
            print("Enabled channels: " + ", ".join([f"Channel {index + 1}" for index in active_channels]))
        else:
            print("ERROR: Active channels not found in self.phasors_data_header.")
            exit(0)               
        # Laser period ns
        if "laser_period_ns" in self.phasors_data_header:
            laser_period_ns = self.phasors_data_header["laser_period_ns"]
            print("Laser period: " + str(laser_period_ns) + "ns") 
        else:
            print("ERROR: Laser period not found in self.phasors_data_header.") 
            exit(0)    
        # Imaging type of experiment
        if "step" in self.phasors_data_header:
            print("Imaging type of experiment: " + self.phasors_data_header["step"]) 
        # Imaging reconstruction
        if "reconstruction" in self.phasors_data_header:
            print("Imaging reconstruction: " + self.phasors_data_header["reconstruction"])   
        # Image width
        if "image_width" in self.phasors_data_header:
            print("Image width: " + str(self.phasors_data_header["image_width"]) + "px") 
        else:
            print("ERROR: Image width not found in self.phasors_data_header.") 
            exit(0)       
        # Image height
        if "image_height" in self.phasors_data_header:
            print("Image height: " + str(self.phasors_data_header["image_height"]) + "px")  
        else:
            print("ERROR: Image height not found in self.phasors_data_header.") 
            exit(0)     
        # Number of frames           
        if "frames" in self.phasors_data_header:
            print("Number of frames: " + str(self.phasors_data_header["frames"]))  
        # Tau (ns)
        if "tau_ns" in self.phasors_data_header:
            print("Tau (ns): " + str(self.phasors_data_header["tau_ns"]))      
        # Harmonics
        if "harmonics" in self.phasors_data_header:
            print("Harmonics: " + str(self.phasors_data_header["harmonics"]))            
   
    def read_image_data(self):
        for channel_index in self.enabled_channels:
            image_data_dict = {}
            decay_curve_dict = {}
            image_data = np.zeros((self.image_height, self.image_width), dtype=np.uint32)
            decay_curve = np.zeros(256, dtype=np.int64)
            channel_data = self.imaging_data[channel_index]
            for pixel_index in range(self.image_width * self.image_height):
                pixel_data = channel_data[pixel_index]
                if pixel_data:
                    for bin_index, bin_count in pixel_data:
                        if bin_index < 256:
                            decay_curve[bin_index] += bin_count

                total_photon_count = (
                    sum(bin_count for _, bin_count in pixel_data) if pixel_data else 0
                )
                row = pixel_index // self.image_width
                col = pixel_index % self.image_width
                image_data[row, col] = total_photon_count

            image_data_dict[channel_index] = image_data
            decay_curve_dict[channel_index] = decay_curve

            self.image_data = image_data_dict
            self.decay_curve_data = decay_curve_dict

            self.df['image_data'] = self.image_data[0].ravel()

        return None
    
    def plot_imaging(self):
        # Imaging
        ch_image_data = self.image_data[self.enabled_channels.index(self.channel)]

        if self.threshold > 0:
            data_2d = np.reshape(
                self.df['metabolic_ratio'],
                ch_image_data.shape
            )
            nan_mask = np.isnan(data_2d)
            ch_image_data = np.where(nan_mask, 0, ch_image_data)

        ax1 = self.fig.add_subplot(self.gs[0, 0])
        data_min = np.min(ch_image_data)
        data_max = np.max(ch_image_data)
        im = ax1.imshow(
            ch_image_data,
            cmap=self.colormap,
            vmin=data_min,
            vmax=data_max
        )
        # ax1.axis('off')
        ax1.set_xlabel("x (pixels)", labelpad=-10)
        ax1.set_ylabel("y (pixels)", labelpad=-10)
        ax1.set_xticks([1, ch_image_data.shape[1]])
        ax1.set_xticklabels(["1", f"{ch_image_data.shape[1]}"])
        ax1.set_yticks([1, ch_image_data.shape[0]])
        ax1.set_yticklabels(["1", str(ch_image_data.shape[0])]) 
        # ax1.set_title(f"Channel {self.channel + 1} - Image ({self.image_width}x{self.image_height})")
        ax1.set_title(
            f"Fluorescence Intensity",
            weight='bold'
        )
        cbar = plt.colorbar(im, ax=ax1)
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(
            "Photon Count",
            fontsize=12,
            weight='bold',
            labelpad=0
        )

        return None
    
    def plot_tcspc(self):
        # TCSPC
        ch_decay_data = self.decay_curve_data[self.enabled_channels.index(self.channel)]
        ax2 = self.fig.add_subplot(self.gs[0, 2]) 
        num_bins = 256
        x_values = np.linspace(0, self.laser_period_ns, num_bins)
        ax2.plot(x_values, ch_decay_data, color="red")
        ax2.set_title(
            f"Channel {self.channel + 1} - TCSPC",
            weight='bold'
        )
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Photon count")
        ax2.set_xlim(0, self.laser_period_ns)
        ticks = np.linspace(0, self.laser_period_ns, 3)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{tick:.0f}" if tick.is_integer() else f"{tick:.1f}" for tick in ticks])

        return None
    
    def plot_blue_semicirc(self, ax):
        """
        Plot the blue circle
        """
        x = np.linspace(0, 1, 1000)
        y = np.sqrt(0.5**2 - (x - 0.5) ** 2)
        ax.plot(x, y)
        ax.set_aspect("equal")

        return None
    
    def plot_metabolic_line(self, ax):
        ax.plot(
            (self.df.iloc[0]['g_glyco'], self.df.iloc[0]['g_ox']),
            (self.df.iloc[0]['s_glyco'], self.df.iloc[0]['s_ox']),
            marker='o',
            linestyle='-',
            color='red',
            linewidth=2
        )
        ax.text(
            0.255,
            0.436,
            f"OX 3.4 ns",
            fontsize=12,
            ha='left',
            va='bottom',
            fontweight='bold',
        )
        ax.text(
            0.961,
            0.193,
            f"GLY 0.4 ns",
            fontsize=12,
            ha='right',
            va='top',
            fontweight='bold',
        )
        ax.text(
            0.449,
            0.275,
            f"Metabolic trajectory",
            fontsize=10,
            ha='left',
            va='bottom',
            # fontweight='bold',
            color='red',
            rotation=340
        )

        return None
    
    def plot_mean(self, ax, g_values, s_values):
        mean_g = np.mean(g_values)
        mean_s = np.mean(s_values)
        freq_mhz = self.ns_to_mhz(self.laser_period_ns)
        tau_phi = (1 / (2 * np.pi * freq_mhz * self.harmonic)) * (mean_s / mean_g) * 1e3
        tau_m_component = (1 / (mean_s**2 + mean_g**2)) - 1
        tau_m = (
            (
                (1 / (2 * np.pi * freq_mhz * self.harmonic))
                * np.sqrt(tau_m_component) * 1e3
            )
            if tau_m_component >= 0
            else None
        )  
        mean_label = f"G (mean): {round(mean_g, 2)}; S (mean): {round(mean_s, 2)}; τϕ={round(tau_phi, 2)} ns"  
        if tau_m is not None:
            mean_label += f"; τm={round(tau_m, 2)} ns"
                        
        ax.scatter(        
            mean_g,
            mean_s,
            color="yellow",
            marker="x",
            s=100,
            zorder=3,
            label=mean_label,
        )

        return None
    
    def plot_phasors(self):
        # Phasors 
        ax3 = self.fig.add_subplot(self.gs[0, 1])

        self.plot_blue_semicirc(ax3)
        self.plot_metabolic_line(ax3)

        """
        Plot the intersections coloring by the value of tau phi
        """
        sc = ax3.scatter(
            self.df['x_int'],
            self.df['y_int'],
            # label=f"Metabolic intersections",
            zorder=2,
            # color="orange",
            c=self.df['tau_phi'],
            cmap="gist_rainbow",
            s=10,
            vmin=0.4,
            vmax=3.4,
            # alpha=0.1
        )

        """
        We could get them from self.df, but it's the same and like this we are sure 
        they are not filtered
        """
        # g_data = np.array(self.phasors_data["g_data"]).ravel()
        # s_data = np.array(self.phasors_data["s_data"]).ravel()
        # mask = (np.abs(g_data) < 1e9) & (np.abs(s_data) < 1e9) & ~((g_data == 0) & (s_data == 0))
        # g_values = g_data[mask]
        # s_values = s_data[mask]
        g_data = self.df["g_data"]
        s_data = self.df["s_data"]
        ch_image_data = self.image_data[self.enabled_channels.index(self.channel)]
        bins = 100
        ch_image_data = ch_image_data.ravel()
        # intensity = ch_image_data[mask]
        intensity = ch_image_data
        # sc =ax3.scatter(
        #     g_data,
        #     s_data,
        #     # label=f"Harmonic: {self.harmonic}",
        #     zorder=2,
        #     c=intensity,
        #     cmap="grey",
        #     s=10
        # )
        # cbar = plt.colorbar(sc, ax=ax3)
        # cbar.ax.yaxis.set_label_position("left")
        # cbar.set_label(
        #     "Photon Count",
        # )

        x_bins = np.linspace(np.min(g_data), np.max(g_data), bins)
        y_bins = np.linspace(np.min(s_data), np.max(s_data), bins)
        x_bins = np.linspace(0, 1, bins)
        y_bins = np.linspace(0, 0.6, bins)
        counts, x_edges, y_edges = np.histogram2d(g_data, s_data, bins=[x_bins, y_bins])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        array_min = np.min(counts)
        array_max = np.max(counts)
        factor = (counts - array_min) / (array_max - array_min)
        intensity = factor + 0.2 * (1 - factor)
        x = x_centers.repeat(bins - 1)
        y = np.tile(y_centers, bins - 1)
        counts = counts.flatten()
        intensity = intensity.flatten()
        x = x[counts > 0]
        y = y[counts > 0]
        intensity = intensity[counts > 0]
        counts = counts[counts > 0]
        sc = plt.scatter(
            x,
            y,
            c=counts,
            cmap='jet',
            # alpha=intensity,
            marker='s',
            s=50
        )

        # df = self.df[self.df['g_data'].notna() & self.df['s_data'].notna()]
        # x = df['g_data'].to_numpy()
        # y = df['s_data'].to_numpy()
        # x = x[~np.isnan(x)]
        # y = y[~np.isnan(y)]
        # plt.hist2d(x, y, bins=50, cmap='Blues')

        cbar = plt.colorbar(sc, ax=ax3)
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(
            "Pixel Count",
            fontsize=12,
            weight='bold',
            labelpad=0
        )

        self.plot_mean(ax3, g_data, s_data)

        ax3.legend(fontsize="small")
        # ax3.set_title(f"Phasor - Channel {self.channel + 1} - Harmonic {self.harmonic}")
        ax3.set_title(
            f"FLIM Phasor Plot",
            weight='bold'
        )
        ax3.set_xlabel("G")
        ax3.set_ylabel("S")
        ax3.grid(True)

        return None
    
    def plot_tau_image(self):
        ax5 = self.fig.add_subplot(self.gs[0, 2])

        ch_image_data = self.image_data[self.enabled_channels.index(self.channel)]
        self.add_black_background(ax5, ch_image_data)
        
        data_masked, intensity = self.get_2d_data_masked_and_intensity(
            self.df['tau_phi'],
            ch_image_data
        )

        cmap = matplotlib.colormaps.get_cmap("gist_rainbow").copy()
        cmap.set_bad(color="black")
        im = ax5.pcolormesh(
            data_masked,
            cmap=cmap,
            # shading='auto',
            vmin=0.4,
            vmax=3.4,
            alpha=intensity
        )

        # ax5.axis('off')
        ax5.invert_yaxis()
        ax5.set_xlabel("x (pixels)", labelpad=-10)
        ax5.set_ylabel("y (pixels)", labelpad=-10)
        ax5.set_xticks([1, data_masked.shape[1]])
        ax5.set_xticklabels(["1", f"{data_masked.shape[1]}"])
        ax5.set_yticks([1, data_masked.shape[0]])
        ax5.set_yticklabels(["1", str(data_masked.shape[0])]) 
        ax5.set_aspect('equal')
        ax5.set_title(
            f"FLIM",
            weight='bold'
        )
        cbar = plt.colorbar(im, ax=ax5)
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(
            "τφ (ns)",
            fontsize=12,
            weight='bold',
            labelpad=0
        )
        cbar.set_ticks([0.4, 3.4])
        cbar.set_ticklabels(['0.4', '3.4'])

        return None
    
    def det(self, a, b):

        return a[0] * b[1] - a[1] * b[0]
    
    #### line 1 and line 2 are in the form ((0, 0), (1, 1))
    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        div = self.det(xdiff, ydiff)
        # if div == 0:
        #     raise Exception('Division by zero: lines do not intersect')

        d = (self.det(*line1), self.det(*line2))
        x = self.det(d, xdiff) / div
        y = self.det(d, ydiff) / div

        x = np.where((x > 1) | (x < 0), np.nan, x)
        y = np.where((y > 0.5) | (y < 0), np.nan, y)
        
        return x, y
    
    def add_metabolic_index_distance(self):
        """
        Here self.harmonic was taken from a df column to perform the calculation
        across multiple harmonics. We keep it like this for the time being.
        """
        freq_mhz = self.ns_to_mhz(self.laser_period_ns) * 1_000_000
        self.df['k_met'] = 1 / (2 * np.pi * (self.harmonic) * freq_mhz)
        for type in ['ox', 'glyco']:
            self.df[f"phi_{type}"] = np.arctan2(self.taus['phase'][type], self.df['k_met'])
            self.df[f"m_{type}"] = np.sqrt(1 / (1 + ((self.taus['modulation'][type] / self.df['k_met'])**2)))
            self.df[f"g_{type}"] = self.df[f"m_{type}"] * np.cos(self.df[f"phi_{type}"])
            self.df[f"s_{type}"] = self.df[f"m_{type}"] * np.sin(self.df[f"phi_{type}"])

        # print('Metabolic calculation of the centroid:')
        # print(f"Oxidative state:")
        # print(f"Coordinates (G, S): ({self.df.iloc[0][f"g_ox"]}, {self.df.iloc[0][f"s_ox"]})")
        # print(f"Glycolytic State:")
        # print(f"Coordinates (G, S): ({self.df.iloc[0][f"g_glyco"]}, {self.df.iloc[0][f"s_glyco"]})")

        taus_distance = np.sqrt(
            (self.df['g_glyco'] - self.df['g_ox'])**2
            + (self.df['s_glyco'] - self.df['s_ox'])**2
        )
        
        self.df['x_int'], self.df['y_int'] = self.line_intersection(
            ((self.df['g_glyco'], self.df['s_glyco']), (self.df['g_ox'], self.df['s_ox'])),
            ((0, 0), (self.df["g_data"], self.df["s_data"]))
        )

        self.df['metabolic_ratio'] = np.sqrt(
            (self.df['g_glyco'] - self.df['x_int'])**2
            + (self.df['s_glyco'] - self.df['y_int'])**2
        ) / taus_distance * 100

        self.df['metabolic_ratio'] = np.where(
            (self.df['metabolic_ratio'] < 100) | (np.isnan(self.df['metabolic_ratio'])),
            self.df['metabolic_ratio'],
            100
        )

        self.df['metabolic_ratio'] = np.where(
            (self.df['metabolic_ratio'] > 0) | (np.isnan(self.df['metabolic_ratio'])),
            self.df['metabolic_ratio'],
            0
        )

        return None
    
    def add_metabolic_index_single(self, g_data, s_data):
        freq_mhz = self.ns_to_mhz(self.laser_period_ns) * 1_000_000
        k_met = 1 / (2 * np.pi * (self.harmonic) * freq_mhz)

        phi_ox = np.arctan2(self.taus['phase']['ox'], k_met)
        m_ox = np.sqrt(1 / (1 + ((self.taus['modulation']['ox'] / k_met)**2)))
        g_ox = m_ox * np.cos(phi_ox)
        s_ox = m_ox * np.sin(phi_ox)

        phi_glyco = np.arctan2(self.taus['phase']['glyco'], k_met)
        m_glyco = np.sqrt(1 / (1 + ((self.taus['modulation']['glyco'] / k_met)**2)))
        g_glyco = m_glyco * np.cos(phi_glyco)
        s_glyco = m_glyco * np.sin(phi_glyco)

        # print('Metabolic calculation of the extremes:')
        # print(f"Oxidative state:")
        # print(f"Coordinates (G, S): ({g_ox}, {s_ox})")
        # print(f"Glycolytic State:")
        # print(f"Coordinates (G, S): ({g_glyco}, {s_glyco})")

        taus_distance = np.sqrt((g_glyco - g_ox)**2 + (s_glyco - s_ox)**2)
        x_int, y_int = self.line_intersection(
            ((g_glyco, s_glyco), (g_ox, s_ox)),
            ((0, 0), (g_data, s_data))
        )
        metabolic_ratio = np.sqrt(
            (g_glyco - x_int)**2
            + (s_glyco - y_int)**2
        ) / taus_distance * 100

        return metabolic_ratio
    
    def add_tau_phi(self):
        freq_mhz = self.ns_to_mhz(self.laser_period_ns)
        self.df['tau_phi'] = (1 / (2 * np.pi * freq_mhz * self.harmonic)) * (self.df['s_data'] / self.df['g_data']) * 1e3

        # self.df['tau_phi'][self.df['tau_phi'] > 3.4] = 3.4
        # self.df['tau_phi'][self.df['tau_phi'] < 0.4] = 0.4
        self.df.loc[self.df['tau_phi'] > 3.4, 'tau_phi'] = 3.4
        self.df.loc[self.df['tau_phi'] < 0.4, 'tau_phi'] = 0.4

        return None
    
    def filter_df(self):
        self.df['g_data'] = np.where(
            self.df['image_data'] >= self.threshold,
            self.df['g_data'],
            np.nan
        )
        self.df['s_data'] = np.where(
            self.df['image_data'] >= self.threshold,
            self.df['s_data'],
            np.nan
        )

        for iteration in range(self.median_filter_iterations):
            self.df["g_data"] = scipy.ndimage.median_filter(
                self.df["g_data"],
                size=self.median_filter_window,
                mode="nearest"
            )
            self.df["s_data"] = scipy.ndimage.median_filter(
                self.df["s_data"],
                size=self.median_filter_window,
                mode="nearest"
            )

        # c1 = self.df['image_data_filtered'] >= 0
        # c2 = self.df['g_data_median_filter'] >= 0
        # c3 = self.df['s_data_median_filter'] >= 0
        # self.df['data_kept'] = True

        return None
    
    def add_metabolic_cbar(self, sc, ax):
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(
            "Metabolic Index (%)",
            fontsize=12,
            weight='bold',
            labelpad=0
        )
        cbar.ax.text(
            0.5,
            1.01,
            "OX",
            ha='center',
            va='bottom',
            transform=cbar.ax.transAxes,
            fontsize=12,
            fontweight='bold'
        )
        cbar.ax.text(
            0.5,
            -0.01,
            "GLY",
            ha='center',
            va='top',
            transform=cbar.ax.transAxes,
            fontsize=12,
            fontweight='bold'
        )

        return None
    
    def add_subplot_metabolic_phasors(self):
        ax4 = self.fig.add_subplot(self.gs[1, 1]) 

        self.plot_blue_semicirc(ax4)
        self.plot_metabolic_line(ax4)

        # nan_mask_x = np.isnan(self.df['x_int'])
        # nan_mask_y = np.isnan(self.df['y_int'])
        # x_int_masked = np.ma.masked_array(self.df['x_int'], mask=nan_mask_x)
        # x_int_masked = np.ma.masked_array(x_int_masked, mask=nan_mask_y)
        # y_int_masked = np.ma.masked_array(self.df['y_int'], mask=nan_mask_x)
        # y_int_masked = np.ma.masked_array(y_int_masked, mask=nan_mask_y)
        # g_masked = np.ma.masked_array(self.df[self.df['data_kept']]['g_data'], mask=nan_mask_x)
        # g_masked = np.ma.masked_array(g_masked, mask=nan_mask_y)
        # s_masked = np.ma.masked_array(self.df[self.df['data_kept']]['s_data'], mask=nan_mask_x)
        # s_masked = np.ma.masked_array(s_masked, mask=nan_mask_y)
        # metabolic_ratio_masked = np.ma.masked_array(self.df[self.df['data_kept']]['metabolic_ratio'], mask=nan_mask_x)
        # metabolic_ratio_masked = np.ma.masked_array(metabolic_ratio_masked, mask=nan_mask_y)

        x_int = self.df['x_int']
        y_int = self.df['y_int']
        g_data = self.df['g_data']
        s_data = self.df['s_data']
        metabolic_ratio = self.df['metabolic_ratio']
        # mask = (np.abs(g_data) < 1e9) | (np.abs(s_data) < 1e9)
        # g_values = g_data[mask]
        # s_values = s_data[mask]
        # g_values = np.where((g_data > 0) & (s_data > 0), g_data, np.nan)
        # s_values = np.where((g_data > 0) & (s_data > 0), s_data, np.nan)
        
        """
        Plot the intersections coloring by the values of the metabolic ratio
        """
        sc = ax4.scatter(
            x_int,
            y_int,
            zorder=2,
            c=metabolic_ratio,
            cmap=self.metabolic_color_map,
            s=10,
            vmin=0
            # alpha=0.1
        )

        sc = ax4.scatter(
            g_data,
            s_data,
            # label=f"Harmonic: {self.harmonic}",
            zorder=2,
            c=metabolic_ratio,
            cmap=self.metabolic_color_map,
            s=10,
            vmin=0,
            vmax=100
            # alpha=0.1
        )
        self.add_metabolic_cbar(sc, ax4)

        """
        Calculation of the mean
        """
        mean_g = np.mean(g_data)
        mean_s = np.mean(s_data)
        self.metabolic_index_mean = self.add_metabolic_index_single(mean_g, mean_s)

        self.plot_mean(
            ax4,
            g_data,
            s_data
        )
        
        ax4.legend(fontsize="small")
        # ax4.set_title(f"Phasor - Channel {self.channel + 1} - Harmonic {self.harmonic}")
        ax4.set_title(
            f"FLIM Metabolic Phasor Plot",
            weight='bold'
        )
        ax4.set_xlabel("G")
        ax4.set_ylabel("S")
        ax4.grid(True)

        return None
    
    def add_subplot_metabolic_text(self):
        ax6 = self.fig.add_subplot(self.gs[1, 2])
        ax6.text(
            0.5,
            0.5,
            # f"Metabolic index\nof the centroid:\n{round(self.metabolic_index_mean, 2)} % OX",
            f"Average\nMetabolic Index\n{round(self.metabolic_index_mean, 2)} % OX",
            fontsize=32,
            ha='center',
            va='center',
            fontweight='bold',
        )
        ax6.axis('off')

        return None
    
    def add_black_background(self, ax, image_data):
        mask_black = np.full_like(image_data, np.nan, dtype=float)
        mask_black = np.ma.masked_invalid(mask_black)
        cmap = matplotlib.colormaps.get_cmap("gray").copy()
        cmap.set_bad(color="black")
        im = ax.pcolormesh(
            mask_black,
            cmap=cmap,
            # shading='auto',
            vmin=0
        )

        return None
    
    def get_2d_data_masked_and_intensity(self, data, image):
        """
        np.ma.masked_array and np.ma.masked_invalid would be equivalent for 
        metabolic_values, but we need anyway nan_mask because we need to filter
        image_intensity. The output must be a 1D array for some reason
        """
        # ch_image_data = np.where(ch_image_data < self.threshold, array_max, ch_image_data)
        data_2d = np.reshape(
            data,
            image.shape
        )
        nan_mask = np.isnan(data_2d)
        data_masked = np.ma.masked_array(data_2d, mask=nan_mask)
        array_min = np.min(image)
        array_max = np.max(image)
        image_normalized = (image - array_min) / (array_max - array_min)
        #### Maybe it is not necessary to do this
        intensity = np.ma.masked_array(image_normalized, mask=nan_mask)

        return data_masked, intensity
    
    def add_subplot_metabolic_image(self):
        ax5 = self.fig.add_subplot(self.gs[1, 0])

        ch_image_data = self.image_data[self.enabled_channels.index(self.channel)]
        self.add_black_background(ax5, ch_image_data)
        
        data_masked, intensity = self.get_2d_data_masked_and_intensity(
            self.df['metabolic_ratio'],
            ch_image_data
        )

        cmap = matplotlib.colormaps.get_cmap(self.metabolic_color_map).copy()
        cmap.set_bad(color="black")
        im = ax5.pcolormesh(
            data_masked,
            cmap=cmap,
            # shading='auto',
            vmin=0,
            vmax=100,
            alpha=intensity
        )

        self.add_metabolic_cbar(im, ax5)

        # ax5.axis('off')
        ax5.invert_yaxis()
        ax5.set_xlabel("x (pixels)", labelpad=-10)
        ax5.set_ylabel("y (pixels)", labelpad=-10)
        ax5.set_xticks([1, data_masked.shape[1]])
        ax5.set_xticklabels(["1", f"{data_masked.shape[1]}"])
        ax5.set_yticks([1, data_masked.shape[0]])
        ax5.set_yticklabels(["1", str(data_masked.shape[0])]) 
        ax5.set_aspect('equal')
        ax5.set_title(
            f"FLIM Metabolic Index",
            weight='bold'
        )

        return None

    """
    This is the full plot with all the subplots
    """
    def plot_data(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(
            2, # rows
            3, # columns
            width_ratios=[2, 3, 2],
            height_ratios=[1, 1]
        )

        self.plot_imaging()
        # self.plot_tcspc()
        self.plot_phasors()
        self.plot_tau_image()

        self.add_subplot_metabolic_phasors()
        self.add_subplot_metabolic_text()
        self.add_subplot_metabolic_image()
                
        # plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
        plt.tight_layout(
            # pad=0.1,
            w_pad=0.1,
            h_pad=0.1
        )
        plt.subplots_adjust(
            hspace=0.1,
            wspace=0.1
        )
        plt.show()

    def update_phasors_file_to_read(self):
        self.phasors_path = self.phasors_file_path.replace("_h1", f"_h{self.harmonic}")
        self.phasors_path = self.phasors_path.replace("_ch1", f"_ch{self.channel + 1}")

        return None    

    def ns_to_mhz(self, laser_period_ns):
        period_s = laser_period_ns * 1e-9
        frequency_hz = 1 / period_s
        frequency_mhz = frequency_hz / 1e6

        return frequency_mhz

    def analyze_imaging_and_phasors_files(self):
        self.update_phasors_file_to_read()
        self.read_phasors_file()
        self.read_phasors_metadata()

        self.read_imaging_file()
        self.read_image_data()

        self.filter_df()
        self.add_tau_phi()
        self.add_metabolic_index_distance()
        print(self.df)

        self.plot_data()

        return None

if __name__ == "__main__":
    metabolic_phasors = MetabolicPhasors()
    metabolic_phasors.analyze_imaging_and_phasors_files()