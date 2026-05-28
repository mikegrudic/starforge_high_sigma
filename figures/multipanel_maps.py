"""Makes a multi-panel timelapse map of the simulation"""

from starforge_tools.plots.multipanel import multipanel_timelapse_map
from matplotlib import pyplot as plt

simulation_path = "/mnt/ceph/users/starforge/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1"
multipanel_timelapse_map(
    simulation_path, cmap_limits={"SurfaceDensity": [1e2, 1e5]}, res=1024, slice_thickness=4, supersample=2
)
