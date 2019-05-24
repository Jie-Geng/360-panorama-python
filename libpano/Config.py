debug = True

frame_margin = 60

# vertical angle of ROI
max_vertical_angle = 1.483  # -85 ~ 85 degree vertically

# minimal pitch for interpolation
min_pitch_needs_interpolation = 0.75

# internal panorama size
internal_panorama_width = 4096
internal_panorama_height = 2048

# mega pixel for seam finding
smp = 2e6

# the order of stitching pipeline
order_warp_first = True

# file names
meta_data_name = 'metadata.txt'
prestitch_result_name = 'framedata'
poststitch_input_name = 'framedata.all'

only_main_frame = False
