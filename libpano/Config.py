debug = True

frame_margin = 60

# vertical angle of ROI
max_vertical_angle = 1.483  # -85 ~ 85 degree vertically

# minimal pitch for interpolation
min_pitch_needs_interpolation = 0.75

# internal panorama size
internal_panorama_width = 4200
internal_panorama_height = 2100

# mega pixel for seam finding
smp = 1e5

# the order of stitching pipeline
order_warp_first = True

# file names
meta_data_name = 'metadata.txt'
register_result_name = 'framedata'
compose_config_name = 'compose-config'

# stitch only middle frames
only_main_frame = False

# stitch main frame first and then others
mainframe_first = True


