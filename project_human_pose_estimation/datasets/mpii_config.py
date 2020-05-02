worldCoors = "data/mpii/mpii_head_worldcoor/worldCoods.mat"
headSize = "data/mpii/mpii_head_worldcoor/headsize.mat"

datadir = "data/mpii/"
annot_dir = datadir + "annot/"
image_dir = datadir + "images/"

max_scale = .25
max_rotate = 30
max_translate = .02

img_res = 280
input_res = 256
output_res = 256

n_joints = 16

gauss = 1

target_type = 'direct'
# target_type = 'heatmap'
