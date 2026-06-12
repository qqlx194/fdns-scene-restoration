import os


def _env_path(name, default):
    return os.getenv(name, default)


gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")

lr = 10 ** (-1.25)
iters = 1500
eps = 0.20
beta = 0.5
gamma = 1e-5

sample_num_images = 0
sample_subset_seed = None

image_index_list = []
image_start_index = 0
image_end_index = -1

experiment_name = os.getenv("EXPERIMENT_NAME", "fdns_scene_restoration")
exp_tag = os.getenv("EXPERIMENT_TAG", "main")

use_paper_rspg_loss = True
alpha = 1.0
Y = 1e-4

psf = 2
complexity_threshold = 0.85
texture_descriptor = "lbp"
texture_entropy_threshold_lbp = 4.0

cover_dataset_dir = _env_path("COVER_DATASET_DIR", os.path.join("data", "cover"))
cover_image_size = 512
secret_dataset_dir = _env_path("SECRET_DATASET_DIR", os.path.join("data", "secret"))
secret_image_size = 128
secret_mask_dataset_dir = _env_path("SECRET_MASK_DATASET_DIR", os.path.join("data", "secret_mask"))

use_grad_signals_in_steganalysis_nets = True
steganalysis_last_n_iters = 100
steganalysis_nets = ["srnet", "siastegnet"]
steganalysis_beta = 0.5
steganalysis_gamma = 2e-5

pre_trained_siastegnet_path = _env_path(
    "SIASTEGNET_CKPT",
    os.path.join("checkpoints", "siastegnet", "model_best.pth.tar"),
)
pre_trained_srnet_path = _env_path(
    "SRNET_CKPT",
    os.path.join("checkpoints", "srnet", "net.pt"),
)
pre_trained_yenet_path = _env_path(
    "YENET_CKPT",
    os.path.join("checkpoints", "yenet", "net.pt"),
)

secret_postprocess = "dncnn"
dncnn_ckpt = _env_path("DNCNN_CKPT", os.path.join("models", "dncnn_color_blind.pth"))

scale_factor = 0.95
max_angle = 0.1
contrast_factor = 0.9
attack_layer = "jpeg"
qf = [90, 80, 70, 60]
add_jpeg_layer = False
salt_prob = pepper_prob = 0.01
brightness_factor = 1.05

num_of_receivers = 2
num_of_secret_imgs = 7

gaussian_std = [0.0001, 0.0002, 0.0003, 0.0004]
sp_density = [0.0005, 0.0010, 0.0020, 0.0030]

robust_jpeg_q_list = [90, 80, 70, 60]
robust_gaussian_std_list = [0.0001, 0.0002, 0.0003, 0.0004]
robust_sp_density_list = [0.0005, 0.0010, 0.0020, 0.0030]

save_images = True
resi_magnification = 10

use_pre_decoder_restorer = False
reset_resume_state = True

pert_restorer_train_online = True
pert_restorer_steps = 200
pert_restorer_lr = 1e-3
pert_restorer_attack_types = ["jpeg", "gaussian", "s&p"]
pert_restorer_q_list = robust_jpeg_q_list
pert_restorer_gaussian_std_list = robust_gaussian_std_list
pert_restorer_sp_density_list = robust_sp_density_list

use_l3_attack = True
l3_attack_types = ["jpeg", "gaussian", "s&p"]
l3_start_ratio = 0.0
l3_prob = 1.0
l3_weight_max = 1
l3_num_attacks_per_iter = 1
l3_attack_agg = "max"

w_rev = 1
hid_warmup_ratio = 0.50
w_hid_base = 0.25
w_hid_max = 1.80
w_tv = 0.03
