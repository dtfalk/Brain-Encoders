"""
Script to train pixel-space class-conditional DDPM using UNet2DModel on the EMNIST dataset
- Author: David Falk
- Organization: APEX Laboratory at The University of Chicago
- Date: 2/11/2026
==================================
Model Training Features:

1. Balanced split (equal samples per class since given dataset class sizes are heavily skewed towars digits)
2. Cosine beta schedule (better fine details)
3. EMA weights (smoother, more stable)
4. CFG dropout (enables guidance at generation time)

Important Notes:
1. I merged letters whose uppercase/lowercase forms are identical/highly-similar (e.g. 's', 'c')
    so that, intuitively speaking, the model doesn't try to learn/internalize some arbitrary difference
    between 's' and 'S'. For inference I have made it so that you can enter either 's' or 'S' and it 
    will cast to the correct class label.
"""
# =====================================================
# IMPORTS (check conda yml for versions)
# =====================================================
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler

# Metrics & logging (factored out — see metrics/ and pretty_logger.py)
from metrics import MetricsTracker, generate_all_plots, generate_report
from metrics.tracker import compute_grad_norm
from pretty_logger import PrettyLogger

# =====================================================
# VARIATION CONFIG
# =====================================================

# What to name the model (relevant for save directory)
MODEL_NAME = "balanced_cfg_cosine_ema_600_steps" 

# Choice of model size from SIZES dict found below 
# Controls image resolution, batch size, and channel size/shape
SIZE_NAME = "large"

# Equalizes equal number of samples in each class
# Equal classes helps ensure that no class dominates the loss function
DATASET_SPLIT = "balanced" 

# Number of training epochs (full passes through the dataset)
EPOCHS = 100

# Learning Rate (size of gradient update)
LEARNING_RATE = 1e-4

# Number of diffusion timesteps
# Gaussian noise is iteratively added to original image over {STEPS} timesteps
STEPS = 600

# Controls how noise increases over timesteps
# Each injection of gaussian noise has variance Beta_t
# This controls how Beta_t grows as t grows
BETA_SCHEDULE = "squaredcos_cap_v2"

# Helps smooth weights updates during training
# Instead of doing x, this does y
EMA_DECAY = 0.999

# Enables classifier free gudiance (cfg)
# {CFG_DROPOUT_PROB * 100} percent of the time a class label is replaced with a null label
# Allows the model to run inference (i.e. predict noise) both with and without conditioning
CFG_DROPOUT_PROB = 0.10

# =====================================================
# PATHS (self-contained — everything lives inside the project root)
# =====================================================

# Project root is one level above this script's directory (emnist-ddpm/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where to save the trained model/checkpoints
CACHE_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")

# Where the EMNIST dataset is stored (downloaded automatically on first run)
EMNIST_DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# =====================================================
# SIZE PRESETS
# =====================================================

# Each entry expands into:
#   (img, batch_size, block_out_channels)
#
# img:
#   Final training image resolution (images resized to img x img).
#
# batch_size:
#   Number of images processed per training step.
#
# block_out_channels:
#   Tuple defining the number of feature channels at each U Net resolution stage.
#   Each number corresponds to one downsampling stage in the U Net.
#   Each value in the 4-tuple is number of feature maps for that convolution block
#
#   Stage 1:
#       Resolution: img x img
#       Channels: block_out_channels[0]
#       Learns low-level features (edges, strokes, perfectly aligned with pixel-level structure).
#
#   Stage 2:
#       Resolution: img/2 x img/2
#       Channels: block_out_channels[1]
#       Learns mid-level features (partial shapes).
#
#   Stage 3:
#       Resolution: img/4 x img/4
#       Channels: block_out_channels[2]
#       Learns higher-level structure (whole character components).
#
#   Stage 4:
#       Resolution: img/8 x img/8
#       Channels: block_out_channels[3]
#       Bottleneck layer with global context for noise prediction.
#   
# Simplified Explanation For Feature Maps (block_out_channels):
#
# At each stage of the U Net, the model creates multiple "feature maps".
# Think of each feature map as shining a different "pattern revealer" laser
# beam across the image.
#
# Each beam slides over the entire image and lights up wherever its learned
# pattern appears.
#
# The brightness at each pixel answers:
#     "How strongly does this specific pattern exist here?"
#
# The number in block_out_channels tells you how many different pattern
# revealers (feature maps) exist at that stage.
#
# Example:
#     64  →  64 different learned pattern detectors
#     256 →  256 different learned pattern detectors
#
# As the U Net goes deeper:
#     - The image resolution becomes smaller
#     - The number of feature maps increases
#
# Early stages detect simple local patterns (edges, strokes).
# Deeper stages detect more abstract structure (shapes, global layout).
#
# Larger numbers here mean:
#     - More pattern detectors
#     - Greater representational capacity
#     - More parameters and memory usage

# Note:
#   As spatial resolution decreases, channel count increases.
#   This preserves model capacity while controlling memory usage.

SIZES = {
    "small":  (28,  2048, (64, 128, 256)),
    "medium": (64,   384, (64, 128, 256, 256)),
    "large":  (96,   96, (96, 192, 384, 384)),
    "xl":     (128,  32, (128, 256, 512, 512)),
}

# =====================================================
# DISTRIBUTED TRAINING SETUP
# =====================================================

# Global rank of this process.
# In multi-GPU / multi-node training, each process gets a unique RANK.
# RANK is used to coordinate communication between processes.
# RANK 0 is considered the main process.
# All other ranks will run their own version of this script, but there is logic
# within the imported packages and within our script such that there are some things
# that only the process with rank 0 will do (e.g. logging)
rank = int(os.environ.get("RANK", "0"))

# Local rank of this process on the current machine.
# If a machine has multiple GPUs, LOCAL_RANK tells this process
# which GPU it should use.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# Assign this process to its designated GPU.
# Ensures each distributed worker uses a different GPU.
torch.cuda.set_device(local_rank)

# Training device (GPU only currently).
device = "cuda"

# Initialize distributed communication backend so the processes can communicate.
# "nccl" is optimized for GPU-to-GPU communication.
# This enables gradient synchronization across processes.
torch.distributed.init_process_group("nccl")


# =====================================================
# LOAD SIZE PRESET
# =====================================================

# Unpack selected preset into:
#   img   --> image resolution
#   batch --> batch size per process
#   chans --> U Net channel widths
img, batch, chans = SIZES[SIZE_NAME]


# =====================================================
# OUTPUT DIRECTORY
# =====================================================

# Construct output directory path:
# checkpoints / SIZE_NAME
# Organized by model size for easy export.
OUT_DIR = os.path.join(CACHE_ROOT, SIZE_NAME)

# Create directory if it does not already exist.
os.makedirs(OUT_DIR, exist_ok = True)

# =====================================================
# METRICS & LOGGER SETUP (rank 0 only writes)
# =====================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics", "output", SIZE_NAME)
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if rank == 0:
    tracker = MetricsTracker(output_dir=METRICS_DIR, model_name=MODEL_NAME)
    logger = PrettyLogger(os.path.join(LOG_DIR, f"train_{SIZE_NAME}.log"))
    logger.start()
    logger.log("Initializing training run...", severity="ok")
else:
    tracker = None
    logger = None

def log(msg, severity="info"):
    """Convenience: log from rank 0 only."""
    if rank == 0 and logger is not None:
        logger.log(msg, severity=severity)


# =====================================================
# DATA LOADING AND PREPROCESSING
# =====================================================

# Log training run metadata from rank 0
log(f"Model: {MODEL_NAME} | Size: {SIZE_NAME} | Img: {img}x{img}", "ok")
log(f"Split: {DATASET_SPLIT} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}", "ok")
log(f"Schedule: {BETA_SCHEDULE} | EMA: {EMA_DECAY} | CFG Drop: {CFG_DROPOUT_PROB}", "ok")

# Builds the image transformation pipeline using the torchvision.transforms package
# As implied by its name, a pipeline is just a set of composable, modular steps that are glued together
# In this case we ".Compose" (glue together) the following image transformations into a single object (the "image_transformation_pipeline" variable)
# We will apply this deterministic set of operations to every image in the dataset.
# We will load EMNIST dataset images as a PIL image (greyscale) and run "image_transformation_pipeline(pil_image)" 
# For each image, the pipeline does the following:
#   1. Resize the EMNSIT image (28 x 28) to target resolution (img x img)   
#   2. Convert the image to a PyTorch Tensor. This changes integer image values into float32 values in the interval [0, 1] (more detail in Training-Explainer file)
#   3. Rotates the image 90 degrees and flips along height axis so that images are upright (EMNIST images are stored rotated and flipped so we undo this)
#   4. Scale the tensors from the interval [0, 1] to the interval [-1 , 1] using f(x) = 2x - 1 (center pixel values around 0)
image_transformation_pipeline = transforms.Compose([
    transforms.Resize((img, img)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flip(torch.rot90(x, 1, [1, 2]), [1])),
    transforms.Lambda(lambda x: x * 2 - 1),
])

# Loads the EMNIST image dataset as a torchvision dataset object
# Basically just a wrapper around the EMNIST files that makes them act like a PyTorch dataset
# There are two internal methods that are of particular interest:
#   1. __len__() --> Returns how many samples exist 
#   2. __getitem__(index) --> Returns a single (image, label) tuple where label is the class name
# __getitem__ will be called repeatedly during the training run. 
dataset = datasets.EMNIST(
    root      = EMNIST_DATA_PATH,               # Path to EMNIST files
    split     = DATASET_SPLIT,                  # Choose the dataset split (EMNIST allows "balanced", "byclass", "letters", "digits", etc...)
    train     = True,                           # EMNIST has both training and testing partions. "train = True" means use the training partition
    download  = False,                          # Controls whether TorchVision should download the dataset if it does not exist
    transform = image_transformation_pipeline,  # Applies the image transformation pipeline (not upfront, runs on the fly)
)

# Initializes the distributed sampler
# Loosely, the sampler decides "In what order should the dataset indices be read?"
# Normally, PyTorch either reads samples sequentially or picks randomly (~ shuffle and then read sequentially).
# However, since this is a distributed training setup, we have multiple processes running simultaneously and we 
# need to make sure that we do not have each GPU read the whole dataset/do duplicate training/mess up gradients
# DistributedSampler fixes this by splitting the indices across processes.
# (Also keeps the shuffling synchronized across processes when we call "sampler.set_epoch" later)
sampler = DistributedSampler(dataset)

# Wraps the dataset object in a DataLoader object.
# Think of DataLoader as a batching and parrallel loading engine.
# It does not change the data in any way.
# It just controls how the data is delivered to the training loop
data_loader = DataLoader(
    dataset     = dataset, # Provides our dataset object
    batch_size  = batch,   # Number of samples per training step per process
    sampler     = sampler, # Provides our distributed sampler
    num_workers = 8,       # Number of cpu subprocesses used to load & preprocess data in parallel
    pin_memory  = True,    # GPU training optimization (locks CPU memory pages, makes transfer to GPU faster)
    drop_last   = True,    # If dataset is not divisible by batch size, then drop the final, smaller batch 
)                          # (It can be a pain if different GPUs try synchronizing gradients when they have different batch sizes)

# Gets number of classes for the given dataset split
# Valid class labels are ints [0, 1, ..., num_classes - 1]
# We add a label, num_classes, so there is a cfg dropout class/label
num_classes = len(dataset.classes)
NULL_CLASS = num_classes

# Log class information
log(f"Classes: {num_classes} | Samples: {len(dataset)} | Null class: {NULL_CLASS}", "ok")

# =====================================================
# MODEL + EMA
# =====================================================

log("Building UNet...", "info")

# -----------------------------------------------------
# UNet2DModel
# -----------------------------------------------------
#
# This is the core neural network that learns to predict noise.
#
# In diffusion training, we corrupt a clean image x₀ into xₜ
# by gradually adding Gaussian noise over many timesteps.
#
# The model learns the function:
#
#     ε_pred = f(x_t, t, class_label)
#
# Where:
#     x_t         = noisy image at timestep t
#     t           = diffusion timestep
#     class_label = conditioning signal (character identity)
#     ε_pred      = predicted noise
#
# The training objective minimizes:
#
#     MSE(ε_pred ε)
#
# where ε is the true Gaussian noise used to corrupt the image.
#
# UNet is a special type of convolutional neural network.
#
# It has three main parts:
#
# 1) Downsampling path (the "compress" part)
#    - The image is gradually reduced in spatial size
#    - Example: 64x64 → 32x32 → 16x16 → 8x8
#    - As the image gets smaller, the number of feature channels increases
#    - The model learns more abstract patterns at each step
#
#    You can think of this as:
#        "Zooming out to understand the overall structure"
#
#
# 2) Bottleneck (the smallest representation)
#    - This is the most compressed version of the image
#    - It contains global information about the character
#    - At this point, the model has a high-level understanding
#
#
# 3) Upsampling path (the "rebuild" part)
#    - The network increases the image size step by step
#    - Example: 8x8 → 16x16 → 32x32 → 64x64
#    - It uses what it learned during compression to reconstruct details
#
#
# Skip connections:
# -----------------
# At every resolution level, the network copies features from the
# downsampling path and sends them directly to the matching
# upsampling stage.
#
# Why?
#
# Because when you compress an image, you lose fine detail.
# Skip connections give the network access to the original
# high-resolution information so it can rebuild sharp results.
#
#
# Why is this good for diffusion?
# --------------------------------
#
# In diffusion training, the model must take a noisy image and
# predict the exact noise that was added.
#
# That requires:
#     - Understanding global structure (what character is this?)
#     - Preserving pixel-level precision (where exactly is the noise?)
#
# The UNet architecture is well suited for this because:
#     - The down path captures overall structure
#     - The up path restores spatial detail
#     - Skip connections prevent loss of fine information
unet = UNet2DModel(
    sample_size        = img,  # Spatial resolution the UNet is built for (inputs will be B x 1 x img x img)
    in_channels        = 1,  # Number of channels in the input image (greyscale => 1, RGB => 3)
    out_channels       = 1,  # Number of channels in the output (predict noise; same shape/channels as input)
    layers_per_block   = 2,  # How many conv/resnet layers to run inside each stage before moving to next resolution
    block_out_channels = chans,  # Channel widths at each UNet stage (controls capacity/memory; one entry per resolution level)
    down_block_types   = ("DownBlock2D",) * len(chans),  # Downsampling path blocks (one per stage; shrink spatial size by 2 each stage)
    up_block_types     = ("UpBlock2D",) * len(chans),  # Upsampling path blocks (one per stage; grow spatial size by 2 each stage)
    num_class_embeds   = num_classes + 1,  # Size of class embedding table (num classes + 1 extra NULL class for CFG dropout)
).to(device)  # Move model weights to GPU

# Wraps model for multi-GPU training 
# Each process uses its local GPU; gradients sync across ranks
unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank])


# -----------------------------------------------------------
# EMA model (Exponential Moving Average of training weights)
# -----------------------------------------------------------
# We create a second copy of the UNet that does NOT train directly.
# Instead, after every optimizer step, we update its weights using:
#
#     ema_weight = decay * ema_weight + (1 - decay) * current_weight
#
# So instead of jumping around like normal SGD updates,
# the EMA model changes slowly and smoothly over time.
#
# Why do this?
#
# During training, model weights fluctuate from step to step.
# Some updates improve things, some slightly overshoot.
#
# The EMA version acts like a "smoothed" version of the model
# across many recent training steps.
#
# Important:
# The optimizer updates the main UNet.
# The EMA UNet is updated manually using the moving average rule.


# EMA model
# (matches initialization for unet)
ema_unet = UNet2DModel(
    sample_size        = img,
    in_channels        = 1,
    out_channels       = 1,
    layers_per_block   = 2,
    block_out_channels = chans,
    down_block_types   = ("DownBlock2D",) * len(chans),
    up_block_types     = ("UpBlock2D",) * len(chans),
    num_class_embeds   = num_classes + 1,
).to(device)

# Initialize EMA model with the exact same weights as the training UNet.
# Because unet is wrapped in DistributedDataParallel, the actual model
# lives inside unet.module.
ema_unet.load_state_dict(unet.module.state_dict())

# Put EMA model into evaluation mode.
# We never train this model directly; it behaves like an inference model.
ema_unet.eval()

# Disable gradient tracking for EMA parameters.
# EMA weights are NOT updated via backprop or the optimizer.
# They are updated manually using the exponential moving average rule
# (We dont need this and certainly don't want to require it)
for p in ema_unet.parameters():
    p.requires_grad_(False)


# -----------------------------------------------------------
# Scheduler, Optimizer & Scaler
# -----------------------------------------------------------

# Diffusion scheduler
# Defines the forward diffusion process (adding noise to clean images):
#   - How many timesteps to diffuse over
#   - How noise variance (beta) increases over time
# This object is used to add noise to clean images during training.
scheduler = DDPMScheduler(
    num_train_timesteps = STEPS,          # Total number of diffusion steps (e.g. 600)
    beta_schedule       = BETA_SCHEDULE,  # Controls how noise grows across timesteps (e.g. cosine schedule)
)

# Optimizer
# Updates the training UNet weights using gradients computed from the loss.
# AdamW is commonly used for diffusion models. Not super familiar with other options.
optimizer = torch.optim.AdamW(unet.parameters(), lr = LEARNING_RATE)

# Automatic Mixed Precision (AMP) gradient scaler
# When training in float16, gradients can underflow
# GradScaler dynamically scales the loss to keep training stable
scaler = torch.cuda.amp.GradScaler()

# =====================================================
# TRAINING - The fun part :)
# =====================================================

# Training start
log(f"Starting training — {EPOCHS} epochs on {torch.cuda.device_count()} GPUs", "ok")

# Build config dict for the report
TRAINING_CONFIG = {
    "model_name": MODEL_NAME,
    "size_name": SIZE_NAME,
    "image_resolution": img,
    "batch_size_per_gpu": batch,
    "block_out_channels": list(chans),
    "dataset_split": DATASET_SPLIT,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "diffusion_steps": STEPS,
    "beta_schedule": BETA_SCHEDULE,
    "ema_decay": EMA_DECAY,
    "cfg_dropout_prob": CFG_DROPOUT_PROB,
    "num_classes": num_classes,
    "null_class_index": NULL_CLASS,
    "num_gpus": torch.cuda.device_count(),
    "optimizer": "AdamW",
    "mixed_precision": "float16",
}

# One epoch = one full pass through the entire dataset.
# The dataset is divided into mini-batches.
#
# Each mini-batch contains `batch_size` images.
# One mini-batch → one forward pass → one loss → one optimizer step.
for epoch in range(EPOCHS):

    # Important for DistributedSampler.
    # Ensures each GPU shuffles data differently each epoch
    # but in a synchronized way across processes.
    sampler.set_epoch(epoch)

    # Get timestamp for start of current epoch
    t0 = time.time()

    # Start metrics tracking for this epoch
    if rank == 0 and tracker is not None:
        tracker.start_epoch()

    # Store one scalar loss per mini-batch.
    # This is used only for computing the average loss at the end of the epoch.
    losses = []
    batch_idx = 0
    total_batches = len(data_loader)

    # =================================================
    # MINI-BATCH LOOP
    # =================================================
    #
    # data_loader yields MINI-BATCHES.
    #
    # images_batch:
    #   Tensor of CLEAN images from the dataset.
    #   Shape: (batch_size, 1, img, img)
    #
    #   In diffusion notation, this is x₀
    #   (the original, uncorrupted image).
    #
    # labels_batch:
    #   Tensor of integer class labels corresponding to each image.
    #   Shape: (batch_size,)
    #
    #   Each entry tells us what character the image represents.
    #
    # Example:
    #   If batch_size = 384,
    #   Then x contains 384 clean training images,
    #   and y contains 384 class labels.
    #
    for images_batch, labels_batch in data_loader:

        # Load images and their labels to GPU
        images_batch = images_batch.to(device, non_blocking = True)
        labels_batch = labels_batch.to(device, non_blocking = True)

        # -------------------------------------------------------------------------
        # CFG dropout 
        # -------------------------------------------------------------------------
        
        # Create mask for switching CFG_DROPOUT_PROB percent of labels to Null Class
        # Looks like ... dropout_mask = [False, True, False, False, True, ...]
        dropout_mask = torch.rand(labels_batch.size(0), device = device) < CFG_DROPOUT_PROB

        # Apply the dropout mask so ~CFG_DROPOUT_PROB percent images are now Null Class
        # (Reminder: This is what allows us to do conditional diffusion later if we so desire)
        labels_cfg = torch.where(dropout_mask, torch.full_like(labels_batch, NULL_CLASS), labels_batch)
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # Forward Diffusion (Add Noise to Clean Images)
        # -------------------------------------------------------------------------
        #
        # At this point:
        #
        #   images_batch = x₀  (clean training images)
        #   labels_cfg   = class labels (possibly replaced with NULL_CLASS)
        #
        # Diffusion training works by:
        #
        #   1) Sampling a random timestep t
        #   2) Adding the correct amount of Gaussian noise
        #      corresponding to that timestep
        #
        # Each image in the mini-batch receives its OWN timestep.
        #
        # This ensures the model learns to denoise
        # at all possible corruption levels.
        #

        # -------------------------------------------------------------------------
        # Sample Diffusion Timesteps
        # -------------------------------------------------------------------------
        #
        # Why are we randomly sampling timesteps?
        #
        # In diffusion, the model must learn to denoise images at
        # *every possible noise level*, not just one.
        #
        # If STEPS = 600, then valid timesteps are:
        #   0, 1, 2, ..., 599
        #
        # Small timestep  → image is lightly noised
        # Large timestep  → image is heavily noised
        #
        # During training, we want the model to learn:
        #
        #   "Given (x_t, t), predict the noise that was added."
        #
        # The true objective is an expectation over all timesteps:
        #
        #   E_t [ MSE(predicted_noise, true_noise) ]
        #
        # Instead of looping over all timesteps sequentially
        # (which would be inefficient),
        # we randomly sample a timestep for each image.
        #
        # This means that within a single mini-batch:
        #   - Some images are lightly corrupted
        #   - Some are moderately corrupted
        #   - Some are heavily corrupted
        #
        # Over many mini-batches, this random sampling
        # approximates the full expectation over timesteps.
        #
        # Intuition:
        # We are training the model to restore images at ALL
        # levels of corruption simultaneously.
        #
        timesteps_batch = torch.randint(
            0,
            STEPS,
            (images_batch.size(0),),
            device=device
        )

        # Sample Gaussian noise with the same shape as the images
        # This represents the "true noise" we will ask the model to predict
        noise_batch = torch.randn_like(images_batch)

        # Generate the noisy images x_t using the scheduler
        #
        # Mathematically:
        #
        #   x_t = sqrt(alpha_bar_t) * x₀
        #         + sqrt(1 - alpha_bar_t) * noise
        #
        # The scheduler handles the correct scaling internally.
        noisy_images_batch = scheduler.add_noise(
            images_batch,
            noise_batch,
            timesteps_batch
        )


        # -------------------------------------------------------------------------
        # Predict Noise with UNet
        # -------------------------------------------------------------------------
        #
        # The model receives:
        #   - noisy_images_batch  (x_t)
        #   - timesteps_batch     (t)
        #   - labels_cfg          (class conditioning signal)
        #
        # It outputs predicted noise for each image.
        #

        optimizer.zero_grad(set_to_none = True)
        with torch.cuda.amp.autocast(dtype = torch.float16):
            predicted_noise_batch = unet(
                noisy_images_batch,
                timesteps_batch,
                class_labels=labels_cfg
            ).sample


            # ---------------------------------------------------------------------
            # Loss Computation
            # ---------------------------------------------------------------------
            #
            # We compute Mean Squared Error between:
            #
            #   predicted_noise_batch
            #   noise_batch (true noise)
            #
            # Important:
            #
            # The MSE is averaged across:
            #   - All pixels
            #   - All channels
            #   - All images in the mini-batch
            #
            # The result is ONE scalar value.
            #
            # This scalar answers:
            #
            #   "How wrong was the model on this entire mini-batch?"
            #
            loss = F.mse_loss(predicted_noise_batch, noise_batch)


        # -------------------------------------------------------------------------
        # Backpropagation + Optimizer Step (Mixed Precision)
        # -------------------------------------------------------------------------
        #
        # We trained under torch.cuda.amp.autocast (float16),
        # which improves speed and reduces memory usage.
        #
        # However, float16 gradients can underflow (become too small to represent).
        #
        # GradScaler solves this by:
        #   1) Scaling the loss upward before backward()
        #   2) Unscaling gradients before optimizer step
        #   3) Dynamically adjusting scaling factor over time
        #
        # This allows stable mixed precision training.
        #
        scaler.scale(loss).backward()     # Compute gradients (scaled)
        scaler.unscale_(optimizer)        # Unscale gradients for accurate norm & clipping
        grad_norm_val = compute_grad_norm(unet.module)
        scaler.step(optimizer)            # Update main UNet weights
        scaler.update()                   # Adjust scaling factor for next step

        # -------------------------------------------------------------------------
        # EMA Update (Exponential Moving Average of Weights)
        # -------------------------------------------------------------------------
        #
        # After the main UNet weights are updated,
        # we update the EMA UNet to be a smoothed version.
        #
        # For each parameter:
        #
        #   ema_weight = EMA_DECAY * ema_weight
        #                + (1 - EMA_DECAY) * current_weight
        #
        # EMA_DECAY is typically close to 1 (e.g. 0.999),
        # meaning:
        #
        #   - The EMA model changes slowly
        #   - It represents a long-term average of recent weights
        #
        # Why do this?
        #
        # During training, weights can fluctuate from mini-batch to mini-batch.
        # EMA smooths those fluctuations.
        #
        # In diffusion models, EMA weights almost always produce:
        #   - More stable generations
        #   - Cleaner samples
        #   - Better generalization
        #
        # Important:
        # The optimizer updates the main UNet.
        # The EMA UNet is NEVER updated by gradients.
        #
        with torch.no_grad():
            for ema_param, current_param in zip(
                ema_unet.parameters(),
                unet.module.parameters()
            ):
                ema_param.mul_(EMA_DECAY).add_(
                    current_param,
                    alpha=1 - EMA_DECAY
                )


        # Store the scalar loss from this mini-batch.
        # This does NOT affect training.
        # It is only used to compute average loss for the epoch.
        batch_loss = loss.item()
        losses.append(batch_loss)
        batch_idx += 1

        # Record batch metrics + update logger progress
        if rank == 0:
            if tracker is not None:
                tracker.log_batch(loss=batch_loss, batch_size=images_batch.size(0), grad_norm=grad_norm_val)
            if logger is not None:
                logger.set_progress(
                    epoch=epoch,
                    batch=batch_idx,
                    total_batches=total_batches,
                    total_epochs=EPOCHS,
                    current_loss=batch_loss,
                )


    # -------------------------------------------------------------------------
    # End of Epoch: Logging + Checkpointing
    # -------------------------------------------------------------------------

    # Compute the average mini-batch loss across the epoch.
    #
    # This gives a rough measure of training progress.
    # It is not a validation metric, just a training signal.
    mean_loss = sum(losses) / len(losses)

    # Compute how long the epoch took
    dt = time.time() - t0


    # Only the main process (rank 0) logs and saves.
    # In distributed training, every GPU runs this script,
    # so we restrict logging/checkpointing to avoid duplication.
    if rank == 0:

        # Finalize epoch metrics
        epoch_metrics = None
        if tracker is not None:
            epoch_metrics = tracker.end_epoch(
                epoch=epoch,
                lr=LEARNING_RATE,
                ema_unet=ema_unet,
                live_unet=unet.module,
            )

        # Log to pretty logger dashboard
        if logger is not None:
            sps = epoch_metrics.samples_per_sec if epoch_metrics else 0.0
            grad = epoch_metrics.grad_norm if epoch_metrics else 0.0
            gpu_peak = epoch_metrics.gpu_mem_peak_mb if epoch_metrics else 0.0
            ema_d = epoch_metrics.ema_delta_norm if epoch_metrics else 0.0
            logger.log_epoch_summary(
                epoch=epoch,
                loss=mean_loss,
                lr=LEARNING_RATE,
                sps=sps,
                dt=dt,
                grad_norm=grad,
                gpu_peak_mb=gpu_peak,
                ema_delta=ema_d,
            )

        # Save EMA checkpoint
        torch.save(
            {
                "unet": ema_unet.state_dict(),
                "classes": dataset.classes,
                "scheduler": scheduler.config,
                "size": img,
                "channels": chans,
                "variation": MODEL_NAME,
                "epoch": epoch,
                "split": DATASET_SPLIT,
                "beta_schedule": BETA_SCHEDULE,
                "ema_decay": EMA_DECAY,
                "cfg_dropout_prob": CFG_DROPOUT_PROB,
                "null_class_index": NULL_CLASS,
                "num_class_embeds": num_classes + 1,
            },
            os.path.join(
                OUT_DIR,
                f"emnist_{MODEL_NAME}_epoch{epoch:03d}.pt"
            ),
        )

# =====================================================
# POST-TRAINING: PLOTS, REPORT, CLEAN SHUTDOWN
# =====================================================
if rank == 0:
    log("Training complete — generating plots and report...", "ok")

    if tracker is not None:
        # Generate all training plots
        try:
            plot_paths = generate_all_plots(tracker)
            log(f"Saved {len(plot_paths)} plots to {METRICS_DIR}/plots", "ok")
        except Exception as e:
            log(f"Plot generation failed: {e}", "warn")
            plot_paths = []

        # Generate Markdown report
        try:
            report_path = generate_report(
                tracker=tracker,
                config=TRAINING_CONFIG,
                plot_paths=plot_paths,
            )
            log(f"Report saved: {report_path}", "ok")
        except Exception as e:
            log(f"Report generation failed: {e}", "warn")

        log(f"CSV metrics: {tracker.csv_path}", "metric")
        log(f"JSON metrics: {tracker.json_path}", "metric")

    log(f"[{MODEL_NAME}] All done.", "ok")

    if logger is not None:
        logger.stop()

torch.distributed.destroy_process_group()

