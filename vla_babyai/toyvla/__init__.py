import os

# Lets the CUDA caching allocator grow segments instead of fragmenting the 8 GB card.
# Read at the first CUDA allocation, so it must be set before any tensor reaches the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
