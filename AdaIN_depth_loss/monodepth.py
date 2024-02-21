from __future__ import absolute_import, division, print_function
import os
from torchvision import transforms
import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mono_640x192"

    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)
    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    return encoder,depth_decoder,feed_height,feed_width

def resize_image(img, size):
    transform = transforms.Resize(size,antialias=True)
    return transform(img)

def Gaussian_blur(img, kernel_size=3):
    transform = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    return transform(img)

def depth_loss(synthesis_img,content_img,encoder,depth_decoder,feed_width,feed_height):
    synthesis_img = resize_image(synthesis_img,(feed_height,feed_width))
    content_img = resize_image(content_img,(feed_height,feed_width))
    # PREDICTION
    synthesis_features = encoder(synthesis_img)
    synthesis_outputs = depth_decoder(synthesis_features)
    syn_dp_norm = synthesis_outputs[("disp", 0)]

    content_features = encoder(content_img)
    content_outputs = depth_decoder(content_features)
    content_dp_norm = content_outputs[("disp", 0)]

    #syn_dp_norm = (syn_dp_norm-torch.min(syn_dp_norm))/(torch.max(syn_dp_norm)-torch.min(syn_dp_norm))
    #resized_content_dp_norm = (resized_content_dp_norm-torch.min(resized_content_dp_norm))/(torch.max(resized_content_dp_norm)-torch.min(resized_content_dp_norm))

    return F.smooth_l1_loss(syn_dp_norm,content_dp_norm)