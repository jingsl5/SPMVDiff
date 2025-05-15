import argparse
import re
import cv2
import numpy as np
from PIL import Image,ImageOps


import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import safetensors
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

import sys
sys.path.append("/home/shenjing/MV-Adapter-main")
from scene_mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline

# parallel
# from scene_mvadapter.pipelines.pipeline_mvadapter_i2mv_sd_6 import MVAdapterI2MVSDPipeline

from scene_mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from scene_mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)

import sys
# add
sys.path.append("/home/shenjing/MV-Adapter-main/IP-Adapter-main")
from ip_adapter.ip_adapter import ImageProjModel


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


ch='100'
color_='black'
# num="0408_parallel_0510"  #  False
num="0408_noaug_0513"   #  True
# num="0408_aug_0510" 
use_lora=False

gen_mask_path = "/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/0f1ba9e425a0452eade2a180cfa41e32_6_bed_250_305_400_450.jpg"
# gen_mask_path = "/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/6c15ae9a3fc24613b9fcedb3da80aee5_6_potted plant_200_367_331_400.jpg"
# gen_mask_path = "/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25_1_car_2_239_122_357.jpg"

# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/2816e163791e4878804d6b7476b9dddd_2_bed_122_308_511_511.jpg"
# roim="/home/shenjing/MV-Adapter-main/results/dog_2_dog_180_353_333_511.png"
# roim="/home/shenjing/mp3d_skybox_roi/ARNzJeq3xxb/20db218ae92c46e6899cd3c0bb1c8d65_4_chair_197_356_315_510.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/ARNzJeq3xxb/d8d73b6740174c2699211fed13e236af_4_chair_290_303_360_448.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/0f1ba9e425a0452eade2a180cfa41e32_7_bed_57_316_427_511.jpg"
roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/8c973e0dd63e490db73c44ff6959854b_4_bed_0_315_135_430.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/6ab39d830183407cb7ec17206889000d_2_chair_180_353_333_511.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/e345211511824219a62f1b8d639c477e_1_chair_121_332_235_452.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/dba514ffecec43bdb0a247cd898d224b_6_tv_0_356_180_491.jpg"
# roim="/home/shenjing/mp3d_skybox_roi/ARNzJeq3xxb/36fa04dd448a44449f4f02628e1d6664_3_chair_154_370_246_511.jpg"
roim="/home/shenjing/mp3d_skybox_roi/ARNzJeq3xxb/4cedb1dfbb2c4bbf89fa6766f46cdd4a_4_bed_197_317_511_511.jpg"
# roim="/home/shenjing/MV-Adapter-main/results/dog3_2_chair_180_353_333_511.png"
# roim="/home/shenjing/MV-Adapter-main/results/dog4_2_chair_180_353_333_511.png"
# roim = "/home/shenjing/MV-Adapter-main/results/dog_2_dog_180_353_333_511.png"
# roim="/home/shenjing/mp3d0310_72/1LXtFkjw3qL/6ab39d830183407cb7ec17206889000d_2_0.8240019679069519_chair_334_372_512_511.jpg"
# roim="/home/shenjing/MV-Adapter-main/results/dog2_2_chair_180_353_333_511.png"
# roim="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/0f7e0af0cb3b4c2abf62bba2fd955702_0_chair_307_340_468_511.jpg"
# roim ="/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/14a8edbbe4b14a05b1b5782a884fb6bf_1_bed_52_261_433_511.jpg"
roim = "/home/shenjing/mp3d0310_72/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25_0_0.862956702709198_car_226_240_337_340.jpg"
roim = "/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25_1_car_0_239_122_357.jpg"


objects = '_'.join(roim.split('_')[-5:-1])
objects = roim.split('_')[-5]
score_ref=1.0

def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    image_encoder,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet",ignore_mismatched_sizes=True,low_cpu_mem_usage=False,)
    if image_encoder is not None:
        pipe_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder)
    # if image_proj_model is not None:
    #     pipe_kwargs["image_proj_model"] = ImageProjModel.from_pretrained(args.image_encoder)
        

    
    

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]
    


    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)  
        
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    # Load the pre-trained token embeddings
    embedding_path = "/home/shenjing/MV-Adapter-main/output{}/learned_embeds-steps-{}.safetensors".format(num,ch)
    embedding_weights = torch.load(embedding_path)
    text_encoder.resize_token_embeddings(len(tokenizer))
    # Update the text_encoder with the new embeddings
    with torch.no_grad(): 
        # Update only the specific embeddings that were trained 
        for token_id, embedding in zip(placeholder_token_ids, embedding_weights['V*']): 
            text_encoder.get_input_embeddings().weight.data[token_id] = torch.tensor(embedding) 
        

    # Prepare pipeline
    pipe: MVAdapterI2MVSDPipeline
    pipe = MVAdapterI2MVSDPipeline.from_pretrained(
        base_model, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
        **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # pipe.scheduler = ShiftSNRScheduler.from_scheduler(
    #     pipe.scheduler,
    #     shift_mode="interpolated",
    #     shift_scale=8.0,
    #     scheduler_class=scheduler_class,
    # )
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="checkpoint-{}.safetensors".format(ch)
    )
    
    # pipe.load_custom_adapter(
    #     adapter_path, weight_name="model.safetensors"
    # )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None and use_lora:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)
        print("lora  loaded ... ")

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


clip_image_processor = CLIPImageProcessor()


def get_mask(selected_file,ymin,ymax, xmin,xmax):
    # 读取图像
    image_sel = Image.open(selected_file).convert('L')
    image_array = np.array(image_sel)

    # 将非零像素填充为1
    binary_image_array = np.where(image_array > 0, 1, 0).astype(np.uint8)

    # 找到最小外接四边形
    coords = cv2.findNonZero(binary_image_array)
    rect = cv2.minAreaRect(coords)

    # 获取四边形的顶点
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 创建一个与原图大小相同的全零图像
    result_image_array = np.zeros_like(binary_image_array)

    # 将最小外接四边形内部所有像素变为1
    cv2.fillPoly(result_image_array, [box], 1)



    # 创建一个512x512的全零图像
    final_image_array = np.zeros((512, 512), dtype=np.uint8)

    # 将结果图像贴到512x512图像中，根据提取的坐标定位
    final_image_array[ymin:ymax, xmin:xmax] = result_image_array

    # 将数组转换回图像
    final_image = Image.fromarray(final_image_array * 255)  # 乘以255以便于可视化

    final_image.save('/home/shenjing/MV-Adapter-main/results3/output_image2.png')
    
    mask = torch.tensor(final_image_array, dtype=torch.float16)
    
    return  mask



def resize_and_pad_image(image, new_width, new_height, background_color=(255, 255, 255)):
    # 获取原始图像的宽度和高度
    original_width, original_height = image.size

    # 计算新的尺寸，保持纵横比
    aspect_ratio = min(new_width / original_width, new_height / original_height)
    new_size = (int(original_width * aspect_ratio), int(original_height * aspect_ratio))

    # 调整图像大小
    resized_image = image.resize(new_size)

    # 创建一个新的白色背景图像
    padded_image = Image.new("RGB", (new_width, new_height), background_color)

    # 计算粘贴位置，使图像居中
    paste_position = ((new_width - new_size[0]) // 2, (new_height - new_size[1]) // 2)

    # 将调整后的图像粘贴到白色背景图像上
    padded_image.paste(resized_image, paste_position)

    return padded_image


def run_pipeline(
    pipe,
    num_views,
    text,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    negative_prompt,
    lora_scale=1.0,
    device="cuda",
    roi_path=None
):
    # 初始化控制图像列表
    control_images = []

    # # 定义图像路径
    image_paths = [
        '/home/shenjing/MV-Adapter-main/tools/perspective72_0.png',
        '/home/shenjing/MV-Adapter-main/tools/perspective72_1.png',
        '/home/shenjing/MV-Adapter-main/tools/perspective72_2.png',
        '/home/shenjing/MV-Adapter-main/tools/perspective72_3.png',
        '/home/shenjing/MV-Adapter-main/tools/perspective72_4.png',
        # '/home/shenjing/MV-Adapter-main/tools/perspective72_5.png'
    ]
    
     # 定义图像路径
    # image_paths = [
    #     '/home/shenjing/MV-Adapter-main/tools/perspective_0.png',
    #     '/home/shenjing/MV-Adapter-main/tools/perspective_1.png',
    #     '/home/shenjing/MV-Adapter-main/tools/perspective_2.png',
    #     '/home/shenjing/MV-Adapter-main/tools/perspective_3.png',
    #     '/home/shenjing/MV-Adapter-main/tools/perspective_4.png',
    #     # '/home/shenjing/MV-Adapter-main/tools/perspective_5.png'
    # ]

    # 定义转换
    transform = transforms.Compose([
        # transforms.Grayscale(),  # 转换为灰度图像
        transforms.ToTensor()    # 转换为张量
    ])

    # 遍历图像路径，将每个图像转换为单通道张量
    for image_path in image_paths:
        # 打开图像
        cim = Image.open(image_path)

        # 将图像转换为单通道张量
        cim_tensor = transform(cim)
        cim_tensor = torch.cat([cim_tensor, cim_tensor], dim=0)

        # 添加到控制图像列表
        control_images.append(cim_tensor.squeeze(0))

    # 将所有单通道张量合并为一个张量
    control_images = torch.stack(control_images).float().to(device)  # 增加一个维度，使形状为 (numviews, 6, H, W)

    pipe_kwargs = {}
    if seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)


    clip_rois = []
    roi_images = []
    roi_texts = []
    roi_text_ids = []
    is_ref_zero = []
    masks = []

    
    for roi_ in roi_path:
        if roi_ is not None:
            cls_name = roi_.split('_')[-5]
            # 使用正则表达式提取坐标
            pattern = r'(\d+)_(\d+)_(\d+)_(\d+)\.jpg$'
            xmin, ymin, xmax, ymax = map(int, re.search(pattern, gen_mask_path).groups())
            # roi_coords = [xmin, ymin, xmax, ymax]
            # mask = get_mask(roi_,ymin,ymax, xmin,xmax)

            # 创建一个512x512的全零彩色图像
            final_image = Image.new('RGB', (512, 512), (255, 255, 255))

            # 在指定区域填充为白色
            final_image_array = np.array(final_image)
            final_image_array[ymin:ymax+1, xmin:xmax+1] = [255, 255, 255]
            # final_image_array[ymin:ymax+1, xmin:xmax+1] = [0, 0, 0]

            # 计算白色区域的宽度和高度
            white_region_width = xmax - xmin
            white_region_height = ymax - ymin
                
                
            roi_image = Image.open(roi_)
            # 调整ROI图像大小以适应白色区域，同时保持纵横比

            

            # 假设 roi_image 是你的原始图像
            # 指定最大宽度和高度
            max_width = white_region_width
            max_height = white_region_height

            # 调整图像大小
            roi_image = resize_and_pad_image(roi_image, max_width, max_height)
            
            final_image0323 = Image.new('L', (512, 512), (0))
            final_image_array0323 = np.array(final_image0323)
            final_image_array0323[ymin:ymax, xmin:xmax]=(1)
            final_image_array0323 = Image.fromarray(final_image_array0323)
            final_image_array0323.save("/home/shenjing/MV-Adapter-main/results3/output0323.jpg")
            
            
            # roi_image = ImageOps.fit(roi_image, (white_region_width, white_region_height), bleed=0.0, centering=(0.5, 0.5))
          
            # 将调整大小后的ROI图像贴到白色区域内
            roi_array = np.array(roi_image)
            mask = (roi_array != [0, 0, 0]).all(axis=-1)  # 创建掩码，只保留非零像素
            final_image_array[ymin:ymax, xmin:xmax][mask] = roi_array[mask]

            # 将数组转换回图像
            roi_image = Image.fromarray(final_image_array)
            
            roi_image.save("/home/shenjing/MV-Adapter-main/results3/output0313.jpg")
            clip_roi_image_ = clip_image_processor(images=roi_image, return_tensors="pt").pixel_values
            

            # # 使用正则表达式提取坐标
            # pattern = r'(\d+)_(\d+)_(\d+)_(\d+)\.jpg$'
            # xmin, ymin, xmax, ymax = map(int, re.search(pattern, roi_).groups())
            # roi_coords = [xmin, ymin, xmax, ymax]

            # # 创建一个 512x512 的全零图像
            # full_image = Image.new('RGB', (512, 512), color=color_)

            # # 将 ROI 贴回到全零图像上
            # full_image.paste(roi_image, (xmin, ymin))
            # roi_image = full_image
            
            
            
            # roi_image = roi_image.resize((512, 512))
            
            roi_text = cls_name+ " V*"
            is_ref_0 = 0
            
        else:
            roi_path = None
            roi_image = Image.new('RGB', (512, 512), (255, 255, 255))
            clip_roi_image_ = clip_image_processor(images=roi_image, return_tensors="pt").pixel_values
                
            roi_text = ""
            is_ref_0 = 1
            
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask = torch.tensor(mask, dtype=torch.float16)
            
        
        masks.append(mask)
            
        is_ref_zero.append(is_ref_0)

        clip_rois.append(clip_roi_image_)
        roi_images.append(roi_image)
        roi_texts.append(roi_text)
        
        roi_text_input_ids = pipe.tokenizer(
                roi_text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                    
            ).input_ids
        roi_text_ids.append(roi_text_input_ids)
        
    clip_rois = torch.cat(clip_rois,dim=0).to(dtype=torch.float16).to(device) 
    roi_text_ids = torch.cat(roi_text_ids,dim=0)


    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=roi_images,
        reference_conditioning_scale=score_ref,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        roi=clip_rois,
        roi_texts=roi_texts,
        roi_text_ids=roi_text_ids,
        is_ref_zero=is_ref_zero,
        masks=masks,
        **pipe_kwargs,
    ).images

    return images


if __name__ == "__main__":
    # txt123 = "there is a living room with a couch, chairs, and a table, arafed room with a couch, chairs, and a mirror, there are four chairs in the room with a mirror on the wall, there is a room with a fireplace and a mirror in it, there is a fireplace in a room with a mirror and a rug, there is a room with a chandelier and a fireplace in it"
    txt123 = "additional view of a grassy area with a fence and a bench, a view of a backyard with a pool and a fence, a view of a backyard with a V* bed, a close up of a pool with a waterfall in the middle of it, a close up of a pool with a waterfall in the middle of it, a view of a pool with a waterfall and a rock wall"
    # txt123="there are two * parked in the parking lot outside of a building, there is a car parked outside of a building with a car parked in front of it, there is a door to a building with a number on it, there is a sign on the side of a building that says no entry, there is a wooden fence with a metal pole on the side, there is a wooden wall with a metal door and a black door"
    
    # txt123 = "additional view of a grassy area with a fence and a bench, a view of a backyard with a pool and a fence, arafed room with a couch, a bed, and a television, there is a large {} bed in a room with a tv and a couch, there is a room with a large window and a tv in it, there is a view of a bedroom with a bed and a window"
    # txt123=txt123.format("V*")
    
    
    # txt123 = "additional view of a grassy area with a fence and a {} V*, a view of a water pool with a pool and a fence, a view of a water pool with a {} V* on the grass, a close up of a pool with a waterfall in the middle of it, a close up of a pool with a waterfall in the middle of it"
    # txt123 = "there is a car V* that is parked in the driveway of a house, there is a hallway with a door and a planter on the floor, there is a door with a number on it and a hand rail, there is a wooden wall with a metal pole on it, there is a car V* parked in a driveway next to a building"

    txt123 = "there is a car V* that is parked in the driveway of a house, there is a hallway with a door and a planter on the floor, there is a door and a car V* and a hand rail, there is a wooden wall with a metal pole on it, there is a car parked in a driveway next to a building"

    # txt123 = "there is a car that is parked in the driveway of a house, there is a hallway with a door and a planter on the floor, there is a door and a car and a hand rail, there is a wooden wall with a metal pole on it, there is a car parked in a driveway next to a building"
    # txt123 = "additional view of a grassy area with a {} V*, a view of a water pool and a fence, a view of a water pool with a {} V* on the grass, a close up of a pool with a waterfall in the middle of it, a close up of a pool with a waterfall in the middle of it"
    
    txt123=txt123.format(objects,objects)
    
    # roisss = [
    #     None,
    #     None,
    #     "/home/shenjing/mp3d_skybox_roi/1LXtFkjw3qL/2816e163791e4878804d6b7476b9dddd_2_bed_122_308_511_511.jpg",
    #     None,
    #     None,
    #     None
    #     ]
    
    roisss = [
        roim,
        None,
        roim,
        None,
        None,
        ]
    
    
    parser = argparse.ArgumentParser()
    # mv Models
    parser.add_argument(
        "--base_model", type=str, default="/home/shenjing/stabilityai/stable-diffusion-2-1-base"
    )
    parser.add_argument("--vae_model", type=str, default=None)
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--image_encoder", type=str, default="/home/shenjing/stabilityai/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--roi_path", type=str, default=roisss)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default="/home/shenjing/code/diffusers-main/examples/text_to_image/sd-naruto-model-lora/checkpoint-60000/pytorch_lora_weights.safetensors")
    parser.add_argument("--adapter_path", type=str, default="/home/shenjing/MV-Adapter-main/output{}".format(num))
    parser.add_argument("--num_views", type=int, default=5)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--text", type=str, default=txt123, required=False)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="V*",
        required=False,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="/home/shenjing/MV-Adapter-main/results3/output0512_6_{}_{}_{}_{}.png".format(ch,objects,score_ref,num))
    args = parser.parse_args()



    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        image_encoder=args.image_encoder,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    images = run_pipeline(
        pipe,
        num_views=args.num_views,
        text=args.text,
        height=512,
        width=512,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        lora_scale=args.lora_scale,
        device=args.device,
        roi_path=args.roi_path
    )
    make_image_grid(images, rows=1,save_dir="/home/shenjing/MV-Adapter-main/results3/1234_6.png").save(args.output)
    
    
