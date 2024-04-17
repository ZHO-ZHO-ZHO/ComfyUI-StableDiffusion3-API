import os
import io
import json
import requests
import torch
import requests
from io import BytesIO
from PIL import Image
import sys
import torch
import numpy as np
import base64

p = os.path.dirname(os.path.realpath(__file__))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_sai_api_key():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        api_key = config["STABILITY_KEY"]
    except:
        print("出错啦 Error: API key is required")
        return ""
    return api_key


class SD3_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"default": "cat", "multiline": True}),
                "negative": ("STRING", {"default": "worst quality, low quality", "multiline": True}),
                "aspect_ratio": (["21:9", "16:9", "5:4", "3:2", "1:1", "2:3", "4:5", "9:16", "9:21"],),
                "mode": (["text-to-image", "image-to-image"],),
                "model": (["sd3", "sd3-turbo"],),
                "seed": ("INT", {"default": 66, "min": 0, "max": 1000000}),
            },
            "optional": {
                "image": ("IMAGE",),  
                "strength": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "🔥SD3"
                       
    def generate_image(self, positive, negative, aspect_ratio, mode, model, seed, image=None, strength=None):
        
        apikey = get_sai_api_key()

        if mode == 'text-to-image':
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": apikey,
                    "accept": "application/json"
                },
                files={"none": ''},
                data={
                    "prompt": positive,
                    "negative_prompt": negative,
                    "aspect_ratio": aspect_ratio,
                    "mode": mode,
                    "model": model,
                    "seed": seed,
                    "output_format": "png",
                },
            )

        elif mode == 'image-to-image':
            pil_image = tensor2pil(image)
            
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            #img_byte_arr = img_byte_arr.getvalue()
            img_byte_arr.seek(0)
            
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": apikey,
                    "accept": "application/json"
                },
                files={
                    "image": ("image.png", img_byte_arr, 'image/png'), 
                },
                data={
                    "prompt": positive,
                    "negative_prompt": negative,
                    "aspect_ratio": aspect_ratio,
                    "mode": mode,
                    "model": model,
                    "seed": seed,
                    "strength": strength,
                    "output_format": "png",
                },
            )
        
        if response.status_code == 200:
            json_data = response.json()
            image_base64 = json_data['image']
            image_bytes = base64.b64decode(image_base64)
            image_data = Image.open(io.BytesIO(image_bytes))
            output_t = pil2tensor(image_data)
            print(output_t.shape)
            return (output_t,)
        else:
            # 错误处理
            if response.headers['Content-Type'] == 'application/json':
                error_info = response.json()
                print("Error name:", error_info.get('name', 'No name provided'))
                print("Error details:", error_info.get('errors', ['No details provided']))
            print("Response Text:", response.text) 
            raise Exception(f"Failed to fetch image: {response.status_code}")
        


NODE_CLASS_MAPPINGS = {
    "SD3_Zho": SD3_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3_Zho": "🔥Stable Diffusion 3",
}
