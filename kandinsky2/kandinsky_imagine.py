from copy import deepcopy
import time
from typing import List

import torch
import torchvision
import PIL
import numpy as np
from PIL import Image
from kandinsky2.model.model_creation import create_gaussian_diffusion
from kandinsky2.utils import prepare_image, q_sample
from kandinsky2 import get_kandinsky2
try:
    import flash_attn
    use_flash_attn = True
except:
    use_flash_attn = False

    
def make_image_text_emb(images_texts, weights, 
                        model, batch_size, prior_cf_scale, 
                        prior_steps, negative_prior_prompt,
                        negative_decoder_prompt):    
    # make sum of weights equal to 1
    weights = np.array(weights).astype(np.float32)
    weights /= weights.sum()
    
    # generate clip embeddings
    generate_clip_emb_kargs = dict(batch_size=batch_size,
                prior_cf_scale=prior_cf_scale,
                prior_steps=prior_steps,
                negative_prior_prompt=negative_prior_prompt)
    image_emb = None
    for img_text, weight in zip(images_texts, weights):
        if torch.is_tensor(img_text):
            encoded = img_text  # was precomputed
        elif type(img_text) == str:
            encoded = model.generate_clip_emb(
                img_text,
                **generate_clip_emb_kargs,
            )
        else:
            encoded = model.encode_images(img_text, is_pil=True)
        encoded = encoded * weight
        image_emb = encoded if image_emb is None else image_emb + encoded
        
    # create negative embedding
    image_emb = image_emb.repeat(batch_size, 1)
    if negative_decoder_prompt == "":
        zero_image_emb = model.create_zero_img_emb(batch_size=batch_size)
    else:
        zero_image_emb = model.generate_clip_emb(
            negative_decoder_prompt,
            **generate_clip_emb_kargs,
        )
    image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(model.device)
    return image_emb


def make_image_emb(model, prompt, batch_size, prior_cf_scale, 
                   prior_steps, negative_prior_prompt,
                   negative_decoder_prompt):
    # generate clip embeddings
    image_emb = model.generate_clip_emb(
        prompt,
        batch_size=batch_size,
        prior_cf_scale=prior_cf_scale,
        prior_steps=prior_steps,
        negative_prior_prompt=negative_prior_prompt,
    )
    if negative_decoder_prompt == "":
        zero_image_emb = model.create_zero_img_emb(batch_size=batch_size)
    else:
        zero_image_emb = model.generate_clip_emb(
            negative_decoder_prompt,
            batch_size=batch_size,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
            negative_prior_prompt=negative_prior_prompt,
        )
    image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(model.device)
    return image_emb


class ImagineKandinsky(torch.nn.Module):
    def __init__(self, width, height, cache_dir="/tmp/kandinsky", device="cuda"):
        super().__init__()
        self.model = get_kandinsky2(device, 
                                    task_type='text2img', 
                                    model_version='2.1', 
                                    use_flash_attention=use_flash_attn,
                                    cache_dir=cache_dir)
        self.model.model.convert_to_fp16()
        self.model.model.dtype = torch.float16
        self.device = "cuda"
        self.w = width
        self.h = height
        
        self.neg_prompt = "ugly, tiling, oversaturated, undersaturated, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"
        self.input_type = torch.float16
    
    def embed_inputs(self, 
                     images_texts,
                     weights,
                     batch_size,
                     prior_cf_scale, 
                     prior_steps,
                     negative_prior_prompt,
                     negative_decoder_prompt
                    ):
        # create img embedding
        image_emb_models = [self.model.prior, self.model.clip_model]
        for m in image_emb_models:
            m.cuda()
        image_emb = make_image_text_emb(images_texts, weights, 
                                        self.model, batch_size, prior_cf_scale, 
                                        prior_steps, negative_prior_prompt,
                                        negative_decoder_prompt)
        #for m in image_emb_models:
        #    m.cpu()

        # create text embedding
        prompt = ""  # TODO: probably needs to be set to something meaningful
        #prompt = images_texts[0]
        self.model.text_encoder.cuda()
        text_emb = self.model.encode_text(
                    text_encoder=self.model.text_encoder,
                    tokenizer=self.model.tokenizer1,
                    prompt=prompt,
                    batch_size=batch_size,
                )
        #self.model.text_encoder.cpu()
        return text_emb, image_emb
    
    def move_encoders(self, *args, **kwargs):
        image_emb_models = [self.model.prior, self.model.clip_model]
        encoder_models = image_emb_models + [self.model.text_encoder]
        for m in encoder_models:
            m.to(*args, **kwargs)
        return self
    
    
    def sample_noise(self, width, height, seed=None, device="cuda", batch_size=1, generator=None):
        # create start noise
        new_h, new_w = self.model.get_new_h_w(height, width)
        shape = (batch_size * 2, 4, new_h, new_w)
        if generator is None:
            g_gpu = torch.Generator(device)
            if seed is not None:
                g_gpu.manual_seed(seed)
        else:
            g_gpu = generator
        noise = torch.randn(*shape, device=device, generator=g_gpu)
        return noise

    
    def gen_img(self,
                prompt=None,
                images_texts=None,
                weights=None,
                embedding=None,
                num_steps=18,
                batch_size=1,
                guidance_scale=4,
                h=None,
                w=None,
                sampler="ddim_sampler",  # plms_sampler or ddim_sampler
                prior_cf_scale=2,
                prior_steps="2",
                negative_prior_prompt="",
                negative_decoder_prompt="",
                seed=1,
                init_step=None,
                inpaint_img=None,
                inpaint_mask=None,
                init_img=None,
                strength=0.9,
                text_emb=None,
                image_emb=None,
                noise=None,
                verbose=True,
                latents=None,
                return_intermediates=False,
               ):
        start_time = time.time()

        if h is None:
            h = self.h
        if w is None:
            w = self.w

        if text_emb is None:
            if images_texts is None:
                assert prompt is not None
                images_texts = [prompt]
                weights = [1.0]
            text_emb, image_emb = self.embed_inputs(images_texts,
                                                    weights,
                                                    batch_size,
                                                    prior_cf_scale, 
                                                    prior_steps, 
                                                    negative_prior_prompt,
                                                    negative_decoder_prompt)
        

        # rescale num steps such that the model actually does the amount of steps the user asked for
        rescale_steps = True
        if rescale_steps and init_img is not None:
            num_steps = round(num_steps / (1 - strength))
            
        # load diffusion
        diffusion, config = self.create_diffusion(num_steps, sampler)
        
        # prep img2img
        if latents is not None:
            noise = latents.cuda()
        elif init_img is None:
            if noise is None:
                noise = self.sample_noise(w, h, seed, device="cuda", batch_size=batch_size)
        else:
            noise, init_step = self.prepare_img2img(init_img, 
                                                    strength,
                                                    h, w, 
                                                    diffusion, 
                                                    config,
                                                    noise)
                        
        if verbose:
            print("Startup time required: ", round(time.time() - start_time, 2))

        # seed diffusion process
        torch.manual_seed(0)

        images, intermediates = self.model.generate_img(
            prompt=prompt,
            image_emb=image_emb,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=h,
            w=w,
            sampler=sampler,
            num_steps=num_steps,
            diffusion=diffusion,
            img_mask=inpaint_mask,
            init_img=inpaint_img,
            init_step=init_step,
            noise=noise,
            # new param:
            text_emb=text_emb,
            verbose=verbose,
            return_intermediates=True,
        )
        if verbose:
            print("Time required: ", round(time.time() - start_time, 2))
        if return_intermediates:
            return images, intermediates  
        else:
            return images
    
    def prepare_img2img(self, init_img, strength, h, w, diffusion, config, noise=None):
        
        
        if not isinstance(init_img, PIL.Image.Image):
            if torch.is_tensor(init_img):
                init_img = init_img.cpu().squeeze().permute(2, 0, 1)
            #print(init_img.shape)
            init_img = torchvision.transforms.ToPILImage()(init_img)
            
        init_img = prepare_image(init_img, h=h, w=w)
       
        if init_img.ndim == 3:
            init_img = init_img.unsqueeze(0)

        if self.model.use_fp16:
            init_img = init_img.half()

        #print("Init img shape: ", init_img.shape)
        encoded_image = self.model.image_encoder.encode(init_img.to(self.model.device)) * self.model.scale

        # use simple noise schedule
        use_noise = False        
        
        if noise is None:
            noise = torch.randn_like(encoded_image)
        else:
            #print(noise.shape)
            if noise.shape[0] == 2:
                #noise = noise.repeat(2, 1, 1, 1)
                noise = noise[0].unsqueeze(0)
            #print(noise.shape)
            
        start_step = round(diffusion.num_timesteps * (1 - strength))
        if use_noise:
            encoded_image = strength * image + (1 - strength) * noise
        else:
            time_step = torch.tensor(diffusion.timestep_map[start_step - 1]).to(self.model.device)
            encoded_image = q_sample(
                encoded_image,
                time_step,
                schedule_name=config["diffusion_config"]["noise_schedule"],
                num_steps=config["diffusion_config"]["steps"],
                noise=noise,
            )

        if encoded_image.shape[0] == 1:
            encoded_image = encoded_image.repeat(2, 1, 1, 1)

        #print("Encoded img shape: ", encoded_image.shape)

        noise = encoded_image
        init_step = start_step
        #print("Start step: ", start_step)
        return noise, init_step
        
    def create_diffusion(self, num_steps, sampler):
        config = deepcopy(self.model.config)
        if sampler == "p_sampler":
            config["diffusion_config"]["timestep_respacing"] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config["diffusion_config"])
        return diffusion, config
    
    
    ### for SD compatibility
    
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self
    
    def init_uncond_embeddings(self, *args, **kwargs):
        pass
    
    def _encode_prompt(self, 
                       prompt_list, 
                       style_images: List = None, 
                       style_image_weights: List = None):
        text_weights = np.ones(len(prompt_list)) / len(prompt_list)
        if style_images is None:
            images_texts = prompt_list
            weights = text_weights
        else:
            # read images as PIL if they are strs
            style_images = [Image.open(style_img) if isinstance(style_img, str) else style_img
                            for style_img in style_images]
            images_texts = prompt_list + style_images
            # set weights
            if style_image_weights is None:
                style_image_weights = np.ones(len(style_images)) / len(style_images)
            weights = np.concatenate((text_weights, style_image_weights))
        # set hyperparams
        batch_size = 1
        prior_cf_scale = 2
        prior_steps = "2"
        negative_prior_prompt = self.neg_prompt
        negative_decoder_prompt = self.neg_prompt
                
        text_emb, image_emb = self.embed_inputs( 
                     images_texts,
                     weights,
                     batch_size,
                     prior_cf_scale, 
                     prior_steps,
                     negative_prior_prompt,
                     negative_decoder_prompt
                    )
        #print(len(text_emb), type(text_emb), len(image_emb), type(image_emb))
        return list(text_emb) + [image_emb]
        
    def forward(self, 
                text_embeddings=None,
                start_img=None,
                img2img_strength=0.0,
                num_inference_steps=30,
                noise=None,
                output_type="torch",
                sampler="ddim_sampler",  # p_sampler or ddim_sampler
                latents=None,
                t_start=None,
                #sampler="p_sampler",
                **diffusion_kwargs):
        
        if t_start == 0:
            t_start = None
        if t_start is not None:
            t_start = num_inference_steps - t_start  # for SD compatibility
            
            if sampler == "ddim_sampler":
                t_start = t_start * (1000 / num_inference_steps)  # for DDIM in the range of 1-1000, instead of 1-num_inference_steps
                
                
                

        # unpack embeddings
        text_emb = text_embeddings[0].cuda(), text_embeddings[1].cuda()
        image_emb = text_embeddings[2].cuda()
        
        if noise is not None:
            noise = noise.cuda()
        
        #self.model.clip_model.to("cpu")
        self.model.model.to("cuda")
        self.model.image_encoder.to("cuda")
                
        # gen pil_img
        pil_imgs, intermediates = self.gen_img(
                prompt=None,
                images_texts=None,
                weights=None,
                embedding=None,
                num_steps=num_inference_steps,
                batch_size=1,
                guidance_scale=4,
                sampler=sampler,
                prior_cf_scale=2,
                prior_steps="2",
                negative_prior_prompt="",
                negative_decoder_prompt="",
            
                seed=1,
                init_step=t_start,
                inpaint_img=None,
                inpaint_mask=None,
            
                init_img=start_img,
                strength=1 - img2img_strength,
                
                text_emb=text_emb,
                image_emb=image_emb,
                noise=noise,
            
                verbose=False,
                latents=latents,
            
                return_intermediates=True
               )
        pil_img = pil_imgs[0]
        
        
        if output_type == "torch":
            img = torchvision.transforms.ToTensor()(pil_img)
        else:
            img = pil_img
            
        return {"images": [img], "latents": intermediates}

    def encode_image(self, pil_img):
        self.model.clip_model.to("cuda")
        if torch.is_tensor(pil_img):
            pil_img = torchvision.transforms.ToPILImage()(pil_img.squeeze().cpu())
        elif isinstance(pil_img, str):
            pil_img = Image.open(pil_img)
        encoded = self.model.encode_images(pil_img, is_pil=True)
        return encoded