import torch
from copy import deepcopy
import time

import PIL
import numpy as np
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
    image_emb = None
    for i in range(len(images_texts)):
        if type(images_texts[i]) == str:
            encoded = model.generate_clip_emb(
                images_texts[i],
                batch_size=1,
                prior_cf_scale=prior_cf_scale,
                prior_steps=prior_steps,
                negative_prior_prompt=negative_prior_prompt,
            )
        else:
            encoded = model.encode_images(images_texts[i], is_pil=True) * weights[i]
        encoded = encoded * weights[i]    
        
        if image_emb is None:
            image_emb = encoded
        else:
            image_emb += encoded
        
    # create negative embedding
    image_emb = image_emb.repeat(batch_size, 1)
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
    def __init__(self):
        self.model = get_kandinsky2('cuda', task_type='text2img', 
                              model_version='2.1', 
                              use_flash_attention=use_flash_attn)
        self.model.model.convert_to_fp16()
        self.model.model.dtype = torch.float16
    
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
                                        self.model, batch_size,prior_cf_scale, 
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
    
    
    def sample_noise(self,batch_size, h, w, seed, device="cuda"):
        # create start noise
        new_h, new_w = self.model.get_new_h_w(h, w)
        shape = (batch_size * 2, 4, new_h, new_w)
        g_gpu = torch.Generator(device)
        g_gpu.manual_seed(seed)
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
                h=576,
                w=768,
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
               ):
        start_time = time.time()

        if text_emb is None:
            if images_texts is None:
                assert prompt is not None
                images_texts = [prompt]
                weights = [1.0]
            text_emb, image_emb = self.embed_inputs(images_texts,
                                                    weights,
                                                    batch_size,
                                                    prior_cf_scale, 
                                                    prior_steps, negative_prior_prompt,
                                                    negative_decoder_prompt)
        

        # rescale num steps such that the model actually does the amount of steps the user asked for
        rescale_steps = True
        if rescale_steps and init_img is not None:
            num_steps = round(num_steps / (1 - strength))
            
        # load diffusion
        diffusion, config = self.create_diffusion(num_steps, sampler)
        
        # prep img2img
        if init_img is None:
            noise = self.sample_noise(batch_size, h, w, seed, device="cuda")
        else:
            noise, init_step = self.prepare_img2img(init_img, 
                                                    strength,
                                                    h, w, 
                                                    diffusion, config)
                        

        print("Startup time required: ", round(time.time() - start_time, 2))

        # seed diffusion process
        torch.manual_seed(0)

        images = self.model.generate_img(
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
        )
        print("Time required: ", round(time.time() - start_time, 2))
        return images[0]
    
    def prepare_img2img(self, init_img, strength, h, w, diffusion, config):
        if isinstance(init_img, PIL.Image.Image):
            init_img = prepare_image(init_img, h=h, w=w)
        elif isinstance(init_img, np.ndarray):
            init_img = torchvision.transforms.ToTensor()(init_img)

        if init_img.ndim == 3:
            init_img = init_img.unsqueeze(0)

        if self.model.use_fp16:
            init_img = init_img.half()

        #print("Init img shape: ", init_img.shape)
        encoded_image = self.model.image_encoder.encode(init_img.to(self.model.device)) * model.scale

        # use simple noise schedule
        use_noise = False        
        
        start_step = round(diffusion.num_timesteps * (1 - strength))
        if use_noise:            
            noise = torch.randn_like(encoded_image)
            encoded_image = strength * image + (1 - strength) * noise
        else:
            encoded_image = q_sample(
                encoded_image,
                torch.tensor(diffusion.timestep_map[start_step - 1]).to(model.device),
                schedule_name=config["diffusion_config"]["noise_schedule"],
                num_steps=config["diffusion_config"]["steps"],
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