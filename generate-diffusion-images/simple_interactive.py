#Main code extracted from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/random_walks_with_stable_diffusion.ipynb#scrollTo=5SnL_YTJ32iS
#https://keras.io/examples/generative/random_walks_with_stable_diffusion/
import os
import copy
import math
import datetime
import argparse
import keras_cv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def get_images(model,prompt,images_per_prompt=1,unconditional_guidance_scale=7.5):
  print("Getting text enconding")
  _encoding = tf.squeeze(model.encode_text(prompt))
  print("Getting images")
  _images = model.generate_image(_encoding,
                                  batch_size=images_per_prompt,
                                  unconditional_guidance_scale=unconditional_guidance_scale)
  print("Images gotten")
  return _images

def save_result(result_images,prompt,output_dir="./results/"):
  _od = prompt.replace(" ","_")
  _odir = os.path.join(output_dir,_od)
  os.makedirs(_odir,exist_ok=True)
  print("Saving on",_odir)
  time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  for ix,image in enumerate(result_images):
    pili = Image.fromarray(image)
    file_name = "generated_image_{}_{}.jpg".format(time_now,ix)
    file_name = os.path.join(_odir,file_name)
    pili.save(file_name, quality=100)
    print(file_name,"saved")

def generate_and_save(model,input_str,
                      output_dir="./results/",
                      split_by=".",
                      images_per_prompt=1,
                      unconditional_guidance_scale = 7.5):
  prompts = [_.strip() for _ in input_str.split(split_by) if len(_)>0]
  for prompt in prompts:
    print("Working on ",prompt)
    result = get_images(model,prompt,images_per_prompt,unconditional_guidance_scale)
    save_result(result,prompt,output_dir)

def set_int(i):
    return int(i)
def set_str(i):
    return str(i)
def set_float(i):
    return float(i)
def get_default_params():
    o = {}
    ks = "input_str,output_dir,split_by,images_per_sentence,unconditional_guidance_scale".split(",")
    _types = [set_str,set_str,set_str,set_int,set_float]
    vs = ["A cat playing guitar","./results/",".",1,7.5]
    input_texts="Text to generate,Output directory,Separate sentences by,Images per sentence,unconditional_guidance_scale".split(",")
    for k,v,t,ts in zip(ks,vs,input_texts,_types):
      o[k]={'value':v,'input_text':t,'type_func':ts}
    return copy.deepcopy(o)
def update_params(params):
    ks = "input_str,output_dir,split_by,images_per_sentence,unconditional_guidance_scale".split(",")
    for k in ks:
      t = params[k]['input_text']
      v = params[k]['value']
      type_func = params[k]['type_func']
      m = "{} [Default: {}]:".format(t,v)
      tmp_v  = input(m).strip()
      if len(tmp_v)>0:
        params[k]['value'] = type_func(tmp_v)

def main():
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("using mixed precision")
    except:
        print("mixed precision not available")
    print("Loading model:")
    model = keras_cv.models.StableDiffusion(jit_compile=True)
    params = get_default_params()
    keep_generation = True
    while keep_generation:
        update_params(params)
        generate_and_save(model,params['input_str']['value'],
                      output_dir=params['output_dir']['value'],
                      split_by=params['split_by']['value'],
                      images_per_prompt=params['images_per_sentence']['value'],
                      unconditional_guidance_scale = params['unconditional_guidance_scale']['value'])
        print("Done.")
        stop = input("Stop generating? Y/N [Default N]")
        if lower(stop)=="y":
            keep_generation=False

def main_plain():
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("using mixed precision")
    except:
        print("mixed precision not available")
    print("Loading model:")
    model = keras_cv.models.StableDiffusion(jit_compile=True)
    params = get_default_params()
    keep_generation = True
    while keep_generation:
        update_params(params)
        print("Getting text enconding")
        images_per_prompt = params['images_per_sentence']['value']
        output_dir = params['output_dir']['value']
        unconditional_guidance_scale = params['unconditional_guidance_scale']['value']
        input_strs = params['input_str']['value'].split(params['split_by']['value'])
        prompts = [_.strip() for _ in input_strs if len(_)>0]
        for prompt in prompts:
            _encoding = tf.squeeze(model.encode_text(prompt))
            print("Getting images")
            result_images = model.generate_image(_encoding,
                                          batch_size=images_per_prompt,
                                          unconditional_guidance_scale=unconditional_guidance_scale)
            save_result(result_images,prompt,output_dir)
        print("Done.")
        stop = input("Stop generating? Y/N [Default N]")
        if stop.lower()=="y":
            keep_generation=False


if __name__ =="__main__":
    main()
