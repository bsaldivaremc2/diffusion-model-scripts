#Main code extracted from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/random_walks_with_stable_diffusion.ipynb#scrollTo=5SnL_YTJ32iS
#https://keras.io/examples/generative/random_walks_with_stable_diffusion/
import os
import math
import datetime
import argparse
import keras_cv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Generate images from keras_cv diffusion')
parser.add_argument('-text-description', type=str,default="A dog surfing. A cat playing guitar",dest="input_str",help='Write what you want to generate')
parser.add_argument('-split-sentence-by', type=str ,default=".",dest="split_by",help='Split each sentence by a delimiter. defaul dot .')
parser.add_argument('-output-dir', type=str,default="./results/",dest="output_dir",help='Where to save the results')
parser.add_argument('-images-per-sentence', type=int,default=1,dest="images_per_sentence",help='How many images generate per sentence written.')
parser.add_argument('-unconditional_guidance_scale', type=float,default=7.5,dest="unconditional_guidance_scale",help='How much the system will try to follow the text. Higher value the more strict. default 7.5')
args = parser.parse_args()

def get_images(model,prompt,images_per_prompt=1,unconditional_guidance_scale=7.5):
  _encoding = tf.squeeze(model.encode_text(prompt))
  _images = model.generate_image(_encoding,
                                  batch_size=images_per_prompt,
                                  unconditional_guidance_scale=unconditional_guidance_scale)
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

def main(args,model):
    generate_and_save(model,args.input_str,
                      output_dir=args.output_dir,
                      split_by=args.split_by,
                      images_per_prompt=args.images_per_sentence,
                      unconditional_guidance_scale = args.unconditional_guidance_scale)

if __name__ =="__main__":
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = keras_cv.models.StableDiffusion(jit_compile=True)
        main(args,model)
    except:
        model = keras_cv.models.StableDiffusion(jit_compile=True)
        main(args,model)
