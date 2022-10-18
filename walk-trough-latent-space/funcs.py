import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from PIL import Image

def get_noise(iseed=12345):
    return tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


# Enable mixed precision
try:#decorator
keras.mixed_precision.set_global_policy("mixed_float16")

model = keras_cv.models.StableDiffusion(jit_compile=True)

prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
interpolation_steps = 5


encoding_1 = tf.squeeze(model.encode_text(prompt_1))
encoding_2 = tf.squeeze(model.encode_text(prompt_2))

interpolated_encodings = tf.linspace(encoding_1, encoding_2, interpolation_steps)
print(f"Encoding shape: {encoding_1.shape}")

seed = 12345
noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

images = model.generate_image(
    interpolated_encodings,
    batch_size=interpolation_steps,
    diffusion_noise=noise,
)


interpolation_steps = 150
batch_size = 3
batches = interpolation_steps // batch_size
noise = get_noise(iseed=12345)

interpolated_encodings = tf.linspace(encoding_1, encoding_2, interpolation_steps)
batched_encodings = tf.split(interpolated_encodings, batches)

images = []
for batch in range(batches):
    images += [
        Image.fromarray(img)
        for img in model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=noise,
        )
    ]



export_as_gif("doggo-and-fruit-150.gif", images, rubber_band=True)
export_as_gif(
    "doggo-and-fruit-5.gif",
    [Image.fromarray(img) for img in images],
    frames_per_second=2,
    rubber_band=True,
)


prompt_1 = "a dragon temple digital art"
num_of_images = 5

encoding_1 = tf.squeeze(model.encode_text(prompt_1))
images = model.generate_image(encoding_1,batch_size=num_of_images)
