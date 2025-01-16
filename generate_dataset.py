from PIL import Image, ImageDraw
import numpy as np
import random
import sys
import os
import json
from tqdm import tqdm

class Square:
    def __init__(self, corner, side):
        self.corner = corner
        self.side = side

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

# PATH = '/home/coby/Repositories/shapecounter/dataset/images'

def render(shapes, filename, path):
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)

    for shape in shapes:
        if isinstance(shape, Square):
            x1, y1 = shape.corner
            x2 = x1 + shape.side
            y2 = y1 + shape.side
            draw.rectangle([(x1, y1), (x2, y2)], fill="black")
        elif isinstance(shape, Circle):
            x, y = shape.center
            radius = shape.radius
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill="black")

    os.makedirs(path, exist_ok=True)
    image.save(f'{path}/{filename}.png')

def create_shapes(num_circles, num_squares):
    shapes = []
    for i in range(num_circles):
        cent_x = random.randint(0, 256)
        cent_y = random.randint(0, 256)
        radius = random.randint(2, 15)
        shapes.append(Circle((cent_x, cent_y), radius))
    for i in range(num_squares):
        corn_x = random.randint(0, 256)
        corn_y = random.randint(0, 256)
        side = random.randint(2, 20)
        shapes.append(Square((corn_x, corn_y), side))
    return shapes

def create_dataset(num_images,
                   mean_num_shapes, std_num_shapes,
                   mean_prop_circles, std_prop_circles,
                   path):
    info_dict = {}
    for i in tqdm(range(num_images)):
        num_shapes = max(int(np.random.normal(mean_num_shapes, std_num_shapes)), 1)
        prop_circles = max(np.random.normal(mean_prop_circles, std_prop_circles), 0)
        num_circles = int(num_shapes * prop_circles)
        num_squares = num_shapes - num_circles
        shapes = create_shapes(num_circles, num_squares)
        render(shapes, f'image_{i}', path)
        info_dict[f'image_{i}'] = (num_circles, num_squares)
    with open(f"{path}/image_data.json", "w") as f:
        json.dump(info_dict, f)

def create_discrim_dataset(num_images, num_shapes, margin, path):
    info_dict = {}
    for i in tqdm(range(num_images)):
        num_circles = int(num_shapes / 2 + margin / 2)
        num_squares = num_shapes - num_circles
        shapes = create_shapes(num_circles, num_squares)
        render(shapes, f'image_{i}', path)
        info_dict[f'image_{i}'] = (num_circles, num_squares)
    with open(f"{path}/image_data.json", "w") as f:
        json.dump(info_dict, f)

# create_dataset(5000, 30, 15, 0.5, 0.25, 'dataset/counting_experiment')