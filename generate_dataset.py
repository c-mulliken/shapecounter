from PIL import Image, ImageDraw
import numpy as np
import random
import sys
import os
import json

class Square:
    def __init__(self, corner, side):
        self.corner = corner
        self.side = side

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

PATH = '/Users/cobymulliken/Desktop/Repositories/shapecounter/dataset/images'

def render(shapes, filename):
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
        elif isinstance(shape, Triangle):
            vertex1, vertex2, vertex3 = shape.vertex1, shape.vertex2, shape.vertex3
            draw.polygon([vertex1, vertex2, vertex3], fill="black")

    os.makedirs(PATH, exist_ok=True)
    image.save(f'{PATH}/{filename}.png')

def create_shapes(num_circles, num_squares):
    shapes = []
    for i in range(num_circles):
        cent_x = random.randint(0