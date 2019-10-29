from random import random
import numpy as np
import soundfile as sf
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import (
    Audio, display, clear_output)


default_widgets_params = {
    'min': 0.1,
    'max': 1.0,
    'step': 0.01,
    'disabled': False,
    'continuous_update': False,
    'orientation': 'horizontal',
    'readout': True,
    'readout_format': '.2f',
}

attack_widgets_params = {
    'value': 0.05,
    'min': 0.01,
    'max': 0.4,
    'step': 0.01,
    'disabled': False,
    'continuous_update': False,
    'orientation': 'horizontal',
    'readout': True,
    'readout_format': '.2f',
    'description': 'Attack'
}

release_widgets_params = {
    'value': 0.5,
    'min': 0.05,
    'max': 0.6,
    'step': 0.01,
    'disabled': False,
    'continuous_update': False,
    'orientation': 'horizontal',
    'readout': True,
    'readout_format': '.2f',
    'description': 'Release'
}

DEFAULT_MODEL = 'log_kicks_high/'
model_widgets_params = {
    'options': [('Kicks norm', 'log_kicks/'), ('Kicks high', 'log_kicks_high/'), ('Kicks full', 'log_kicks_full'),
                ('FS norm', 'log_free/'), ('FS high', 'log_free_high/'), ('FS full', 'log_free_full/')],
    'value': DEFAULT_MODEL,
    'description': 'Model',
    'disabled': False,
}

CURRENT_MODEL_NAME = DEFAULT_MODEL


def generate_envelope(attack, release, sr=16000):
    envelope = []
    if attack + release <= 1 :
        envelope = np.concatenate((np.linspace(0, 1, attack*sr), np.linspace(1, 0, release*sr)))
        result = np.zeros(16000)
        result[:len(envelope)] =  envelope
    return result


def return_widgets(callback):
    model_name = widgets.Dropdown(**model_widgets_params)
    attack = widgets.FloatSlider(**attack_widgets_params)
    release = widgets.FloatSlider(**release_widgets_params)
    brightness = widgets.FloatSlider(description='Brightness', value=random(), **default_widgets_params)
    hardness = widgets.FloatSlider(description='Hardness', value=random(), **default_widgets_params)
    depth = widgets.FloatSlider(description='Depth', value=random(), **default_widgets_params)
    roughness = widgets.FloatSlider(description='Roughness', value=random(), **default_widgets_params)
    boominess = widgets.FloatSlider(description='Boominess', value=random(), **default_widgets_params)
    warmth = widgets.FloatSlider(description='Warmth', value=random(), **default_widgets_params)
    sharpness = widgets.FloatSlider(description='Sharpness', value=random(), **default_widgets_params)

    out = widgets.interactive_output(callback, {
        'model_name': model_name,
        'attack': attack,
        'release': release,
        'brightness': brightness,
        'hardness': hardness,
        'depth': depth,
        'roughness': roughness,
        'boominess': boominess,
        'warmth': warmth,
        'sharpness': sharpness
    })

    return model_name, attack, release, brightness, hardness, depth, roughness, boominess, warmth, sharpness, out

