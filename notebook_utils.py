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


# not used, just here for keeping good example values
DEFAULT_KICKS_VALUES = [0.46533436, 0.6132435, 0.6906892, 0.5227648, 0.6955591, 0.733622, 0.4321724]
DEFAULT_FS_VALUES = [0.7279065, 0.7787189, 0.4476448, 0.7108625, 0.39033762, 0.4943667, 0.5660683]


def generate_envelope(attack, release, sr=16000):
    envelope = []
    if attack + release <= 1 :
        envelope = np.concatenate((np.linspace(0, 1, int(attack*sr)), np.linspace(1, 0, int(release*sr))))
        result = np.zeros(16000)
        result[:len(envelope)] =  envelope
    return result


def return_widgets(callback):
    model_name = widgets.Dropdown(**model_widgets_params)
    attack = widgets.FloatSlider(**attack_widgets_params)
    release = widgets.FloatSlider(**release_widgets_params)
    brightness = widgets.FloatSlider(description='Brightness', value=0.46, **default_widgets_params)
    hardness = widgets.FloatSlider(description='Hardness', value=0.61, **default_widgets_params)
    depth = widgets.FloatSlider(description='Depth', value=0.69, **default_widgets_params)
    roughness = widgets.FloatSlider(description='Roughness', value=0.52, **default_widgets_params)
    boominess = widgets.FloatSlider(description='Boominess', value=0.69, **default_widgets_params)
    warmth = widgets.FloatSlider(description='Warmth', value=0.73, **default_widgets_params)
    sharpness = widgets.FloatSlider(description='Sharpness', value=0.43, **default_widgets_params)

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

