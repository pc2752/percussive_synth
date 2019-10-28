<h1>NeuroDrum</h1>

<h4>António Ramires, Pritish Chandna, Xavier Favory, Emilia Gomez, Xavier Serra</h2>

<h4>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for NeuroDrum, a parametric percussion synthesis using the [Wave-U-Net](https://github.com/f90/Wave-U-Net) architecture. The syntheser is controlled using only high-level timbral characteristics: the envelope and the sounds hardness, depth, brightness, roughness, boominess, warmth and sharpness. An interactive example of this synthesiser is available [here](TODO) and a selected set of sound examples [here](TODO).

<h3>Installation</h3>
To install NeuroDrum and its dependencies, clone the repository and use: 
<pre><code>pip install -r percussive_synth/requirements.txt </code></pre>

Then, you will have to download the [model weights](TODO) which you will link on the generation process. 

<h3>Generation</h3>

Sounds can be generated within Python. 

The following example shows how to generate and save a sound from NeuroDrum
```python
# Import the models module and create an instance of it, this part should only be ran once
import models
model = models.PercSynth()

# Load one of the pre-trained models
sess = model.load_sess(log_dir="/percussive_synth/log_free_full/")

# Generate the sound:
# env should have 16000 elements from 0 to 1
# parameters should be an array with values from 0 to 1 corresponding to each of the following features:
# ['brightness', 'hardness', 'depth', 'roughness', 'boominess', 'warmth', 'sharpness']
output = model.get_output(envelope, parameters , sess)
sf.write('audio.wav', output, 16000)
```

<h3>Training</h3>

If you would like to train a model with a private dataset, the following steps should be taken:

The sounds should be in 16kHz sample rate and be cut or padded to have 1 second of length.
Perform the analysis of the dataset using the [ac-audio-extractor](https://github.com/AudioCommons/ac-audio-extractor).

Prepare the data for use, set the `wav_dir` and the `ana_dir` in the [config.py](config.py) and run [prep_data.py](prep_data.py).

Once setup, you can run the following command to train the model: 
<pre><code>python main.py -t</code></pre>

To generate examples from the validation set from the command line, the following command can be used:

```
python main.py -e
```


<h2>Acknowledgments</h2>
This work is partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No765068, [MIP-Frontiers](https://mip-frontiers.eu).

This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project.

The TITANX used for this research was donated by the NVIDIA Corporation. 
