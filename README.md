<h1>NeuroDrum</h1>

<h4>António Ramires, Pritish Chandna, Xavier Favory, Emilia Gomez, Xavier Serra</h2>

<h4>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for NeuroDrum, a parametric percussion synthesis using the [Wave-U-Net](https://github.com/f90/Wave-U-Net) architecture. The syntheser is controlled using only high-level timbral characteristics: the envelope and the sounds hardness, depth, brightness, roughness, boominess, warmth and sharpness.

<h3>Installation</h3>
To install NeuroDrum and its dependencies, clone the repository and use: 
<pre><code>pip install -r percussive_synth/requirements.txt </code></pre>

Then, you will have to download the [model weights](TODO) which you will link on the generation process. 

<h3>Generation</h3>

Sounds can be generated from the command line, or within Python. 

To generate from the command line, the following commands should be ran:

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

To prepare the data for use, please use *prep_data_nus.py*.


Once setup, you can run the following commands. 
To train the model: 
<pre><code>python main.py -t</code></pre>. 
To synthesize a .lab file:
Use <pre><code>python main.py -e <i>filename</i> <i>alternate_singer_name</i> </code></pre> 

If no alternate singer is given then the original singer will be used for synthesis. A list of valid singer names will be displayed if an invalid singer is entered. 

You will also be prompted on wether plots showed be displayed or not, press *y* or *Y* to view plots.



<h2>Acknowledgments</h2>
This work is partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No765068, MIP-Frontiers.
This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project.
The TITANX used for this research was donated by the NVIDIA Corporation. 
