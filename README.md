# Real-Time Voice Cloning With a Few Samples


This is a quick guideline for running the voice cloner pipeline.
This work implements and follows the technique introduced in [this paper](https://arxiv.org/abs/1806.04558).
The voice cloner system consists of three independent modules called: 1. Encoder, 2. Synthesizer, and 3. Vocoder.
Each module is performing a different task and can be trained sequentially (one after another) on different datasets.


## Environment and dependencies
`requirements.txt` and `requirements_gpu.txt` list the requirements with the proper version.
In addition to these requirements, make sure you have `ffmpeg` and `PyTorch>=1.01` installed.
The author's suggestion is to create a conda environment and install all the requirements within that environment.
Currently, the conda environment `voice_cloning` has all the dependencies installed.
So one can only run the following command:

```
conda activate voice_cloning
```

If you are not using conda, after installing ffmeg and PyTorch run either of the two following commands:
`pip install -r requirements.txt` or `pip install -r requirements_gpu.txt`.


## Encoder
The first and most important module of this voice cloning system is the encoder.
The file `encoder/params_data.py` lists all the parameteres of the encoedr preprocessing. Fill free to set your desired configurations.

Before training the encoder, the datasets must go through a preprocessing step which happens in `encoder_preprocess.py`
```
python encoder_preprocess.py <dataset_root> -o <output_dir> -d <datasets> -s <skip_existing> --no_trim
```


Arguments:

**datasets_root**: Path to the directory containing your LibriSpeech/TTS and VoxCeleb datasets.

**out_dir**: Path to the output directory that will contain the mel spectrograms. If left out, defaults to `<datasets_root>/SV2TTS/encoder/`.

**datasets**: Comma-separated list of the name of the datasets you want to preprocess. Only the train set of these datasets will be used. Possible names: librispeech_other, voxceleb1, voxceleb2, timit, vctk.

**skip_existing**: Whether to skip existing output files with the same name. Useful if this script was interrupted.

**no_trim**: Preprocess audio without trimming silences. (NOTE: not recommended, leave this to default)

This command preprocesses audio files from datasets, encodes them as mel spectrograms, and writes them to the disk.
Normally, the outputs are going to be stored in `<dataset_root/SV2TTS/encoder>`.
This will allow you to train the encoder. The datasets required are at least one of VoxCeleb1, VoxCeleb2, TIMIT, and LibriSpeech.
Ideally, you should have all three. You should extract them as they are after having downloaded them and put them in a same directory, e.g.:
-[datasets_root]\
&nbsp;&nbsp;**LibriSpeech**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-train-other-500\
&nbsp;&nbsp;**VoxCeleb1**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-wav\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-vox1_meta.csv\
&nbsp;&nbsp;**VoxCeleb2**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-dev\
&nbsp;&nbsp;**TIMIT**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-ALL

Currently, these datasets are stored with the mentioned structure under `/data/voice_cloning`.


After preprocessing the data for encoder, now you should be able to train the encoder model using the following command:
```
python encoder_train.py <run_id> <clean_data_root> -m <models_dir> -v <vis_every> -u <umap_every> -s <save_every> -b <backup_every> -f <force_restart> --visdom_server --novisdom 
```
The file `encoder/params_model.py` sets the hyper-parameters of the encoder model. All the descriptions are also included.

Arguments:

**run_id**: Name for this model instance. If a model state from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved states and restart from scratch.

**clean_data_root**: Path to the output directory of encoder_preprocess.py. If you left the default output directory when preprocessing, it should be `<datasets_root>/SV2TTS/encoder/`.

**models_dir**: Path to the output directory that will contain the saved model weights, as well as backups of those weights and plots generated during training.

**vis_every**: Number of steps between updates of the loss and the plots.

**umap_every**: Number of steps between updates of the umap projection. Set to 0 to never update the projections.

**save_every**: Number of steps between updates of the model on the disk. Set to 0 to never save the model.

**backup_every**: Number of steps between backups of the model. Set to 0 to never make backups of the model.

**force_restart**: Do not load any saved model.

**visdom_server**: [http://localhost](http://localhost). (NOTE: leave to default)

**no_visdom**: Disable visdom. (NOTE: leave to default)

Once the model is trained, you can see the saved weights under `<models_dir>/<run_id>.pt`. All the visualizations are then stored under `<models_dir>/<run_id>_backups`.


## Synthesizer

Since the synthesizer is receiving two input streams, we take two different preprocessing steps for this module.

1. Preprocess audio: Here, we preprocess audio files from datasets, encode them as mel spectrograms and write them to the disk. Audio files are also saved, to be used later by the vocoder for training. 

**NOTE: This step requires word alignments.**

This step is done using the following command:
```
python synthesizer_preprocess_audio.py <datasets_root> -o <out_dir> -n <n_processes> -s <skip_existing> --hparams --no_trim
```
Arguments:

**datasets_root**: Path to the directory containing your LibriSpeech/TTS datasets.

**out_dir**: Path to the output directory that will contain the mel spectrograms, the audios, and the embeds. Defaults to `<datasets_root>/SV2TTS/synthesizer/`.

**n_process**: Number of processes in parallel.

**skip_existing**: Whether to overwrite existing files with the same name. Useful if the preprocessing was interrupted.

**hparams**: Hyperparameter overrides as a comma-separated list of name-value pairs.

**no_trim**: Preprocess audio without trimming silences. (NOTE: not recommended)




After training the encoder, and doing the audio preprocessing for the synthesizer, we should be able to preprocess embedding data for training the synthesizer.

2. Preprocess embedding: In order to do the transfer learning on the module encoder, we use a trained encoder (with frozen parameters) and generate embeddings for the synthesizer.
This step can be done using the following command:
```
python synthesizer_preprocess_embeds.py <synthesizer_root> -e <encoder_model_fpath> -n <n_processes>
```
Arguments:

**synthesizer_root**: Path to the synthesizer training data that contains the audios and the train.txt file. 
If you let everything as default, it should be `<datasets_root>/SV2TTS/synthesizer/`.

**encoder_model_fpath**: Path your trained encoder model.

**n_processes**: Number of parallel processes. An encoder is created for each, so you may need to lower this value on GPUs with low memory. (NOTE: Set it to 1 if CUDA is unhappy)


once both preprocessing steps are successfully done, you must have `<datasets_root>/SV2TTS/synthesizer/mel/`, `<datasets_root>/SV2TTS/synthesizer/audios/`,  `<datasets_root>/SV2TTS/synthesizer/embeds/` and `<datasets_root>/SV2TTS/synthesizer/train.txt`. Then you are good to train the synthesizer as follows:

```
python synthesizer_train.py <name> <synthesizer_root> <models_dir> --mode --GTA --restore --summary_interval --embeding_interval --checkpoint_interval --eval_interval --tacotron_train_steps --tf_log_level --slack_url --hparams
```

Arguments:

**name**: Name of the run and the logging directory.

**synthesizer_root**: Path to the synthesizer training data that contains the audios and the train.txt file. If you let everything as default, it should be `<datasets_root>/SV2TTS/synthesizer/`.

**models_dir**: Path to the output directory that will contain the saved model weights and the logs.

**mode**: Mode for the synthesis of tacotron after training (NOTE: leave to default)

**GTA**: Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode. (NOTE: leave to default)

**restore**: Set this to False to do a fresh training.

**summary_interval**: Steps between running summary ops.

**embedding_interval**: Steps between updating embeddings projection visualization.

**checkpoint_interval**: Steps between writing checkpoints.

**eval_interval**: Steps between eval on test data.

**tacotron_train_steps**: Total number of tacotron training steps.

**tf_log_level**: Tensorflow C++ log level.

**slack_url**: Slack webhook notification destination link.

**hparams**: Hyperparameter overrides as a comma-separated list of name=value pairs.


Within `synthesizer/hparams.py` you can change the set of hyper-parameters. All the descriptions are also included.

## Vocoder

The last step is to preprocess and train the vocoder. It is necessary to preprocess and train the synthesizer to be able to pass this step.

Here we preprocess the data to create ground-truth aligned (GTA) spectrograms from the vocoder.
In order to do so, the following command must be run:
```
python vocoder_preprocess.py <datasets_root> --model_dir -i <in_dir> -o <-out_dir> --hparams --no_trim
```

Arguments: 

**dataset_root**: Path to the directory containing your SV2TTS directory. If you specify both --in_dir and --out_dir, this argument won't be used.

**model_dir**: Path to the pretrained model directory. The default should be `<datasets_root>/SV2TTS/synthesizer/saved_models/logs-pretrained/`.

**in_dir**: Path to the synthesizer directory that contains the mel spectrograms, the wavs, and the embeds. Defaults to `<datasets_root>/SV2TTS/synthesizer/`.

**out_dir**: Path to the output vocoder directory that will contain the ground truth aligned mel spectrograms. Defaults to `<datasets_root>/SV2TTS/vocoder/`.

**hparams**: Hyperparameter overrides as a comma-separated list of name=value pairs.

**no_trim**: Preprocess audio without trimming silences (NOTE: not recommended).

Once the proper data are generated within `<datasets_root>/SV2TTS/vocoder/`, we can train the vocoder as the last step.

```
python vocoder_train.py <run_id> <dataset_root> --syn_dir --voc_dir -m <-models_dir> -g <ground_truth> -s <save_every> -b <backup_every> -f <force_restart>
```

Arguments:

**run_id**: Name for this model instance. If a model state from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved states and restart from scratch.

**dataset_root**: Path to the directory containing your SV2TTS directory. Specifying `--syn_dir` or `--voc_dir` will take priority over this argument.

**syn_dir**: Path to the synthesizer directory that contains the ground truth mel spectrograms, the wavs, and the embeds. Defaults to `<datasets_root>/SV2TTS/synthesizer/`.

**voc_dir**: Path to the vocoder directory that contains the GTA synthesized mel spectrograms. Defaults to `<datasets_root>/SV2TTS/vocoder/`. **Unused if --ground_truth is passed**.

**models_dir**: Path to the directory that will contain the saved model weights, as well as backups of those weights and wavs generated during training.

**ground_truth**: Train on ground truth spectrograms which are under `<datasets_root>/SV2TTS/synthesizer/mels`.

**save_every**: Number of steps between updates of the model on the disk. Set to 0 to never save the model.

**backup_every**: Number of steps between backups of the model. Set to 0 to never make backups of the model.

**force_restart**: Do not load any saved model and restart from scratch.

## Cloning voice with given references

After training all three modules, you can clone a reference voice and vocalize any given input text.
This process can be done using `all_together.py` script as follows:

```
python all_together.py -e <enc_model_fpath> -s <syn_model_dir> -v <voc_model_fpath> -i <input_dir> -o <output_dir>
```

Arguments:
**enc_model_fpath**: Path to a saved encoder model. Default is `encoder/saved_models/pretrained.pt`.

**syn_model_dir**: Path to a directory contataining synthesizer logs and models. Default is `synthesizer/saved_models/logs-pretrained/`.

**voc_model_fpath**: Path to a saved vocoder. Default is `vocoder/saved_models/pretrained/pretrained.pt`.

**input_dir**: Path to a directory consisting of audio references. Each file must be in `.wav` format.

**outpu_dir**: Path to a directory where you want to save you cloned (synthesized) samples.

Within `vocoder/hparams.py` you can change the set of hyper-parameters. All the descriptions are also included.

## Data
In this section, we briefly explain the datasets that have been used in this project so far (as of September 2020)
The publicly available datasets are:
1. [LibriSpeech](http://www.openslr.org/12)\
&nbsp;&nbsp;&nbsp;&nbsp;clean-100\
&nbsp;&nbsp;&nbsp;&nbsp;clean-360\
&nbsp;&nbsp;&nbsp;&nbsp;other-500
2. [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
3. [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
4. [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
5. [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)

### Encoder
For training the encoder, we used all the aforementioned datasets.
Since this is a text-independent task, there is no need to provide alignments.

### Synthesizer
For training the synthesizer, this is important to use clean utterances. 
Other than that, the limited time did not allow us to go further than using 
only libri-clean-100 and libri-clean-360.
Training synthesizer requires word-level alignments. These alignments can be generated using [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/) software. Then the outputs must be reformated
to the same format as libriSpeech alignments. More details on this are included in the alignment section.

It is necessary to note that the main paper had used libri-100, 360, and VCTK to train the synthesizer. So, there is no need to add more datasets on an
English language-based voice cloner.

### Vocoder
For training WaveNet vocoder, libri-100 and libri-360 were used. The same datasets were used in the main paper.
