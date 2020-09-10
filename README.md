# Real Time Voice Cloning With a Few Samples


This is a quick guidline for running the voice cloner pipeline.
This work implements and follows the technique introduced in [this paper](https://arxiv.org/abs/1806.04558).
The voice cloner system consits of three independent modules called: 1. Encoder, 2. Synthesizer, and 3. Vocoder.
Each module is performing a differnet task and can be trained sequantially (one after another) on different datasets.

## Encoder
The first and most important module of this voice cloning system is the encoder.
Before training encoder, the datasets must go through a preprocessing step which happens in `encoder_preprocess.py`
```

```











## Data
In this section we briefly explain the datasets that have been used in this project so far (as of September 2020)
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
Since this is a text independent task, there is no need to provide alignments.

### Synthesizer
For training the synthesizer, this is important to use clean utterances. 
Other than that, the limited time did not allow us to go further than using 
only libri-clean-100 and libri-clean-360.
Training synthesizer, requires word level alignements. These alignements can 
be generated using [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/) software. Then the outputs must be reformated
to the same format as libriSpeech alignements. More details on this are included in the alignement section.

It is necessary to note that the main paper had used libri-100, 360, and VCTK to train the synthesizer. So, there is no need to add more datasets on an
English language based voice cloner.

### Vocoder
For training WaveNet vocoder, libri-100 and libri-360 were used. Same datasets were used in the main paper.








![](header.png)

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
