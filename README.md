# Speech emotion recognition

**Can a model hear emotion in a stranger's voice?** Less than it seems. A 1D CNN on MFCC features scores 65.1% on a random split of IEMOCAP, and 58.0% when the test speakers were never heard in training.

<img src="doc/experiments/speaker_plain_mfcc_head/confusion_matrix.png" width="440" alt="Confusion matrix on unseen speakers: angry and neutral are recognized best, happy is mostly heard as neutral">

*PyTorch · librosa · scikit-learn* · [Full report (Persian)](doc/report_fa.md) · [All results](doc/tables/experiment_results.md)

## What I did

- Took four emotions (angry, happy, neutral, sad) from IEMOCAP: 4,901 utterances from 10 speakers.
- Turned each clip into a fixed-length sequence of 40 MFCCs per frame, optionally with deltas.
- Built plain and residual 1D CNNs and ran six experiments: random against speaker-independent splits, feature sets, head or tail truncation, masked pooling.
- Held out two speakers entirely (946 test clips) to see what the model does with voices it has never heard.

## What I found

| Setting | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Random split, plain CNN | 65.1% | 0.62 |
| Unseen speakers, plain CNN | 58.0% | 0.52 |
| Unseen speakers, plain CNN with MFCC deltas | 60.0% | 0.49 |
| Unseen speakers, residual CNN (2.5× the parameters) | 57.3% | 0.47 |

- **Random splits flatter the model.** When the same speakers appear in training and test, the model can lean on voices it already knows. On new speakers, seven points disappear.
- **Bigger isn't better here.** The residual network has 2.5 times the parameters and does worse.
- **Accuracy can hide the hardest class.** *Happy* is the smallest class (13% of the data). On unseen speakers the plain CNN hears it as *neutral* half the time, and with deltas, accuracy goes up while the model stops predicting *happy* at all.

## Try it

The trained speaker-independent model is included, so you can run it on your own WAV file without downloading the dataset:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
.\.venv\Scripts\python.exe code\predict.py doc\experiments\speaker_plain_mfcc_head\best_checkpoint.pt path\to\sample.wav
```

To rebuild everything from the data:

```powershell
.\.venv\Scripts\python.exe code\download_data.py
.\.venv\Scripts\python.exe code\prepare_data.py
.\.venv\Scripts\python.exe code\extract_features.py --workers 4
.\.venv\Scripts\python.exe code\run_experiments.py --include-optional
.\.venv\Scripts\python.exe code\summarize_results.py
```

## Layout

| Path | What's there |
| --- | --- |
| [`code/ser/`](code/ser/) | Data, features, models and training |
| [`code/predict.py`](code/predict.py) | The emotion of one WAV file |
| [`doc/experiments/`](doc/experiments/) | Six runs: metrics, curves, confusion matrices and best checkpoints |
| [`doc/report_fa.md`](doc/report_fa.md) | The full analysis, in Persian |

Natural language processing course project, University of Isfahan, 2026.
