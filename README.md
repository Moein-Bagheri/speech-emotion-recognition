# Speech emotion recognition

**Can a model hear emotion in a stranger's voice?** Less than it seems. A 1D CNN on MFCC features scores 66.7% on a random split of IEMOCAP, and 52.6% when the test speakers were never heard in training.

<img src="doc/experiments/speaker_plain_mfcc_head/confusion_matrix.png" width="440" alt="Confusion matrix on unseen speakers: angry, neutral and sad are recognized best; happy is mostly heard as neutral or sad">

*PyTorch · librosa · scikit-learn* · [Full report (Persian)](doc/report_fa.md) · [All results](doc/tables/experiment_results.md)

## What I did

- Took four emotions (angry, happy, neutral, sad) from IEMOCAP: 4,901 utterances from 10 speakers.
- Turned each clip into a fixed-length sequence of 40 MFCCs per frame, optionally with deltas.
- Built plain and residual 1D CNNs and ran six experiments: random against speaker-independent splits, feature sets, head or tail truncation, masked pooling.
- Held out two speakers entirely (974 test clips) to see what the model does with voices it has never heard.

## What I found

| Setting | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Random split, plain CNN | 66.7% | 0.63 |
| Unseen speakers, plain CNN | 52.6% | 0.44 |
| Unseen speakers, plain CNN with MFCC deltas | 49.3% | 0.46 |
| Unseen speakers, residual CNN (2.5× the parameters) | 55.0% | 0.49 |

- **Random splits flatter the model.** When the same speakers appear in training and test, the model can lean on voices it already knows. On new speakers, fourteen points disappear. Three training seeds agree: 66–67% against 53–54%.
- **Bigger helps, a little.** The residual network has 2.5 times the parameters and gains about three points on unseen speakers (55–57% against 53–54%), in all three seeds.
- **Accuracy can hide the hardest class.** *Happy* is the smallest class (13% of the data). On unseen speakers the plain CNN recognizes 3–14% of it across three seeds, and mostly hears it as *neutral* or *sad*.
- **A bug made the gap look smaller.** The first version read the speaker from the dialog's name instead of the utterance's, so test voices were also in training, and it reported 58.0%. The split and all six runs are redone (test speakers Ses02F and Ses03M).

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
