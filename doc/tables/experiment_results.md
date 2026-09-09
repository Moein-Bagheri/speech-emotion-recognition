| experiment | split | model | features | truncation | masked_GAP | accuracy | weighted_F1 | macro_F1 | best_epoch | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_plain_mfcc_head | random | plain | mfcc | head | False | 0.6514 | 0.6516 | 0.6224 | 25 | 198724 |
| speaker_plain_mfcc_delta2_head | speaker | plain | mfcc_delta2 | head | False | 0.6004 | 0.563 | 0.4857 | 2 | 214084 |
| speaker_plain_mfcc_head | speaker | plain | mfcc | head | False | 0.5803 | 0.5718 | 0.5204 | 16 | 198724 |
| speaker_plain_mfcc_head_masked | speaker | plain | mfcc | head | True | 0.556 | 0.5449 | 0.4937 | 2 | 198724 |
| speaker_plain_mfcc_tail | speaker | plain | mfcc | tail | False | 0.5687 | 0.5659 | 0.5171 | 16 | 198724 |
| speaker_residual_mfcc_head | speaker | residual | mfcc | head | False | 0.5729 | 0.5371 | 0.471 | 3 | 502532 |
