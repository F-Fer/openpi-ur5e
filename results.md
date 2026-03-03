# Memory Results

## Uniform

### FFT

{'total_params': 3238048528, 'trainable_params': 3238048528, 'params_gib': 12.06, 'opt_gib': 24.13, 'ema_gib': 12.06, 'bytes_in_use_gib': 36.21860647201538}


### Rank 8 | uniform

07:05:36.374 [I] Parameter summary: total=3,257,512,720, trainable=437,516,048 (13.43%), frozen=2,819,996,672 (1551:train.py:77)
07:05:36.375 [I] Memory (params only): total=6.88 GiB, trainable=1.63 GiB, frozen=5.25 GiB        (1551:train.py:84)
07:05:36.375 [I] Memory (optimizer state): 3.26 GiB                                               (1551:train.py:90)

### Rank 16 | uniform

07:11:02.373 [I] Parameter summary: total=3,276,976,912, trainable=456,980,240 (13.95%), frozen=2,819,996,672 (8126:train.py:77)
07:11:02.374 [I] Memory (params only): total=6.96 GiB, trainable=1.70 GiB, frozen=5.25 GiB        (8126:train.py:84)
07:11:02.374 [I] Memory (optimizer state): 3.40 GiB                                               (8126:train.py:90)

### Rank 32 | uniform

07:17:08.571 [I] Parameter summary: total=3,315,905,296, trainable=495,908,624 (14.96%), frozen=2,819,996,672 (7416:train.py:77)
07:17:08.572 [I] Memory (params only): total=7.10 GiB, trainable=1.85 GiB, frozen=5.25 GiB        (7416:train.py:84)
07:17:08.573 [I] Memory (optimizer state): 3.69 GiB                                               (7416:train.py:90)

### Rank 64 | uniform

07:15:18.672 [I] Parameter summary: total=3,393,762,064, trainable=573,765,392 (16.91%), frozen=2,819,996,672 (8193:train.py:77)
07:15:18.673 [I] Memory (params only): total=7.39 GiB, trainable=2.14 GiB, frozen=5.25 GiB        (8193:train.py:84)
07:15:18.673 [I] Memory (optimizer state): 4.27 GiB                                               (8193:train.py:90)

## Rank 128 | uniform

06:46:14.327 [I] Parameter summary: total=3,549,475,600, trainable=729,478,928 (20.55%), frozen=2,819,996,672 (2146:train.py:77)
06:46:14.327 [I] Memory (params only): total=7.97 GiB, trainable=2.72 GiB, frozen=5.25 GiB        (2146:train.py:84)
06:46:14.327 [I] Memory (optimizer state): 5.44 GiB                                               (2146:train.py:90)

## Rank 256 | uniform

06:49:28.312 [I] Parameter summary: total=3,860,902,672, trainable=1,040,906,000 (26.96%), frozen=2,819,996,672 (945:train.py:77)
06:49:28.312 [I] Memory (params only): total=9.13 GiB, trainable=3.88 GiB, frozen=5.25 GiB        (945:train.py:84)
06:49:28.313 [I] Memory (optimizer state): 7.76 GiB                                               (945:train.py:90)


## Asymmetric

### Action Expert Rank 16 | VLM Rank 128

07:08:09.752 [I] Parameter summary: total=3,472,061,200, trainable=652,064,528 (18.78%), frozen=2,819,996,672 (1728:train.py:77)
07:08:09.752 [I] Memory (params only): total=7.68 GiB, trainable=2.43 GiB, frozen=5.25 GiB        (1728:train.py:84)
07:08:09.753 [I] Memory (optimizer state): 4.86 GiB                                               (1728:train.py:90)

### Action Expert Rank 128| VLM Rank 16

07:09:10.369 [I] Parameter summary: total=3,354,391,312, trainable=534,394,640 (15.93%), frozen=2,819,996,672 (2251:train.py:77)
07:09:10.369 [I] Memory (params only): total=7.24 GiB, trainable=1.99 GiB, frozen=5.25 GiB        (2251:train.py:84)
07:09:10.369 [I] Memory (optimizer state): 3.98 GiB                                               (2251:train.py:90)

### Freeze VLM | Action Expert FFT

07:33:33.398 [I] Parameter summary: total=3,238,048,528, trainable=314,713,120 (9.72%), frozen=2,923,335,408 (918:train.py:77)
07:33:33.399 [I] Memory (params only): total=6.62 GiB, trainable=1.17 GiB, frozen=5.45 GiB        (918:train.py:84)
07:33:33.399 [I] Memory (optimizer state): 2.34 GiB                                               (918:train.py:90)

## SigLIP Abilation

### Apply LoRA to SigLIP

14:03:23.972 [I] Parameter summary: total=3,325,333,264, trainable=90,532,896 (2.72%), frozen=3,234,800,368 (6688:train.py:77)
14:03:23.972 [I] Memory (params only): total=6.36 GiB, trainable=345.36 MiB, frozen=6.03 GiB      (6688:train.py:84)
14:03:23.972 [I] Memory (optimizer state): 690.71 MiB                                             (6688:train.py:90)

### Freeze SigLIP entirely

14:17:08.409 [I] Parameter summary: total=3,315,905,296, trainable=81,104,928 (2.45%), frozen=3,234,800,368 (937:train.py:77)
14:17:08.410 [I] Memory (params only): total=6.33 GiB, trainable=309.39 MiB, frozen=6.03 GiB      (937:train.py:84)
14:17:08.410 [I] Memory (optimizer state): 618.78 MiB                                             (937:train.py:90)
