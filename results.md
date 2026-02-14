# Memory Results

## Uniform

´´´
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_ur_tasks_merged_lora_r16_uniform --exp-name=r16_uniform_0 --overwrite
´´´

### Rank 16 | uniform

09:57:47.677 [I] Parameter summary: total=3,276,976,912, trainable=456,980,240 (13.95%), frozen=2,819,996,672 (1614:train.py:77)
09:57:47.677 [I] Memory (params only): total=6.96 GiB, trainable=1.70 GiB, frozen=5.25 GiB        (1614:train.py:84)
09:57:47.677 [I] Memory (optimizer state): 3.40 GiB                                               (1614:train.py:90)

### Rank 32 | uniform

10:13:08.122 [I] Parameter summary: total=3,315,905,296, trainable=495,908,624 (14.96%), frozen=2,819,996,672 (957:train.py:77)
10:13:08.122 [I] Memory (params only): total=7.10 GiB, trainable=1.85 GiB, frozen=5.25 GiB        (957:train.py:84)
10:13:08.122 [I] Memory (optimizer state): 3.69 GiB                                               (957:train.py:90)

### Rank 64 | uniform

10:21:31.614 [I] Parameter summary: total=3,393,762,064, trainable=573,765,392 (16.91%), frozen=2,819,996,672 (913:train.py:77)
10:21:31.615 [I] Memory (params only): total=7.39 GiB, trainable=2.14 GiB, frozen=5.25 GiB        (913:train.py:84)
10:21:31.615 [I] Memory (optimizer state): 4.27 GiB                                               (913:train.py:90)
