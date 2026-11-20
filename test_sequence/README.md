# test_sequence

Builds the fixed test sequence that every experiment run plays back, so all treatments see exactly the same audio.

| | |
| --- | --- |
| Source | Speech Commands v2, **test split only** (never seen in training) |
| Content | 100 keywords (10 per class) + 50 unknown words spread evenly over the 25 non-keyword words |
| Speakers | one utterance per speaker (150 different speakers) |
| Order | shuffled with seed `2026` (fixed for the whole project) |
| Layout | `[1 s gap][1 s utterance slot]` × 150 = **300 s** over a constant −60 dBFS noise floor |
| Format | 16 kHz, 16-bit little-endian mono, 9.6 MB (fits the 10 MB `sequence` partition) |

## Build (Colab)

Open [`colab_sequence.ipynb`](colab_sequence.ipynb) in Colab (CPU runtime) and *Run all*. It downloads `kws_sequence.zip`.

Locally (needs only numpy; the dataset is ~2.3 GB):

```bash
python build_sequence.py --download --data-dir ../training/data/speech_commands
python -m unittest discover -s tests
```

Output goes to `test_sequence/out/` (not versioned):

| File | Use |
| --- | --- |
| `sequence.raw` | audio written to the `sequence` flash partition |
| `sequence.wav` | the same audio, to listen to |
| `labels.csv` | ground truth: `label`, `start`, `end`, `voiced_start`, `voiced_end` (samples), `source` |
| `kws_sequence_labels.h` | the same table as C constants; T1 pulses the marker pin (GPIO 15) at each `voiced_end` |
| `manifest.json` | parameters, counts, size and SHA-256 of `sequence.raw` |

`start`/`end` delimit the clip; `voiced_start`/`voiced_end` estimate the spoken word inside it: the run of 20 ms frames around the loudest one that are within 25 dB of it and at least 10 dB above the clip's own noise floor, bridging pauses up to 100 ms. On the real sequence the words last 350 ms (median; 200–560 ms for 80 % of them); 9 clips are shorter than 150 ms because the dataset recording itself is truncated or nearly silent, and they are kept as they are. **Detection latency is measured from `voiced_end`**; any bias of this estimate is the same for every treatment, so it cancels out when treatments are compared.

The committed build is [`labels.csv`](labels.csv), [`manifest.json`](manifest.json) and [`firmware/main/kws_sequence_labels.h`](../firmware/main/kws_sequence_labels.h). `sequence.raw` is not versioned: rebuild it with the notebook (same seed) and check that its SHA-256 matches `manifest.json`:

```
b8220a2fba3f6d9e3d2f189a93ab3dd4927040da91c1ef02d5abe0eeefbedaca
```

`python build_sequence.py --relabel` recomputes the voiced ranges of an existing `out/` from `sequence.raw` alone (no dataset).

## Flash

From an ESP-IDF terminal, after flashing the firmware (the partition table must already be on the board):

```bash
python $IDF_PATH/components/partition_table/parttool.py --port COMx \
    --partition-table-file firmware/build/partition_table/partition-table.bin \
    write_partition --partition-name sequence --input test_sequence/out/sequence.raw
```

The partition is only rewritten when the sequence changes; reflashing the app does not erase it.
