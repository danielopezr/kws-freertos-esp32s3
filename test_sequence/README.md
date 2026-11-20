# test_sequence

Builds the fixed test sequence that every run receives: 5 minutes with 150 utterances from the
test split (100 keywords and 50 unknown words) separated by silence, plus a label file with the
start and end of each utterance.

The sequence (16 kHz, 16-bit, about 9.6 MB) is written to a Flash partition on the ESP32-S3.
Output goes to `test_sequence/out/` (not versioned).
