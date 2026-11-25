// The real-time keyword spotting pipeline (tasks T1-T4) and the board-side commands.
#pragma once

#include "run_config.h"

// Map the `sequence` flash partition (the fixed 5-minute test sequence). False if it is missing.
bool pipeline_init(void);

// One experiment run: prints OK, then DET / TASK / RESULT lines and DONE (protocol.py).
// In demo mode it never returns.
void pipeline_run(const kws_cfg_t *cfg);

// Offline check of the whole chain without real time: MFCC + model on each 1 s clip of the
// sequence. Prints one EVAL line per clip and an EVAL_SUMMARY line.
void pipeline_eval(int model, bool psram);

// Load each model with a large PSRAM arena and print the arena it really uses.
void pipeline_arena(void);
