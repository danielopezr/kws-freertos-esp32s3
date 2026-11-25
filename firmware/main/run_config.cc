#include "run_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct cfg_key_t {
    const char *name;
    const char *const *values;  // allowed values; their index is stored
    int n;
};

const char *const kModels[] = {"dnn", "cnn", "dscnn"};
const char *const kAffinity[] = {"same", "split", "none"};
const char *const kPrio[] = {"lower", "equal", "higher"};
const char *const kOnOff[] = {"off", "on"};
const char *const kMem[] = {"sram", "psram"};
const char *const kAlloc[] = {"static", "dynamic"};
const char *const kFrag[] = {"0", "1", "2"};
const char *const kSeq[] = {"none", "full"};

// Same order as PARAMS in experiments/protocol.py
const cfg_key_t kKeys[] = {
    {"model", kModels, 3}, {"affinity", kAffinity, 3}, {"prio", kPrio, 3},
    {"load", kOnOff, 2},   {"mem", kMem, 2},           {"alloc", kAlloc, 2},
    {"frag", kFrag, 3},    {"seq", kSeq, 2},           {"demo", kOnOff, 2},
};
constexpr int kNKeys = sizeof(kKeys) / sizeof(kKeys[0]);

}  // namespace

bool kws_parse_cfg(const char *line, kws_cfg_t *cfg, char *err, size_t errlen)
{
    char buf[256];
    if (strncmp(line, "CFG", 3) != 0) {
        snprintf(err, errlen, "not a CFG line");
        return false;
    }
    strncpy(buf, line + 3, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    int value[kNKeys];
    bool seen[kNKeys] = {};
    bool seed_seen = false;
    *cfg = kws_cfg_t{};
    cfg->threshold = KWS_DEFAULT_THRESHOLD;

    char *save = nullptr;
    for (char *tok = strtok_r(buf, " \t\r\n", &save); tok; tok = strtok_r(nullptr, " \t\r\n", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) {
            snprintf(err, errlen, "expected key=value, got %.32s", tok);
            return false;
        }
        *eq = '\0';
        const char *k = tok, *v = eq + 1;
        if (strcmp(k, "seed") == 0) {
            cfg->seed = (uint32_t)strtoul(v, nullptr, 10);
            seed_seen = true;
            continue;
        }
        if (strcmp(k, "thr") == 0) {
            cfg->threshold = strtof(v, nullptr);
            if (!(cfg->threshold > 0.0f && cfg->threshold <= 1.0f)) {
                snprintf(err, errlen, "thr=%.16s out of (0, 1]", v);
                return false;
            }
            continue;
        }
        int i = 0;
        while (i < kNKeys && strcmp(k, kKeys[i].name) != 0) {
            i++;
        }
        if (i == kNKeys) {
            snprintf(err, errlen, "unknown key %.32s", k);
            return false;
        }
        int j = 0;
        while (j < kKeys[i].n && strcmp(v, kKeys[i].values[j]) != 0) {
            j++;
        }
        if (j == kKeys[i].n) {
            snprintf(err, errlen, "%s=%.16s not allowed", k, v);
            return false;
        }
        value[i] = j;
        seen[i] = true;
    }
    for (int i = 0; i < kNKeys; i++) {
        if (!seen[i]) {
            snprintf(err, errlen, "missing key %s", kKeys[i].name);
            return false;
        }
    }
    if (!seed_seen) {
        snprintf(err, errlen, "missing key seed");
        return false;
    }
    cfg->model = value[0];
    cfg->affinity = value[1];
    cfg->prio = value[2];
    cfg->load = value[3] == 1;
    cfg->psram = value[4] == 1;
    cfg->dynamic_alloc = value[5] == 1;
    cfg->frag = value[6];
    cfg->seq_full = value[7] == 1;
    cfg->demo = value[8] == 1;
    return true;
}
