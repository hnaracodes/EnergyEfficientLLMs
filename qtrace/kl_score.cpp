// Teacher-forced KL(P||Q) between two GGUF models over stored token sequences. Driven by
// qtrace/kl_gguf.py, which documents the build command.
//
// Usage: kl_score [--cpu] [--top-k K] P.gguf Q.gguf JOBS [n_batch]
//
// JOBS has one run per line, "<tokens> <idx> <out>" (paths without spaces). tokens holds int32 token
// ids; idx holds sorted, unique int32 positions to score, where the logit at position i predicts
// token i + 1. out receives one row per idx entry: float32 kl, flip, entropy_p, nll_p, nll_q, the
// fields of compare() in qtrace/kl.py, then with --top-k K the K most likely token ids under P (int32)
// and their log-probabilities (float32), most likely first. Both models load once and every run
// reuses them, because a load costs more than scoring a short run.
//
// --cpu keeps every layer and op on the CPU backend. Flash attention is pinned on and the KV cache is
// llama.cpp's default F16, so the scores do not depend on llama.cpp's auto choices. The settings used
// are printed to stdout as one JSON line after both models load.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static void die(const std::string & msg) {
    fprintf(stderr, "kl_score: %s\n", msg.c_str());
    fflush(stderr);
    // _exit, not exit: exit runs the backend destructors, and Metal's asserts that no buffers are
    // still allocated, so every error became a SIGABRT (exit code 134) instead of 1.
    _exit(1);
}

static std::vector<int32_t> read_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot read " + path);
    const std::streamoff n = f.tellg();
    if (n % 4) die(path + " is not a whole number of int32 values");
    std::vector<int32_t> v(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), n);
    if (!f) die("short read on " + path);
    return v;
}

struct Model {
    llama_model * model;
    llama_context * ctx;
};

static ggml_backend_dev_t no_devices[] = {nullptr};

static llama_context_params context_params(uint32_t n_ctx, uint32_t n_batch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 1;
    cp.no_perf = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return cp;
}

static Model load(const char * path, const llama_context_params & cp, bool cpu) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cpu ? 0 : -1;  // all layers
    if (cpu) mp.devices = no_devices;  // n_gpu_layers 0 alone still offloads large matmuls to a GPU
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) die(std::string("cannot load ") + path);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die(std::string("cannot create a context for ") + path);
    return {model, ctx};
}

// Identical token lists, so scoring the same ids under both models means the same text.
static void check_same_vocab(const llama_vocab * a, const llama_vocab * b) {
    const int n = llama_vocab_n_tokens(a);
    if (n != llama_vocab_n_tokens(b) || llama_vocab_type(a) != llama_vocab_type(b)) {
        die("P and Q have different vocabularies (size or type)");
    }
    for (int i = 0; i < n; ++i) {
        if (std::string(llama_vocab_get_text(a, i)) != llama_vocab_get_text(b, i)) {
            die("P and Q have different vocabularies: token " + std::to_string(i) + " differs");
        }
    }
}

// The compare() fields for one position, then the top_k ids and log-probabilities of a. Sums over
// the vocabulary run in double.
static void row_stats(const float * a, const float * b, int n_vocab, int y, int top_k, float * out) {
    int ia = 0, ib = 0;  // first index of the max, like torch.argmax
    for (int i = 1; i < n_vocab; ++i) {
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    double sa = 0, sb = 0;
    for (int i = 0; i < n_vocab; ++i) {
        sa += std::exp(double(a[i]) - a[ia]);
        sb += std::exp(double(b[i]) - b[ib]);
    }
    const double za = a[ia] + std::log(sa), zb = b[ib] + std::log(sb);  // log-sum-exp
    double kl = 0, ent = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const double la = a[i] - za, lb = b[i] - zb, p = std::exp(la);
        kl += p * (la - lb);
        ent -= p * la;
    }
    out[0] = float(kl);
    out[1] = ia != ib;
    out[2] = float(ent);
    out[3] = float(za - a[y]);
    out[4] = float(zb - b[y]);
    if (!top_k) return;
    std::vector<int32_t> order(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [a](int32_t i, int32_t j) { return a[i] > a[j] || (a[i] == a[j] && i < j); });
    int32_t * ids = reinterpret_cast<int32_t *>(out + 5);
    for (int r = 0; r < top_k; ++r) {
        ids[r] = order[r];
        out[5 + top_k + r] = float(a[order[r]] - za);
    }
}

static std::vector<float> score(Model & P, Model & Q, int n_vocab, int n_batch, int top_k, size_t job,
                                const std::vector<int32_t> & tok, const std::vector<int32_t> & idx) {
    const std::string where = "job " + std::to_string(job) + ": ";
    const int n = int(tok.size());
    const size_t width = 5 + 2 * size_t(top_k);
    std::vector<float> out(idx.size() * width);
    if (idx.empty()) return out;

    llama_memory_clear(llama_get_memory(P.ctx), true);
    llama_memory_clear(llama_get_memory(Q.ctx), true);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t k = 0;  // next idx entry
    for (int s = 0; s < n && k < idx.size(); s += n_batch) {
        const int m = std::min(n_batch, n - s);
        const size_t k0 = k;
        batch.n_tokens = m;
        for (int j = 0; j < m; ++j) {
            batch.token[j] = tok[s + j];
            batch.pos[j] = s + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = k < idx.size() && idx[k] == s + j;
            if (batch.logits[j]) ++k;
        }
        if (llama_decode(P.ctx, batch) != 0 || llama_decode(Q.ctx, batch) != 0) die(where + "llama_decode failed");
        std::vector<const float *> lp(k - k0), lq(k - k0);
        for (size_t r = k0; r < k; ++r) {
            lp[r - k0] = llama_get_logits_ith(P.ctx, idx[r] - s);
            lq[r - k0] = llama_get_logits_ith(Q.ctx, idx[r] - s);
            if (!lp[r - k0] || !lq[r - k0]) die(where + "missing logits");
        }
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < n_threads; ++w) {
            pool.emplace_back([&, w] {
                for (size_t r = k0 + w; r < k; r += n_threads) {
                    row_stats(lp[r - k0], lq[r - k0], n_vocab, tok[idx[r] + 1], top_k, &out[r * width]);
                }
            });
        }
        for (auto & t : pool) t.join();
    }
    llama_batch_free(batch);
    return out;
}

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

// The devices the default (non --cpu) load offloads to: every GPU llama.cpp found, else the CPU.
static std::string devices_json(bool cpu) {
    std::string s;
    for (size_t i = 0; !cpu && i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        std::string name = std::string(ggml_backend_dev_name(d)) + " (" + ggml_backend_dev_description(d) + ")";
        name.erase(std::remove_if(name.begin(), name.end(), [](char c) { return c == '"' || c == '\\'; }), name.end());
        s += (s.empty() ? "\"" : ", \"") + name + "\"";
    }
    return "[" + (s.empty() ? std::string("\"CPU\"") : s) + "]";
}

int main(int argc, char ** argv) {
    bool cpu = false;
    int top_k = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu") {
            cpu = true;
        } else if (a == "--top-k" && i + 1 < argc) {
            top_k = atoi(argv[++i]);
            if (top_k < 1) die("--top-k must be a positive integer");
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() < 3 || pos.size() > 4) die("usage: kl_score [--cpu] [--top-k K] P.gguf Q.gguf JOBS [n_batch]");
    const int n_batch = pos.size() == 4 ? atoi(pos[3].c_str()) : 512;
    if (n_batch < 1) die("n_batch must be a positive integer");

    struct Job {
        std::string tok, idx, out;
    };
    std::vector<Job> jobs;
    std::ifstream f(pos[2]);
    if (!f) die("cannot read " + pos[2]);
    for (Job j; f >> j.tok >> j.idx >> j.out;) jobs.push_back(j);
    if (jobs.empty()) die("no jobs in " + pos[2]);

    // Every input is read, and every idx checked, before the models load, so a bad input fails in seconds.
    std::vector<std::vector<int32_t>> toks, idxs;
    size_t n_ctx = 1;  // the longest run sets the context size
    for (const Job & j : jobs) {
        toks.push_back(read_i32(j.tok));
        idxs.push_back(read_i32(j.idx));
        n_ctx = std::max(n_ctx, toks.back().size());
        const std::vector<int32_t> & idx = idxs.back();
        const long n = long(toks.back().size());
        for (size_t k = 0; k < idx.size(); ++k) {
            if (idx[k] < 0 || idx[k] >= n - 1 || (k && idx[k] <= idx[k - 1])) {
                die("job " + std::to_string(toks.size() - 1) + ": idx must be sorted, unique and in [0, n_tokens - 1)");
            }
        }
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    const llama_context_params cp = context_params(uint32_t(n_ctx), uint32_t(n_batch));
    Model P = load(pos[0].c_str(), cp, cpu);
    Model Q = load(pos[1].c_str(), cp, cpu);
    const llama_vocab * vocab = llama_model_get_vocab(P.model);
    check_same_vocab(vocab, llama_model_get_vocab(Q.model));
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (top_k > n_vocab) die("--top-k is larger than the vocabulary");
    // Every job's ids are checked before any is scored, so a bad id in the last job does not fail
    // the batch after the others were scored.
    for (size_t j = 0; j < toks.size(); ++j) {
        for (int32_t t : toks[j]) {
            if (t < 0 || t >= n_vocab) {
                die("job " + std::to_string(j) + ": token id " + std::to_string(t) + " is outside the vocabulary");
            }
        }
    }
    printf("{\"devices\": %s, \"n_batch\": %u, \"n_ubatch\": %u, \"n_ctx\": %u, \"type_k\": \"%s\", "
           "\"type_v\": \"%s\", \"flash_attn\": \"%s\", \"n_vocab\": %d, \"top_k\": %d}\n",
           devices_json(cpu).c_str(), llama_n_batch(P.ctx), llama_n_ubatch(P.ctx), llama_n_ctx(P.ctx),
           ggml_type_name(cp.type_k), ggml_type_name(cp.type_v), llama_flash_attn_type_name(cp.flash_attn_type),
           n_vocab, top_k);
    fflush(stdout);

    for (size_t j = 0; j < jobs.size(); ++j) {
        const std::vector<float> out = score(P, Q, n_vocab, n_batch, top_k, j, toks[j], idxs[j]);
        std::ofstream o(jobs[j].out, std::ios::binary);
        o.write(reinterpret_cast<const char *>(out.data()), std::streamsize(out.size() * sizeof(float)));
        if (!o) die("job " + std::to_string(j) + ": cannot write " + jobs[j].out);
    }

    llama_free(P.ctx);
    llama_free(Q.ctx);
    llama_model_free(P.model);
    llama_model_free(Q.model);
    llama_backend_free();
    return 0;
}
