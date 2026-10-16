# Separating vocals with BS-RoFormer

**BS-RoFormer** takes a song apart: a mixed track in, the **vocals** and
the **accompaniment** out, each the full length of the song. It is the
Band-Split RoPE Transformer of Lu et al., *Music Source Separation with
Band-Split RoPE Transformer* (2023): it cuts the spectrum into frequency
bands and attends across time and across bands in turn.

vpipe runs it on-device through its **metal-compute** backend: its own
Metal kernels, no Python in the loop. It reads checkpoints in the format of
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MSST) — a PyTorch `.ckpt` beside the YAML file that describes it — which is
how BS-RoFormer weights are published. Its output matches MSST's own to
within the reference's own half-precision rounding.

The examples use **[pcunwa/BS-Roformer-Resurrection](https://huggingface.co/pcunwa/BS-Roformer-Resurrection)**,
which holds two models: one trained to extract **vocals** and one trained
to extract the **instrumental**.

**That repository states no license.** It has no model card and no
license tag, so the terms under which you may use or publish what you make
with it are unstated. vpipe can fetch it; vpipe does not distribute it.
Decide for yourself before using it for anything beyond your own listening.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **Under 1 GB**: 99 MB of weights (published as 32-bit floats, held as 16-bit) and a working set of about 0.7 GB, whatever the song's length — the track streams through. |
| **Disk** | **~390 MiB** for both models, 195 MiB each. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | None. The repository is public and ungated. |

## The pipelines

- **[`prepare-bs-roformer.vpipeline`](pipelines/prepare-bs-roformer.vpipeline)**
  — fetch both models. Run once.
- **[`bs-roformer-separate-vocals.vpipeline`](pipelines/bs-roformer-separate-vocals.vpipeline)**
  — `song.m4a` in, `song-vocals.m4a` and `song-instrumental.m4a` out, with
  the **vocals** model.
- **[`bs-roformer-instrumental.vpipeline`](pipelines/bs-roformer-instrumental.vpipeline)**
  — the same two files from the **instrumental** model. Use it when the
  accompaniment is what you are after — a karaoke track, say.

Any of them runs from the terminal with `vpipe --launch <file>` or opens with
**Load** in the web UI's Pipeline Manager.

## Step 1 — get the models

```sh
vpipe --launch docs/pipelines/prepare-bs-roformer.vpipeline
```

Two fetches from one repository, each taking only its own checkpoint and
YAML and registering under its own name:

| registered as | the model |
|---|---|
| `pcunwa/BS-Roformer-Resurrection-vocals` | extracts the vocals |
| `pcunwa/BS-Roformer-Resurrection-inst` | extracts the instrumental |

Both land in one folder, `models/pcunwa/BS-Roformer-Resurrection/`; the
registered name is what says which of the two a pipeline means.

## Step 2 — separate a song

Put a song beside the pipeline as `song.m4a`, or point `input_url` at any
file — mp3, m4a/AAC, FLAC, WAV, Ogg, Opus, or **the soundtrack of a video
file**:

```sh
vpipe --launch docs/pipelines/bs-roformer-separate-vocals.vpipeline \
      --stage-cfg load-audio::input_url=/path/to/song.flac
```

It writes `song-vocals.m4a` and `song-instrumental.m4a`, each exactly as
long as the input.

### The shape of the graph

```
load-audio ─▸ audio-to-pcm ─▸ audio-separate
                               ├─ stem ─────▸ temporal-stack ─▸ save-audio
                               └─ residual ─▸ temporal-stack ─▸ save-audio
```

- **`load-audio`** reads the file's encoded packets; **`audio-to-pcm`**
  decodes them and resamples to **44.1 kHz stereo**, the rate the model was
  trained at. A mono file is fine; it comes out as stereo.
- **`audio-separate`** is the model. It has two outputs:
  - **`stem`** — the source the model was trained to extract: the
    vocals for the vocals model, the instrumental for the instrumental one;
  - **`residual`** — the song minus that source: the other half.
- **`temporal-stack`** gathers each output into one waveform, so that
  **`save-audio`** writes one file per output rather than one per piece.

### How it works through a long song

The model looks at **about 17 seconds** at a time (17.8 for the vocals
model, 17.0 for the instrumental one) and hears everything in that window
at once. A song is therefore cut into overlapping windows, each
separated, and the results cross-faded back together. `audio-separate`
does this as the audio arrives: it trails the input by about one window,
holds only a couple of windows in memory, and its output is the same,
sample for sample, as separating the whole song in one go.

## Choosing a model

| | `stem` port | `residual` port | example |
|---|---|---|---|
| **vocals** (`…-vocals`) | vocals | instrumental | `bs-roformer-separate-vocals` |
| **instrumental** (`…-inst`) | instrumental | vocals | `bs-roformer-instrumental` |

Each model is better at the half it was trained to produce. Take the
vocals from the vocals model and the accompaniment from the instrumental
model when either half has to be as clean as possible.

## The knobs

`audio-separate`:

| key | default | what it does |
|---|---|---|
| `hf_dir` | — | The model: a registered name, a folder holding an MSST checkpoint and its YAML, or the `.ckpt` itself. |
| `checkpoint` | | The `.ckpt` to use when the folder holds several and the registered name does not pin one. |
| `config` | | The YAML, when it is not named after the checkpoint (`<name>-Config.yaml`, `<name>.yaml`, …) or the only one beside it. |
| `stem` | first | For a model that extracts several sources, which one goes to the `stem` port — by its name in the YAML or by index. |
| `num_overlap` | the YAML's (2) | How many windows cover each moment of the song. Higher smooths the joins between windows; the time grows in proportion. |

`audio-to-pcm` must produce the model's rate — 44100 for every MSST model so
far. Feed it another rate and `audio-separate` stops with a message naming
the fix rather than separating the wrong sound.

`save-audio` writes `.m4a` (AAC; the examples use 256 kbps), `.mp3`, `.aac`
or `.wav`. Its WAV is 16-bit, which clips anything past full scale — and a
separated stem can peak a little above the mixture it came from — so the
examples write `.m4a`.

## How long it takes

A 3 minute 34 second song, the whole pipeline:

| Mac | vocals model | instrumental model |
|---|---|---|
| M4 Pro | 52–55 s | 69 s |
| M5 (10-core GPU, fanless) | 37 s | |

The instrumental model is slower because it slices time more finely:
each of its windows holds 1700 time steps against the vocals model's
1536, and the windows are a little shorter, so a song needs more of them.
`num_overlap` multiplies either figure: 4 takes twice as long as the
default 2.

## Other BS-RoFormer checkpoints

Any **BS-RoFormer** published in MSST's format runs: point `hf_dir` at the
`.ckpt` (or its folder) and keep its YAML beside it. Models that separate
several sources at once (drums, bass, vocals, other) load too; `stem`
picks the one to emit.

The YAML is read, not trusted. A checkpoint that asks for something this
implementation does not have — MSST's linear-attention layers, the PoPE
positional encoding, hyper-connections or value residuals from
lucidrains' newer code, or a **Mel-Band RoFormer** (a sibling
architecture with overlapping mel-scale bands) — is refused at load with a
message naming the setting, rather than run as a different network.

## Troubleshooting

- **"the input is 48000 Hz and the model was trained at 44100 Hz"** — set
  `output_sample_rate: 44100` on `audio-to-pcm` (the examples do).
- **"… holds several checkpoints"** — the folder has more than one model;
  use its registered name (`…-vocals` / `…-inst`), or name the file with
  `checkpoint`.
- **"no YAML config beside …"** — the checkpoint needs its YAML; put it
  beside the `.ckpt` or name it with `config`.
