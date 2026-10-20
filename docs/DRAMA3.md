# Fish Audio Drama 3 on Apple Silicon

**Drama 3** is Fish Audio's text-to-speech model for scenes and long-form
reading: up to sixteen speakers in one text, inline cues such as `[sighs]`
or `[laughs]` placed where they should land, and voice cloning from a short
clip and its transcript. It comes from
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) as one
repository: a 35-billion-parameter mixture-of-experts language model (3
billion active per token), a small codebook decoder beside it, and the
44.1 kHz codec that turns its codes into sound.

vpipe runs all three on-device through its **metal-compute** backend: its
own Metal kernels, no Python and no third-party tensor runtime in the
forward pass. Either published checkpoint runs **as published**:

- **[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)**, the
  release: its routed experts are 8-bit floating point (FP8), everything
  else bf16. About 36 GB held.
- **[fishaudio/drama-3-int4](https://huggingface.co/fishaudio/drama-3-int4)**,
  the same model at int4: a community quantization published in Fish
  Audio's own organization. The experts and the attention projections are
  int4 (GPTQ format), the rest bf16. About 22 GB held. This is the one for
  a 32 GB Mac.

**The weights are under the Fish Audio Community License.** It allows
research and non-commercial use. Commercial use is allowed only while the
user's annual revenue is under USD 500,000. Offering it as a hosted
text-to-speech service needs a separate licence from Fish Audio. The int4
checkpoint carries the same licence. This is the model's licence, not
vpipe's. Read `LICENSE` in the model directory before you publish anything
you make with it.

## What you need

| | |
|---|---|
| **Machine** | An Apple Silicon Mac with **64 GB** of memory to run the release at full speed, or **32 GB** for the int4 checkpoint. A smaller Mac runs either one more slowly: the experts that do not fit are read from the drive as they are needed (see [Smaller Macs](#smaller-macs)). |
| **Memory** | Release: **35.8 GB** for the language model, plus 0.4 GB for the codec (0.8 GB with its encoder, which voice cloning loads). The whole process peaked at 39.4 GB cloning a voice. Int4: **22.4 GB** for the language model; the process peaked at 24.8 GB, and the memory plan admits it on a 32 GB box with little to spare (tested by simulation, `VPIPE_RAM_LIMIT_MB=32768`; no 32 GB Mac was at hand). On a Mac too small for that, the experts that do not fit stay on the drive and the process holds 8–9 GB (see [Smaller Macs](#smaller-macs)). |
| **Disk** | Release **~39 GB**; int4 **~22 GB**. Each includes the 1.8 GB `codec.pth`, which is the same file in both. |
| **Build** | An Apple Silicon build of vpipe, the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | None. Both repositories are public and ungated. |

## The pipelines

- **[`prepare-drama3.vpipeline`](pipelines/prepare-drama3.vpipeline)** and
  **[`prepare-drama3-int4.vpipeline`](pipelines/prepare-drama3-int4.vpipeline)**:
  fetch one checkpoint. Run once.
- **[`drama3-speak.vpipeline`](pipelines/drama3-speak.vpipeline)**: turns
  each text file into one `.wav` in the model's own voice.
- **[`drama3-clone-voice.vpipeline`](pipelines/drama3-clone-voice.vpipeline)**:
  does the same in the voice of `reference.wav`.
- **[`drama3-scene.vpipeline`](pipelines/drama3-scene.vpipeline)**: a scene
  with two characters, each with a voice cloned from its own file.

Every pipeline names the release in `hf_dir`. Set it to
`fishaudio/drama-3-int4` to run the int4 checkpoint instead; nothing else
changes. Run any of them from the terminal with `vpipe --launch <file>`, or
open it with **Load** in the web UI's Pipeline Manager.

## Step 1: get the model

```sh
vpipe --launch docs/pipelines/prepare-drama3.vpipeline        # 64 GB Macs
vpipe --launch docs/pipelines/prepare-drama3-int4.vpipeline   # 32 GB Macs
```

Each fetch is pinned to a list of files: the weight shards and their index,
the tokenizer files, `codec.pth`, the licence and the README. The int4
repository also holds evaluation clips, a demo video and its quantization
scripts, none of which is fetched.

## Step 2: speak

Put one text in each of `speech-1.txt` and `speech-2.txt` (or edit the
`load-text` stage's `path` list), then:

```sh
vpipe --launch docs/pipelines/drama3-speak.vpipeline
```

```
load-text ─▸ text-to-speech ─▸ save-audio
```

`text-to-speech` reads the model's `config.json` and recognises Drama 3 by
it. Drama 3 needs no `codec_dir`, because its codec is the `codec.pth` in
the same directory. Each text beat becomes one 44.1 kHz mono waveform.

### Writing the text

The text is passed to the model **as written**. Nothing is normalised,
because the markup is part of the model's vocabulary:

- **Speakers.** `<|speaker:0|>` … `<|speaker:15|>` start a speaker's
  lines. A text without one is spoken as speaker 0.
- **Cues.** Bracketed directions such as `[sighs]`, `[laughs]` or
  `[whispers]` go at the point in the text where the event should happen.
- **Refused.** The prompt's own structural tokens (`<|im_start|>`,
  `<|im_end|>`, `<|voice|>`, `<|text|>`, `<|interleave|>`, `<|semantic:…|>`)
  are refused, and so are speaker tags outside 0–15 or with a leading zero.
  In either case the beat is dropped with a warning naming the problem.

## Long text

A long text is spoken **in chunks**, cut the way Fish Audio's own server
cuts it (`chunk_text`, on by default):

- About **300 bytes** a chunk and at most **3 speakers**. Cuts fall at
  sentence ends first, then at clause marks, then at spaces. A '.' between
  digits is not a sentence end.
- Never inside a special token, a paired tag such as
  `<whisper>…</whisper>`, or a `[bracketed cue]`.
- A tagged text is regrouped by speaker, one segment a line, so short
  turns from A, B and A again can share a chunk.

The chunks are spoken one after another. **Each one is spoken after the
first chunk and the latest one**, which go back into the prompt as
finished turns, text and audio. That keeps the voice and pace steady across
the whole text. If a prompt would pass the reference's 8192-row limit,
history is dropped (the latest turns first, then the first) until it fits.
Each chunk is decoded on its own and the audio is joined into **one beat
per text**, as the reference does.

`max_frames` is the budget **per chunk**. Fish Audio's server allows 600
(about 28 s), and the shipped pipelines set the same. `chunk_text: false`
speaks the whole text as one take.

`context: run` carries the same rule across beats: the run's first take
stays in the context with the latest, so a reading fed in pieces (one
paragraph a beat from `load-text`) keeps one voice and pace throughout.
The default, `context: text`, starts fresh for each text beat.

## Step 3: clone a voice

Put a clip of the voice in `reference.wav`, and set `reference_text` on the
`speak` stage to **exactly what is said in it**:

```sh
vpipe --launch docs/pipelines/drama3-clone-voice.vpipeline
```

```
load-audio ─▸ audio-to-pcm ─┐
                            ├─▸ text-to-speech ─▸ save-audio
             load-text  ────┘
```

Drama 3 clones from the clip's sound **and** its words. A reference without
a transcript is not used, and the stage says so. The clip is resampled to
44.1 kHz, cut to `voice_ref_seconds` (12 s by default) and encoded once.
A few seconds of clean speech is enough: Fish Audio's own examples clone
from a 3.9 s clip, and so do the tests here. If the clip runs longer than
`voice_ref_seconds`, the transcript has to cover only the part that is
kept.

The voice then speaks every later text, until another clip for the same
speaker arrives on the `audio-ref` port. `wait_for_reference: true` makes
the first text wait for the clip, so a reference read from a file is not
missed by an utterance that starts before it is decoded. A text beat can
carry its own `"reference_text"`, which overrides the config for that beat.

`voice_lock: true` does without a clip. The first take, with the words it
said, becomes the reference for every take after it. A transcript has to
match its audio exactly, so a first take longer than `voice_ref_seconds`
cannot be cut to fit, and the lock waits for a take short enough.

## A scene with several voices

`references` clones voices from files, one per speaker:

```json
"references": [
  {"audio": "narrator.wav",  "text": "What narrator.wav says."},
  {"audio": "character.wav", "text": "<|speaker:1|> What character.wav says."}
]
```

A reference's transcript names its speaker with a leading tag (speaker 0
without one), and the text then gives each line its speaker:

```
<|speaker:0|> The door opened slowly. <|speaker:1|> Is anyone there?
```

[`drama3-scene.vpipeline`](pipelines/drama3-scene.vpipeline) reads a scene
from `scene.txt` this way. The files are decoded (any format FFmpeg reads),
cut to `voice_ref_seconds` and encoded once, at load. A voice that arrives
on the `audio-ref` port replaces the one for its own speaker, so a live
reference and the configured ones can be mixed.

## Editing by continuation

A text beat sent as an object can name earlier takes of the same run in
`"previous_takes"`. Their audio goes into the prompt ahead of the new text,
so a changed line can be re-said in the delivery of the first:

```json
{"text": "The report is due on Thursday, so please finish it by noon.",
 "previous_takes": ["The report is due on Wednesday, so please finish it by noon."]}
```

A previous take is matched by its exact text (white space at the ends
ignored) among the takes this stage spoke in the run, the newest first. The
stage keeps the last 64. A text it has not spoken is skipped with a warning.

## The model decides the length

Drama 3 ends a take when it chooses to; `max_frames` only caps it. On some
lines it stops early (after the first speaker, say), or stops speaking and
runs to the cap in silence. This happens in the reference implementation
too, and the vLLM-Omni recipe for this model notes it. Measured here:

- An 87-character English sentence in the model's own voice reached a
  400-frame cap on 3 of 8 seeds. A cloned voice is much steadier.
- A two-speaker line with two cloned voices, 12 seeds a session: in the
  per-speaker layout the chunker gives it (which the stage sends), 0–1
  takes stopped early and 3–6 ran silent to the cap. Sent as one raw line
  instead, 5 of 12 stopped early.

If a take goes wrong, try another seed. `max_frames` bounds the cost either
way.

## Sampling

Unwired, the stage samples the way the reference server does: temperature
1.0, top-p 0.9, top-k 30, with the same anti-repetition re-draw (the window
starts from the prompt's last 16 rows, as every reference seeds it). It
uses a fixed seed, so a rerun of the same text gives the same take. To
change any of this, wire a `sampler-select` stage into the `audio-sampler`
port. Its temperature, top-p, top-k (capped at 64) and seed replace the
defaults, and a seed of 0 draws a fresh one for each take.
`repetition_penalty` has no counterpart in the model and is ignored with a
warning.

## Streaming

`stream_chunk_frames` (16 by default) emits audio every 16 frames, about
0.74 s, while a take is still being generated. The first audio leaves about
one piece after generation starts. The codec decodes the pieces with its
windowed history carried between them, and starts afresh at each chunk of a
long text. The streamed waveform is therefore **sample-for-sample the
one-shot one**. A new text beat cuts the current one short (after flushing
what it has) unless `interrupt_on_new_text` is false. The shipped pipelines
set `stream_chunk_frames: 0` and write one file per text.

## Speed

Measured on an M4 Pro (64 GB), checkpoints on the internal SSD:

| | Release (FP8) | int4 |
|---|---|---|
| Load | 13.8 s (codec 0.4–0.8 s), then a warm-up take | 7.9 s |
| Generation | 32–33 frames a second | 34–35 frames a second |
| Decode | ~150 ms for a 3.5 s take | the same |

The codec runs at 21.5 frames a second, so either checkpoint generates
about **1.5× faster than real time**. The first frame follows the prompt
by 12–20 ms.

The prompt's cost grows with what it carries. A sentence is 130–200 ms. A
3.9 s cloned voice brings it to 370–440 ms, and two voices to 700–950 ms.
A chunk with two earlier takes in its context is about 1100 rows and takes
1.7 s. The release's experts take a grouped GEMM over the prompt (each
expert's weights read once), about 1.6–2.0 ms a row. The int4 experts run
the existing affine int4 MoE kernels.

## Smaller Macs

When a checkpoint does not fit beside everything else with room to spare,
vpipe keeps everything except the routed experts in memory and holds the
experts in a **cache**. A frame routes each of its 40 layers through 8 of
256 experts, and some experts are chosen far more often than others, so
the cache keeps the ones chosen most and reads the rest from the drive
when they are needed. While a frame runs one layer, vpipe can also read
the experts the next layer will probably pick, and it measures how many
are worth reading ahead on your drive as it goes. vpipe decides all of
this when it loads the model, and the output is the same as with every
expert in memory, bit for bit. There is nothing to configure.

It is slower, by how much depends on how much of the model fits. Measured
on an M4 Pro, the checkpoints on its internal SSD, with smaller Macs
simulated by limiting the memory vpipe plans for. The text was a
212-character paragraph of about 280 frames:

| Mac | Checkpoint | Experts kept | Chosen from the cache | Generation | Process peak |
|---|---|---|---|---|---|
| 32 GB | int4 | all (it fits) | — | 35 frames/s | 24.7 GB |
| 32 GB | release | 58% (18.6 GB) | 89% | 9.4 frames/s | 25.1 GB |
| 16 GB | int4 | 8% (1.8 GB) | 40% | 5.4 frames/s | 8.4 GB |
| 16 GB | release | 5% (2.2 GB) | 38% | 3.4 frames/s | 8.7 GB |

(The 16 GB int4 row was measured before the read-ahead.)

Real time is 21.5 frames a second, so on a 16 GB Mac the int4 checkpoint
generates at about a quarter of real time. That is fine for a recording
and too slow for live speech. It is still the checkpoint to choose there:
its experts are half the size, so the drive reads half as much per frame.
The drive sets the pace: an external drive reads several times slower
than the internal SSD, and so is the model. Each text's log line says
what the cache did (`experts N% hits, M MB read, T ms waited for`, and
how much it read ahead).

## Accuracy

Each checkpoint was checked against the reference implementation in fp32,
running **that checkpoint's own weights**, with the reference's own bf16
run as the bar:

- **Release.** Taught a prompt and a cloned take row by row, the main
  logits are 0.0168 from fp32 (the reference's bf16: 0.0209). The model
  picks the same token at every position where fp32's choice is decisive
  (38 of 38), and each codebook head the same at its own. The step-by-step
  decode path agrees, and the sampled path matches it exactly.
- **Int4.** Main logits 0.0180 from fp32 (its reference's bf16: 0.0148),
  every decisive pick the same (37 of 37, and every head). Its experts are
  converted exactly into vpipe's int4 layout; the only rounding is of
  their scales to bf16.
- **Codec.** Decode is 57 dB from the reference. Streamed decode is
  bit-identical to whole decode.
- **Text and prompts.** The chunker matches fish-speech's own on 57 cases
  (tags, cues, paired tags, decimals, CJK and Korean punctuation, emoji).
  A prompt with references and earlier takes matches its prompt builder
  row for row.
- **End to end.** Qwen3-ASR hears both checkpoints, in their own voices
  and a cloned one, say the requested text (WER 0). A three-chunk
  paragraph with two cloned voices comes back at WER 0.03, the difference
  being numerals ("9" for "nine"); an edited line through
  `previous_takes`, at WER 0.

## Not supported

- **Real time on a 16 GB Mac.** The expert cache lets it run (see
  [Smaller Macs](#smaller-macs)), but a frame that misses the cache reads
  the experts it needs from the drive, and the drive cannot keep up with
  real time.
- **Quantizing the release.** `model-quantize` refuses it by name and
  points at the int4 checkpoint.
