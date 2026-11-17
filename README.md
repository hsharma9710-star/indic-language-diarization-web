# Indic language diarization - web app

Live page: https://hsharma9710-star.github.io/indic-language-diarization-web/

Upload speech that mixes English and Indian languages and get an RTTM file, a language-coloured waveform and one pooled audio
file per language. Two models can be chosen with the tabs at the top, and a reference RTTM can be loaded to get the DER.
Everything runs **inside your browser** (ONNX Runtime Web, WebAssembly): your audio is never uploaded or stored.

* `index.html` / `app.js` - the page (decoding, plotting, RTTM and WAV writers, tabs, DER card)
* `model/model_int8.part0..2` - the CLSRIL-Transformer-LD model (CLSRIL-23 encoder + head), 8-bit-quantised ONNX, split into three files that the page joins
* `ort.min.js`, `ort-wasm-simd-threaded.*` - ONNX Runtime Web 1.20.1 (MIT)
* `coi-serviceworker.js` - lets the page use several threads on a static host

Model, training and accuracy of our model: https://github.com/hsharma9710-star/indic-language-diarization
(on 27 real recordings about 26 % diarization error with the languages named; the 8-bit model scored 27.6 % vs 28.6 % for the full-precision
one on an 8-recording check). Research prototype.

English check (CLSRIL-Transformer-LD): Whisper-tiny's language detector (ONNX, fp32, `model/lid_fp32.part0..1`, 72 MB, downloaded on first use and cached; `mel_filters.bin` is its log-Mel filter bank) runs on 4 s windows every 2 s after the main model. A speech bin that the main model calls an Indian language is switched to English where the interpolated English probability is at least 0.3. It is on by default and can be switched off with the tick-box. On 6 held-out real recordings (English + Malayalam / Kannada, 4-minute excerpts) it lowered DER from 24.7 % to 16.4 %; on a 30 s Hindi news clip it finds most of the English, but not all.

## Reference RTTM and DER

Load a reference RTTM (before or after diarizing) and the page shows the DER and, for each language, how much of it is missed speech, false alarm and wrong language (`der.js`, a JavaScript port of the NIST `md-eval-22` scorer that `dscore` calls; 0.25 s collar, overlapped speech excluded). It was checked against `dscore` on 56 reference/system pairs; the largest difference was 0.005 DER points.

## Second model: Whisper-Linear-LD

The second tab runs **Whisper-Linear-LD** by Mohit (https://github.com/MohitAiry/Whisper-Linear-LD): a frozen Whisper-large-v3 encoder cut after block 28 of 32, a LayerNorm + linear probe (5-seed ensemble), session language pruning and Viterbi decoding, for ten languages (EN HI KN ML TA TE AS BN MR OD) without naming any of them.

* `wlld.js` - the pipeline except the encoder: 128-band log-Mel, the 30 s / 25 s-hop windows with 2.5 s margins, the pair-averaged 25 Hz float16-rounded features, the five probes, language pruning (10 %), Viterbi (switch penalty 32 nats) and RTTM output. Checked against his Python code: windows and Viterbi paths identical, probe log-probabilities within 1e-6, log-Mel within 4e-5.
* `wlld/probe.bin`, `wlld/mel128.bin` - his five linear-probe checkpoints (`checkpoints/linear_p11_s{0..4}/sel.pt`) and the 128-band mel filter bank of `WhisperFeatureExtractor`, as raw float32.
* `model/wl_shard{0..3}.part*` - the Whisper-large-v3 encoder (blocks 1-28) as four ONNX shards of seven blocks, weights stored in 4-bit blocks (activations stay fp32), split into files of at most 45 MB that the page joins; downloaded on first use and cached.

The encoder is a quantised copy, so results can differ slightly from his GPU code; see the comparison in the repository notes.
