# Indic language diarization - web app

Live page: https://hsharma9710-star.github.io/indic-language-diarization-web/

Upload speech that mixes English and one or two Indian languages and get an RTTM file, a language-coloured waveform and one pooled audio
file per language. The model runs **inside your browser** (ONNX Runtime Web, WebAssembly): your audio is never uploaded or stored.

* `index.html` / `app.js` - the page (decoding, plotting, RTTM and WAV writers)
* `model/model_int8.part0..2` - the model (CLSRIL-23 encoder + head), 8-bit-quantised ONNX, split into three files that the page joins
* `ort.min.js`, `ort-wasm-simd-threaded.*` - ONNX Runtime Web 1.20.1 (MIT)
* `coi-serviceworker.js` - lets the page use several threads on a static host

Model, training and accuracy: https://github.com/hsharma9710-star/indic-language-diarization
(on 27 real recordings about 26 % diarization error with the languages named; the 8-bit model scored 27.6 % vs 28.6 % for the full-precision
one on an 8-recording check). Research prototype.
