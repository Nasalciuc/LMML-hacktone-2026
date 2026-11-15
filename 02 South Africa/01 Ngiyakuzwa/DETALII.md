# South Africa — Ngiyakuzwa?

**Task.** A 10.5 min, 16 kHz mono voice note: a woman reads short isiZulu phrases in a loud stadium.
Somewhere she says *ikhodi* ("code"); the next word is the flag, `SIGMOID_<WORD>`, in standard
isiZulu spelling, uppercase. Done for practice, not for score.

**Answer: `SIGMOID_ULWAZI`** (*ulwazi* = "knowledge, information"). Confidence is high.
The first answer, ISISHO, was wrong: it came from segment 067, which does not contain *ikhodi* at all
(see "Where ikhodi really is" below).

Clips: `C:\Users\user\Downloads\LMML_upload\South Africa\asculta\`
`IKHODI_phrase_07m14-07m21_{clean,strong}[_0.75x].wav` (the whole phrase) and
`NEXT_WORD_07m19-07m21_{clean,strong}[_0.75x].wav` (the word after *ikhodi*).

## Where ikhodi really is

The phrases in the note are prompts of the public NCHLT isiZulu speech corpus
(HF dataset `Max5ive/nchlt_speech_zulu`), so their text can be looked up.

1. `corpus_ikhodi.py`: full-text search of the corpus for *ikhodi*. 83 utterances, only three texts:
   "khona ikhodi lomgwaqo" (44), "wentela uzosebenzisa ikhodi" (38), "ikhodi engagunyaziwe elimazayo" (1).
2. `find_phrase.py`: likelihood of each text on each of the 138 segments, two isiZulu Whisper models,
   summed. A known prompt as control: "kutshengisa ikhono elihle".

| text | best segment | z vs the other segments | next best |
|---|---|---|---|
| kutshengisa ikhono elihle (control) | 077 | −6.4 | 067: −4.7 |
| wentela uzosebenzisa ikhodi + ulwazi | **097** | **−5.1** | 023: −2.6 |
| khona ikhodi lomgwaqo | 034 | −2.7 | no match |
| ikhodi engagunyaziwe elimazayo | 118 | −2.8 | no match |
| ulwazi oluthile mayelana (a prompt that is not in the note) | 118 | −2.7 | noise level |

   So segments 067 and 077 are the same prompt, "kutshengisa **ikhono** elihle" (*ikhono*, not *ikhodi*):
   two decoys. That is why every model heard a final -o in 067 (ikhodo, ikhoto, inkolo).
   The real *ikhodi* is in segment 097 (07:14.78–07:20.75): "wentela uzosebenzisa ikhodi", and the
   word after it, in the same breath, is the key.
3. `next_word.py`: cost of one more word after "wentela uzosebenzisa ikhodi" in that window
   (2 models × 3 cleaned versions, lower is better): uluwazi 41, ucwazi 54, ulwasi 57, ulazi 64,
   **ulwazi 64**, ngolwazi 71, uyazi 73, umlazi 76, ukwazi 78, ubuntu 87, isisho 120, lomgwaqo 205.
   The first four are spellings of the same sound that are not words; *ulwazi* is the best real word.
4. `mms_check.py`: MMS-1b (zul adapter), a letter model with no language prior, reads the window as
   "wenela uzosobenziisiee nikunji uulwazu" (clean) and "wenena izosebenzise ikunji undwazi" (strong).
   The same model reads 077 as "kutjengisa ikholo elihla".

What is left open: only the spelling of the last word. *ulwazi* is the standard form and the only
dictionary word among the top readings.

## Earlier reading (superseded)

Everything below describes the first attempt, which read the word after the *ikhono* decoy in segment 067
and gave ISISHO. Kept for the record.

## The audio

- Speech: 267 s active, split into 138 phrases (`segments/`, times in `segments/index.csv`).
- Two vuvuzela-like horns: harmonic combs at 217.5 Hz × k and 233.1 Hz × k, switching on and off in
  blocks of a few seconds (69 switches). They cover most of the voice band.
- Other steady and slowly gliding whistles, broadband crowd roar, a hiss band at 5–7 kHz, claps.
- *ikhodi* is clearest in segment 067 (05:04.53–05:07.78). The phrase reads "kutshengisa ikhodi X".
  Letter timings from the MMS CTC model: *kutshengisa* 304.8–305.6, *ikhodi* 305.7–306.3,
  **the word 306.37–307.0**, then a pause until about 308.2.
- Spectrogram of the word (`work/zoomword.png`): a vowel at 306.4–306.65, then strong 2–6 kHz
  frication at 306.65–307.0. That fits /s/ and /ʃ/ (s, sh). It does not fit /l/, and a weak /f/ fits poorly.

## Method

1. `clean.py`: STFT noise floor (running 25th percentile), removal of steady tonal lines and both
   horn combs, Wiener gain. Writes `cleaned/voice_note_clean.wav`, `_strong` (harder), `_voiceband`
   (250–3800 Hz).
2. `segment.py`: speech activity, then 138 phrases plus an acoustic *ikhodi* candidate list. The list
   is weak; the actual hit came from ASR.
3. ASR over all segments (Whisper large-v3-turbo, isiZulu Whisper fine-tunes) to find *ikhodi* → seg 067.
4. `enhance.py`: neural enhancement of 290–320 s (HTDemucs vocals, SpeechBrain MetricGAN+,
   SepFormer-WHAM16k, SepFormer-DNS4), applied to both the raw and the cleaned audio → `work/enh/`.
5. `asr_all.py`: free transcription of the phrase with 4 isiZulu models on every audio version.
6. `ctc_zul.py`: greedy letter decoding with times (MMS-1b-all + `zul` adapter, w2v-BERT isiZulu).
7. `decide.py`: forced-alignment cost of the word only, under all 4 models (details below).
8. `tts_dtw.py`: synthesized candidates matched by DTW. This attempt failed (see below).

## Evidence

### A. Free transcription of the word after *ikhodi* (`asr_ikhodi_phrase.tsv`, `ctc_zul.py`)

Only the three cleaned versions are shown. Raw audio and the neural-enhanced versions produce garbage
or loops (table D).

| model | clean | strong | voiceband |
|---|---|---|---|
| zionia/whisper-small-isizulu | **isisho** | ikisho | eisisi |
| zionia/whisper-small-isizulu-noisy | ufishu | isishizwa | **esisho** |
| facebook/mms-1b-all (zul), greedy letters | ekisu | efisi | esishi |
| kesbeast23/isizulu-w2v-bert-2.0, greedy | elishe | eblishe… | elishe |

What the models agree on: vowel, fricative, *i*, *s/sh*, vowel. Five of the twelve outputs contain *-sho/-shi*.
The w2v-BERT model hears an *l*. The spectrogram shows frication at that spot, so the *l* is probably wrong.

### B. Forced alignment, word cost only (`decide.py` → `work/decide_out.tsv`)

Cost = NLL("kutshengisa ikhodi W") − NLL("kutshengisa ikhodi") over 304.4–307.7 s, averaged over
clean/strong/voiceband (Whisper models: also over the prefixes *kucingisa* / *kutshengisa*). Each model
ranks the words by cost per letter; the table averages those ranks. Lower is better.

| word | in isiZulu Wikipedia (count) | MMS | w2v-BERT | Whisper-zu | Whisper-zu-noisy | avg rank |
|---|---|---|---|---|---|---|
| esisi | no | 1 | 2 | 3 | 2 | 2.00 |
| isisi | no | 3 | 6 | 1 | 1 | 2.75 |
| **isisho** | yes (4) | 6 | 7 | 2 | 3 | **4.50** |
| esisho | yes (15) | 4 | 3 | 11 | 7 | 6.25 |
| ifishi | no | 8 | 4 | 6 | 8 | 6.50 |
| ifu | yes | 2 | 12 | 4 | 9 | 6.75 |
| elisha | yes (101) | 7 | 1 | 10 | 10 | 7.00 |
| ifishu | no | 9 | 9 | 9 | 6 | 8.25 |
| elihle | yes (14) | 11 | 5 | 15 | 15 | 11.50 |
| imfihlo | yes (10) | 15 | 16 | 16 | 16 | 15.75 |
| eyimfihlo | yes (7) | 17 | 17 | 17 | 17 | 17.00 |

17 candidates were scored; the table shows the main ones. Reading the table:

- **isisho is the best real word**: average rank 4.50, against 6.25 for esisho, 6.75 for ifu and 7.00 for elisha.
  It is the best real word under both Whisper models (per letter 0.25 vs 0.48 for ifu, and −0.38 vs −0.21 for isiswe).
- The "secret" reading (*imfihlo*, *eyimfihlo*) comes last under every model. The *fish* outputs
  (ifishu, ifishi) are not words and still rank below *isi-*.
- *isisi* and *esisi* score a bit better than isisho. They are the same *isi-s-* skeleton with the last
  syllable heard as -si, and neither appears in the 29.7k-word isiZulu Wikipedia list. Short strings also get a
  small bonus in these scores. MMS on the voiceband audio does hear *s-h-i* there.
- Initial vowel i or e: both Whisper models (which have an isiZulu text prior that favours *esisho*,
  15 vs 4 Wikipedia hits) still pick *isisho*; both CTC models lean to *e-*. *ikhodi* is class 9,
  and *esisho* is a class-7 relative that would not agree with it. So the standard noun *isisho* is the best spelling.

Whisper-only totals (`work/score_whisper_out.txt`, unnormalized): isisi 5.7, esisi 21.5,
**isisho 30.1**, ifu 56.9, ifiso 59.0, elisha 78.0, ifishu 81.4. imfihlo is not in the top 15.

### C. Attempts that did not work

| attempt | result |
|---|---|
| Free lexicon search (`lexicon_search.py`, `lexicon_scores.tsv`): CTC loss of "ikhodi W" for 29.7k words | Useless. The top 30 are all two-letter strings (ie, ii, ni, is…) because CTC loss summed over a fixed window favours short targets. `decide.py` fixes this (prefix cost subtracted, per letter, real candidates only). |
| `score_words.py`: same idea over all 26 versions incl. enhanced | Rankings dominated by the destroyed enhanced versions (raw__metricgan etc. all pick "ifu"). Only the 3 cleaned versions are usable. Output in `work/score_words_out.txt`. |
| TTS + DTW with `facebook/mms-tts-zul` | The model does not exist on HF (401). No Zulu, Xhosa, Ndebele, Swati, Sotho or Tswana MMS-TTS either. `tts_zul.py` is left as written but cannot run. |
| TTS + DTW with Xitsonga / Swahili MMS voices (`tts_dtw.py`) | No signal. All candidates fall within 5.85–6.70 (normalized MFCC DTW), the order differs between the two voices, and isisho ranks near the bottom in both. A different-language voice against this noisy speaker is too far off. `work/tts_dtw_tso.txt`, `work/tts_dtw_swh.txt`. |
| Neural enhancement before ASR (HTDemucs, MetricGAN+, SepFormer WHAM/DNS4; table D) | Made it worse. On raw audio the horns are taken as the "voice". On cleaned audio the result is muffled, and Whisper loops ("kuthi kuthi…"). Only sepdns on the cleaned audio stays readable ("kudingi samikholo isivisa", "sabukhodo isisa"). |
| Not tried | DeepFilterNet, MossFormer2 (not installed, and the enhancement above showed this audio suffers more from added artefacts than from leftover noise); lsgmgbl/wav2vec2-xls-r-isizulu. |

### D. Same phrase on the neural-enhanced versions (`asr_ikhodi_phrase.tsv`, excerpt)

| version | whisper-isizulu | MMS |
|---|---|---|
| clean__sepdns | kudingi samikholo isivisa kuthi | kuqengisa ikikhotho ekuzengiso |
| clean__demucs | u kusengisa dukota u | kucengesali nkono uuu |
| clean__metricgan | kuthi kuthi kuthi … | kukhengezamnimuuakne |
| raw__demucs | kuthi angis angis … | kutshingia |
| raw__sepdns | ucingis … uccingis … | a |

## Reproduce

Python with GPU torch + transformers: `C:\Users\user\miniconda3\python.exe` (below: `py`).
Models download from HF on first use. VRAM stays under ~6 GB.

```
.venv\Scripts\python clean.py            # cleaned/  (~3 min, numpy/scipy only)
.venv\Scripts\python segment.py          # segments/, ikhodi_candidates/
py enhance.py                            # work/enh/  (needs demucs, speechbrain)
py asr_all.py                            # asr_ikhodi_phrase.tsv
py ctc_zul.py mms 304.0 308.2 cleaned/voice_note_clean_voiceband.wav   # letters + times
py ctc_zul.py w2vbert 304.0 308.2 cleaned/voice_note_clean.wav
py decide.py                             # work/decide_out.tsv  (main ranking, ~3 min)
py score_whisper.py cleaned/voice_note_clean.wav:0 cleaned/voice_note_clean_strong.wav:0 cleaned/voice_note_clean_voiceband.wav:0
py tts_dtw.py facebook/mms-tts-tso       # failed check, kept for the record
py work/zoomword.py                      # work/zoomword.png
py work/makeclips.py                     # clips in LMML_upload\South Africa\asculta\
```

## Files

| file | what |
|---|---|
| `clean.py`, `segment.py` | DSP cleaning; speech activity, phrase cutting, acoustic *ikhodi* candidates |
| `cleaned/` | clean / strong / voiceband versions of the whole note |
| `segments/`, `segments/index.csv` | 138 phrases (wav + spectrogram png) with times |
| `ikhodi_candidates/` | 10 acoustic guesses (weak, superseded by ASR) |
| `asr.py`, `asr_segments.py`, `asr_pick.py` | Whisper large-v3-turbo over the note / segments → `transcript.*`, `segments_sw.tsv` |
| `asr_zulu.py`, `asr_slide.py` | isiZulu Whisper fine-tunes over segments / sliding windows → `segments_whisper-small-isizulu*.tsv`, `slide_clean.txt` |
| `asr_prompt.py`, `asr_var.py`, `asr_var2.py`, `asr_words.py` | prompted, beam, sampled, word-timestamp decoding of the *ikhodi* phrase |
| `asr_mms.py`, `asr_all.py` | MMS zul / all-model transcription of the phrase → `asr_ikhodi_phrase.tsv` |
| `enhance.py` | HTDemucs / MetricGAN+ / SepFormer enhancement → `work/enh/` |
| `ctc_zul.py` | CTC model loading, greedy letters with times, CTC scoring helper |
| `decide.py` | **final ranking** → `work/decide_out.tsv`, `work/decide_log.txt` |
| `score_cand.py`, `score_ctc.py`, `score_whisper.py`, `score_words.py` | earlier candidate scorers (summed, not length-fair) |
| `make_lexicon.py`, `lexicon_search.py`, `lexicon_scores.tsv`, `work/zulu_words.tsv` | isiZulu Wikipedia word list (29.7k) and the failed free lexicon search |
| `tts_zul.py`, `tts_dtw.py` | TTS + DTW check (zul voice does not exist; tso/swh gave no signal) |
| `work/zoomword.png`, `work/z067*.png` | spectrograms of the phrase / word |
| `work/makeclips.py` | writes the listening clips |
