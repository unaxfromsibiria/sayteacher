import asyncio
import io
import logging
import os
import re
import struct
import time
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .common import TARGET_SAMPLE_RATE
except ImportError:
    from common import TARGET_SAMPLE_RATE

log = logging.getLogger("transcreenpt")

# Recognition text quality controls.
RATIO_MIN = 0.10            # words per char; below this -> letter runs / long garbage tokens (few words, many chars)
# When False the recognized-text quality checks are skipped
RATIO_MAX = 0.27            # above this -> short-token spam; generous because function-word-heavy text (English) runs high
GROWTH_LIMIT_FACTOR = 10.0  # reject when latest segment adds more than this * mean previous
MIN_WORDS_FOR_RATIO = 12    # ratio check needs enough words to be meaningful
MIN_CHARS_FOR_RATIO = 40    # ratio check needs enough chars to be meaningful
MAX_WORD_FREQ = 0.25        # reject when the most frequent word makes up more than this share of all words
MIN_WORDS_FOR_FREQ = 10     # frequency check needs enough words to be meaningful
MIN_GROWTH_BASE = 5         # floor for the growth baseline so short/silent previous segments set no tiny threshold
MIN_DIVERSITY = 0.35        # reject when unique/total word ratio is below this
MIN_WORDS_FOR_DIVERSITY = 20  # diversity check needs enough words to be meaningful
_PUNCT = ".,!?;:()[]{}\"'—–-…«»"


def count_words(text: str) -> int:
    return len((text or "").split())


def word_ratio(text: str) -> float:
    """Words per character (chars counted with spaces); sensible text spans ~0.13..0.25 by language/word length."""
    if not text:
        return 0.0
    return count_words(text) / max(1, len(text))


def ratio_ok(ratio: float) -> bool:
    """Word/char ratio of sensible text sits in [RATIO_MIN, RATIO_MAX]; letter runs fall below, short-token spam above."""
    return RATIO_MIN <= ratio <= RATIO_MAX


def ratio_check_applicable(text: str) -> bool:
    """The ratio is only meaningful when the text has enough words or chars; short texts are noisy."""
    return count_words(text) >= MIN_WORDS_FOR_RATIO or len(text) >= MIN_CHARS_FOR_RATIO


def _norm_words(text: str) -> list[str]:
    """Words with edge punctuation stripped; empty tokens dropped."""
    out = []
    for w in (text or "").split():
        w = w.strip(_PUNCT)
        if w:
            out.append(w)
    return out


def top_word_freq(text: str) -> float:
    """Share of the most frequent word in the text (0.0 for empty text)."""
    words = _norm_words(text)
    if not words:
        return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)


def word_freq_ok(text: str) -> bool:
    """Reject texts where one word dominates the segment (model repetition loops).

    Skipped for short texts where a repeated word is more likely legitimate."""
    if count_words(text) < MIN_WORDS_FOR_FREQ:
        return True
    return top_word_freq(text) <= MAX_WORD_FREQ


def word_diversity(text: str) -> float:
    """Share of distinct words in the text (1.0 for all-unique, lower for repetition loops)."""
    words = _norm_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def word_diversity_ok(text: str) -> bool:
    """Reject texts where a short phrase repeats so few distinct words dominate

    Skipped for short texts where repetition is more likely legitimate."""
    if count_words(text) < MIN_WORDS_FOR_DIVERSITY:
        return True
    return word_diversity(text) >= MIN_DIVERSITY


def _increments(totals: list[int]) -> list[int]:
    """Per-segment word increments from cumulative totals recorded per buffer element."""
    out = []
    prev = 0
    for t in totals:
        if t > prev:
            out.append(t - prev)
        prev = t
    return out


def growth_ok(latest_added: int, prev_totals: list[int]) -> bool:
    """Words added by the latest segment must not greatly exceed mean of previous segments."""
    increments = _increments(prev_totals)
    if not increments or latest_added <= 0:
        return True
    avg = sum(increments) / len(increments)
    # Floor the baseline so a run of short/silent segments (avg < MIN_GROWTH_BASE) does not set an
    # unreasonably tiny threshold that rejects a normal phrase.
    return latest_added <= max(avg, MIN_GROWTH_BASE) * GROWTH_LIMIT_FACTOR


def validate_recognition(text: str, prev_totals: list[int]) -> tuple[bool, str]:
    """Check recognized text; returns (ok, reason). Empty reason means ok."""
    problems = []
    if ratio_check_applicable(text):
        ratio = word_ratio(text)
        if not ratio_ok(ratio):
            problems.append(f"word/char ratio {ratio:.3f} outside [{RATIO_MIN:.3f}..{RATIO_MAX:.3f}]")
    if not word_freq_ok(text):
        problems.append(f"top word frequency {top_word_freq(text):.3f} above {MAX_WORD_FREQ}")
    if not word_diversity_ok(text):
        problems.append(f"word diversity {word_diversity(text):.3f} below {MIN_DIVERSITY}")
    latest = max(0, count_words(text) - sum(prev_totals))
    if not growth_ok(latest, prev_totals):
        increments = _increments(prev_totals) or [sum(prev_totals)]
        avg = sum(increments) / len(increments)
        problems.append(f"latest segment adds {latest} words vs mean previous segment ~{avg:.1f}")
    return (not problems, "; ".join(problems))


# Per-worker-process cache of loaded ASR models, keyed by (model, device, compute_type).
# ProcessPoolExecutor reuses worker processes, so each worker loads the model once.
_WHISPER_CACHE: dict[tuple, object] = {}


def _worker_transcribe(model: str, num_beams: int, samples: np.ndarray) -> str:
    """Runs in a worker process: load/cache the faster-whisper model and transcribe samples.

    Module-level so it is picklable for ProcessPoolExecutor; only settings and the numpy
    samples cross the process boundary, never the service instance."""
    try:
        import torch
        from faster_whisper import WhisperModel
    except Exception as e:
        log.error("failed to import ASR deps for %s: %s", model, e)
        return ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16"
    key = (model, device, compute_type)
    pipe = _WHISPER_CACHE.get(key)
    if pipe is None:
        try:
            pipe = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as e:
            log.error("failed to load ASR model %s: %s", model, e)
            return ""
        _WHISPER_CACHE[key] = pipe
        log.info("ASR model loaded %s (device=%s compute=%s)", model, device, compute_type)
    try:
        segments, info = pipe.transcribe(samples, beam_size=num_beams)
        # segments is a generator; transcription starts when it is iterated.
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.warning("transcribe error: %s", e)
        return ""


_NLLB_LANG_CODES = {
    "russian": "rus_Cyrl",
    "english": "eng_Latn",
    "german": "deu_Latn",
    "french": "fra_Latn",
    "spanish": "spa_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "ukrainian": "ukr_Cyrl",
    "polish": "pol_Latn",
    "chinese": "zho_Hans",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
}


def nllb_lang_code(name: str) -> str:
    """Map a common language name to its NLLB code; unknown names pass through unchanged."""
    key = (name or "").lower()
    if key in _NLLB_LANG_CODES:
        return _NLLB_LANG_CODES[key]
    return name or ""


_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping ending punctuation attached; [] for empty input"""
    if not text:
        return []
    out = []
    for part in _SENTENCE_RE.findall(text):
        s = part.strip()
        if s:
            out.append(s)
    return out


# Per-worker-process cache of loaded NLLB translators, keyed by model id.
_TRANSLATE_CACHE: dict[tuple, object] = {}


def _worker_translate(model: str, lang_code: str, text: str) -> str:
    """Runs in a worker process: load/cache the NLLB translator and translate text.

    Module-level so it is picklable for ProcessPoolExecutor; only settings and the source
    text cross the process boundary, never the service instance."""
    key = (model,)
    cached = _TRANSLATE_CACHE.get(key)
    if cached is None:
        try:
            from transformers import AutoModelForSeq2SeqLM
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
            translator = AutoModelForSeq2SeqLM.from_pretrained(model)
        except Exception as e:
            log.error("failed to load translator %s: %s", model, e)
            return text
        cached = (tokenizer, translator)
        _TRANSLATE_CACHE[key] = cached
        log.info("translator loaded %s", model)
    tokenizer, translator = cached
    try:
        inputs = tokenizer(text, return_tensors="pt")
        tgt_id = tokenizer.convert_tokens_to_ids(lang_code)
        translated_tokens = translator.generate(
            **inputs, forced_bos_token_id=tgt_id
        )
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
    except Exception as e:
        log.debug("translate error: %s", e)
        return text


class AudioBuffer:
    """Ring buffer holding up to N audio segments with stable group counter."""

    def __init__(self, size: int = 3) -> None:
        self.size = size
        self._slots = [None] * size
        # Recognized word count stored per element (in fill order of the current group).
        self._word_counts: list[int] = []
        # Recognized text per slot in fill order; joined for translation.
        self._text_slots: list[str | None] = [None] * size
        # Per-group sentence -> translation cache so unchanged sentences are not re-translated.
        self._translations: dict[str, str] = {}
        self.group_id = 0

    def add(self, data: bytes) -> None:
        # buffer full -> start new group; cumulative word totals restart from zero too
        if all(s is not None for s in self._slots):
            self.clear_slots()
            self._word_counts = []
            self.group_id += 1
        # first item in a fresh group
        if all(s is None for s in self._slots):
            if self.group_id == 0:
                self.group_id = 1
            else:
                # rolled over: cumulative totals restart from zero with the new group
                self._word_counts = []
        # fill first empty slot
        for i in range(self.size):
            if self._slots[i] is None:
                self._slots[i] = data
                log.debug("buffer slot %d filled, group=%d", i, self.group_id)
                return

    def clear_slots(self) -> None:
        self._slots = [None] * self.size
        self._text_slots = [None] * self.size
        self._translations = {}

    def clear(self) -> None:
        self.clear_slots()
        self._word_counts = []
        self.group_id = 0

    def set_word_count(self, index: int, count: int) -> None:
        """Store the recognized word total for the group state that ended at `index` (1-based)."""
        while len(self._word_counts) < index:
            self._word_counts.append(0)
        self._word_counts[index - 1] = max(0, int(count))

    def word_count_at(self, index: int) -> int | None:
        """Stored total for the group state that ended at `index` (1-based); None if absent."""
        if 1 <= index <= len(self._word_counts):
            return self._word_counts[index - 1]
        return None

    def add_text(self, index: int, text: str) -> None:
        """Store recognized text for the slot that ended at `index` (1-based fill order)."""
        if 1 <= index <= self.size:
            self._text_slots[index - 1] = text

    def collect_text(self) -> str:
        """All stored segment texts joined by a space, in fill order; "" when empty."""
        return " ".join(t for t in self._text_slots if t)

    def collect(self) -> bytes:
        parts = [b for b in self._slots if b is not None]
        return b"".join(parts)

    def latest(self) -> bytes:
        """Data of the most recently added segment (last filled slot); b"" when empty."""
        for i in range(self.size - 1, -1, -1):
            if self._slots[i] is not None:
                return self._slots[i]
        return b""

    def has_data(self) -> bool:
        return any(s is not None for s in self._slots)

    def filled_count(self) -> int:
        """How many buffer slots currently hold a segment (0..size)."""
        return sum(1 for s in self._slots if s is not None)

    def file_name(self, audio_out_dir: str) -> str | None:
        """Return full path for WAV file named audio_<group_id>_segment_X_Y.wav.

        Returns None if audio_out_dir is empty.
        """
        if not audio_out_dir:
            return None
        # filled slot count doubles as the "which part" index of the last added segment
        slot_index = self.filled_count()
        if slot_index == 0:
            return None
        group_str = f"{self.group_id:03d}"
        fname = f"audio_{group_str}_segment_{slot_index}_{self.size}.wav"
        return os.path.join(audio_out_dir, fname)


class RecognitionService:
    """ASR + translation service with 3-segment audio buffer."""

    def __init__(
        self,
        *names: str,
        model: str = "",
        translate_model: str = "",
        from_lang: str = "english",
        to_lang: str = "english",
        num_beams: int = 1,
        audio_out: str = "",
        parallel: int = 1,
        validate: bool = True,
    ) -> None:
        self.model_id = model
        # Separate NLLB translator for the target language; empty disables translation.
        self.translate_model_id = translate_model
        self.from_lang = from_lang.lower()
        self.to_lang = to_lang.lower()
        # Greedy (num_beams=1) is the fastest decoding mode; recommended for short audio.
        self.num_beams = max(1, int(num_beams))
        self.audio_out = audio_out
        self.parallel = max(1, int(parallel))
        # When False the recognized-text quality checks are skipped.
        self.validate = validate

        self._buffers: dict[str, AudioBuffer] = {}
        self.outboxes: dict[str, asyncio.Queue] = {}
        for name in names:
            self._buffers[name] = AudioBuffer()
            self.outboxes[name] = asyncio.Queue()

        # The ASR model is created inside each worker process; the main
        # process only holds settings and dispatches numpy samples to the pool.
        self._executor = ProcessPoolExecutor(max_workers=self.parallel)

    async def reset_session(self, name: str) -> None:
        buf = self._buffers.get(name)
        if buf:
            buf.clear()
        q = self.outboxes.get(name)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self._clear_audio_dir()
        log.info("session reset for %s", name)

    def _clear_audio_dir(self) -> None:
        """Remove leftover audio_*.wav files from a previous session."""
        if not self.audio_out:
            return
        d = Path(self.audio_out)
        if not d.is_dir():
            return
        removed = 0
        for p in d.iterdir():
            name = p.name
            if name.startswith("audio_") and name.endswith(".wav") and p.is_file():
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    log.debug("could not remove %s: %s", p, e)
        if removed:
            log.info("removed %d stale audio file(s) from %s", removed, self.audio_out)

    def shutdown(self) -> None:
        """Shut down the worker process pool, cancelling pending recognition tasks."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def process_segment(self, name: str, frame_name: str, audio: bytes) -> None:
        buf = self._buffers.get(name)
        if not buf:
            raise KeyError(name)
        outbox = self.outboxes[name]

        buf.add(audio)

        if not self.model_id:
            await outbox.put((buf.group_id, ""))
            return

        try:
            # Recognize only the newest segment; the model no longer re-processes earlier ones.
            latest_audio = buf.latest()
            # Save audio before recognition using collected data (full group picture for debugging).
            if self.audio_out:
                self._save_wav_for_buffer(buf, buf.collect())

            # Every audio slot gets a matching text entry (empty when there is no usable text) so the
            # text buffer stays in lockstep with the continuously filled audio slots.
            slot_index = buf.filled_count()

            samples = np.frombuffer(latest_audio, dtype="<f4")
            if samples.size == 0:
                buf.add_text(slot_index, "")
                return

            # Only the latest segment reaches the model this round.
            log.info(
                "recognition input for %s group=%d: %.2fs",
                name, buf.group_id, samples.size / TARGET_SAMPLE_RATE,
            )

            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            text = await loop.run_in_executor(
                self._executor,
                _worker_transcribe,
                self.model_id,
                self.num_beams,
                samples,
            )
            elapsed = time.perf_counter() - t0
            log.info(
                "recognition took %.2fs for %s group=%d",
                elapsed, name, buf.group_id,
            )

            if not text:
                # Silence: record an empty entry so this slot still matches its audio segment.
                buf.add_text(slot_index, "")
                return

            # Control the recognized text before sending; skipped when disabled.
            if self.validate:
                ok, reason = self._validate_text(buf, text)
                if not ok:
                    log.warning(
                        "validation failed for %s group=%d segment=%d: %s; text=%r",
                        name,
                        buf.group_id,
                        slot_index,
                        reason,
                        text[:120],
                    )
                    # Rejected: no contribution to the accumulated group, but keep the slot aligned.
                    buf.add_text(slot_index, "")
                    return
            else:
                log.debug("segment validation disabled for %s group=%d", name, buf.group_id)

            buf.set_word_count(slot_index, count_words(text))
            # Store the recognized segment text so translation can work on the accumulated group.
            buf.add_text(slot_index, text)
            # Surface per-slot state to spot any desync between audio and text slots.
            log.debug("buffer text state group=%d slots=%r", buf.group_id, buf._text_slots)

            # The client receives the full accumulated group content each step; when a translator is
            # configured it is translated first. NLLB only translates a single sentence
            # reliably, so the group text is split into sentences and each is translated separately
            # before being joined back together
            full_text = buf.collect_text()
            msg_text = full_text
            if self._should_translate():
                # Log exactly what goes into the translator so it can be compared with the result.
                log.info("translation input for %s group=%d: %r", name, buf.group_id, full_text)
                t0 = time.perf_counter()
                msg_text = await self._translate_sentences(loop, buf, full_text)
                log.info(
                    "translation took %.2fs for %s group=%d", time.perf_counter() - t0, name, buf.group_id,
                )

            await outbox.put((buf.group_id, msg_text))
            log.info(
                "transcription for %s group=%d segment=%d words=%d text=%r src=%r",
                name, buf.group_id, slot_index, count_words(msg_text), msg_text, text,
            )
        except Exception as e:
            log.warning("processing failed for %s: %s", name, e)

    async def _translate_sentences(self, loop, buf: "AudioBuffer", full_text: str) -> str:
        """Translate each sentence of `full_text` and join in order.
        NLLB emits EOS after a single sentence on multi-sentence input, so the group text is split into
        sentences; already-translated ones are cached per group to avoid re-translation."""
        lang_code = nllb_lang_code(self.to_lang)
        sentences = split_sentences(full_text)
        todo = list(dict.fromkeys(s for s in sentences if s not in buf._translations))
        if todo:
            results = await asyncio.gather(*[
                loop.run_in_executor(
                    self._executor, _worker_translate, self.translate_model_id, lang_code, s,
                )
                for s in todo
            ])
            buf._translations.update(zip(todo, results))
        return " ".join(buf._translations[s] for s in sentences if buf._translations.get(s))

    def _validate_text(self, buf: "AudioBuffer", text: str) -> tuple[bool, str]:
        """Word/char ratio check + growth of the latest segment over recorded previous totals."""
        prev_totals = []
        for idx in range(1, max(buf.size - 1, 1)):
            val = buf.word_count_at(idx)
            if val is None:
                break
            prev_totals.append(val)
        return validate_recognition(text, prev_totals)

    def _should_translate(self) -> bool:
        """Translate only when a translator model and a non-English target language are set"""
        return bool(self.translate_model_id) and self.to_lang not in ("", "english")

    def _save_wav_for_buffer(self, buf: "AudioBuffer", data: bytes) -> None:
        """Save merged buffer data as WAV to audio_out directory.

        Uses buf.file_name() for naming. Called before recognition.
        """
        if not self.audio_out or not data:
            return
        path = buf.file_name(self.audio_out)
        if not path:
            return
        os.makedirs(self.audio_out, exist_ok=True)
        try:
            samples = np.frombuffer(data, dtype="<f4")
        except ValueError:
            log.warning("RecognitionService._save_wav_for_buffer: data length not multiple of 4")
            return
        buf_io = io.BytesIO()
        with wave.open(buf_io, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SAMPLE_RATE)
            for s in samples:
                s_clamped = max(-1.0, min(1.0, float(s)))
                w.writeframes(struct.pack("<h", int(s_clamped * 32767)))
        with open(path, "wb") as f:
            f.write(buf_io.getvalue())
        log.debug("saved audio to %s (%d bytes)", path, len(data))


