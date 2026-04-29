import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pretty_midi
import torch
from torch import nn


DEFAULT_CACHE = Path("data/project_outputs/cache/sequence_cache_256seq.json")
DEFAULT_CHECKPOINT = Path(
    "data/project_outputs/cache/07_transformer_20260428_212800__final_transformer_quantized_time.pt"
)
DEFAULT_TIME_BIN_SEC = 0.01
DEFAULT_SEQ_LEN = 256
DEFAULT_GEN_TOKENS = 128
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLE_RATE = 22050

DEFAULT_MAX_TIME_SHIFT_STEP = 8
# After this many consecutive TIME_SHIFT_* tokens in the generated suffix alone,
# the next step cannot sample any TIME_SHIFT_* (forces a note/off or other event).
DEFAULT_MAX_CONSECUTIVE_TIME_SHIFTS = 2


def build_time_shift_id_mask(id2tok: Dict[int, str], vocab_size: int, device: torch.device) -> torch.Tensor:
    """True for token ids whose string starts with TIME_SHIFT_ (any variant)."""
    m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        t = id2tok.get(i, "")
        if isinstance(t, str) and t.startswith("TIME_SHIFT_"):
            m[i] = True
    return m


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerNextToken(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos = PositionalEncoding(emb_dim, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.pos(h)
        h = self.enc(h)
        h = self.drop(h[:, -1, :])
        return self.fc(h)


def load_cache(cache_path: Path) -> Dict:
    with cache_path.open("r") as f:
        cache = json.load(f)
    if "tok2id" not in cache or "id2tok" not in cache:
        raise ValueError("Cache must contain tok2id and id2tok mappings.")
    cache["id2tok"] = {int(k): v for k, v in cache["id2tok"].items()}
    return cache


def parse_checkpoint(ckpt_obj: object) -> Tuple[Dict, Dict]:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"], ckpt_obj
        if any(k.startswith("emb.") or k.startswith("fc.") for k in ckpt_obj.keys()):
            return ckpt_obj, {}
    raise ValueError("Unsupported checkpoint format. Expected state_dict or plain model weights dict.")


def load_model(
    ckpt_path: Path,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
    emb_dim: int,
    nhead: int,
    num_layers: int,
    ff_mult: int,
    dropout: float,
) -> TransformerNextToken:
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict, meta = parse_checkpoint(ckpt_obj)
    cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
    emb_dim = int(cfg.get("emb_dim", emb_dim))
    nhead = int(cfg.get("nhead", nhead))
    num_layers = int(cfg.get("num_layers", num_layers))
    ff_mult = int(cfg.get("ff_mult", ff_mult))
    dropout = float(cfg.get("dropout", dropout))
    if "pos.pe" in state_dict:
        max_len = int(state_dict["pos.pe"].shape[1])
    else:
        max_len = max(int(cfg.get("seq_len", seq_len)), seq_len, 512)

    model = TransformerNextToken(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        nhead=nhead,
        num_layers=num_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        max_len=max_len,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def notes_from_midi(midi_path: str) -> List[Tuple[float, float, int, int]]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    notes.sort(key=lambda x: x[0])
    return notes


def qtime(x: float, bin_sec: float) -> float:
    return float(np.round(x / bin_sec) * bin_sec)


def notes_to_events(notes: List[Tuple[float, float, int, int]], time_bin: float) -> List[str]:
    events = []
    raw_events = []
    for s, e, p, _v in notes:
        s = qtime(s, time_bin)
        e = float(max(qtime(e, time_bin), s + time_bin))
        raw_events.append((s, f"NOTE_ON_{p}"))
        raw_events.append((e, f"NOTE_OFF_{p}"))
    raw_events.sort(key=lambda x: x[0])

    prev_t = 0.0
    for t, ev in raw_events:
        dt = max(0.0, t - prev_t)
        if dt > 0:
            step = int(max(1, round(dt / time_bin)))
            events.append(f"TIME_SHIFT_{step}")
        events.append(ev)
        prev_t = t
    return events


def tokens_to_midi(tokens: List[str], time_bin: float) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    current_t = 0.0
    active: Dict[int, List[Tuple[float, int]]] = {}

    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            try:
                step = int(tok.split("_")[-1])
            except ValueError:
                continue
            current_t += max(1, step) * time_bin
            continue

        if tok.startswith("NOTE_ON_"):
            try:
                pitch = int(tok.split("_")[-1])
            except ValueError:
                continue
            active.setdefault(pitch, []).append((current_t, 100))
            continue

        if tok.startswith("NOTE_OFF_"):
            try:
                pitch = int(tok.split("_")[-1])
            except ValueError:
                continue
            if pitch in active and active[pitch]:
                start, velocity = active[pitch].pop(0)
                end = max(current_t, start + time_bin)
                inst.notes.append(
                    pretty_midi.Note(velocity=int(velocity), pitch=pitch, start=float(start), end=float(end))
                )

    for pitch, starts in active.items():
        for start, velocity in starts:
            inst.notes.append(
                pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=float(start), end=float(start + time_bin))
            )

    pm.instruments.append(inst)
    return pm


def render_audio(pm: pretty_midi.PrettyMIDI, sample_rate: int) -> np.ndarray:
    audio = None
    try:
        audio = pm.fluidsynth(fs=sample_rate)
    except Exception:
        try:
            audio = pm.synthesize(fs=sample_rate)
        except Exception as exc:
            raise RuntimeError(
                "Could not render audio. Install pyfluidsynth+fluidsynth or ensure scipy is available for pretty_midi synthesis."
            ) from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        audio = np.zeros(sample_rate, dtype=np.float32)
    mx = np.max(np.abs(audio))
    if mx > 1e-6:
        audio = audio / mx
    return audio


def extract_excerpt(pm: pretty_midi.PrettyMIDI, start_sec: float, duration_sec: float) -> pretty_midi.PrettyMIDI:
    out = pretty_midi.PrettyMIDI()
    end_sec = start_sec + duration_sec
    for inst in pm.instruments:
        out_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if n.end <= start_sec or n.start >= end_sec:
                continue
            new_start = max(0.0, n.start - start_sec)
            new_end = min(end_sec, n.end) - start_sec
            out_inst.notes.append(
                pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=float(new_start), end=float(max(new_end, new_start + 0.01)))
            )
        out.instruments.append(out_inst)
    return out


def token_times(tokens: List[str], time_bin: float) -> List[float]:
    times = []
    cur = 0.0
    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            try:
                step = int(tok.split("_")[-1])
                cur += max(1, step) * time_bin
            except ValueError:
                pass
        times.append(cur)
    return times


def choose_context_at_time(token_ids: List[int], tokens: List[str], start_sec: float, seq_len: int, time_bin: float) -> List[int]:
    times = token_times(tokens, time_bin)
    if not times:
        raise ValueError("No token times could be computed from MIDI.")
    total_dur = times[-1]
    if start_sec < 0 or start_sec > total_dur:
        raise ValueError(f"start_sec={start_sec:.3f} is out of range [0, {total_dur:.3f}]")

    # Rule: choose context ending at the last token whose running token time is <= start_sec.
    end_idx = 0
    for i, t in enumerate(times):
        if t <= start_sec:
            end_idx = i
        else:
            break
    lo = max(0, end_idx - seq_len + 1)
    ctx = token_ids[lo : end_idx + 1]
    if not ctx:
        raise ValueError("Could not build non-empty context from tokenized MIDI.")
    return ctx


def sample_transformer(
    model: TransformerNextToken,
    prompt_ids: List[int],
    steps: int,
    temperature: float,
    seq_len: int,
    device: torch.device,
    rng: np.random.Generator,
) -> List[int]:
    out = list(prompt_ids)
    for _ in range(steps):
        x = torch.tensor([out[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs = probs / probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return out[len(prompt_ids) :]


def sample_transformer_capped_timeshift(
    model: TransformerNextToken,
    prompt_ids: List[int],
    steps: int,
    temperature: float,
    seq_len: int,
    device: torch.device,
    rng: np.random.Generator,
    id2tok: Dict[int, str],
    time_shift_id_mask: torch.Tensor,
    max_time_shift_step: int = DEFAULT_MAX_TIME_SHIFT_STEP,
    max_consecutive_time_shifts: int = DEFAULT_MAX_CONSECUTIVE_TIME_SHIFTS,
) -> List[int]:
    """
    Autoregressive sampling with two decoding rules (streak counts only *new* tokens, not the prompt):

    1) Per-step cap: if TIME_SHIFT_n has n > max_time_shift_step, mask that id once and resample.

    2) Consecutive cap: after MAX_CONSECUTIVE_TIME_SHIFTS (default 2) generated TIME_SHIFT_* in a row,
       mask *all* TIME_SHIFT_* ids for that step, softmax, multinomial.
       If no valid probability mass (numerical edge), argmax on log_work (deterministic).

    Example: gen TIME_SHIFT, gen TIME_SHIFT -> third generated token cannot be TIME_SHIFT
    (streak resets only after a non–TIME_SHIFT token).
    """
    _ = rng  # unigram uses numpy RNG; this path uses torch.multinomial on device
    out = list(prompt_ids)
    streak = 0

    for _ in range(steps):
        x = torch.tensor([out[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0] / max(float(temperature), 1e-6)

        log_work = logits.clone()
        if streak >= int(max_consecutive_time_shifts):
            log_work = log_work.masked_fill(time_shift_id_mask, float("-inf"))

        probs = torch.softmax(log_work, dim=-1)
        if torch.isnan(probs).any() or (probs.sum().item() <= 0) or not torch.isfinite(probs).all():
            idx = int(torch.argmax(log_work).item())
        else:
            idx = int(torch.multinomial(probs, num_samples=1).item())

        tok = id2tok.get(idx, "")
        if isinstance(tok, str) and tok.startswith("TIME_SHIFT_"):
            try:
                step = int(tok.split("_")[-1])
                if step > int(max_time_shift_step):
                    log2 = log_work.clone()
                    log2[idx] = float("-inf")
                    p2 = torch.softmax(log2, dim=-1)
                    if torch.isfinite(p2).all() and p2.sum().item() > 0 and not torch.isnan(p2).any():
                        idx = int(torch.multinomial(p2, num_samples=1).item())
                    else:
                        idx = int(torch.argmax(log2).item())
            except ValueError:
                pass

        if bool(time_shift_id_mask[idx].item()):
            streak += 1
        else:
            streak = 0
        out.append(idx)

    return out[len(prompt_ids) :]


def sample_unigram(prompt_ids: List[int], steps: int, vocab_size: int, rng: np.random.Generator) -> List[int]:
    if not prompt_ids:
        raise ValueError("Prompt is empty; cannot fit unigram baseline.")
    counts = np.bincount(np.array(prompt_ids, dtype=np.int64), minlength=vocab_size).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    if probs.sum() <= 0:
        probs = np.ones(vocab_size, dtype=np.float64) / vocab_size
    out = rng.choice(vocab_size, size=steps, p=probs)
    return [int(x) for x in out.tolist()]


def generated_duration_sec(gen_tokens: List[str], time_bin: float) -> float:
    total = 0.0
    for tok in gen_tokens:
        if tok.startswith("TIME_SHIFT_"):
            try:
                total += max(1, int(tok.split("_")[-1])) * time_bin
            except ValueError:
                pass
    return max(total, 3.0)


def resolve_cache_path(user_path: Optional[str]) -> Path:
    p = Path(user_path) if user_path else DEFAULT_CACHE
    if p.exists():
        return p
    raise FileNotFoundError(f"Cache path not found: {p}")


def build_app(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = resolve_cache_path(args.cache)
    cache = load_cache(cache_path)
    tok2id = cache["tok2id"]
    id2tok = cache["id2tok"]
    vocab_size = len(cache["vocab"])
    time_shift_id_mask = build_time_shift_id_mask(id2tok, vocab_size, device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model_state: Dict[str, object] = {
        "path": str(ckpt_path.resolve()),
        "model": load_model(
            ckpt_path,
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            device=device,
            emb_dim=args.emb_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
        ),
    }

    def _resolve_checkpoint_path(ckpt_path_str: Optional[str]) -> Path:
        if ckpt_path_str is None or not str(ckpt_path_str).strip():
            return Path(args.checkpoint).expanduser().resolve()
        return Path(str(ckpt_path_str).strip()).expanduser().resolve()

    def _resolve_ckpt_from_ui(ckpt_path_str: Optional[str], ckpt_upload) -> Path:
        """Upload wins if present; else optional text path; else CLI --checkpoint default."""
        if ckpt_upload is not None:
            up = ckpt_upload if isinstance(ckpt_upload, str) else getattr(ckpt_upload, "name", None)
            if up:
                return Path(str(up)).expanduser().resolve()
        return _resolve_checkpoint_path(ckpt_path_str)

    def _ensure_model_for_checkpoint(resolved: Path) -> None:
        key = str(resolved)
        if key == model_state["path"]:
            return
        if not resolved.exists():
            raise gr.Error(f"Checkpoint not found: {resolved}")
        try:
            model_state["model"] = load_model(
                resolved,
                vocab_size=vocab_size,
                seq_len=args.seq_len,
                device=device,
                emb_dim=args.emb_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                ff_mult=args.ff_mult,
                dropout=args.dropout,
            )
            model_state["path"] = key
        except Exception as exc:
            raise gr.Error(f"Failed to load checkpoint: {exc}") from exc

    def on_generate(
        ckpt_path_str: str,
        ckpt_upload,
        midi_file,
        start_sec: float,
        gen_tokens: int,
        temperature: float,
        seed: Optional[int],
    ):
        resolved_ckpt = _resolve_ckpt_from_ui(ckpt_path_str, ckpt_upload)
        _ensure_model_for_checkpoint(resolved_ckpt)
        model = model_state["model"]  # type: ignore[assignment]

        if midi_file is None:
            raise gr.Error("Please upload a MIDI file.")
        rng = np.random.default_rng(seed if seed is not None and seed >= 0 else None)

        midi_path = midi_file if isinstance(midi_file, str) else getattr(midi_file, "name", None)
        if midi_path is None:
            raise gr.Error("Could not read uploaded MIDI path.")

        try:
            notes = notes_from_midi(midi_path)
            pm_input = pretty_midi.PrettyMIDI(midi_path)
        except Exception as exc:
            raise gr.Error(f"Failed to parse MIDI: {exc}") from exc

        events = notes_to_events(notes, time_bin=args.time_bin)
        token_ids = [tok2id[t] for t in events if t in tok2id]
        tokens_kept = [t for t in events if t in tok2id]
        if len(token_ids) < 2:
            raise gr.Error("Tokenized MIDI is too short or mostly out-of-vocabulary for this model/cache.")

        try:
            prompt_ids = choose_context_at_time(token_ids, tokens_kept, float(start_sec), args.seq_len, args.time_bin)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        try:
            baseline_ids = sample_unigram(prompt_ids, int(gen_tokens), vocab_size, rng)
            transformer_ids = sample_transformer_capped_timeshift(
                model,
                prompt_ids=prompt_ids,
                steps=int(gen_tokens),
                temperature=float(temperature),
                seq_len=args.seq_len,
                device=device,
                rng=rng,
                id2tok=id2tok,
                time_shift_id_mask=time_shift_id_mask,
                max_time_shift_step=args.max_time_shift_step,
                max_consecutive_time_shifts=args.max_consecutive_time_shifts,
            )
        except Exception as exc:
            raise gr.Error(f"Generation failed: {exc}") from exc

        try:
            baseline_tokens = [id2tok[i] for i in baseline_ids]
            transformer_tokens = [id2tok[i] for i in transformer_ids]
        except KeyError as exc:
            raise gr.Error(f"Checkpoint/cache vocab mismatch: missing token id {exc}") from exc

        baseline_pm = tokens_to_midi(baseline_tokens, args.time_bin)
        transformer_pm = tokens_to_midi(transformer_tokens, args.time_bin)
        clip_dur = max(
            generated_duration_sec(baseline_tokens, args.time_bin),
            generated_duration_sec(transformer_tokens, args.time_bin),
        )
        original_pm = extract_excerpt(pm_input, float(start_sec), clip_dur)

        try:
            full_audio = render_audio(pm_input, args.sample_rate)
            orig_audio = render_audio(original_pm, args.sample_rate)
            base_audio = render_audio(baseline_pm, args.sample_rate)
            tf_audio = render_audio(transformer_pm, args.sample_rate)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return (
            (args.sample_rate, full_audio),
            (args.sample_rate, orig_audio),
            (args.sample_rate, base_audio),
            (args.sample_rate, tf_audio),
        )

    with gr.Blocks(title="MIDI Continuation Demo") as demo:
        gr.Markdown(
            "## MIDI Continuation Demo\n"
            "Upload MIDI, choose start time, compare baseline vs trained Transformer.\n\n"
            "**Checkpoint:** upload a `.pt` file *or* edit the path below. "
            "If you upload a file, it overrides the path for that run. "
            "Changing checkpoint reloads the model (only when the path changes)."
        )
        with gr.Row():
            ckpt_upload = gr.File(
                label="Optional: upload Transformer checkpoint (.pt)",
                file_types=[".pt"],
            )
            ckpt_path_in = gr.Textbox(
                label="Checkpoint path (.pt on disk)",
                value=str(ckpt_path.resolve()),
                lines=1,
            )
        with gr.Row():
            midi_in = gr.File(label="MIDI file (.mid/.midi)", file_types=[".mid", ".midi"])
            start_in = gr.Number(label="Start time (seconds)", value=0.0, precision=3)
        out_full = gr.Audio(label="Original uploaded MIDI (full)", type="numpy")
        with gr.Row():
            gen_in = gr.Slider(label="Generation length (tokens)", minimum=32, maximum=512, step=16, value=args.gen_tokens)
            temp_in = gr.Slider(label="Temperature", minimum=0.2, maximum=2.0, step=0.05, value=args.temperature)
            seed_in = gr.Number(label="Seed (optional, -1 for random)", value=-1, precision=0)

        btn = gr.Button("Generate")
        with gr.Row():
            out_orig = gr.Audio(label="Original excerpt", type="numpy")
            out_base = gr.Audio(label="Baseline continuation (unigram)", type="numpy")
            out_tf = gr.Audio(label="Transformer continuation", type="numpy")

        btn.click(
            on_generate,
            inputs=[ckpt_path_in, ckpt_upload, midi_in, start_in, gen_in, temp_in, seed_in],
            outputs=[out_full, out_orig, out_base, out_tf],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Gradio MIDI continuation demo.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to Transformer checkpoint (.pt).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(DEFAULT_CACHE),
        help="Path to sequence_cache JSON (vocab + tok2id/id2tok).",
    )
    parser.add_argument("--time-bin", type=float, default=DEFAULT_TIME_BIN_SEC, help="Tokenization time bin in seconds.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Context length used for Transformer sampling.")
    parser.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS, help="Default generation length slider value.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Default temperature slider value.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio output sample rate.")
    parser.add_argument("--emb-dim", type=int, default=192, help="Fallback emb_dim if checkpoint has no config metadata.")
    parser.add_argument("--nhead", type=int, default=6, help="Fallback nhead if checkpoint has no config metadata.")
    parser.add_argument("--num-layers", type=int, default=3, help="Fallback num_layers if checkpoint has no config metadata.")
    parser.add_argument("--ff-mult", type=int, default=4, help="Fallback feed-forward multiplier if checkpoint has no config metadata.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Fallback dropout if checkpoint has no config metadata.")
    parser.add_argument(
        "--max-time-shift-step",
        type=int,
        default=DEFAULT_MAX_TIME_SHIFT_STEP,
        help="Resample if sampled TIME_SHIFT_n has n above this cap.",
    )
    parser.add_argument(
        "--max-consecutive-time-shifts",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_TIME_SHIFTS,
        help="After this many consecutive TIME_SHIFT_* in generated tokens, forbid another TIME_SHIFT for one step.",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode (useful in Colab).")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_app(args)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
