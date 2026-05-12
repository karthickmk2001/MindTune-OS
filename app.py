"""MindTune OS — Hugging Face Spaces demo.

Simulates an EEG replay session:
  calm → relaxed → stressed → AI music suggestion → calm

No Spotify or Arduino required. Only the Groq API key is needed
(free tier at console.groq.com). Set it as a Space secret named
GROQ_API_KEY, or enter it directly in the UI.
"""

import os
import re
import random
import time

import gradio as gr
from groq import Groq

# ── EEG simulation constants ──────────────────────────────────────────────────

STATE_COLORS  = {"calm": "#22c55e", "relaxed": "#3b82f6", "stressed": "#ef4444"}
STATE_EMOJI   = {"calm": "😌", "relaxed": "😊", "stressed": "😰"}
STATE_LABELS  = {"calm": "CALM", "relaxed": "RELAXED", "stressed": "STRESSED"}

# Neuroscience-based band power profiles (0 = minimal activity, 1 = peak activity)
#   Alpha ↓  during stress (suppressed by cortisol)
#   Beta  ↑  during stress (active thinking / anxiety marker)
#   Theta ↑  during mental load
BAND_PROFILES = {
    "calm":     {"Delta": 0.33, "Theta": 0.38, "Alpha": 0.74, "Beta": 0.22, "Gamma": 0.18},
    "relaxed":  {"Delta": 0.41, "Theta": 0.52, "Alpha": 0.56, "Beta": 0.43, "Gamma": 0.32},
    "stressed": {"Delta": 0.58, "Theta": 0.80, "Alpha": 0.17, "Beta": 0.88, "Gamma": 0.72},
}

# 40-tick demo scenario with two stress episodes
_DEMO_SEQ = (
    ["calm"]    * 8  +
    ["relaxed"] * 4  +
    ["stressed"]* 7  +   # first stress episode → triggers AI suggestion
    ["relaxed"] * 5  +
    ["calm"]    * 5  +
    ["stressed"]* 5  +   # second (brief) stress episode
    ["relaxed"] * 6
)
DEMO_SEQ = _DEMO_SEQ * 2   # ~80-second demo


# ── Groq / fallback suggestion ────────────────────────────────────────────────

_FALLBACK = [
    ("ambient piano stress relief",       "Fallback — gentle piano proven for relaxation"),
    ("nature sounds rain meditation",     "Fallback — rain sounds calm the nervous system"),
    ("Weightless Marconi Union",          "Fallback — clinically studied stress-reduction track"),
    ("soft classical strings calm",       "Fallback — slow strings lower cortisol"),
    ("binaural beats alpha waves focus",  "Fallback — alpha-frequency audio for relaxed state"),
]


def _fallback(tried):
    tried_lower = {q.lower() for q in tried}
    for q, r in _FALLBACK:
        if q.lower() not in tried_lower:
            return q, r
    return _FALLBACK[0]


def get_suggestion(stress_history, tried, wins, fails, groq_key):
    """Ask Groq for a Spotify search query; fall back gracefully on any error."""
    key = groq_key.strip() or os.getenv("GROQ_API_KEY", "")
    if not key:
        return _fallback(tried)

    tried_str = ", ".join(tried[-10:]) if tried else "None yet"
    wins_str  = "\n".join(wins[-5:])   if wins  else "None yet"
    fails_str = "\n".join(fails[-3:])  if fails else "None yet"

    prompt = (
        f"The system is in CALM mode.\n"
        f"User EEG readings: {stress_history}\n"
        f"Queries already tried — do NOT repeat: {tried_str}\n\n"
        f"GOAL: Stress relief. Suggest low-BPM, acoustic, gentle music "
        f"(ambient, piano, nature sounds) to aid relaxation.\n\n"
        f"Past tracks that WORKED:\n{wins_str}\n\n"
        f"Past tracks that FAILED:\n{fails_str}\n\n"
        f"Respond in EXACTLY this format and nothing else:\n"
        f"QUERY: <Spotify search query, 3–6 words>\n"
        f"REASON: <one sentence explaining the choice>"
    )

    try:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100,
        )
        text = resp.choices[0].message.content
        query = reason = None
        for raw in text.splitlines():
            line = re.sub(r"[\*\_]+", "", raw).strip()
            if re.match(r"QUERY\s*:", line, re.IGNORECASE):
                query  = re.split(r"QUERY\s*:",  line, maxsplit=1, flags=re.IGNORECASE)[1].strip()
            elif re.match(r"REASON\s*:", line, re.IGNORECASE):
                reason = re.split(r"REASON\s*:", line, maxsplit=1, flags=re.IGNORECASE)[1].strip()
        if query and reason:
            return query, reason
    except Exception as exc:
        pass   # fall through to fallback

    return _fallback(tried)


# ── HTML rendering helpers ────────────────────────────────────────────────────

def _state_card(state):
    c = STATE_COLORS[state]
    return (
        f'<div style="text-align:center;padding:22px 10px;background:{c}18;'
        f'border:2px solid {c};border-radius:14px;">'
        f'<div style="font-size:2.6em;margin-bottom:4px;">{STATE_EMOJI[state]}</div>'
        f'<div style="font-size:1.5em;font-weight:700;color:{c};">{STATE_LABELS[state]}</div>'
        f'</div>'
    )


def _stress_meter(n):
    c = "#ef4444" if n >= 3 else "#f59e0b" if n >= 2 else "#22c55e"
    dots = "●" * n + "○" * (5 - n)
    return (
        f'<div style="text-align:center;font-size:1.6em;letter-spacing:5px;color:{c};">{dots}</div>'
        f'<div style="text-align:center;color:#94a3b8;font-size:0.85em;margin-top:4px;">'
        f'{n}/5 stress score</div>'
    )


def _band_bars(bands):
    rows = ""
    for band, score in bands.items():
        pct = int(score * 100)
        if band == "Alpha":
            # Alpha: high = calm (green), low = stressed (red)
            hue = int(120 * score)
        elif band in ("Beta", "Gamma"):
            # Beta/Gamma: high = stressed (red), low = calm (green)
            hue = int(120 * (1 - score))
        else:
            hue = 220   # purple-blue for Delta/Theta
        color = f"hsl({hue},65%,50%)"
        rows += (
            f'<div style="display:flex;align-items:center;margin:5px 0;">'
            f'<span style="width:52px;color:#94a3b8;font-size:0.9em;">{band}</span>'
            f'<div style="background:#1e293b;border-radius:4px;flex:1;height:16px;margin:0 8px;">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;'
            f'transition:width 0.4s ease;"></div></div>'
            f'<span style="width:38px;color:#e2e8f0;font-size:0.85em;text-align:right;">'
            f'{score:.2f}</span></div>'
        )
    return f'<div style="font-family:monospace;padding:6px 0;">{rows}</div>'


# ── Simulation generator ──────────────────────────────────────────────────────

def simulate(groq_key):
    """Yields (state_html, stress_html, bands_html, suggestion_text, log_text)."""
    recent = []
    tried  = []
    wins   = []
    fails  = []
    log    = []
    last_suggestion  = "Waiting for EEG data…"
    intervention_on  = False

    for tick, state in enumerate(DEMO_SEQ):
        bands = {
            k: round(min(1.0, max(0.0, v + random.gauss(0, 0.045))), 3)
            for k, v in BAND_PROFILES[state].items()
        }

        recent = (recent + [state])[-5:]
        n_stress = recent.count("stressed")

        # Trigger intervention when stress hits 3 / 5
        if n_stress >= 3 and not intervention_on:
            intervention_on = True
            query, reason = get_suggestion(recent, tried, wins, fails, groq_key)
            tried.append(query)
            last_suggestion = (
                f"🎵  Search Spotify for:\n\"{query}\"\n\n"
                f"💭  Groq reasoning:\n{reason}"
            )
            log.append(f"[{tick:03d}] ⚠  STRESS detected — AI suggests: {query[:45]}")

        elif n_stress <= 1 and intervention_on:
            intervention_on = False
            wins.append(f"WIN: Stress resolved after {query!r}")
            last_suggestion = f"✅  Stress reduced!\n\n{last_suggestion}"
            log.append(f"[{tick:03d}] ✓  Calm restored")

        log.append(
            f"[{tick:03d}] {state:8s}  stress={n_stress}/5  "
            f"β={bands['Beta']:.2f}  α={bands['Alpha']:.2f}"
        )
        log = log[-22:]

        yield (
            _state_card(state),
            _stress_meter(n_stress),
            _band_bars(bands),
            last_suggestion,
            "\n".join(log[-16:]),
        )
        time.sleep(1.0)

    # End of demo
    yield (
        '<div style="text-align:center;padding:22px;color:#94a3b8;">Demo complete — click Start to replay</div>',
        '<div style="text-align:center;color:#94a3b8;">–</div>',
        "",
        last_suggestion,
        "\n".join(log[-16:]) + "\n\n[Demo finished]",
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = """
body, .gradio-container { background: #0f172a !important; }
.gr-textbox textarea { font-family: monospace; font-size: 0.85em; }
"""

with gr.Blocks(
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    title="MindTune OS",
    css=_CSS,
) as demo:
    gr.Markdown(
        """
# 🧠 MindTune OS — EEG Adaptive Music Demo
**MSc Foundations of AI · National College of Ireland**

Replays a pre-recorded EEG session. When sustained stress is detected (≥ 3/5),
a **Groq LLM** (Llama 3.3 70B) suggests a Spotify search query personalised to your mental state history.
        """
    )

    with gr.Row():
        groq_input = gr.Textbox(
            label="Groq API Key  (free at console.groq.com — or set as Space secret GROQ_API_KEY)",
            placeholder="gsk_…",
            type="password",
            scale=4,
        )
        start_btn = gr.Button("▶  Start Demo", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            state_out   = gr.HTML(
                value='<div style="text-align:center;padding:22px;color:#94a3b8;">Press Start</div>',
                label="Mental State",
            )
            stress_out  = gr.HTML(
                value='<div style="text-align:center;color:#94a3b8;font-size:1.6em;">○○○○○</div>',
                label="Stress Level",
            )
        with gr.Column(scale=1):
            bands_out   = gr.HTML(label="EEG Frequency Bands")

    suggestion_out = gr.Textbox(
        label="AI Music Suggestion  (Groq → Llama 3.3 70B)",
        lines=4,
        interactive=False,
    )
    log_out = gr.Textbox(
        label="Session Log",
        lines=10,
        interactive=False,
    )

    gr.Markdown(
        """
---
**Architecture:** EEG replay → SGD stress classifier → Groq agent → Spotify query
**EEG dataset:** [birdy654/eeg-brainwave-dataset-mental-state](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state)
**Source:** [github.com/karthickmk2001/MindTune-OS](https://github.com/karthickmk2001/MindTune-OS)
        """
    )

    start_btn.click(
        fn=simulate,
        inputs=[groq_input],
        outputs=[state_out, stress_out, bands_out, suggestion_out, log_out],
    )

if __name__ == "__main__":
    demo.launch()
