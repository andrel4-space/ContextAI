import html
import json
import re
import streamlit as st
import streamlit.components.v1 as components
import os
import sqlite3
from datetime import datetime
from google import genai
from groq import Groq
from dotenv import load_dotenv

# =====================================================================
# 1. DATABASE SYSTEM LAYER
# =====================================================================
def initialize_database():
    conn = sqlite3.connect("context_vault.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            energy_score INTEGER,
            pressure_threshold TEXT,
            raw_prompt TEXT,
            optimized_output TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_session_to_database(energy, pressure, original_text, final_output):
    try:
        conn = sqlite3.connect("context_vault.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO cognitive_logs (timestamp, energy_score, pressure_threshold, raw_prompt, optimized_output)
            VALUES (?, ?, ?, ?, ?)
        """, (now, energy, pressure, original_text, final_output))
        conn.commit()
        conn.close()
    except Exception as db_err:
        st.warning(f"Could not save this session: {db_err}")

def fetch_historical_metrics():
    try:
        conn = sqlite3.connect("context_vault.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, energy_score, pressure_threshold, raw_prompt, optimized_output "
            "FROM cognitive_logs ORDER BY id DESC"
        )
        logs = cursor.fetchall()
        conn.close()
        return logs
    except Exception:
        return []

initialize_database()

# =====================================================================
# 2. CONFIGURATION
# =====================================================================
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

primary_model = "gemini-2.5-flash"
backup_model = "llama-3.1-8b-instant"

ENERGY_LABELS = {
    1: "Running on empty",
    2: "Very tired",
    3: "Okay",
    4: "Focused",
    5: "Wide awake",
}

def is_low_energy(energy_level):
    return energy_level <= 2

def build_meta_synthesis_prompt(energy_level, pressure_state, raw_prompt):
    low_energy_rules = ""
    if is_low_energy(energy_level):
        low_energy_rules = """
                CRITICAL — LOW ENERGY STUDENT MODE:
                The user is a tired student (often after work or a night shift). The secondary LLM MUST:
                - Output EXACTLY 3 bullet points. No more, no fewer.
                - Each bullet is ONE short next step (under 15 words if possible).
                - No paragraphs, no essays, no preamble, no "here's a summary of the assignment."
                - Use plain language (8th-grade reading level).
                - If the task is an assignment, break it into the smallest doable steps for tonight only.
                """

    return f"""
                You are the delivery planner for ContextAI — built for students who are tired and do not want to read long AI answers.

                Your job is NOT to answer the assignment. Your job is to write strict instructions for a secondary LLM about HOW to answer.

                STUDENT STATE:
                - Energy tonight: {energy_level}/5 — {ENERGY_LABELS.get(energy_level, "")}
                - Pressure: {pressure_state}

                ASSIGNMENT OR QUESTION THEY PASTED:
                "{raw_prompt}"
                {low_energy_rules}
                GENERAL RULES:
                - Protect them from information overload.
                - Prefer action over explanation.
                - Match tone to pressure (High Pressure = calm and direct, Chill = friendly but still short).

                Output ONLY the instruction block for the secondary LLM. No greeting, no markdown titles.
                """

def build_execution_payload(dynamic_system_instruction, energy_level, pressure_state, raw_prompt):
    if is_low_energy(energy_level):
        format_lock = """
                OUTPUT FORMAT (mandatory):
                Reply with exactly these three lines and nothing else:
                1. [first next step]
                2. [second next step]
                3. [third next step]
                """
    else:
        format_lock = """
                OUTPUT FORMAT:
                - Start with 3 numbered next steps.
                - Then at most 3 short bullets of help if needed.
                - Keep total response under 150 words unless they asked for more detail.
                """

    return f"""SYSTEM RULES FROM CONTEXTAI:
{dynamic_system_instruction}
{format_lock}

STUDENT ENERGY: {energy_level}/5 ({ENERGY_LABELS.get(energy_level, "")})
PRESSURE: {pressure_state}

THEIR ASSIGNMENT OR QUESTION:
{raw_prompt}
"""

def build_fallback_payload(energy_level, pressure_state, raw_prompt):
    if is_low_energy(energy_level):
        return f"""You help exhausted students. Energy {energy_level}/5, pressure {pressure_state}.

Assignment: {raw_prompt}

Give EXACTLY 3 numbered next steps only. Each step one short line. No intro, no essay."""
    return f"""You help tired students study. Energy {energy_level}/5, pressure {pressure_state}.

Assignment: {raw_prompt}

Give 3 numbered next steps, then brief bullets only if needed. Under 150 words."""

def run_primary_model(payload):
    g_client = genai.Client(api_key=gemini_key)
    response = g_client.models.generate_content(model=primary_model, contents=payload)
    return response.text

def run_backup_model(payload):
    groq_client = Groq(api_key=groq_key)
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": payload}],
        model=backup_model,
    )
    return chat_completion.choices[0].message.content

def is_dev_mode():
    return os.getenv("CONTEXTAI_DEBUG", "").lower() in ("1", "true", "yes")

def api_keys_configured():
    return bool(gemini_key or groq_key)

def parse_next_steps(text):
    """Try to split AI output into three numbered steps for card UI."""
    if not text:
        return None
    steps = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^[\-*•]?\s*([1-3])\s*[\.\)\:\-]\s*(.+)$", line)
        if match:
            steps.append(match.group(2).strip())
            continue
        match = re.match(r"^[\-*•]\s*(.+)$", line)
        if match and len(steps) < 3:
            steps.append(match.group(1).strip())
    if len(steps) >= 3:
        return steps[:3]
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
    if len(chunks) >= 3:
        return chunks[:3]
    return None

def steps_plain_text(steps):
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))

def render_output_actions(plain_text):
    payload = json.dumps(plain_text)
    components.html(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.5rem 0 1rem 0;">
          <button id="copySteps" style="
            padding:0.45rem 0.9rem;border-radius:10px;border:1px solid rgba(126,184,218,0.4);
            background:rgba(126,184,218,0.15);color:#e8ecf4;cursor:pointer;font-size:0.9rem;">
            Copy steps
          </button>
          <button id="readSteps" style="
            padding:0.45rem 0.9rem;border-radius:10px;border:1px solid rgba(126,184,218,0.4);
            background:rgba(126,184,218,0.15);color:#e8ecf4;cursor:pointer;font-size:0.9rem;">
            Read aloud
          </button>
          <span id="copyStatus" style="color:#7eb8da;font-size:0.85rem;align-self:center;"></span>
        </div>
        <script>
        const text = {payload};
        document.getElementById("copySteps").onclick = async () => {{
          try {{
            await navigator.clipboard.writeText(text);
            document.getElementById("copyStatus").textContent = "Copied!";
            setTimeout(() => document.getElementById("copyStatus").textContent = "", 2000);
          }} catch (e) {{
            document.getElementById("copyStatus").textContent = "Copy failed — use Download below";
          }}
        }};
        document.getElementById("readSteps").onclick = () => {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(text);
          u.rate = 0.92;
          window.speechSynthesis.speak(u);
        }};
        </script>
        """,
        height=70,
    )

def render_step_cards(steps):
    for i, step in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div class="step-card">
              <div class="step-number">{i}</div>
              <div class="step-text">{html.escape(step)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_results(output_text):
    steps = parse_next_steps(output_text)
    plain = steps_plain_text(steps) if steps else output_text
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
    st.markdown("#### Your next steps")
    render_output_actions(plain)
    if steps:
        render_step_cards(steps)
    else:
        st.markdown(
            f'<div class="next-steps-card">{html.escape(output_text)}</div>',
            unsafe_allow_html=True,
        )
    st.download_button(
        label="Download steps",
        data=plain,
        file_name="contextai-next-steps.txt",
        mime="text/plain",
        use_container_width=True,
    )

def show_unavailable_message():
    if is_dev_mode():
        st.error("API keys missing. Set GEMINI_API_KEY and/or GROQ_API_KEY in `.env` or host settings.")
    else:
        st.error("ContextAI is briefly unavailable. Please try again in a few minutes.")

# =====================================================================
# 3. UI
# =====================================================================
def inject_theme():
    st.markdown(
        """
        <style>
        /* Full-page seamless gradient */
        .stApp {
            background: linear-gradient(165deg, #080b12 0%, #0f1624 38%, #121a2e 72%, #0d1219 100%);
            background-attachment: fixed;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(12, 16, 28, 0.97) 0%, rgba(18, 26, 42, 0.92) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] > div {
            background: transparent;
        }
        /* Softer blocks — no harsh white panels */
        [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 0.25rem 0.5rem;
            backdrop-filter: blur(12px);
        }
        h1, h2, h3, label, p, span, .stMarkdown {
            color: #e8ecf4 !important;
        }
        .hero-tagline {
            color: #9aa8bc !important;
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 42rem;
        }
        .low-energy-chip {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.5rem 0.85rem;
            border-radius: 999px;
            background: rgba(126, 184, 218, 0.12);
            border: 1px solid rgba(126, 184, 218, 0.28);
            color: #b8d4e8 !important;
            font-size: 0.9rem;
        }
        .next-steps-card {
            margin-top: 1rem;
            padding: 1.25rem 1.35rem;
            border-radius: 16px;
            background: rgba(126, 184, 218, 0.08);
            border: 1px solid rgba(126, 184, 218, 0.22);
            color: #e8ecf4;
            font-size: 1.05rem;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        div[data-testid="stTextArea"] textarea {
            background: rgba(0, 0, 0, 0.25) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: #e8ecf4 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] {
            margin-top: 0.25rem;
        }
        .stButton > button[kind="primary"] {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #5a8fb8 0%, #7eb8da 100%);
            color: #0c1018;
            font-weight: 600;
            padding: 0.65rem 1rem;
            box-shadow: 0 4px 20px rgba(126, 184, 218, 0.25);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6a9fc8 0%, #8ec8ea 100%);
            box-shadow: 0 6px 24px rgba(126, 184, 218, 0.35);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
        }
        .main-wrap {
            max-width: 42rem;
            margin: 0 auto;
            padding: 0 0.5rem 2rem 0.5rem;
        }
        .step-card {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.75rem;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(126, 184, 218, 0.08);
            border: 1px solid rgba(126, 184, 218, 0.2);
        }
        .step-number {
            flex-shrink: 0;
            width: 2rem;
            height: 2rem;
            line-height: 2rem;
            text-align: center;
            border-radius: 50%;
            background: rgba(126, 184, 218, 0.35);
            color: #0c1018 !important;
            font-weight: 700;
            font-size: 1rem;
        }
        .step-text {
            color: #e8ecf4 !important;
            font-size: 1.08rem;
            line-height: 1.5;
            padding-top: 0.15rem;
        }
        .font-large .step-text { font-size: 1.22rem !important; }
        .font-large .hero-tagline { font-size: 1.15rem !important; }
        .preset-row button {
            font-size: 0.88rem !important;
        }
        @media (max-width: 768px) {
            .main-wrap { padding: 0 0.25rem 2rem 0.25rem; }
            .step-text { font-size: 1.12rem !important; }
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="ContextAI — Study when you're tired",
    layout="centered",
    page_icon="🌙",
)

inject_theme()

if "active_response" not in st.session_state:
    st.session_state.active_response = None
if "energy" not in st.session_state:
    st.session_state.energy = 2
if "pressure" not in st.session_state:
    st.session_state.pressure = "Moderate"
if "font_large" not in st.session_state:
    st.session_state.font_large = False

with st.sidebar:
    st.title("🌙 ContextAI")
    st.caption("For students working late.")

    if not api_keys_configured() and is_dev_mode():
        st.error("Add GEMINI_API_KEY and/or GROQ_API_KEY to `.env` or host settings.")

    st.markdown("### Recent sessions")
    history = fetch_historical_metrics()[:5]

    if not history:
        st.caption("After your first run, your last sessions appear here.")
    else:
        for item in history:
            preview = item[3][:48] + ("…" if len(item[3]) > 48 else "")
            with st.expander(f"{item[0]} · ⚡{item[1]}/5"):
                st.caption(preview)
                st.write(f"**Pressure:** {item[2]}")
                parsed = parse_next_steps(item[4])
                if parsed:
                    for i, step in enumerate(parsed, 1):
                        st.write(f"**{i}.** {step}")
                else:
                    st.write(item[4])

font_class = "font-large" if st.session_state.font_large else ""
st.markdown(f'<div class="main-wrap {font_class}">', unsafe_allow_html=True)

st.markdown("## 🌙 ContextAI")
st.markdown(
    '<p class="hero-tagline">Paste your assignment. Get <strong>3 next steps</strong> '
    "you can start tonight — sized to how tired you are.</p>",
    unsafe_allow_html=True,
)

fa, fb = st.columns(2)
with fa:
    if st.button("A", help="Normal text size", use_container_width=True):
        st.session_state.font_large = False
        st.rerun()
with fb:
    if st.button("A+", help="Larger text size", use_container_width=True):
        st.session_state.font_large = True
        st.rerun()

st.markdown("#### 1. Your assignment")
raw_prompt = st.text_area(
    "Assignment",
    placeholder="Paste the assignment, rubric, or what you're stuck on…",
    height=130,
    label_visibility="collapsed",
    key="assignment_input",
)

st.markdown("#### 2. Quick mood")
st.caption("Tap a preset or adjust the sliders.")
p1, p2, p3 = st.columns(3)
with p1:
    if st.button("I'm wiped", use_container_width=True):
        st.session_state.energy = 1
        st.session_state.pressure = "Moderate"
        st.rerun()
with p2:
    if st.button("Normal night", use_container_width=True):
        st.session_state.energy = 3
        st.session_state.pressure = "Moderate"
        st.rerun()
with p3:
    if st.button("Due tomorrow", use_container_width=True):
        st.session_state.energy = 2
        st.session_state.pressure = "High Pressure"
        st.rerun()

energy_level = st.slider(
    "Energy tonight (1 = exhausted, 5 = wide awake)",
    min_value=1,
    max_value=5,
    key="energy",
)
st.caption(ENERGY_LABELS.get(energy_level, ""))
if is_low_energy(energy_level):
    st.markdown(
        '<p class="low-energy-chip">Tonight: exactly 3 short steps — nothing extra</p>',
        unsafe_allow_html=True,
    )

pressure_state = st.select_slider(
    "Deadline pressure",
    options=["Chill", "Moderate", "High Pressure"],
    key="pressure",
)

run_col, clear_col = st.columns([3, 1])
with run_col:
    run_clicked = st.button("Get my next steps", type="primary", use_container_width=True)
with clear_col:
    if st.button("Start over", use_container_width=True):
        st.session_state.active_response = None
        st.rerun()

if run_clicked:
    if not api_keys_configured():
        show_unavailable_message()
    elif not raw_prompt.strip():
        st.warning("Paste your assignment first.")
    else:
        output_text = None
        with st.spinner("Breaking this into tonight's steps…"):
            try:
                if gemini_key:
                    meta_synthesis_prompt = build_meta_synthesis_prompt(
                        energy_level, pressure_state, raw_prompt
                    )
                    dynamic_system_instruction = run_primary_model(meta_synthesis_prompt)
                    final_engineered_payload = build_execution_payload(
                        dynamic_system_instruction,
                        energy_level,
                        pressure_state,
                        raw_prompt,
                    )
                    output_text = run_primary_model(final_engineered_payload)
                elif groq_key:
                    final_engineered_payload = build_fallback_payload(
                        energy_level, pressure_state, raw_prompt
                    )
                    output_text = run_backup_model(final_engineered_payload)
            except Exception:
                try:
                    final_engineered_payload = build_fallback_payload(
                        energy_level, pressure_state, raw_prompt
                    )
                    if groq_key:
                        output_text = run_backup_model(final_engineered_payload)
                    elif gemini_key:
                        output_text = run_primary_model(final_engineered_payload)
                except Exception as err:
                    if is_dev_mode():
                        st.error(f"AI error: {err}")
                    else:
                        st.error("Could not get steps right now. Wait a moment and try again.")

        if output_text:
            log_session_to_database(energy_level, pressure_state, raw_prompt, output_text)
            st.session_state.active_response = output_text
            st.toast("Your next steps are ready.")
            st.rerun()

if st.session_state.active_response:
    st.markdown("---")
    render_results(st.session_state.active_response)
    if st.button("Start over with a new assignment", use_container_width=True):
        st.session_state.active_response = None
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
