import html
import streamlit as st
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
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="ContextAI — Study when you're tired",
    layout="wide",
    page_icon="🌙",
)

inject_theme()

if "active_response" not in st.session_state:
    st.session_state.active_response = None

with st.sidebar:
    st.title("🌙 ContextAI")
    st.caption("For students working late — short answers, clear next steps.")

    if not gemini_key and not groq_key:
        st.error("Add GEMINI_API_KEY and/or GROQ_API_KEY to your `.env` file or hosting settings.")

    st.markdown("### Past sessions")
    history = fetch_historical_metrics()

    if not history:
        st.caption("Your study sessions will show up here.")
    else:
        st.metric(label="Sessions saved", value=len(history))
        st.markdown("---")
        for item in history:
            with st.expander(f"🕒 {item[0]} · energy {item[1]}/5"):
                st.write(f"**Pressure:** {item[2]}")
                st.write(f"**You asked:** {item[3]}")
                st.markdown("**Next steps you got:**")
                st.markdown(
                    f'<div class="next-steps-card">{html.escape(item[4])}</div>',
                    unsafe_allow_html=True,
                )

st.markdown("## 🌙 ContextAI")
st.markdown(
    '<p class="hero-tagline">Turn one assignment into what you can actually do tonight. '
    "Built for night-shift and working students — when your energy is low, you get "
    "<strong>only 3 short next steps</strong>, not a wall of text.</p>",
    unsafe_allow_html=True,
)

st.markdown("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### How are you right now?")
    energy_level = st.slider(
        "Energy tonight (1 = exhausted, 5 = wide awake):",
        min_value=1,
        max_value=5,
        value=2,
    )
    st.caption(ENERGY_LABELS.get(energy_level, ""))
    if is_low_energy(energy_level):
        st.markdown(
            '<p class="low-energy-chip">Low energy — exactly 3 short next steps, nothing extra</p>',
            unsafe_allow_html=True,
        )

    pressure_state = st.select_slider(
        "Deadline pressure:",
        options=["Chill", "Moderate", "High Pressure"],
    )

with col2:
    st.markdown("#### Your assignment")
    raw_prompt = st.text_area(
        "Paste the assignment, rubric, or what you're stuck on:",
        placeholder=(
            "Example: Write a 500-word essay on the causes of WWI. "
            "Due Friday. I haven't started."
        ),
        height=140,
        label_visibility="collapsed",
    )

st.markdown("")

if st.button("Get my next steps", type="primary", use_container_width=True):
    if not gemini_key and not groq_key:
        st.error("API keys are missing. Add them to `.env` or your host's environment variables.")
    elif not raw_prompt.strip():
        st.warning("Paste your assignment or question first.")
    else:
        output_text = None

        with st.spinner("Figuring out how much you can handle tonight..."):
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
                else:
                    final_engineered_payload = build_fallback_payload(
                        energy_level, pressure_state, raw_prompt
                    )
            except Exception:
                final_engineered_payload = build_fallback_payload(
                    energy_level, pressure_state, raw_prompt
                )

        with st.spinner("Writing your next steps..."):
            try:
                if gemini_key:
                    output_text = run_primary_model(final_engineered_payload)
                    st.success("Ready — sized for your energy tonight.")
                elif groq_key:
                    output_text = run_backup_model(final_engineered_payload)
                    st.success("Ready — served via backup.")
                else:
                    st.error("No API keys configured.")
            except Exception:
                if groq_key:
                    try:
                        output_text = run_backup_model(final_engineered_payload)
                        st.success("Primary AI was busy — used backup.")
                    except Exception as groq_error:
                        st.error(f"Could not reach AI right now: {groq_error}")
                else:
                    st.error("Could not reach AI. Check your API keys and try again.")

        if output_text:
            log_session_to_database(energy_level, pressure_state, raw_prompt, output_text)
            st.session_state.active_response = output_text
            st.rerun()

if st.session_state.active_response:
    st.markdown("#### Your next steps")
    st.markdown(
        f'<div class="next-steps-card">{html.escape(st.session_state.active_response)}</div>',
        unsafe_allow_html=True,
    )
