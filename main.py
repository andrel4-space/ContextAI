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
st.set_page_config(
    page_title="ContextAI — Study when you're tired",
    layout="wide",
    page_icon="🌙",
)

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
        st.info("Your study sessions will show up here.")
    else:
        st.metric(label="Sessions saved", value=len(history))
        st.markdown("---")
        for item in history:
            with st.expander(f"🕒 {item[0]} · energy {item[1]}/5"):
                st.write(f"**Pressure:** {item[2]}")
                st.write(f"**You asked:** {item[3]}")
                st.markdown("**Next steps you got:**")
                st.info(item[4])

st.title("🌙 ContextAI")
st.subheader("Turn one assignment into what you can actually do tonight.")
st.write(
    "Built for night-shift and working students. Paste your assignment — "
    "when your energy is low, you get **only 3 short next steps**, not a wall of text."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("How are you right now?")
    energy_level = st.slider(
        "Energy tonight (1 = exhausted, 5 = wide awake):",
        min_value=1,
        max_value=5,
        value=2,
    )
    st.caption(ENERGY_LABELS.get(energy_level, ""))
    if is_low_energy(energy_level):
        st.info("Low energy mode: you'll get exactly **3 short next steps** — nothing extra.")

    pressure_state = st.select_slider(
        "Deadline pressure:",
        options=["Chill", "Moderate", "High Pressure"],
    )

with col2:
    st.header("Your assignment")
    raw_prompt = st.text_area(
        "Paste the assignment, rubric, or what you're stuck on:",
        placeholder=(
            "Example: Write a 500-word essay on the causes of WWI. "
            "Due Friday. I haven't started."
        ),
        height=140,
    )

st.markdown("---")

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
    st.markdown("### Your next steps")
    st.success(st.session_state.active_response)
