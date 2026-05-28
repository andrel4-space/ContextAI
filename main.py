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
    conn = sqlite3.connect("context_vault.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO cognitive_logs (timestamp, energy_score, pressure_threshold, raw_prompt, optimized_output)
        VALUES (?, ?, ?, ?, ?)
    """, (now, energy, pressure, original_text, final_output))
    conn.commit()
    conn.close()

def fetch_historical_metrics():
    conn = sqlite3.connect("context_vault.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, energy_score, pressure_threshold, raw_prompt, optimized_output FROM cognitive_logs ORDER BY id DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs

initialize_database()

# =====================================================================
# 2. CONFIGURATION & STATE INITIALIZATION
# =====================================================================
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# Extract environment models with safety fallbacks
primary_model = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
backup_model = os.getenv("BACKUP_MODEL", "llama3-8b-8192")

st.set_page_config(page_title="ContextAI Management Console", layout="wide")

if "active_response" not in st.session_state:
    st.session_state.active_response = None

# =====================================================================
# 3. GRAPHICAL USER INTERFACE SIDEBAR
# =====================================================================
with st.sidebar:
    st.title("🛡️ ContextAI Vault")
    st.write("Secure, local analytical metric logging system.")
    
    st.markdown("### Historical Analytics Log")
    history = fetch_historical_metrics()
    
    if not history:
        st.info("No prior sessions recorded in local data schema.")
    else:
        st.metric(label="Total Optimizations Executed", value=len(history))
        st.markdown("---")
        for item in history:
            with st.expander(f"🕒 {item[0]}"):
                st.write(f"**Energy Bandwidth:** {item[1]}/5")
                st.write(f"**Pressure Context:** {item[2]}")
                st.write(f"**Raw Objective:** *{item[3]}*")
                st.markdown("---")
                st.write(f"**Saved AI Output:**")
                st.write(item[4])

# =====================================================================
# 4. MAIN INTERFACE ORCHESTRATION LAYER
# =====================================================================
st.title("🧠 ContextAI Core Console")
st.write("Production-grade human-centric middleware deployment.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Human Telemetry Ingestion")
    energy_level = st.slider("Select current cognitive energy:", min_value=1, max_value=5, value=3)
    pressure_state = st.select_slider("Current system pressure:", options=["Chill", "Moderate", "High Pressure"])

with col2:
    st.header("2. Objective Payload Input")
    raw_prompt = st.text_area("Input raw target prompt or operational objective:", placeholder="Paste objective parameters here...", height=115)

st.markdown("---")

if st.button("Execute Human-Aware Strategic Optimization", use_container_width=True):
    if not gemini_key:
        st.error("Missing Primary Security Token. Verify .env file state.")
    elif not raw_prompt.strip():
        st.warning("Please deliver an objective payload before forcing execution.")
    else:
        context_modifier = ""
        if energy_level <= 2:
            context_modifier += "CRITICAL FRAMEWORK: User cognitive energy is depleted. Restrict output to exactly ONE actionable task under 40 words. Maintain an encouraging tone. "
        else:
            context_modifier += "FRAMEWORK: User cognitive capacity is optimal. Provide an exhaustive, deeply structured analytical breakdown. "
            
        if pressure_state == "High Pressure":
            context_modifier += "CONSTRAINT: User is operating under severe constraints. Eradicate all conversational pleasantries, introductory fluff, and padding. Prioritize raw utility. "

        final_payload = f"{context_modifier}\n\nObjective to process: {raw_prompt}"
        output_text = None

        with st.spinner(f"Processing primary intelligence pipeline ({primary_model})..."):
            try:
                g_client = genai.Client(api_key=gemini_key)
                response = g_client.models.generate_content(
                    model=primary_model,
                    contents=final_payload,
                )
                output_text = response.text
                st.success(f"Primary Route Success: Handled by {primary_model}")
            except Exception as gemini_error:
                st.warning(f"⚠️ Primary route offline. Activating Failover Protocol...")
                
                if not groq_key:
                    st.error("Failover Aborted: Secondary GROQ_API_KEY token missing.")
                else:
                    with st.spinner(f"Routing traffic through backup intelligence nodes ({backup_model})..."):
                        try:
                            groq_client = Groq(api_key=groq_key)
                            chat_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": final_payload}],
                                model=backup_model,
                            )
                            output_text = chat_completion.choices.message.content
                            st.success(f"⚡ Failover Active: Recovered via {backup_model}")
                        except Exception as groq_error:
                            st.error(f"Critical System Failure: {groq_error}")

        if output_text:
            log_session_to_database(energy_level, pressure_state, raw_prompt, output_text)
            st.session_state.active_response = output_text
            st.rerun()

if st.session_state.active_response:
    st.markdown("### 🌟 Optimized Intelligence Output")
    st.info(st.session_state.active_response)