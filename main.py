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
                st.write("**Saved AI Output:**")
                st.write(item[4])

# =====================================================================
# 4. MAIN INTERFACE LAYER
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
        output_text = None
        final_engineered_payload = f"Energy Level: {energy_level}, Pressure: {pressure_state}\n\nObjective: {raw_prompt}"
        
        # =====================================================================
        # 5. DYNAMIC PROMPT SYNTHESIS ENGINE
        # =====================================================================
        with st.spinner("Synthesizing custom meta-cognitive prompting blueprint..."):
            try:
                g_client = genai.Client(api_key=gemini_key)
                
                meta_synthesis_prompt = f"""
                You are the core Dynamic Prompt Synthesis Layer for ContextAI. 
                Your job is NOT to answer the user's prompt. Your job is to analyze the human user's metrics and generate a highly custom system prompt engineering layout tailored to their cognitive capacity.
                
                HUMAN METRICS:
                - Cognitive Energy: {energy_level}/5 (1 is completely burned out/exhausted, 5 is peak focus)
                - Environmental Pressure: {pressure_state}
                
                USER'S TARGET OBJECTIVE:
                "{raw_prompt}"
                
                INSTRUCTIONS:
                Based algorithmically on these human metrics, generate a precise, un-padded block of operational instructions that commands a secondary LLM exactly how to structure its delivery tone, length limits, information density, and actionability to protect the user's mind from information overload. Focus entirely on the human psychological constraints. Do not output conversational text or introductions. Output only the generated prompt constraint block.
                """
                
                synthesis_response = g_client.models.generate_content(
                    model=primary_model,
                    contents=meta_synthesis_prompt,
                )
                dynamic_system_instruction = synthesis_response.text
                final_engineered_payload = f"CRITICAL SYSTEM OPERATIONAL BLUEPRINT:\n{dynamic_system_instruction}\n\nUSER CORE TASK TO EXECUTE:\n{raw_prompt}"
                
            except Exception as synthesis_error:
                st.warning("⚠️ Synthesis node error. Falling back to baseline matrix rules...")

        # =====================================================================
        # 6. RUN EXECUTABLE PIPELINE WITH FIXED VARIABLE NAME
        # =====================================================================
        with st.spinner(f"Routing engineered payload through {primary_model}..."):
            try:
                response = g_client.models.generate_content(
                    model=primary_model,
                    contents=final_engineered_payload,
                )
                output_text = response.text
                st.success(f"Primary Route Success: Handled dynamically by {primary_model}")
            except Exception as gemini_error:
                st.warning(f"⚠️ Primary route offline. Activating Failover Protocol...")
                
                if not groq_key:
                    st.error("Failover Aborted: Secondary GROQ_API_KEY token missing.")
                else:
                    with st.spinner(f"Routing traffic through backup intelligence nodes ({backup_model})..."):
                        try:
                            groq_client = Groq(api_key=groq_key)
                            chat_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": final_engineered_payload}], # FIXED THE VARIABLE TYPO HERE
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