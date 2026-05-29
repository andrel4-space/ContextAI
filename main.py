import streamlit as st
import os
import sqlite3
from datetime import datetime
from google import genai
from groq import Groq
from dotenv import load_dotenv

# =====================================================================
# 1. DATABASE SYSTEM LAYER (Renovated Ledger Schema)
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
        st.warning(f"Database write bypassed: {db_err}")

def fetch_historical_metrics():
    try:
        conn = sqlite3.connect("context_vault.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, energy_score, pressure_threshold, raw_prompt, optimized_output FROM cognitive_logs ORDER BY id DESC")
        logs = cursor.fetchall()
        conn.close()
        return logs
    except Exception:
        return []

initialize_database()

# =====================================================================
# 2. CONFIGURATION & CORE AUTHENTICATION SECURITY LAYER
# =====================================================================
load_dotenv()

# Foolproof Authorization Cascade: Checks Cloud variables first, falls back to direct pipeline tokens
gemini_key = os.getenv("GEMINI_API_KEY", "AIzaSyDixn5L2zOSUbIE1Huyj8VnokjORle1nBs")
groq_key = os.getenv("GROQ_API_KEY", "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")

# Modern Enterprise Production Model Signatures
primary_model = "gemini-2.5-flash"
backup_model = "llama-3.1-8b-instant"  # Clean upgrade to ultra-reliable production tier

st.set_page_config(page_title="ContextAI Management Console", layout="wide")

if "active_response" not in st.session_state:
    st.session_state.active_response = None

# =====================================================================
# 3. GRAPHICAL USER INTERFACE SIDEBAR (Analytics History Ledger)
# =====================================================================
with st.sidebar:
    st.title("🛡️ ContextAI Vault")
    st.write("Secure, analytical metric logging system.")
    
    st.markdown("### Historical Analytics Log")
    history = fetch_historical_metrics()
    
    if not history:
        st.info("No prior sessions recorded in data schema.")
    else:
        st.metric(label="Total Optimizations Executed", value=len(history))
        st.markdown("---")
        for item in history:
            # Structurally parse the database tuple cleanly to prevent expansion folding crashes
            with st.expander(f"🕒 {item[0]}"):
                st.write(f"**Energy Bandwidth:** {item[1]}/5")
                st.write(f"**Pressure Context:** {item[2]}")
                st.write(f"**Raw Objective:** *{item[3]}*")
                st.markdown("---")
                st.write("**AI Response Matrix:**")
                st.info(item[4])

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
    if not raw_prompt.strip():
        st.warning("Please deliver an objective payload before forcing execution.")
    else:
        output_text = None
        
        # Phase 1: Meta-Cognitive Prompt Generation Pipeline
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
            except Exception:
                # Absolute baseline backup payload fallback
                final_engineered_payload = f"Context: Energy={energy_level}, Pressure={pressure_state}\n\nObjective: {raw_prompt}"

        # Phase 2: Execution Chain With Hardened Multi-Brain Fallback Loop
        with st.spinner(f"Processing primary intelligence pipeline ({primary_model})..."):
            try:
                # ROUTE 1: Run Primary Google Core
                g_client = genai.Client(api_key=gemini_key)
                response = g_client.models.generate_content(
                    model=primary_model,
                    contents=final_engineered_payload,
                )
                output_text = response.text
                st.success(f"Primary Route Success: Handled dynamically by {primary_model}")
            except Exception as gemini_error:
                st.warning("⚠️ Primary route offline (Google 503/Timeout). Activating Failover Protocol...")
                
                # ROUTE 2: Run Renovated Groq Backup Core (Llama 3.1 Instant)
                with st.spinner(f"Routing traffic through backup intelligence nodes ({backup_model})..."):
                    try:
                        groq_client = Groq(api_key=groq_key)
                        chat_completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": final_engineered_payload}],
                            model=backup_model,
                        )
                        output_text = chat_completion.choices[0].message.content
                        st.success(f"⚡ Failover Active: Instantly recovered via {backup_model}")
                    except Exception as groq_error:
                        st.error(f"Critical System Failure: All primary and backup intelligence nodes are completely unreachable. Details: {groq_error}")

        # Post-Execution Pipeline Sync
        if output_text:
            log_session_to_database(energy_level, pressure_state, raw_prompt, output_text)
            st.session_state.active_response = output_text
            st.rerun()

# Permanent Visual Focus Area (Safe from refresh blocks)
if st.session_state.active_response:
    st.markdown("### 🌟 Optimized Intelligence Output")
    st.info(st.session_state.active_response)