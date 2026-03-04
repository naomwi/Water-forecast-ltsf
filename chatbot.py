import os
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from data_loader import get_global_kpi_summary

# Load environment variables (e.g. API Key)
load_dotenv()

def init_gemini():
    """Initializes the Gemini client using API key from .env or st.secrets."""
    load_dotenv()
    
    api_key = None
    secret_err = None
    # 1. Try st.secrets first (Cloud)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = str(st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        secret_err = str(e)
        
    # 2. Try OS Environment Variables (Local / Docker)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # 3. Clean string
    if api_key:
        api_key = api_key.strip(" '\"")
        
    if not api_key:
        st.error("Missing GEMINI_API_KEY in Streamlit Secrets or .env file.")
        if secret_err:
            st.error(f"Streamlit Cloud Secret Parsing Error: {secret_err}")
        st.info("💡 Hint: If you just updated the Streamlit Secrets, it takes 1-2 minutes to propagate. Try `Manage app` -> `Reboot app`.")
        return None
        
    client = genai.Client(api_key=api_key)
    
    # Extract the entire Context once to put into System Instruction
    global_data_context = get_global_kpi_summary()
    
    # Fetch PDF Report Context if available
    report_context = ""
    # Use relative pathing so it works on Streamlit Community Cloud
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(dashboard_dir)
    report_path = os.path.join(project_dir, "documents", "report_context.txt")
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_context = f.read()

    system_instruction = (
        "You are HydroBot, an expert AI Water Quality Data Analyst designed for the FPT University Capstone Project.\n"
        "Your role is to help users understand water quality metrics (EC, pH) and the performance of your group's predictive models.\n\n"
        "I am providing you with the ENTIRE BENCHMARK RESULTS of all models built in this project across multiple sites and horizons. "
        "You must use this data to confidently answer any question about which model is best, what the MSE/R2 is, and how the models compare.\n\n"
        f"--- BENCHMARK RESULTS ---\n{global_data_context}\n\n"
        f"--- PROJECT REPORT CONTEXT ---\n{report_context}\n\n"
        "Speak professionally but be helpful. Provide concise, bolded, and data-backed answers based tightly on the provided report context and benchmark data."
    )
    
    # Config contains System Instructions instead of passing into model constructor as before
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    
    return {"client": client, "config": config}

def get_chat_session():
    """ 
    Initialize or load the chat session from Streamlit status (st.session_state) 
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am **HydroBot**, the intelligent assistant for the FPT University Water Quality Forecasting Project.\nI have studied all the benchmark data for our models and read the entire team project report. Ask me anything!"}
        ]
    
    # Reload chatbot if it failed previously (e.g. user just added API key)
    if st.session_state.get("chat_session") is None:
        setup = init_gemini()
        if setup:
            try:
                # Initialize a chat session (Context is already in System Instruction config)
                client = setup["client"]
                config = setup["config"]
                
                # CRITICAL: Save Client to session state so it doesn't get garbage collected or drop connections
                st.session_state.gemini_client = client
                
                # Update model to gemini-3.1-pro-preview per latest Google requirements
                st.session_state.chat_session = st.session_state.gemini_client.chats.create(
                    model="gemini-3.1-pro-preview",
                    config=config
                )
            except Exception as e:
                st.error(f"Chatbot initialize error: {e}")
                return None, []
    
    return st.session_state.get("chat_session"), st.session_state.chat_history

def stream_generator(response_stream):
    """Yield chunks from the Gemini stream."""
    for chunk in response_stream:
        yield chunk.text

def display_chat():
    """ Display main Chatbot UI. """
    
    chat_session, chat_history = get_chat_session()
    
    if not chat_session:
        return
        
    # Logo Avatar
    ai_avatar = "dashboard/assets/logo.png"

    # Container for chat history
    for message in chat_history:
        avatar = ai_avatar if message["role"] == "assistant" else "user"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Handle new chat input
    prompt = st.chat_input("Ask HydroBot about any specific model's performance...")
    if prompt:
        # Save to history for UI rendering
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Instantly render to screen (UI)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar=ai_avatar):
            try:
                # ---- Smart Prediction Detection ----
                from prediction_loader import detect_intent, build_prediction_context
                intent = detect_intent(prompt)
                
                if intent["is_prediction"]:
                    # Load model predictions and build context for Gemini
                    pred_context = build_prediction_context(
                        intent["features"], intent["horizon"]
                    )
                    enhanced_prompt = (
                        f"[LIVE PREDICTION DATA]\n{pred_context}\n\n"
                        f"[USER QUESTION]\n{prompt}"
                    )
                else:
                    enhanced_prompt = prompt
                
                # Send query and activate streaming mode with new SDK
                response_stream = chat_session.send_message_stream(enhanced_prompt)
                full_response = st.write_stream(stream_generator(response_stream))
                
                # Save to log
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error handling request: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": "I encountered an error. Please try again."})