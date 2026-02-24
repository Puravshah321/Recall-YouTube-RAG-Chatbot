import streamlit as st
import os
from dotenv import load_dotenv
from youtube_utils import extract_video_id, get_transcript
from chatbot_engine import create_chatbot_engine

load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recall — YouTube AI",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# PREMIUM UI DESIGN (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">

<style>
    /* Theme Variables */
    :root {
        --primary: #FF0000;
        --primary-glow: rgba(255, 0, 0, 0.2);
        --bg-dark: #0A0A0B;
        --card-bg: rgba(28, 28, 30, 0.7);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-main: #FFFFFF;
        --text-muted: #9BA1A6;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* Global Base Styling */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1a1c 0%, var(--bg-dark) 100%) !important;
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* Typography */
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Premium Glass Cards */
    .premium-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .premium-card:hover {
        border: 1px solid rgba(255, 0, 0, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6);
    }

    /* Custom Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px var(--primary-glow);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--primary-glow);
        background: linear-gradient(135deg, #FF1A1A 0%, #E60000 100%);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Input Styling */
    .stTextInput input {
        background-color: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px var(--primary-glow) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Chat Elements */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background-color: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 18px !important;
        padding: 1.2rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* User Message Specifics */
    [data-testid="stChatMessage"][data-testid="user"] 
    [data-testid="stChatMessageContent"] {
        border-left: 4px solid var(--primary) !important;
    }

    /* Sidebar/Left Panel Video */
    .video-container {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--card-border);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #444;
    }

    /* Metric Cards */
    .metric-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: var(--text-main);
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "ready" not in st.session_state:
    st.session_state.ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
        <div style="background: var(--primary); width: 45px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px var(--primary-glow);">
            <span style="color: white; font-size: 18px;">▶</span>
        </div>
        <div>
            <h1 style="margin: 0; line-height: 1; font-size: 2rem;">Recall <span style="color: var(--primary);">AI</span></h1>
            <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">Intelligent Video Context Engine</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LANDING UI
# ─────────────────────────────────────────────────────────────
def landing_ui():
    render_header()

    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("""
        <h3 style="margin-top: 0;">Initialize Intelligence</h3>
        <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
            Paste any YouTube URL below to begin analyzing the content with our neural context engine.
        </p>
        """, unsafe_allow_html=True)

        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        if st.button("Generate Neural Context"):
            if not url:
                st.warning("Please enter a valid YouTube URL.")
                return

            with st.spinner("Decoding video signal & indexing context..."):
                video_id = extract_video_id(url)

                if not video_id:
                    st.error("Protocol Error: Invalid YouTube URL format.")
                    return

                text, err = get_transcript(video_id)

                if err and not text:
                    st.error(f"Transcript Failure: {err}")
                    return

                engine = create_chatbot_engine(text)

                st.session_state.core_engine = engine
                st.session_state.current_video_url = url
                st.session_state.chunk_count = len(text.split()) // 150
                st.session_state.messages = []
                st.session_state.ready = True

                st.toast("Neural Engine Ready")
                st.rerun()
        
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--glass-border);">
            <p style="font-size: 0.8rem; color: var(--text-muted); text-align: center;">
                Supports multi-lingual transcripts and long-form content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DASHBOARD UI
# ─────────────────────────────────────────────────────────────
def dashboard_ui():
    render_header()

    left, right = st.columns([0.45, 0.55], gap="large")

    # ───── LEFT PANEL ─────
    with left:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(st.session_state.current_video_url)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Metadata", unsafe_allow_html=True)
        
        # Styled Metrics
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-label">Context Engine</div>
            <div class="metric-value">FAISS with HuggingFace</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Language Model</div>
            <div class="metric-value">Llama 3.1 8B (via Groq)</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Dialogue State</div>
            <div class="metric-value">{len(st.session_state.messages)} Messages</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("New Video"):
                st.session_state.ready = False
                st.rerun()
        with col_b:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    # ───── RIGHT PANEL ─────
    with right:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### Video Dialogue", unsafe_allow_html=True)

        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Query the video context..."):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    response = st.session_state.core_engine.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if not st.session_state.ready:
    landing_ui()
else:
    dashboard_ui()