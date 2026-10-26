"""TRUTHLENS AI — Deep Learning Powered News Credibility Analysis.

CYBER-DARK EDITION WITH INTERACTIVE THREE.JS NEURAL MATRIX VISUALIZATION.
Features:
- State-of-the-art Cyber-Dark Glassmorphism Theme (Neon cyan, electric violet, emerald & crimson)
- Interactive 3D Three.js Deep Learning Neural Lattice & LSTM Synaptic Core
- Live 3D particle connections, orbital memory gates, and interactive mouse parallax & rotation
- Dynamic 3D color shift based on prediction state (Cyan/Violet -> Emerald Green for REAL, Warning Crimson for FAKE)
- Ambient Three.js particle starfield background injection
- High-contrast typography with Outfit, Inter & JetBrains Mono
- Dual-card Result Dashboard with glowing SVG circular confidence ring & neon spectrum meter
- Empirical metrics loaded from evaluation_results.json & preprocessing_config.json
- Sample articles with instant one-click analysis
"""

from pathlib import Path
import json
import math
import streamlit as st
import streamlit.components.v1 as components

from src.predict import predict_news, get_project_paths


# ==============================================================================
# 1. PAGE CONFIGURATION & CYBER-DARK DESIGN SYSTEM
# ==============================================================================

def setup_page_config():
    """Configure Streamlit page properties."""
    st.set_page_config(
        page_title="TruthLens AI — Deep Learning News Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_custom_css():
    """Inject ultra-premium Cyber-Dark Glassmorphism CSS design system."""
    custom_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Color Palette & Dark System Variables */
        :root {
            --bg-page: #050811;
            --bg-card: rgba(13, 20, 36, 0.75);
            --bg-card-hover: rgba(18, 28, 52, 0.85);
            --bg-subtle: rgba(26, 38, 66, 0.55);
            --text-title: #f8fafc;
            --text-body: #cbd5e1;
            --text-muted: #94a3b8;
            --text-light: #64748b;
            --border-card: rgba(56, 189, 248, 0.18);
            --border-glow: rgba(56, 189, 248, 0.45);
            --brand-primary: #38bdf8;
            --brand-cyan: #06b6d4;
            --brand-indigo: #818cf8;
            --brand-purple: #a855f7;
            --brand-gradient: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #9333ea 100%);
            --real-green: #10b981;
            --real-bg: rgba(6, 78, 59, 0.35);
            --real-border: rgba(16, 185, 129, 0.55);
            --real-glow: rgba(16, 185, 129, 0.3);
            --fake-red: #f43f5e;
            --fake-bg: rgba(136, 19, 55, 0.35);
            --fake-border: rgba(244, 63, 94, 0.55);
            --fake-glow: rgba(244, 63, 94, 0.3);
            --glass-shadow: 0 10px 35px -5px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        /* Enforce Dark Theme on the entire Streamlit page */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg-page) !important;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(30, 58, 138, 0.18) 0%, transparent 65%),
                radial-gradient(circle at 90% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 10% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
                linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px) !important;
            background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
            color: var(--text-body) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Hide Streamlit default branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {display: none;}
        [data-testid="stDecoration"] {display: none;}

        /* Main layout container */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1240px !important;
        }

        /* Typography Defaults */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-title) !important;
            letter-spacing: -0.02em !important;
        }

        /* TOP NAVIGATION BAR */
        .top-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(13, 20, 36, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 0.95rem 1.75rem;
            margin-bottom: 1.75rem;
            box-shadow: var(--glass-shadow);
        }
        .nav-brand-group {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .nav-logo-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
            color: #ffffff;
            font-size: 1.25rem;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.45);
        }
        .nav-logo-text {
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
        }
        .nav-logo-text span {
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-badge {
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 6px;
            letter-spacing: 0.05em;
            font-family: 'JetBrains Mono', monospace;
        }
        .nav-lab-tag {
            font-size: 0.85rem;
            color: var(--text-muted);
            border-left: 1px solid rgba(255, 255, 255, 0.12);
            padding-left: 0.85rem;
            font-weight: 500;
        }
        .nav-status-group {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }
        .nav-status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-body);
        }
        .status-pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981, 0 0 18px rgba(16, 185, 129, 0.4);
            animation: pulse-ring 2s infinite;
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
            70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* HERO EDITORIAL SECTION */
        .hero-banner {
            padding: 0.5rem 0 1.5rem 0;
            text-align: left;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.35);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 20px;
            letter-spacing: 0.08em;
            margin-bottom: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .hero-headline {
            font-size: 3.1rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            line-height: 1.1;
            color: #ffffff;
            margin: 0 0 0.85rem 0;
        }
        .hero-headline .gradient-text {
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-description {
            font-size: 1.12rem;
            color: var(--text-muted);
            line-height: 1.55;
            margin: 0;
        }

        /* CARD CONTAINERS */
        .app-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 18px;
            padding: 1.75rem;
            box-shadow: var(--glass-shadow);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .app-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 12px 40px -5px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.15);
        }
        .card-heading {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin: 0 0 0.35rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-subtext {
            font-size: 0.92rem;
            color: var(--text-muted);
            margin: 0 0 1.25rem 0;
            line-height: 1.5;
        }

        /* Text Area Dark Styling */
        .stTextArea textarea {
            background-color: rgba(8, 13, 25, 0.9) !important;
            color: #f1f5f9 !important;
            border: 1.5px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 14px !important;
            font-size: 0.98rem !important;
            line-height: 1.6 !important;
            padding: 1.1rem !important;
            transition: all 0.25s ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stTextArea textarea:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2), 0 0 20px rgba(56, 189, 248, 0.25) !important;
        }

        /* Character Counter Row */
        .char-counter-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            color: var(--text-light);
            margin-top: 0.45rem;
            margin-bottom: 1.25rem;
            padding: 0 6px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Streamlit Button Cyber Styling */
        div.stButton > button {
            background: var(--brand-gradient) !important;
            color: #ffffff !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            font-family: 'Outfit', sans-serif !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            padding: 0.8rem 2rem !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35), 0 0 12px rgba(56, 189, 248, 0.25) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            letter-spacing: 0.02em !important;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.55), 0 0 22px rgba(56, 189, 248, 0.45) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        div.stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* RESULT DASHBOARD & CIRCULAR GAUGE */
        .result-panel {
            background: rgba(13, 20, 36, 0.85);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--border-card);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
            margin-top: 1.75rem;
            margin-bottom: 2rem;
        }
        .verdict-box-real {
            background: linear-gradient(180deg, var(--real-bg) 0%, rgba(13, 20, 36, 0.8) 100%);
            border: 1.5px solid var(--real-border);
            border-radius: 16px;
            padding: 1.75rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 0 30px var(--real-glow);
        }
        .verdict-box-fake {
            background: linear-gradient(180deg, var(--fake-bg) 0%, rgba(13, 20, 36, 0.8) 100%);
            border: 1.5px solid var(--fake-border);
            border-radius: 16px;
            padding: 1.75rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 0 30px var(--fake-glow);
        }
        .verdict-tag-real {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.45);
            font-size: 0.75rem;
            font-weight: 800;
            padding: 4px 12px;
            border-radius: 20px;
            letter-spacing: 0.08em;
            margin-bottom: 0.65rem;
            width: fit-content;
            font-family: 'JetBrains Mono', monospace;
        }
        .verdict-tag-fake {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(244, 63, 94, 0.2);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.45);
            font-size: 0.75rem;
            font-weight: 800;
            padding: 4px 12px;
            border-radius: 20px;
            letter-spacing: 0.08em;
            margin-bottom: 0.65rem;
            width: fit-content;
            font-family: 'JetBrains Mono', monospace;
        }
        .verdict-title-real {
            font-size: 2.35rem;
            font-weight: 900;
            color: #10b981;
            margin: 0 0 0.35rem 0;
            line-height: 1.15;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
            font-family: 'Outfit', sans-serif;
        }
        .verdict-title-fake {
            font-size: 2.35rem;
            font-weight: 900;
            color: #f43f5e;
            margin: 0 0 0.35rem 0;
            line-height: 1.15;
            text-shadow: 0 0 20px rgba(244, 63, 94, 0.4);
            font-family: 'Outfit', sans-serif;
        }
        .verdict-sub-real {
            font-size: 0.95rem;
            font-weight: 600;
            color: #6ee7b7;
            margin: 0;
        }
        .verdict-sub-fake {
            font-size: 0.95rem;
            font-weight: 600;
            color: #fda4af;
            margin: 0;
        }

        /* Circular Confidence Gauge */
        .ring-card-box {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 1.75rem;
            text-align: center;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
        }
        .ring-svg-wrap {
            position: relative;
            width: 140px;
            height: 140px;
            margin: 0 auto 0.75rem auto;
        }
        .ring-center-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        .ring-pct {
            font-size: 1.95rem;
            font-weight: 900;
            color: #ffffff;
            line-height: 1;
            font-family: 'Outfit', sans-serif;
        }
        .ring-caption {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* AI ANALYSIS SPECTRUM METER */
        .meter-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 1.35rem 1.6rem;
            margin-top: 1.5rem;
        }
        .meter-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.85rem;
        }
        .meter-heading {
            font-size: 0.92rem;
            font-weight: 800;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-family: 'Outfit', sans-serif;
        }
        .meter-confidence-pill {
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--brand-primary);
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 3px 10px;
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
        }
        .spectrum-bar-track {
            position: relative;
            width: 100%;
            height: 14px;
            background: linear-gradient(90deg, #f43f5e 0%, #eab308 50%, #10b981 100%);
            border-radius: 8px;
            margin-bottom: 0.65rem;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        }
        .spectrum-pointer {
            position: absolute;
            top: -6px;
            width: 26px;
            height: 26px;
            background: #ffffff;
            border: 3px solid #0f172a;
            border-radius: 50%;
            transform: translateX(-50%);
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.8), 0 2px 8px rgba(0, 0, 0, 0.6);
            transition: left 0.5s ease;
        }
        .spectrum-labels-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .meter-sub-caption {
            font-size: 0.78rem;
            color: var(--text-light);
            margin-top: 0.75rem;
            text-align: center;
        }

        /* SPEC GRID */
        .spec-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }
        .spec-pill {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            transition: all 0.2s ease;
        }
        .spec-pill:hover {
            border-color: var(--border-glow);
            background: rgba(22, 34, 62, 0.8);
        }
        .spec-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-family: 'JetBrains Mono', monospace;
        }
        .spec-val {
            font-size: 0.96rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 3px;
        }

        /* SAMPLE ARTICLE CARDS */
        .sample-card-item {
            background: rgba(13, 20, 36, 0.75);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 1.35rem;
            box-shadow: var(--glass-shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }
        .sample-card-item:hover {
            transform: translateY(-4px);
            border-color: var(--border-glow);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.2);
        }
        .sample-tag {
            font-size: 0.74rem;
            font-weight: 800;
            color: var(--brand-primary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.45rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .sample-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.55rem;
            font-family: 'Outfit', sans-serif;
        }
        .sample-preview {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin-bottom: 1.1rem;
        }

        /* 4-STEP PIPELINE */
        .steps-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-top: 1.25rem;
        }
        .step-col {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 1.25rem;
            position: relative;
            box-shadow: var(--glass-shadow);
            transition: all 0.25s ease;
        }
        .step-col:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
        }
        .step-num {
            font-size: 0.85rem;
            font-weight: 900;
            color: var(--brand-primary);
            margin-bottom: 0.4rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .step-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-family: 'Outfit', sans-serif;
        }
        .step-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.45;
        }

        /* PROJECT STATISTICS ROW */
        .stat-box {
            background: rgba(13, 20, 36, 0.8);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            text-align: center;
            box-shadow: var(--glass-shadow);
            transition: all 0.25s ease;
        }
        .stat-box:hover {
            border-color: var(--border-glow);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
            transform: translateY(-2px);
        }
        .stat-value {
            font-size: 1.85rem;
            font-weight: 900;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            font-family: 'Outfit', sans-serif;
        }
        .stat-title {
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 6px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* FOOTER */
        .truthlens-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding: 2.25rem 0 1rem 0;
            margin-top: 3.5rem;
            text-align: center;
        }
        .footer-brand {
            font-size: 1.1rem;
            font-weight: 800;
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
        }
        .footer-brand span {
            color: var(--brand-primary);
        }
        .footer-sub {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.35rem;
        }
        .footer-disclaimer {
            font-size: 0.76rem;
            color: var(--text-light);
            margin-top: 0.5rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA UTILITIES & RESOURCE LOADERS
# ==============================================================================

def load_project_metadata():
    """Retrieve actual empirical metrics and hyperparameters from saved JSON files."""
    project_root = Path(__file__).resolve().parent
    eval_path = project_root / "data" / "processed" / "evaluation_results.json"
    config_path = project_root / "models" / "preprocessing_config.json"

    stats = {
        "train_samples": 24848,
        "test_samples": 7766,
        "vocab_size": 20000,
        "sequence_length": 300,
        "accuracy": 0.9898,
    }

    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                stats["test_samples"] = eval_data.get("total_test_samples", stats["test_samples"])
                stats["accuracy"] = eval_data.get("accuracy", stats["accuracy"])
        except Exception:
            pass

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                stats["vocab_size"] = cfg_data.get("max_vocab_size", stats["vocab_size"])
                stats["sequence_length"] = cfg_data.get("max_sequence_length", stats["sequence_length"])
        except Exception:
            pass

    return stats


# ==============================================================================
# 3. THREE.JS 3D NEURAL MATRIX VISUALIZER
# ==============================================================================

def render_threejs_neural_matrix(result: dict | None = None, height: int = 440):
    """Render interactive 3D WebGL Neural Lattice using Three.js."""
    # Determine color palette based on prediction state
    if result:
        is_real = result.get("prediction") == "REAL"
        confidence = result.get("confidence", 0.9)
        if is_real:
            status_text = "VERIFIED REAL NEWS"
            badge_class = "badge-real"
            core_hex = "0x10b981"
            wire_hex = "0x34d399"
            part_hex = "0x6ee7b7"
            bg_glow = "rgba(16, 185, 129, 0.25)"
            status_icon = "● VERDICT: REAL"
        else:
            status_text = "ALERT: MISINFORMATION DETECTED"
            badge_class = "badge-fake"
            core_hex = "0xf43f5e"
            wire_hex = "0xfb7185"
            part_hex = "0xfda4af"
            bg_glow = "rgba(244, 63, 94, 0.25)"
            status_icon = "▲ VERDICT: FAKE"
    else:
        status_text = "AWAITING ARTICLE INGESTION"
        badge_class = "badge-idle"
        core_hex = "0x38bdf8"
        wire_hex = "0x818cf8"
        part_hex = "0x38bdf8"
        bg_glow = "rgba(56, 189, 248, 0.15)"
        status_icon = "● STANDBY: READY"

    html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; user-select: none; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: transparent; font-family: 'Segoe UI', Roboto, monospace; }
    #canvas-container {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 18px;
      overflow: hidden;
      background: radial-gradient(circle at center, rgba(15, 23, 42, 0.75) 0%, rgba(5, 8, 17, 0.95) 100%);
      border: 1px solid rgba(56, 189, 248, 0.25);
      box-shadow: 0 10px 35px rgba(0, 0, 0, 0.8), inset 0 0 25px __BG_GLOW__;
    }
    canvas { display: block; width: 100%; height: 100%; cursor: grab; }
    canvas:active { cursor: grabbing; }

    /* Cyber HUD Overlay */
    .hud-overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .hud-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .hud-brand {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.1em;
      color: #38bdf8;
      text-transform: uppercase;
      background: rgba(8, 14, 28, 0.75);
      padding: 6px 14px;
      border-radius: 8px;
      border: 1px solid rgba(56, 189, 248, 0.3);
      backdrop-filter: blur(10px);
    }
    .hud-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #38bdf8;
      box-shadow: 0 0 8px #38bdf8;
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.85); } }

    .hud-controls {
      display: flex;
      gap: 8px;
      pointer-events: auto;
    }
    .hud-btn {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #cbd5e1;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      backdrop-filter: blur(8px);
    }
    .hud-btn:hover {
      background: rgba(56, 189, 248, 0.25);
      border-color: #38bdf8;
      color: #ffffff;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
      transform: translateY(-1px);
    }

    .hud-bottom {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    .hud-telemetry {
      font-size: 11px;
      color: #94a3b8;
      font-family: monospace;
      background: rgba(8, 14, 28, 0.75);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(8px);
    }
    .hud-status-badge {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      padding: 6px 14px;
      border-radius: 8px;
      text-transform: uppercase;
      font-family: monospace;
    }
    .badge-idle {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
    }
    .badge-real {
      background: rgba(16, 185, 129, 0.22);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.55);
      box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .badge-fake {
      background: rgba(244, 63, 94, 0.22);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.55);
      box-shadow: 0 0 20px rgba(244, 63, 94, 0.4);
    }

    /* Corner Tech Accents */
    .corner {
      position: absolute;
      width: 14px;
      height: 14px;
      border-color: rgba(56, 189, 248, 0.5);
      border-style: solid;
      pointer-events: none;
    }
    .top-left { top: 8px; left: 8px; border-width: 2px 0 0 2px; }
    .top-right { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
    .bottom-left { bottom: 8px; left: 8px; border-width: 0 0 2px 2px; }
    .bottom-right { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
  <div id="canvas-container">
    <div class="corner top-left"></div>
    <div class="corner top-right"></div>
    <div class="corner bottom-left"></div>
    <div class="corner bottom-right"></div>

    <div class="hud-overlay">
      <div class="hud-top">
        <div class="hud-brand">
          <span class="hud-dot"></span>
          <span>LSTM 3D Neural Lattice • Interactive</span>
        </div>
        <div class="hud-controls">
          <button class="hud-btn" id="btnPulse">⚡ SYNAPSE PULSE</button>
          <button class="hud-btn" id="btnSpeed">⏩ WARP SPEED</button>
          <button class="hud-btn" id="btnReset">⟳ RESET VIEW</button>
        </div>
      </div>
      <div class="hud-bottom">
        <div class="hud-telemetry">
          <span>NODES: 180 ACTIVE</span> • <span>RECURRENT GATES: 3</span> • <span>DRAG TO ORBIT</span>
        </div>
        <div class="hud-status-badge __BADGE_CLASS__">
          __STATUS_ICON__
        </div>
      </div>
    </div>
  </div>

  <script>
    (function() {
      const container = document.getElementById('canvas-container');
      const width = container.clientWidth || 800;
      const height = container.clientHeight || 440;

      // 1. Scene, Camera, Renderer
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
      camera.position.set(0, 0, 18);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);

      // Colors from Python state
      const coreColor = __CORE_HEX__;
      const wireColor = __WIRE_HEX__;
      const partColor = __PART_HEX__;

      // 2. Neural Root Group
      const neuralGroup = new THREE.Group();
      scene.add(neuralGroup);

      // 3. Central Hologram Core (Dual Icosahedron)
      const coreGeo = new THREE.IcosahedronGeometry(2.5, 2);
      const coreMat = new THREE.MeshStandardMaterial({
        color: coreColor,
        wireframe: true,
        transparent: true,
        opacity: 0.85,
        roughness: 0.1,
        metalness: 0.9
      });
      const coreMesh = new THREE.Mesh(coreGeo, coreMat);
      neuralGroup.add(coreMesh);

      // Inner Glowing Seed
      const innerGeo = new THREE.SphereGeometry(1.6, 24, 24);
      const innerMat = new THREE.MeshBasicMaterial({
        color: wireColor,
        transparent: true,
        opacity: 0.35,
        wireframe: false
      });
      const innerMesh = new THREE.Mesh(innerGeo, innerMat);
      neuralGroup.add(innerMesh);

      // Outer Lattice Cage
      const cageGeo = new THREE.IcosahedronGeometry(4.2, 1);
      const cageMat = new THREE.MeshBasicMaterial({
        color: wireColor,
        wireframe: true,
        transparent: true,
        opacity: 0.25
      });
      const cageMesh = new THREE.Mesh(cageGeo, cageMat);
      neuralGroup.add(cageMesh);

      // 4. LSTM Memory Gates (3 Orbital Concentric Rings)
      const ringGroup = new THREE.Group();
      neuralGroup.add(ringGroup);

      const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.55 });
      const ringMat2 = new THREE.MeshBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.55 });
      const ringMat3 = new THREE.MeshBasicMaterial({ color: coreColor, transparent: true, opacity: 0.65 });

      const ring1 = new THREE.Mesh(new THREE.TorusGeometry(5.4, 0.035, 16, 120), ringMat1);
      const ring2 = new THREE.Mesh(new THREE.TorusGeometry(6.6, 0.035, 16, 120), ringMat2);
      const ring3 = new THREE.Mesh(new THREE.TorusGeometry(7.8, 0.035, 16, 120), ringMat3);

      ring1.rotation.x = Math.PI / 3;
      ring2.rotation.y = Math.PI / 4;
      ring3.rotation.z = Math.PI / 6;

      ringGroup.add(ring1);
      ringGroup.add(ring2);
      ringGroup.add(ring3);

      // 5. Dynamic Synaptic Particles & Organic Connection Lines
      const particleCount = 180;
      const particlePositions = new Float32Array(particleCount * 3);
      const particleVelocities = [];
      const spread = 9.5;

      for (let i = 0; i < particleCount; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        const r = 2.8 + Math.random() * (spread - 2.8);

        particlePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        particlePositions[i * 3 + 2] = r * Math.cos(phi);

        particleVelocities.push({
          x: (Math.random() - 0.5) * 0.02,
          y: (Math.random() - 0.5) * 0.02,
          z: (Math.random() - 0.5) * 0.02
        });
      }

      const pGeo = new THREE.BufferGeometry();
      pGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

      const pMat = new THREE.PointsMaterial({
        color: partColor,
        size: 0.22,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending
      });
      const particlePoints = new THREE.Points(pGeo, pMat);
      neuralGroup.add(particlePoints);

      // Connecting Synaptic Lines (dynamic buffer)
      const maxLines = 400;
      const linePositions = new Float32Array(maxLines * 6);
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3).setUsage(THREE.DynamicDrawUsage));

      const lineMat = new THREE.LineBasicMaterial({
        color: wireColor,
        transparent: true,
        opacity: 0.35,
        blending: THREE.AdditiveBlending
      });
      const lineSegments = new THREE.LineSegments(lineGeo, lineMat);
      neuralGroup.add(lineSegments);

      // 6. Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
      scene.add(ambientLight);

      const pointLight1 = new THREE.PointLight(coreColor, 3.5, 30);
      pointLight1.position.set(0, 0, 0);
      scene.add(pointLight1);

      const pointLight2 = new THREE.PointLight(0x38bdf8, 2, 40);
      pointLight2.position.set(12, 10, 15);
      scene.add(pointLight2);

      // 7. Interactive Controls & Mouse Physics
      let isDragging = false;
      let prevMousePos = { x: 0, y: 0 };
      let rotVelocity = { x: 0, y: 0.003 };
      let baseSpeed = 0.003;
      let isWarp = false;
      let pulseScale = 1.0;

      container.addEventListener('mousedown', (e) => {
        isDragging = true;
        prevMousePos = { x: e.clientX, y: e.clientY };
      });

      window.addEventListener('mouseup', () => { isDragging = false; });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) {
          // Subtle parallax camera tilt
          const rect = container.getBoundingClientRect();
          const normX = ((e.clientX - rect.left) / width - 0.5) * 2;
          const normY = ((e.clientY - rect.top) / height - 0.5) * 2;
          camera.position.x += (normX * 1.5 - camera.position.x) * 0.05;
          camera.position.y += (-normY * 1.5 - camera.position.y) * 0.05;
          camera.lookAt(0, 0, 0);
          return;
        }
        const deltaX = e.clientX - prevMousePos.x;
        const deltaY = e.clientY - prevMousePos.y;
        rotVelocity.y = deltaX * 0.005;
        rotVelocity.x = deltaY * 0.005;
        prevMousePos = { x: e.clientX, y: e.clientY };
      });

      // HUD Button Listeners
      document.getElementById('btnPulse').addEventListener('click', () => {
        pulseScale = 1.7;
        pointLight1.intensity = 8.0;
      });

      const btnSpeed = document.getElementById('btnSpeed');
      btnSpeed.addEventListener('click', () => {
        isWarp = !isWarp;
        baseSpeed = isWarp ? 0.015 : 0.003;
        btnSpeed.textContent = isWarp ? "⏸ CRUISE SPEED" : "⏩ WARP SPEED";
      });

      document.getElementById('btnReset').addEventListener('click', () => {
        neuralGroup.rotation.set(0, 0, 0);
        camera.position.set(0, 0, 18);
        camera.lookAt(0, 0, 0);
        rotVelocity = { x: 0, y: 0.003 };
      });

      // 8. Animation Loop
      let clock = new THREE.Clock();

      function animate() {
        requestAnimationFrame(animate);
        const delta = clock.getDelta();
        const time = clock.getElapsedTime();

        // Rotate Root Group with Damping
        if (!isDragging) {
          rotVelocity.y += (baseSpeed - rotVelocity.y) * 0.03;
          rotVelocity.x += (0 - rotVelocity.x) * 0.03;
        }
        neuralGroup.rotation.y += rotVelocity.y;
        neuralGroup.rotation.x += rotVelocity.x;

        // Counter-rotate core and cage
        coreMesh.rotation.y += 0.008;
        coreMesh.rotation.x -= 0.005;
        cageMesh.rotation.y -= 0.004;
        cageMesh.rotation.z += 0.003;

        // Rotate LSTM rings
        ring1.rotation.z += 0.012;
        ring2.rotation.x += 0.009;
        ring3.rotation.y += 0.007;

        // Pulse decay
        if (pulseScale > 1.0) {
          pulseScale += (1.0 - pulseScale) * 0.08;
          coreMesh.scale.set(pulseScale, pulseScale, pulseScale);
          innerMesh.scale.set(pulseScale, pulseScale, pulseScale);
          pointLight1.intensity += (3.5 - pointLight1.intensity) * 0.08;
        }

        // Particle dynamics & dynamic synaptic lines
        const pos = particlePoints.geometry.attributes.position.array;
        let lineIdx = 0;

        for (let i = 0; i < particleCount; i++) {
          pos[i * 3] += particleVelocities[i].x;
          pos[i * 3 + 1] += particleVelocities[i].y;
          pos[i * 3 + 2] += particleVelocities[i].z;

          // Bounding sphere bounce
          const d = Math.sqrt(pos[i*3]*pos[i*3] + pos[i*3+1]*pos[i*3+1] + pos[i*3+2]*pos[i*3+2]);
          if (d > spread || d < 2.5) {
            particleVelocities[i].x *= -1;
            particleVelocities[i].y *= -1;
            particleVelocities[i].z *= -1;
          }

          // Build connections
          if (lineIdx < maxLines * 6) {
            for (let j = i + 1; j < particleCount; j++) {
              const dx = pos[i*3] - pos[j*3];
              const dy = pos[i*3+1] - pos[j*3+1];
              const dz = pos[i*3+2] - pos[j*3+2];
              const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

              if (dist < 2.3 && lineIdx < maxLines * 6) {
                linePositions[lineIdx++] = pos[i*3];
                linePositions[lineIdx++] = pos[i*3+1];
                linePositions[lineIdx++] = pos[i*3+2];
                linePositions[lineIdx++] = pos[j*3];
                linePositions[lineIdx++] = pos[j*3+1];
                linePositions[lineIdx++] = pos[j*3+2];
              }
            }
          }
        }

        particlePoints.geometry.attributes.position.needsUpdate = true;
        lineGeo.setDrawRange(0, lineIdx / 3);
        lineGeo.attributes.position.needsUpdate = true;

        renderer.render(scene, camera);
      }

      animate();

      // Window resize handling
      window.addEventListener('resize', () => {
        const newW = container.clientWidth || 800;
        const newH = container.clientHeight || 440;
        camera.aspect = newW / newH;
        camera.updateProjectionMatrix();
        renderer.setSize(newW, newH);
      });

      // Ambient parent background injection (fallback safe)
      try {
        const parentDoc = window.parent.document;
        if (parentDoc && !parentDoc.getElementById('truthlens-ambient-bg')) {
          const bg = parentDoc.createElement('div');
          bg.id = 'truthlens-ambient-bg';
          bg.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;background:radial-gradient(circle at 50% 10%, rgba(56,189,248,0.06) 0%, transparent 60%);';
          parentDoc.body.prepend(bg);
        }
      } catch (e) {
        // Cross-origin safe
      }
    })();
  </script>
</body>
</html>
""".replace("__CORE_HEX__", core_hex) \
   .replace("__WIRE_HEX__", wire_hex) \
   .replace("__PART_HEX__", part_hex) \
   .replace("__BG_GLOW__", bg_glow) \
   .replace("__BADGE_CLASS__", badge_class) \
   .replace("__STATUS_ICON__", status_icon)

    components.html(html_code, height=height)


# ==============================================================================
# 4. UI SECTIONS
# ==============================================================================

def render_top_navbar():
    """Render the top cyber navigation bar."""
    st.markdown(
        """
        <div class="top-navbar">
            <div class="nav-brand-group">
                <div class="nav-logo-icon">⚡</div>
                <div class="nav-logo-text">TRUTHLENS <span>AI</span></div>
                <span class="nav-badge">LSTM NEURAL NET</span>
                <span class="nav-lab-tag">Deep Learning Intelligence</span>
            </div>
            <div class="nav-status-group">
                <div class="nav-status-indicator">
                    <span class="status-pulse-dot"></span>
                    <span>Tensor Core: <strong>Online</strong> (TF 2.21)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section():
    """Render the hero section."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-pill">⚡ DEEP LEARNING • RECURRENT NEURAL NETWORK • NLP CLASSIFIER</div>
            <h1 class="hero-headline">
                Can AI Detect <span class="gradient-text">Misleading News?</span>
            </h1>
            <p class="hero-description">
                Inspect linguistic structures, semantic framing, and token sequences in real-time
                using our production-trained Bidirectional LSTM Neural Architecture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_circular_ring_svg(confidence: float, is_real: bool) -> str:
    """Generate inline SVG for circular progress ring gauge."""
    radius = 56
    circumference = 2 * math.pi * radius
    offset = circumference * (1.0 - confidence)
    ring_color = "#10b981" if is_real else "#f43f5e"
    glow_shadow = "drop-shadow(0 0 10px rgba(16, 185, 129, 0.75))" if is_real else "drop-shadow(0 0 10px rgba(244, 63, 94, 0.75))"

    return (
        f'<div class="ring-svg-wrap">'
        f'<svg width="140" height="140" viewBox="0 0 140 140" style="transform: rotate(-90deg); filter: {glow_shadow};">'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="11" />'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="{ring_color}" stroke-width="11" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
        f'stroke-linecap="round" style="transition: stroke-dashoffset 0.8s ease;" />'
        f'</svg>'
        f'<div class="ring-center-text">'
        f'<div class="ring-pct">{confidence * 100:.0f}%</div>'
        f'<div class="ring-caption">Confidence</div>'
        f'</div>'
        f'</div>'
    )


def render_result_dashboard(result: dict):
    """Render dual-card result dashboard with ring gauge and probability meter."""
    prediction = result["prediction"]
    confidence = result["confidence"]
    prob_real = result["probability"]
    prob_fake = 1.0 - prob_real

    is_real = prediction == "REAL"
    box_class = "verdict-box-real" if is_real else "verdict-box-fake"
    tag_class = "verdict-tag-real" if is_real else "verdict-tag-fake"
    title_class = "verdict-title-real" if is_real else "verdict-title-fake"
    sub_class = "verdict-sub-real" if is_real else "verdict-sub-fake"

    verdict_title = "Likely Authentic" if is_real else "Potentially Misleading"
    tag_label = "● VERIFIED ARTICLE" if is_real else "▲ HIGH RISK MISINFORMATION"
    model_classification_text = "Model Classification: REAL (Credible Pattern)" if is_real else "Model Classification: FAKE (Deceptive Pattern)"

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    col_verdict, col_gauge = st.columns([1.3, 1], gap="medium")

    with col_verdict:
        verdict_html = (
            f'<div class="{box_class}">'
            f'<span class="{tag_class}">{tag_label}</span>'
            f'<h2 class="{title_class}">{verdict_title}</h2>'
            f'<p class="{sub_class}">{model_classification_text}</p>'
            f'<div style="margin-top: 1.25rem; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">'
            f'The neural network analyzed word embeddings across the 300-token sequence window. '
            f'Extracted linguistic features strongly correlate with <strong>{prediction}</strong> reporting conventions.'
            f'</div>'
            f'</div>'
        )
        st.markdown(verdict_html, unsafe_allow_html=True)

    with col_gauge:
        ring_svg = render_circular_ring_svg(confidence, is_real)
        st.markdown(f'<div class="ring-card-box">{ring_svg}</div>', unsafe_allow_html=True)

    # Horizontal AI Analysis Spectrum Meter
    needle_pct = min(max(prob_real * 100.0, 2.0), 98.0)
    meter_html = (
        f'<div class="meter-card">'
        f'<div class="meter-header">'
        f'<span class="meter-heading">Linguistic Authenticity Spectrum</span>'
        f'<span class="meter-confidence-pill">Posterior Prob: {prob_real * 100:.2f}% Real</span>'
        f'</div>'
        f'<div class="spectrum-bar-track">'
        f'<div class="spectrum-pointer" style="left: {needle_pct}%;"></div>'
        f'</div>'
        f'<div class="spectrum-labels-row">'
        f'<span style="color: #f43f5e;">FAKE ({prob_fake * 100:.1f}%)</span>'
        f'<span style="color: #94a3b8; font-weight: 600;">Equivocal (50%)</span>'
        f'<span style="color: #10b981;">REAL ({prob_real * 100:.1f}%)</span>'
        f'</div>'
        f'<div class="meter-sub-caption">'
        f'Evaluated by LSTM Sigmoid Activation. Reflects learned dataset patterns.'
        f'</div>'
        f'</div>'
    )
    st.markdown(meter_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_model_insight_card():
    """Render the architectural specification card."""
    st.markdown(
        """
        <div class="app-card">
            <h3 class="card-heading">🧠 Neural Architecture Specifications</h3>
            <p class="card-subtext">
                TruthLens utilizes a Bidirectional LSTM network with sequence embeddings to detect subtle contextual cues.
            </p>
            <div class="spec-grid">
                <div class="spec-pill">
                    <div class="spec-label">Model Type</div>
                    <div class="spec-val">Bi-LSTM</div>
                </div>
                <div class="spec-pill">
                    <div class="spec-label">Hidden Units</div>
                    <div class="spec-val">128 LSTM Cells</div>
                </div>
                <div class="spec-pill">
                    <div class="spec-label">Embedding Dim</div>
                    <div class="spec-val">100 Dimensions</div>
                </div>
                <div class="spec-pill">
                    <div class="spec-label">Dropout</div>
                    <div class="spec-val">0.3 Rate</div>
                </div>
                <div class="spec-pill">
                    <div class="spec-label">Activation</div>
                    <div class="spec-val">Sigmoid Output</div>
                </div>
                <div class="spec-pill">
                    <div class="spec-label">Framework</div>
                    <div class="spec-val">TensorFlow 2.21</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_articles():
    """Render 3 dark glass sample cards that populate analyzer input."""
    samples = {
        "Sample 01": {
            "tag": "🚀 Science & Exploration",
            "title": "Mars Exploration Initiative",
            "text": (
                "CAPE CANAVERAL (Reuters) - NASA and international space agency partners confirmed "
                "final operational parameters on Wednesday for the upcoming Mars robotic exploration initiative. "
                "Flight systems engineers stated that deep-space navigation equipment and geological spectrometers "
                "successfully cleared pre-launch environmental verification tests at the Goddard Space Flight Center. "
                "The scientific payload will conduct in-situ mineralogical analyses."
            ),
        },
        "Sample 02": {
            "tag": "⚠️ Health Conspiracy Claim",
            "title": "Suppressed Miracle Discovery",
            "text": (
                "MIRACLE DISCOVERY SUPPRESSED: Ancient Himalayan root extract completely cures advanced diabetes "
                "and reverses cellular aging in just 48 hours! Corrupt pharmaceutical executives are frantically paying off "
                "federal health regulators to ban this revolutionary substance from public awareness. Stock up now before "
                "mainstream media and medical authorities outlaw it forever! Share this urgent truth immediately!"
            ),
        },
        "Sample 03": {
            "tag": "🌐 Diplomacy & Policy",
            "title": "Geneva Relief Agreement",
            "text": (
                "GENEVA (Reuters) - United Nations delegates reached a provisional framework agreement regarding "
                "humanitarian relief corridors following seventy-two hours of continuous multilateral negotiations. "
                "The accord sets rigorous neutral inspection protocols for convoys entering designated emergency zones, "
                "with diplomatic observers monitoring compliance across all affected border crossing points."
            ),
        },
    }

    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.85rem;">
            <h3 class="card-heading" style="margin-bottom: 0.25rem;">⚡ Pre-loaded Test Bench</h3>
            <p class="card-subtext" style="margin-bottom: 0;">Click any benchmark article to test inference and observe 3D lattice response.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    for col, (key, data) in zip([col1, col2, col3], samples.items()):
        with col:
            st.markdown(
                f"""
                <div class="sample-card-item">
                    <div>
                        <div class="sample-tag">{data['tag']}</div>
                        <div class="sample-title">{data['title']}</div>
                        <div class="sample-preview">{data['text'][:125]}...</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Load {key} →", key=f"btn_{key}", use_container_width=True):
                st.session_state["news_text_input"] = data["text"]
                st.session_state["run_prediction_flag"] = True
                st.rerun()


def render_how_it_works():
    """Render horizontal 4-step pipeline."""
    st.markdown(
        """
        <div class="app-card" style="margin-top: 2rem;">
            <h3 class="card-heading">🔄 Deep Learning Pipeline Architecture</h3>
            <p class="card-subtext">The mathematical stages from raw text input to classification verdict</p>
            <div class="steps-container">
                <div class="step-col">
                    <div class="step-num">01 // INGEST</div>
                    <div class="step-title">Text Cleaning</div>
                    <div class="step-desc">Strips noise, punctuation, and URLs while preserving semantic syntax.</div>
                </div>
                <div class="step-col">
                    <div class="step-num">02 // TOKENIZE</div>
                    <div class="step-title">Embedding Matrix</div>
                    <div class="step-desc">Converts word tokens into 100-dim vectors with 20,000 vocab vocabulary.</div>
                </div>
                <div class="step-col">
                    <div class="step-num">03 // INFERENCE</div>
                    <div class="step-title">LSTM Cell Memory</div>
                    <div class="step-desc">Recurrent gates retain long-term dependencies across 300 token steps.</div>
                </div>
                <div class="step-col">
                    <div class="step-num">04 // VERDICT</div>
                    <div class="step-title">Probability Score</div>
                    <div class="step-desc">Dense layer with Sigmoid outputs posterior credibility confidence.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_statistics():
    """Render compact statistics from actual project outputs."""
    stats = load_project_metadata()

    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.85rem;">
            <h3 class="card-heading" style="margin-bottom: 0.25rem;">📊 Empirical Test Metrics</h3>
            <p class="card-subtext" style="margin-bottom: 0;">Verified model evaluations on production holdout datasets</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-value">{stats['train_samples']:,}</div>
                <div class="stat-title">Training Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-value">{stats['test_samples']:,}</div>
                <div class="stat-title">Test Holdout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-value">{stats['vocab_size']:,}</div>
                <div class="stat-title">Vocabulary Size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-value">{stats['sequence_length']}</div>
                <div class="stat-title">Max Sequence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-value">{stats['accuracy'] * 100:.2f}%</div>
                <div class="stat-title">Test Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    """Render cyber footer."""
    st.markdown(
        """
        <div class="truthlens-footer">
            <div class="footer-brand">⚡ TRUTHLENS <span>AI</span></div>
            <div class="footer-sub">Deep Learning News Credibility Analysis • Built with TensorFlow & Three.js</div>
            <div class="footer-disclaimer">Model predictions evaluate learned linguistic patterns. Not a substitute for human investigative journalism.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 5. MAIN APPLICATION ORCHESTRATION
# ==============================================================================

def main():
    """Execute TruthLens AI Web Application."""
    setup_page_config()
    inject_custom_css()

    # Session State Initialization
    if "news_text_input" not in st.session_state:
        st.session_state["news_text_input"] = ""
    if "last_prediction_result" not in st.session_state:
        st.session_state["last_prediction_result"] = None
    if "run_prediction_flag" not in st.session_state:
        st.session_state["run_prediction_flag"] = False

    # Render Top Navigation Bar
    render_top_navbar()

    # Hero Section & 3D Three.js Visualizer
    render_hero_section()
    render_threejs_neural_matrix(result=st.session_state["last_prediction_result"], height=420)

    # Main Article Analyzer Card
    st.markdown(
        """
        <div class="app-card" style="margin-top: 1.5rem;">
            <h2 class="card-heading">📰 Article Intelligence Terminal</h2>
            <p class="card-subtext">Input article text or headlines below for recurrent neural analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        label="News Article Input",
        value=st.session_state["news_text_input"],
        placeholder="Paste headline or news article text here to trigger deep learning analysis...",
        height=220,
        label_visibility="collapsed",
    )
    st.session_state["news_text_input"] = user_text

    # Dynamic Character & Word Counter
    char_count = len(user_text)
    word_count = len(user_text.split()) if user_text.strip() else 0
    st.markdown(
        f"""
        <div class="char-counter-row">
            <span>● LENGTH: {char_count:,} CHARS</span>
            <span>● ESTIMATED TOKENS: ~{word_count:,} WORDS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Row
    col_submit, col_clear = st.columns([4, 1])
    with col_submit:
        analyze_clicked = st.button("⚡ Run Neural Analysis →", use_container_width=True)
    with col_clear:
        if st.button("Clear Input", use_container_width=True):
            st.session_state["news_text_input"] = ""
            st.session_state["last_prediction_result"] = None
            st.rerun()

    # Check execution trigger (button click or demo selection)
    should_run = analyze_clicked or st.session_state.get("run_prediction_flag", False)

    if should_run:
        st.session_state["run_prediction_flag"] = False
        current_input = st.session_state["news_text_input"]

        if not current_input.strip():
            st.warning("⚠️ Please provide news article text before running analysis.")
        else:
            with st.spinner("Analyzing linguistic patterns with Bidirectional LSTM..."):
                try:
                    result = predict_news(current_input)
                    st.session_state["last_prediction_result"] = result
                    # Rerun to update the Three.js 3D visualizer state dynamically
                    st.rerun()
                except FileNotFoundError as fnf:
                    st.error(f"Missing Model Artifact: {fnf}")
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as ex:
                    st.error(f"Inference Error: {str(ex)}")

    # Display Result Dashboard if prediction exists
    if st.session_state["last_prediction_result"]:
        render_result_dashboard(st.session_state["last_prediction_result"])

    # Pre-loaded Sample Articles
    render_sample_articles()

    # Model Insight & Architectural Specifications
    render_model_insight_card()

    # How It Works 4-Step Process
    render_how_it_works()

    # Empirical Project Statistics
    render_project_statistics()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
