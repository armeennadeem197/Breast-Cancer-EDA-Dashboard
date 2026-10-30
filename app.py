from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer EDA Dashboard",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Brand tokens and dataset contract
# -----------------------------------------------------------------------------
COLORS = {
    # Dark premium reference palette. Keep these tokens central so the
    # sidebar, cards, charts, hover states, and empty states stay coherent.
    "navy": "#172B3A",
    "dark_navy": "#14213D",
    "navy_2": "#263F50",
    "teal": "#36CFC9",
    "cyan": "#20D5E5",
    "light_teal": "#1A4A53",
    "blue": "#42A5F5",
    "purple": "#C77DFF",
    "coral": "#E78BEA",
    "pink": "#E78BEA",
    "background": "#172B3A",
    "card": "#203747",
    "secondary_card": "#263F50",
    "text": "#F4F7FB",
    "muted": "#A8BAC8",
    "line": "#345064",
    "slate": "#6D8EA3",
    "soft_teal": "#1A4A53",
    "soft_blue": "#1D4662",
    "soft_purple": "#453A62",
    "soft_coral": "#553C5D",
}

DATA_FILENAME = "Breast_Cancer.csv"
DATA_PATH = Path(__file__).resolve().parent / DATA_FILENAME

ORIGINAL_COLUMNS = [
    "Age",
    "Race",
    "Marital Status",
    "T Stage ",
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Tumor Size",
    "Estrogen Status",
    "Progesterone Status",
    "Regional Node Examined",
    "Reginol Node Positive",
    "Survival Months",
    "Status",
]

STAGE_ORDER = ["IIA", "IIB", "IIIA", "IIIB", "IIIC"]
T_STAGE_ORDER = ["T1", "T2", "T3", "T4"]
N_STAGE_ORDER = ["N1", "N2", "N3"]
AGE_GROUP_ORDER = ["<40", "40-49", "50-59", "60-69"]
STATUS_ORDER = ["Alive", "Dead"]
RACE_ORDER = ["White", "Black", "Other"]
GRADE_ORDER = ["1", "2", "3", "anaplastic; Grade IV"]

PAGE_CONFIG = [
    ("Overview", "dashboard"),
    ("Patient Demographics", "groups"),
    ("Cancer Stage Analysis", "account_tree"),
    ("Survival Analysis", "monitor_heart"),
    ("Tumor & Lymph Node Analysis", "biotech"),
    ("Correlation Analysis", "hub"),
    ("Data Explorer", "table_view"),
    ("Insights & Recommendations", "lightbulb"),
]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bc-navy: #172B3A;
            --bc-navy-2: #263F50;
            --bc-dark: #14213D;
            --bc-teal: #36CFC9;
            --bc-cyan: #20D5E5;
            --bc-blue: #42A5F5;
            --bc-purple: #C77DFF;
            --bc-coral: #E78BEA;
            --bc-bg: #172B3A;
            --bc-card: #203747;
            --bc-secondary-card: #263F50;
            --bc-text: #F4F7FB;
            --bc-muted: #A8BAC8;
            --bc-line: #345064;
            --bc-soft-teal: #1A4A53;
        }

        html, body, .stApp {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--bc-text);
        }

        .stApp {
            background: var(--bc-bg);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1.2rem;
        }

        .block-container {
            max-width: 1500px;
            padding: 1.35rem 2.7rem 3rem;
        }

        /* The reference shell uses a persistent dark navigation rail and a light
           analytics canvas. Keep the shell quiet so the data cards do the work. */
        section[data-testid="stSidebar"] {
            width: 274px !important;
            background: var(--bc-dark);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] > div:first-child {
            background: var(--bc-dark);
            padding: 1.25rem 0.95rem 1rem;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] label {
            color: #D7E2F2;
            font-size: 0.76rem;
        }

        section[data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: rgba(255, 255, 255, 0.11);
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.05rem 0.2rem 1.4rem;
        }

        .brand-mark {
            width: 2.35rem;
            height: 2.35rem;
            display: grid;
            place-items: center;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 0.78rem;
            color: #FFFFFF;
            background: var(--bc-teal);
            font-size: 0.8rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            box-shadow: 0 9px 24px rgba(54, 207, 201, 0.2);
        }

        .brand-name {
            color: #FFFFFF;
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            margin-top: 0.12rem;
            color: #8FA7C7;
            font-size: 0.67rem;
            line-height: 1.35;
        }

        .sidebar-label {
            margin: 0.55rem 0 0.48rem;
            color: #7F98B9;
            font-size: 0.63rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 0.18rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            min-height: 2.28rem;
            border: 1px solid transparent;
            border-radius: 0.62rem;
            padding: 0.45rem 0.7rem;
            transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.055);
            border-color: rgba(255, 255, 255, 0.08);
            transform: translateX(1px);
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(54, 207, 201, 0.14);
            border-color: rgba(54, 207, 201, 0.28);
            box-shadow: inset 3px 0 0 var(--bc-teal);
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
            color: #FFFFFF;
            font-weight: 750;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="input"] > div {
            min-height: 2.35rem;
            background: #203747;
            border: 1px solid #304564;
            border-radius: 0.58rem;
            color: #F8FAFC;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div,
        section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within > div {
            border-color: var(--bc-teal);
            box-shadow: 0 0 0 1px var(--bc-teal);
        }

        section[data-testid="stSidebar"] [data-baseweb="tag"] {
            background: rgba(54, 207, 201, 0.18);
            border: 1px solid rgba(54, 207, 201, 0.3);
            color: #CFFAF3;
        }

        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            background: var(--bc-teal);
            border-color: var(--bc-teal);
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button,
        section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
            min-height: 2.35rem;
            border: 1px solid #304564;
            border-radius: 0.58rem;
            background: #203747;
            color: #E7F2FF;
            font-weight: 650;
            box-shadow: none;
            transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
        section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
            background: rgba(54, 207, 201, 0.14);
            border-color: var(--bc-teal);
            transform: translateY(-1px);
        }

        .active-filter-panel, .dataset-panel {
            margin-top: 0.7rem;
            padding: 0.78rem 0.8rem;
            border: 1px solid #263B5D;
            border-radius: 0.72rem;
            background: rgba(39, 64, 96, 0.42);
            color: #C8D8EC;
        }

        .active-filter-title, .dataset-panel-title {
            color: #FFFFFF;
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .filter-chip {
            display: inline-block;
            margin: 0.34rem 0.24rem 0 0;
            border-radius: 999px;
            padding: 0.18rem 0.44rem;
            background: rgba(54, 207, 201, 0.15);
            color: #D8FCF7;
            font-size: 0.62rem;
        }

        .dataset-panel p {
            margin: 0.34rem 0 0;
            color: #9FB2D1 !important;
            font-size: 0.67rem !important;
            line-height: 1.45;
        }

        /* Overview hero */
        .hero {
            position: relative;
            min-height: 9.6rem;
            overflow: hidden;
            margin: 0 0 1.15rem;
            padding: 1.55rem 1.7rem;
            border: 1px solid rgba(20, 33, 61, 0.1);
            border-radius: 1.05rem;
            background: #203747;
            box-shadow: 0 16px 36px rgba(20, 33, 61, 0.14);
            color: #FFFFFF;
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 14rem;
            height: 14rem;
            right: -3rem;
            bottom: -9rem;
            border: 1px solid rgba(116, 229, 215, 0.25);
            border-radius: 50%;
            box-shadow: 0 0 0 1.35rem rgba(116, 229, 215, 0.05), 0 0 0 2.7rem rgba(116, 229, 215, 0.035);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 0.58rem;
            height: 0.58rem;
            right: 17%;
            top: 25%;
            border-radius: 50%;
            background: var(--bc-teal);
            box-shadow: 1.55rem 1.4rem 0 rgba(59, 130, 246, 0.72), -1.9rem 2.3rem 0 rgba(139, 92, 246, 0.65);
        }

        .hero > * {
            position: relative;
            z-index: 1;
        }

        .hero-eyebrow {
            margin-bottom: 0.45rem;
            color: #74E5D7;
            font-size: 0.64rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0;
            color: #FFFFFF;
            font-size: clamp(1.8rem, 3.2vw, 2.55rem);
            font-weight: 780;
            letter-spacing: -0.05em;
            line-height: 1.04;
        }

        .hero p {
            max-width: 44rem;
            margin: 0.55rem 0 0;
            color: #D8E5F7;
            font-size: 0.9rem;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.92rem;
        }

        .hero-meta span {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.56rem;
            background: rgba(255, 255, 255, 0.1);
            color: #D8E5F7;
            font-size: 0.65rem;
            font-weight: 650;
        }

        .hero-meta .hero-status {
            background: rgba(54, 207, 201, 0.18);
            color: #CFFAF3;
        }

        .hero-status::before {
            content: "";
            width: 0.38rem;
            height: 0.38rem;
            margin-right: 0.35rem;
            border-radius: 50%;
            background: var(--bc-teal);
            box-shadow: 0 0 0 3px rgba(54, 207, 201, 0.13);
        }

        .dashboard-toolbar {
            display: flex;
            align-items: center;
            gap: 1.3rem;
            min-height: 2.6rem;
            padding: 0.42rem 0.75rem;
            border: 1px solid var(--bc-line);
            border-radius: 0.7rem;
            background: rgba(32, 55, 71, 0.68);
        }

        .dashboard-toolbar > div {
            display: flex;
            flex-direction: column;
            gap: 0.08rem;
        }

        .toolbar-label {
            color: var(--bc-muted);
            font-size: 0.57rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .toolbar-value {
            color: var(--bc-text);
            font-size: 0.7rem;
            font-weight: 700;
        }

        .mini-chart-label {
            margin: 0.2rem 0 0.05rem;
            color: var(--bc-muted);
            font-size: 0.62rem;
            font-weight: 750;
            letter-spacing: 0.03em;
        }

        /* Page headers, metric strip, and content cards */
        .page-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.15rem 0 1rem;
            padding: 0.2rem 0 0.88rem;
            border-bottom: 1px solid var(--bc-line);
        }

        .page-heading h1 {
            margin: 0;
            color: var(--bc-navy);
            font-size: clamp(1.5rem, 2.5vw, 2rem);
            font-weight: 780;
            letter-spacing: -0.045em;
        }

        .page-heading p {
            margin: 0.34rem 0 0;
            color: var(--bc-muted);
            font-size: 0.82rem;
        }

        .data-badge {
            display: inline-flex;
            flex: 0 0 auto;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid #C6ECE6;
            border-radius: 999px;
            padding: 0.36rem 0.65rem;
            background: var(--bc-soft-teal);
            color: var(--bc-navy-2);
            font-size: 0.68rem;
            font-weight: 750;
        }

        .data-badge span {
            color: var(--bc-teal);
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.15rem 0 1.22rem;
        }

        .kpi-card {
            min-height: 7.35rem;
            padding: 0.9rem 0.95rem 0.82rem;
            border: 1px solid var(--bc-line);
            border-top: 3px solid var(--bc-teal);
            border-radius: 0.82rem;
            background: var(--bc-card);
            box-shadow: 0 8px 22px rgba(20, 33, 61, 0.055);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .kpi-card:nth-child(2) { border-top-color: var(--bc-blue); }
        .kpi-card:nth-child(3) { border-top-color: var(--bc-coral); }
        .kpi-card:nth-child(4) { border-top-color: var(--bc-purple); }
        .kpi-card:nth-child(5) { border-top-color: var(--bc-navy-2); }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(20, 33, 61, 0.1);
        }

        .kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .kpi-label {
            color: var(--bc-muted);
            font-size: 0.7rem;
            font-weight: 700;
        }

        .kpi-icon {
            display: grid;
            width: 1.78rem;
            height: 1.78rem;
            place-items: center;
            border-radius: 0.52rem;
            font-size: 0.9rem;
        }

        .kpi-value {
            margin-top: 0.68rem;
            color: var(--bc-navy);
            font-size: clamp(1.25rem, 2vw, 1.7rem);
            font-weight: 820;
            letter-spacing: -0.045em;
            line-height: 1;
            overflow-wrap: anywhere;
        }

        .kpi-hint {
            margin-top: 0.44rem;
            color: var(--bc-muted);
            font-size: 0.66rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--bc-line);
            border-radius: 0.86rem;
            background: var(--bc-card);
            box-shadow: 0 7px 20px rgba(20, 33, 61, 0.04);
        }

        .chart-title {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.12rem;
        }

        .chart-title h3 {
            margin: 0;
            color: var(--bc-navy);
            font-size: 0.91rem;
            font-weight: 780;
            letter-spacing: -0.015em;
        }

        .chart-title span {
            color: var(--bc-teal);
            font-size: 0.6rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .chart-subtitle {
            margin: 0 0 0.28rem;
            color: var(--bc-muted);
            font-size: 0.7rem;
            line-height: 1.45;
        }

        .section-kicker {
            margin: 1.08rem 0 0.55rem;
            color: var(--bc-navy-2);
            font-size: 0.63rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.78rem 0 1.1rem;
        }

        .insight-card {
            padding: 0.95rem 1rem;
            border: 1px solid var(--bc-line);
            border-radius: 0.78rem;
            background: var(--bc-card);
            box-shadow: 0 5px 16px rgba(20, 33, 61, 0.025);
        }

        .insight-card .icon {
            display: inline-grid;
            width: 1.82rem;
            height: 1.82rem;
            place-items: center;
            margin-bottom: 0.54rem;
            border-radius: 0.52rem;
        }

        .insight-card h4 {
            margin: 0;
            color: var(--bc-navy);
            font-size: 0.84rem;
        }

        .insight-card p {
            margin: 0.32rem 0 0;
            color: var(--bc-muted);
            font-size: 0.76rem;
            line-height: 1.52;
        }

        .empty-state {
            display: grid;
            min-height: 12rem;
            place-items: center;
            padding: 1.5rem;
            border: 1px dashed #B8C5D9;
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.72);
            text-align: center;
        }

        .empty-state strong {
            color: var(--bc-navy);
            font-size: 1rem;
        }

        .empty-state p {
            max-width: 28rem;
            margin: 0.35rem 0 0;
            color: var(--bc-muted);
            font-size: 0.8rem;
        }

        .callout {
            margin: 0.15rem 0 0.95rem;
            border-left: 3px solid var(--bc-teal);
            border-radius: 0 0.72rem 0.72rem 0;
            padding: 0.78rem 0.95rem;
            background: var(--bc-soft-teal);
            color: var(--bc-navy-2);
            font-size: 0.76rem;
            line-height: 1.55;
        }

        .footer-note {
            padding-top: 1.25rem;
            color: var(--bc-muted);
            font-size: 0.68rem;
            text-align: right;
        }

        .material-symbols-rounded {
            font-family: "Material Symbols Rounded";
            font-weight: normal;
            font-style: normal;
            font-size: 1em;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-feature-settings: "liga";
            -webkit-font-smoothing: antialiased;
        }

        /* Native controls inherit the same card language on inner pages. */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            border-color: var(--bc-line);
            border-radius: 0.62rem;
            background: var(--bc-card);
        }

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            border-radius: 0.62rem;
            border: 1px solid var(--bc-line);
            background: var(--bc-secondary-card);
            color: var(--bc-text);
            font-weight: 700;
        }

        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            border-color: var(--bc-cyan);
            background: rgba(32, 213, 229, 0.12);
            color: var(--bc-text);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--bc-line);
            border-radius: 0.8rem;
            overflow: hidden;
        }

        @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .block-container { padding-left: 1.7rem; padding-right: 1.7rem; }
        }

        @media (max-width: 720px) {
            section[data-testid="stSidebar"] { width: 100% !important; }
            .block-container { padding: 1rem 1rem 2rem; }
            .hero { padding: 1.2rem 1.1rem; }
            .page-heading { align-items: flex-start; flex-direction: column; }
            .kpi-grid, .insight-grid { grid-template-columns: 1fr; }
        }

        /* Dark premium analytics reference layer */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: var(--bc-bg);
        }

        .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--bc-text);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--bc-line);
            background: var(--bc-card);
            box-shadow: 0 9px 24px rgba(5, 15, 22, 0.16);
        }

        .hero {
            border-color: var(--bc-line);
            border-left: 3px solid var(--bc-cyan);
            background: var(--bc-card);
            box-shadow: 0 16px 34px rgba(5, 15, 22, 0.2);
        }

        .hero::before {
            border-color: rgba(32, 213, 229, 0.24);
            box-shadow: 0 0 0 1.35rem rgba(32, 213, 229, 0.05), 0 0 0 2.7rem rgba(32, 213, 229, 0.025);
        }

        .hero h1, .page-heading h1, .chart-title h3, .insight-card h4,
        .empty-state strong, .kpi-value {
            color: var(--bc-text);
        }

        .hero-eyebrow, .chart-title span, .section-kicker {
            color: var(--bc-cyan);
        }

        .hero p, .page-heading p, .chart-subtitle, .kpi-label, .kpi-hint,
        .insight-card p, .empty-state p {
            color: var(--bc-muted);
        }

        .hero-meta span {
            background: rgba(52, 80, 100, 0.72);
            color: var(--bc-muted);
        }

        .hero-meta .hero-status {
            background: rgba(54, 207, 201, 0.16);
            color: #B8F4F1;
        }

        .data-badge {
            border-color: rgba(54, 207, 201, 0.42);
            background: rgba(54, 207, 201, 0.12);
            color: #B8F4F1;
        }

        .data-badge span { color: var(--bc-cyan); }

        .kpi-card, .insight-card {
            border-color: var(--bc-line);
            background: var(--bc-card);
            box-shadow: 0 8px 22px rgba(5, 15, 22, 0.14);
        }

        .kpi-card:nth-child(1) { border-top-color: var(--bc-cyan); }
        .kpi-card:nth-child(2) { border-top-color: var(--bc-teal); }
        .kpi-card:nth-child(3) { border-top-color: var(--bc-pink); }
        .kpi-card:nth-child(4) { border-top-color: var(--bc-purple); }
        .kpi-card:nth-child(5) { border-top-color: var(--bc-blue); }

        .overview-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.62rem;
            margin: 0;
        }

        .overview-kpi-grid .kpi-card {
            min-height: 6.7rem;
        }

        .empty-state {
            border-color: var(--bc-line);
            background: rgba(32, 55, 71, 0.62);
        }

        .callout {
            border-left-color: var(--bc-cyan);
            background: rgba(32, 213, 229, 0.1);
            color: #CFEFF5;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stDateInput"] input {
            border-color: var(--bc-line);
            background: var(--bc-card);
            color: var(--bc-text);
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
            border-color: var(--bc-cyan);
            box-shadow: 0 0 0 1px var(--bc-cyan);
        }

        [data-baseweb="select"] *, [data-baseweb="input"] *,
        [data-testid="stTextInput"] input::placeholder {
            color: var(--bc-text);
        }

        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"] {
            border-color: var(--bc-line);
            background: var(--bc-card);
            color: var(--bc-text);
        }

        [role="option"]:hover, [role="option"][aria-selected="true"] {
            background: rgba(32, 213, 229, 0.12);
            color: var(--bc-text);
        }

        [data-testid="stAlert"] {
            border-color: var(--bc-line);
            background: var(--bc-secondary-card);
            color: var(--bc-text);
        }

        [data-testid="stDataFrame"] {
            border-color: var(--bc-line);
            background: var(--bc-card);
        }

        [data-testid="stDataFrame"] * {
            color: var(--bc-text);
        }

        [data-testid="stExpander"] {
            border-color: var(--bc-line);
            background: var(--bc-card);
        }

        @media (max-width: 720px) {
            .overview-kpi-grid { grid-template-columns: 1fr; }
            .dashboard-toolbar { align-items: flex-start; flex-direction: column; gap: 0.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data preparation and calculations
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the source file, follow the notebook's duplicate handling, and add derived fields."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_FILENAME} was not found beside app.py. Place the original CSV in the project folder."
        )

    # Use the exact requested filename when the app is launched from this project.
    # The absolute-path fallback keeps the app reliable when launched from elsewhere.
    if Path(DATA_FILENAME).exists():
        df = pd.read_csv(DATA_FILENAME)
    else:
        df = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in ORIGINAL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing expected columns: {', '.join(missing_columns)}")

    df = df[ORIGINAL_COLUMNS].drop_duplicates(keep="first").reset_index(drop=True).copy()
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()

    numeric_columns = [
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Survival Months",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Grade Numeric"] = pd.to_numeric(df["Grade"], errors="coerce")
    df.loc[df["Grade"].eq("anaplastic; Grade IV"), "Grade Numeric"] = 4
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 39, 49, 59, 69],
        labels=AGE_GROUP_ORDER,
        include_lowest=True,
    ).astype("string")
    df["Node Positivity Rate"] = np.where(
        df["Regional Node Examined"].gt(0),
        df["Reginol Node Positive"] / df["Regional Node Examined"] * 100,
        np.nan,
    )
    df["Patient #"] = np.arange(1, len(df) + 1)
    return df


def ordered_values(series: pd.Series, preferred: Iterable[str]) -> list[str]:
    values = [value for value in preferred if value in set(series.dropna().astype(str))]
    remaining = sorted(set(series.dropna().astype(str)) - set(values))
    return values + remaining


def initialise_filter_state(df: pd.DataFrame) -> None:
    defaults = {
        "filter_age": (int(df["Age"].min()), int(df["Age"].max())),
        "filter_stage": ordered_values(df["6th Stage"], STAGE_ORDER),
        "filter_race": ordered_values(df["Race"], RACE_ORDER),
        "filter_status": ordered_values(df["Status"], STATUS_ORDER),
        "filter_grade": ordered_values(df["Grade"], GRADE_ORDER),
        "filter_estrogen": sorted(df["Estrogen Status"].dropna().unique().tolist()),
        "filter_progesterone": sorted(df["Progesterone Status"].dropna().unique().tolist()),
        "corr_features": [
            "Age",
            "Tumor Size",
            "Regional Node Examined",
            "Reginol Node Positive",
            "Survival Months",
            "Grade Numeric",
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_filters(df: pd.DataFrame) -> None:
    st.session_state["filter_age"] = (int(df["Age"].min()), int(df["Age"].max()))
    st.session_state["filter_stage"] = ordered_values(df["6th Stage"], STAGE_ORDER)
    st.session_state["filter_race"] = ordered_values(df["Race"], RACE_ORDER)
    st.session_state["filter_status"] = ordered_values(df["Status"], STATUS_ORDER)
    st.session_state["filter_grade"] = ordered_values(df["Grade"], GRADE_ORDER)
    st.session_state["filter_estrogen"] = sorted(df["Estrogen Status"].dropna().unique().tolist())
    st.session_state["filter_progesterone"] = sorted(df["Progesterone Status"].dropna().unique().tolist())


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    initialise_filter_state(df)
    stage_options = ordered_values(df["6th Stage"], STAGE_ORDER)
    race_options = ordered_values(df["Race"], RACE_ORDER)
    status_options = ordered_values(df["Status"], STATUS_ORDER)
    grade_options = ordered_values(df["Grade"], GRADE_ORDER)
    hormone_options = sorted(df["Estrogen Status"].dropna().unique().tolist())
    progesterone_options = sorted(df["Progesterone Status"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-mark">BC</div>
                <div>
                    <div class="brand-name">CareLens Analytics</div>
                    <div class="brand-subtitle">Breast cancer data intelligence</div>
                </div>
            </div>
            <div class="sidebar-label">Workspace</div>
            """,
            unsafe_allow_html=True,
        )

        page_labels = [f":material/{icon}: {page}" for page, icon in PAGE_CONFIG]
        selected_label = st.radio(
            "Dashboard sections",
            page_labels,
            key="active_page",
            label_visibility="collapsed",
        )
        selected_page = selected_label.split(": ", 1)[1]

        st.divider()
        st.markdown('<div class="sidebar-label">Filter population</div>', unsafe_allow_html=True)
        if st.button(
            "Reset all filters",
            key="reset_filters",
            icon=":material/refresh:",
            width="stretch",
            on_click=reset_filters,
            args=(df,),
        ):
            st.toast("Filters reset to the full dataset.", icon=":material/check_circle:")

        st.slider(
            "Age range",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            key="filter_age",
            help="Inclusive age range applied to every page.",
        )
        st.multiselect("Cancer stage", stage_options, key="filter_stage")
        st.multiselect("Race", race_options, key="filter_race")
        st.multiselect("Patient status", status_options, key="filter_status")
        st.multiselect("Grade", grade_options, key="filter_grade")
        st.multiselect("Estrogen status", hormone_options, key="filter_estrogen")
        st.multiselect("Progesterone status", progesterone_options, key="filter_progesterone")

    filters = {
        "age": st.session_state["filter_age"],
        "stage": st.session_state["filter_stage"],
        "race": st.session_state["filter_race"],
        "status": st.session_state["filter_status"],
        "grade": st.session_state["filter_grade"],
        "estrogen": st.session_state["filter_estrogen"],
        "progesterone": st.session_state["filter_progesterone"],
    }
    filtered = apply_filters(df, filters)

    active_filter_chips: list[str] = []
    age_bounds = (int(df["Age"].min()), int(df["Age"].max()))
    if tuple(filters["age"]) != age_bounds:
        active_filter_chips.append(f"Age {filters['age'][0]}–{filters['age'][1]}")
    for label, key, options in [
        ("Stage", "stage", stage_options),
        ("Race", "race", race_options),
        ("Status", "status", status_options),
        ("Grade", "grade", grade_options),
        ("ER", "estrogen", hormone_options),
        ("PR", "progesterone", progesterone_options),
    ]:
        selected = filters[key]
        if set(selected) != set(options):
            if not selected:
                value = "None"
            elif len(selected) <= 2:
                value = ", ".join(str(item) for item in selected)
            else:
                value = f"{len(selected)} selected"
            active_filter_chips.append(f"{label}: {value}")

    with st.sidebar:
        st.divider()
        if len(filtered) == len(df):
            st.caption(f"Showing all {len(df):,} patient records")
        else:
            st.caption(f"Showing {len(filtered):,} of {len(df):,} patient records")
        st.download_button(
            "Download filtered CSV",
            data=filtered[ORIGINAL_COLUMNS].to_csv(index=False).encode("utf-8"),
            file_name="Breast_Cancer_filtered.csv",
            mime="text/csv",
            key="sidebar_download",
            icon=":material/download:",
            width="stretch",
        )
        chip_html = "".join(
            f'<span class="filter-chip">{escape(str(chip))}</span>'
            for chip in active_filter_chips
        ) or '<span class="filter-chip">All records</span>'
        st.markdown(
            f'<div class="active-filter-panel"><div class="active-filter-title">Active filters</div>{chip_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'''
            <div class="dataset-panel">
                <div class="dataset-panel-title">Dataset</div>
                <p><strong>Breast_Cancer.csv</strong><br>{len(df):,} unique records · {len(ORIGINAL_COLUMNS)} source fields<br>Values retained from the supplied file.</p>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="footer-note" style="text-align:left; color:#8FA7C7;">CareLens Analytics · descriptive view</div>',
            unsafe_allow_html=True,
        )

    return filtered, {"page": selected_page, **filters}


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    age_min, age_max = filters["age"]
    mask = df["Age"].between(age_min, age_max, inclusive="both")
    for column, key in [
        ("6th Stage", "stage"),
        ("Race", "race"),
        ("Status", "status"),
        ("Grade", "grade"),
        ("Estrogen Status", "estrogen"),
        ("Progesterone Status", "progesterone"),
    ]:
        values = filters[key]
        if not values:
            return df.iloc[0:0].copy()
        mask &= df[column].isin(values)
    return df.loc[mask].copy()


def add_page_heading(title: str, description: str, filtered: pd.DataFrame, total: int) -> None:
    scope = f"{len(filtered):,} patients in view"
    if len(filtered) == total:
        scope = f"{total:,} patient records"
    st.markdown(
        f"""
        <div class="page-heading">
            <div>
                <h1>{title}</h1>
                <p>{description}</p>
            </div>
            <div class="data-badge"><span>●</span> {scope}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">HEALTHCARE DATA ANALYTICS</div>
            <h1>Breast Cancer EDA Dashboard</h1>
            <p>Exploratory Data Analysis of Patient Demographics, Cancer Stages, Tumor Characteristics and Survival Patterns.</p>
            <div class="hero-meta">
                <span class="hero-status">Live source dataset</span>
                <span>{len(df):,} records in the current view</span>
                <span>Filters update every analysis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    toolbar_left, toolbar_right = st.columns([4.3, 1], gap="medium")
    with toolbar_left:
        st.markdown(
            f'<div class="dashboard-toolbar"><div><span class="toolbar-label">Dataset</span><span class="toolbar-value">Breast_Cancer.csv</span></div><div><span class="toolbar-label">Current view</span><span class="toolbar-value">{len(df):,} patient records</span></div><div><span class="toolbar-label">Mode</span><span class="toolbar-value">Descriptive analytics</span></div></div>',
            unsafe_allow_html=True,
        )
    with toolbar_right:
        if st.button(
            "Reset filters",
            key="hero_reset_filters",
            icon=":material/refresh:",
            width="stretch",
            on_click=reset_filters,
            args=(load_data(),),
        ):
            st.toast("Filters reset to the full dataset.", icon=":material/check_circle:")


def format_number(value: float | int, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def common_value(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return "N/A"
    counts = df[column].value_counts(dropna=True)
    return str(counts.index[0]) if not counts.empty else "N/A"


def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    alive = int((df["Status"] == "Alive").sum()) if total else 0
    dead = int((df["Status"] == "Dead").sum()) if total else 0
    average_survival = df["Survival Months"].mean() if total else np.nan
    stage = common_value(df, "6th Stage")
    items = [
        ("Total patients", format_number(total), "Filtered records", "groups", COLORS["soft_blue"], COLORS["blue"]),
        ("Alive patients", format_number(alive), f"{alive / total * 100:.1f}% of view" if total else "No records", "favorite", COLORS["soft_teal"], COLORS["teal"]),
        ("Dead patients", format_number(dead), f"{dead / total * 100:.1f}% of view" if total else "No records", "monitor_heart", COLORS["soft_coral"], COLORS["coral"]),
        ("Average survival", f"{format_number(average_survival, 1)} mo", "Observed average", "schedule", COLORS["soft_purple"], COLORS["purple"]),
        ("Most common stage", stage, "6th stage classification", "account_tree", COLORS["secondary_card"], COLORS["blue"]),
    ]
    card_columns = st.columns(2, gap="small")
    for index, (label, value, hint, icon, bg, accent) in enumerate(items):
        with card_columns[index % 2]:
            kpi_html = f"""
                <div class="kpi-card">
                    <div class="kpi-top">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-icon" style="background:{bg}; color:{accent};">{material_icon(icon)}</div>
                    </div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-hint">{hint}</div>
                </div>
            """
            # This is intentionally the only KPI output call: render the
            # trusted HTML string directly, never through write/code/text.
            st.markdown(kpi_html.strip(), unsafe_allow_html=True)


def material_icon(name: str) -> str:
    return f'<span class="material-symbols-rounded">{name}</span>'


def style_figure(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(family="Inter, ui-sans-serif, sans-serif", color=COLORS["text"], size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10, color=COLORS["muted"]),
            bgcolor="rgba(32,55,71,0)",
        ),
        hoverlabel=dict(bgcolor=COLORS["secondary_card"], bordercolor=COLORS["cyan"], font_size=11, font_color=COLORS["text"]),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=COLORS["line"],
        zeroline=False,
        tickfont=dict(color=COLORS["muted"], size=10),
        title_font=dict(color=COLORS["muted"], size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(52,80,100,0.62)",
        linecolor=COLORS["line"],
        zeroline=False,
        tickfont=dict(color=COLORS["muted"], size=10),
        title_font=dict(color=COLORS["muted"], size=10),
    )
    return fig


def plot_card(title: str, subtitle: str, fig: go.Figure, alt: str, key: str) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="chart-title"><h3>{title}</h3><span>Interactive</span></div><p class="chart-subtitle">{subtitle}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            theme=None,
            key=key,
            alt=alt,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )


def empty_state(message: str = "No records match the current filters.") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div>
                <div style="font-size:2rem; margin-bottom:0.4rem; color:{COLORS['teal']};">{material_icon('filter_alt_off')}</div>
                <strong>No matching patient records</strong>
                <p>{message} Use the sidebar filters to broaden the population and try again.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_counts(df: pd.DataFrame) -> pd.DataFrame:
    result = df["6th Stage"].value_counts().reindex(STAGE_ORDER, fill_value=0).rename_axis("Stage").reset_index(name="Patients")
    result["Stage"] = pd.Categorical(result["Stage"], categories=STAGE_ORDER, ordered=True)
    return result.sort_values("Stage")


def render_overview(df: pd.DataFrame, total: int) -> None:
    render_hero(df)
    if df.empty:
        empty_state()
        return

    # Row 1: compact KPI stack beside the status distribution, matching the
    # reference's dense two-column analytics layout.
    overview_left, overview_right = st.columns([1.02, 1.18], gap="medium")
    with overview_left:
        st.markdown('<div class="section-kicker">Patient overview</div>', unsafe_allow_html=True)
        render_kpis(df)

    status_counts = df["Status"].value_counts().reindex(STATUS_ORDER, fill_value=0).rename_axis("Status").reset_index(name="Patients")
    status_fig = px.pie(
        status_counts,
        names="Status",
        values="Patients",
        hole=0.68,
        color="Status",
        color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["pink"]},
    )
    status_fig.update_traces(textinfo="percent", textposition="inside", hovertemplate="%{label}<br>%{value:,} patients<extra></extra>")
    status_fig.add_annotation(text=f"{len(df):,}<br><span style='font-size:11px'>patients</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=21, color=COLORS["text"]))
    status_fig = style_figure(status_fig, 286)
    with overview_right:
        st.markdown('<div class="section-kicker">Patient status analysis</div>', unsafe_allow_html=True)
        plot_card("Patient status", "Alive and dead records in the selected population.", status_fig, "Donut chart showing patient status shares", "overview_status")

    # Row 2: stage mix and observed survival.
    st.markdown('<div class="section-kicker">Clinical profile</div>', unsafe_allow_html=True)
    stage_fig = px.bar(
        stage_counts(df),
        x="Stage",
        y="Patients",
        text="Patients",
        color="Stage",
        color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]],
    )
    stage_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="Stage %{x}<br>%{y:,} patients<extra></extra>")
    stage_fig.update_layout(showlegend=False, yaxis_title="Patients", xaxis_title="")
    stage_fig = style_figure(stage_fig, 282)

    survival_summary = (
        df.groupby("6th Stage", observed=False)
        .agg(Average_Survival=("Survival Months", "mean"), Patients=("Survival Months", "size"))
        .reindex(STAGE_ORDER)
        .dropna(subset=["Average_Survival"])
        .reset_index()
    )
    survival_fig = px.bar(
        survival_summary,
        x="6th Stage",
        y="Average_Survival",
        text="Average_Survival",
        color="6th Stage",
        color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]],
        hover_data={"Patients": True, "Average_Survival": ":.1f"},
    )
    survival_fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside", hovertemplate="Stage %{x}<br>Average survival %{y:.1f} months<br>Patients %{customdata[0]:,}<extra></extra>")
    survival_fig.update_layout(showlegend=False, xaxis_title="6th stage", yaxis_title="Average survival (months)")
    survival_fig = style_figure(survival_fig, 282)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        plot_card("Cancer stage distribution", "6th-stage classification across the selected population.", stage_fig, "Bar chart showing patient counts by cancer stage", "overview_stage")
    with col2:
        plot_card("Average survival by stage", "Observed survival months, summarized by 6th-stage classification.", survival_fig, "Bar chart of average survival months by cancer stage", "overview_survival")

    # Row 3: demographics and a combined tumor/node card.
    st.markdown('<div class="section-kicker">Population detail</div>', unsafe_allow_html=True)
    demographics_fig = px.histogram(
        df,
        x="Age",
        color="Status",
        nbins=18,
        marginal="rug",
        barmode="overlay",
        opacity=0.74,
        color_discrete_map={"Alive": COLORS["cyan"], "Dead": COLORS["pink"]},
    )
    demographics_fig.update_traces(hovertemplate="Age %{x}<br>%{y:,} patients<extra></extra>")
    demographics_fig.update_layout(bargap=0.08, xaxis_title="Age (years)", yaxis_title="Patients")
    demographics_fig = style_figure(demographics_fig, 280)

    tumor_fig = px.histogram(
        df,
        x="Tumor Size",
        color="Status",
        nbins=18,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["pink"]},
    )
    tumor_fig.update_layout(showlegend=False, xaxis_title="Tumor size", yaxis_title="Patients")
    tumor_fig = style_figure(tumor_fig, 210)

    node_stage = (
        df.groupby("6th Stage", observed=False)
        .agg(Examined=("Regional Node Examined", "mean"), Positive=("Reginol Node Positive", "mean"))
        .reindex(STAGE_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    node_long = node_stage.melt(id_vars=["6th Stage"], var_name="Measure", value_name="Average Nodes")
    node_fig = px.bar(
        node_long,
        x="6th Stage",
        y="Average Nodes",
        color="Measure",
        barmode="group",
        color_discrete_map={"Examined": COLORS["cyan"], "Positive": COLORS["purple"]},
    )
    node_fig.update_layout(xaxis_title="Stage", yaxis_title="Average nodes")
    node_fig = style_figure(node_fig, 210)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        plot_card("Patient demographics", "Age distribution split by current patient status.", demographics_fig, "Histogram of patient ages split by status", "overview_demographics")
    with col2:
        with st.container(border=True):
            st.markdown('<div class="chart-title"><h3>Tumor &amp; lymph node analysis</h3><span>Interactive</span></div><p class="chart-subtitle">Tumor size distribution and average regional node counts.</p>', unsafe_allow_html=True)
            mini_left, mini_right = st.columns(2, gap="small")
            with mini_left:
                st.markdown('<div class="mini-chart-label">Tumor size</div>', unsafe_allow_html=True)
                st.plotly_chart(tumor_fig, width="stretch", theme=None, key="overview_tumor", alt="Tumor size distribution by status", config={"displaylogo": False, "responsive": True, "scrollZoom": True})
            with mini_right:
                st.markdown('<div class="mini-chart-label">Regional nodes</div>', unsafe_allow_html=True)
                st.plotly_chart(node_fig, width="stretch", theme=None, key="overview_nodes", alt="Average regional nodes examined and positive by stage", config={"displaylogo": False, "responsive": True, "scrollZoom": True})

    # Row 4: full-width correlation matrix and the source records.
    st.markdown('<div class="section-kicker">Relationships &amp; records</div>', unsafe_allow_html=True)
    correlation_columns = [
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Survival Months",
        "Grade Numeric",
    ]
    correlation = df[correlation_columns].corr(numeric_only=True)
    display_names = {
        "Age": "Age",
        "Tumor Size": "Tumor size",
        "Regional Node Examined": "Nodes examined",
        "Reginol Node Positive": "Nodes positive",
        "Survival Months": "Survival months",
        "Grade Numeric": "Grade",
    }
    correlation_display = correlation.copy()
    correlation_display.index = [display_names[column] for column in correlation_display.index]
    correlation_display.columns = [display_names[column] for column in correlation_display.columns]
    heatmap = go.Figure(
        go.Heatmap(
            z=correlation_display.values,
            x=correlation_display.columns,
            y=correlation_display.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, COLORS["pink"]], [0.5, COLORS["card"]], [1, COLORS["cyan"]]],
            text=np.round(correlation_display.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} x %{x}<br>Correlation %{z:.3f}<extra></extra>",
            colorbar=dict(title="r", thickness=12, tickfont=dict(color=COLORS["muted"])),
        )
    )
    heatmap = style_figure(heatmap, 350)
    heatmap.update_traces(hovertemplate="%{y} x %{x}<br>Correlation %{z:.3f}<extra></extra>")
    plot_card("Correlation analysis", "Descriptive relationships among the numeric fields in the current view.", heatmap, "Correlation heatmap of selected numerical breast cancer fields", "overview_correlation")

    with st.container(border=True):
        st.markdown('<div class="chart-title"><h3>Detailed patient records</h3><span>Source fields</span></div><p class="chart-subtitle">Filtered rows from Breast_Cancer.csv. Original column names and values are retained.</p>', unsafe_allow_html=True)
        st.dataframe(
            df[ORIGINAL_COLUMNS],
            column_config={
                "Age": st.column_config.NumberColumn("Age", format="%d"),
                "Tumor Size": st.column_config.NumberColumn("Tumor size", format="%d"),
                "Regional Node Examined": st.column_config.NumberColumn("Regional nodes examined", format="%d"),
                "Reginol Node Positive": st.column_config.NumberColumn("Regional nodes positive", format="%d"),
                "Survival Months": st.column_config.NumberColumn("Survival months", format="%d"),
            },
            hide_index=True,
            height=330,
            width="stretch",
            alt="Filtered breast cancer patient records",
        )
        st.caption(f"Showing {len(df):,} filtered records from the supplied dataset.")

    # Keep one compact insight strip at the end of the overview for fast scan.
    comparison = pd.crosstab(df["6th Stage"], df["Status"]).reindex(STAGE_ORDER, fill_value=0).reset_index().melt(id_vars="6th Stage", var_name="Status", value_name="Patients")
    alive_pct = (df["Status"].eq("Alive").mean() * 100) if len(df) else np.nan
    top_stage = common_value(df, "6th Stage")
    top_age = common_value(df, "Age")
    stage_alive = comparison.loc[comparison["Status"].eq("Alive"), "Patients"].sum()
    st.markdown('<div class="section-kicker">Quick insights</div>', unsafe_allow_html=True)
    insights = [
        ("favorite", COLORS["soft_teal"], "Alive share", f"{alive_pct:.1f}% of the selected records are alive."),
        ("account_tree", COLORS["soft_blue"], "Most represented stage", f"Stage {top_stage} is the most common 6th-stage classification in view."),
        ("person", COLORS["soft_purple"], "Most common age", f"Age {top_age} appears most frequently; {stage_alive:,} alive records are in view."),
    ]
    cards = [f'<div class="insight-card"><div class="icon" style="background:{bg}; color:{COLORS["text"]};">{material_icon(icon)}</div><h4>{title}</h4><p>{text}</p></div>' for icon, bg, title, text in insights]
    st.markdown(f'<div class="insight-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_demographics(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Patient demographics", "Understand the composition and age profile of the selected patient population.", df, total)
    if df.empty:
        empty_state()
        return

    median_age = df["Age"].median()
    common_race = common_value(df, "Race")
    common_marital = common_value(df, "Marital Status")
    st.markdown(
        f'<div class="callout">The selected population has a median age of <strong>{median_age:.0f} years</strong>. {common_race} is the most represented race and {common_marital} is the most common recorded marital status in this view.</div>',
        unsafe_allow_html=True,
    )

    age_fig = px.histogram(df, x="Age", color="6th Stage", nbins=20, marginal="rug", color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]])
    age_fig.update_layout(barmode="stack", xaxis_title="Age (years)", yaxis_title="Patients")
    age_fig = style_figure(age_fig, 330)

    age_groups = df["Age Group"].value_counts().reindex(AGE_GROUP_ORDER, fill_value=0).rename_axis("Age Group").reset_index(name="Patients")
    age_group_fig = px.bar(age_groups, x="Age Group", y="Patients", text="Patients", color="Patients", color_continuous_scale=[COLORS["soft_teal"], COLORS["teal"], COLORS["cyan"]])
    age_group_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="%{x}<br>%{y:,} patients<extra></extra>")
    age_group_fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Age group", yaxis_title="Patients")
    age_group_fig = style_figure(age_group_fig, 330)

    col1, col2 = st.columns(2)
    with col1:
        plot_card("Age distribution", "Age histogram colored by 6th stage.", age_fig, "Histogram of age distribution colored by cancer stage", "demo_age")
    with col2:
        plot_card("Age group distribution", "Patient counts using consistent age bands.", age_group_fig, "Bar chart showing patient counts by age group", "demo_age_group")

    race_counts = df["Race"].value_counts().reindex(RACE_ORDER, fill_value=0).rename_axis("Race").reset_index(name="Patients")
    race_fig = px.bar(race_counts, x="Patients", y="Race", orientation="h", text="Patients", color="Race", color_discrete_sequence=[COLORS["cyan"], COLORS["blue"], COLORS["teal"]])
    race_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="%{y}<br>%{x:,} patients<extra></extra>")
    race_fig.update_layout(showlegend=False, xaxis_title="Patients", yaxis_title="")
    race_fig = style_figure(race_fig, 320)

    marital_counts = df["Marital Status"].value_counts().rename_axis("Marital Status").reset_index(name="Patients").sort_values("Patients")
    marital_fig = px.bar(marital_counts, x="Patients", y="Marital Status", orientation="h", text="Patients", color="Patients", color_continuous_scale=[COLORS["soft_purple"], COLORS["purple"], COLORS["blue"]])
    marital_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="%{y}<br>%{x:,} patients<extra></extra>")
    marital_fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Patients", yaxis_title="")
    marital_fig = style_figure(marital_fig, 320)

    col1, col2 = st.columns(2)
    with col1:
        plot_card("Race distribution", "Recorded race categories in the selected data.", race_fig, "Horizontal bar chart showing patient counts by race", "demo_race")
    with col2:
        plot_card("Marital status distribution", "Recorded marital status categories.", marital_fig, "Horizontal bar chart showing patient counts by marital status", "demo_marital")

    stage_age_fig = px.box(df, x="6th Stage", y="Age", color="6th Stage", category_orders={"6th Stage": STAGE_ORDER}, color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]])
    stage_age_fig.update_layout(showlegend=False, xaxis_title="6th stage", yaxis_title="Age (years)")
    stage_age_fig = style_figure(stage_age_fig, 330)

    age_counts = df["Age"].value_counts().sort_index().rename_axis("Age").reset_index(name="Patients")
    age_count_fig = px.area(age_counts, x="Age", y="Patients", markers=True, color_discrete_sequence=[COLORS["teal"]])
    age_count_fig.update_traces(hovertemplate="Age %{x}<br>%{y:,} patients<extra></extra>")
    age_count_fig.update_layout(xaxis_title="Age (years)", yaxis_title="Patients")
    age_count_fig = style_figure(age_count_fig, 330)

    col1, col2 = st.columns(2)
    with col1:
        plot_card("Patient age vs cancer stage", "Age spread within each 6th-stage classification.", stage_age_fig, "Box plot showing age by cancer stage", "demo_stage_age")
    with col2:
        plot_card("Patient count by age", "Exact age counts for the selected population.", age_count_fig, "Area chart showing patient count for each age", "demo_age_count")


def render_stage_analysis(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Cancer stage analysis", "Compare stage, grade, and recorded patient status across the selected population.", df, total)
    if df.empty:
        empty_state()
        return

    counts = stage_counts(df)
    stage_fig = px.bar(counts, x="Stage", y="Patients", color="Stage", text="Patients", color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]])
    stage_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="Stage %{x}<br>%{y:,} patients<extra></extra>")
    stage_fig.update_layout(showlegend=False, xaxis_title="6th stage", yaxis_title="Patients")
    stage_fig = style_figure(stage_fig, 320)

    t_counts = df["T Stage "].value_counts().reindex(T_STAGE_ORDER, fill_value=0).rename_axis("T Stage").reset_index(name="Patients")
    t_fig = px.bar(t_counts, x="T Stage", y="Patients", text="Patients", color="Patients", color_continuous_scale=[COLORS["soft_blue"], COLORS["blue"], COLORS["cyan"]])
    t_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="%{x}<br>%{y:,} patients<extra></extra>")
    t_fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="T stage", yaxis_title="Patients")
    t_fig = style_figure(t_fig, 320)

    n_counts = df["N Stage"].value_counts().reindex(N_STAGE_ORDER, fill_value=0).rename_axis("N Stage").reset_index(name="Patients")
    n_fig = px.bar(n_counts, x="N Stage", y="Patients", text="Patients", color="Patients", color_continuous_scale=[COLORS["soft_purple"], COLORS["purple"], COLORS["pink"]])
    n_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="%{x}<br>%{y:,} patients<extra></extra>")
    n_fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="N stage", yaxis_title="Patients")
    n_fig = style_figure(n_fig, 320)

    grade_counts = df["Grade"].value_counts().reindex(GRADE_ORDER, fill_value=0).rename_axis("Grade").reset_index(name="Patients")
    grade_fig = px.bar(grade_counts, x="Grade", y="Patients", text="Patients", color="Grade", color_discrete_sequence=[COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["coral"]])
    grade_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="Grade %{x}<br>%{y:,} patients<extra></extra>")
    grade_fig.update_layout(showlegend=False, xaxis_title="Grade", yaxis_title="Patients")
    grade_fig = style_figure(grade_fig, 320)

    row1 = st.columns(2)
    with row1[0]:
        plot_card("6th stage distribution", "Patient counts across the primary stage field.", stage_fig, "Bar chart showing patient counts by 6th stage", "stage_sixth")
    with row1[1]:
        plot_card("T stage distribution", "Tumor classification counts.", t_fig, "Bar chart showing patient counts by T stage", "stage_t")
    row2 = st.columns(2)
    with row2[0]:
        plot_card("N stage distribution", "Node classification counts.", n_fig, "Bar chart showing patient counts by N stage", "stage_n")
    with row2[1]:
        plot_card("Grade distribution", "Recorded grade values, including the anaplastic category.", grade_fig, "Bar chart showing patient counts by grade", "stage_grade")

    status_stage = pd.crosstab(df["6th Stage"], df["Status"]).reindex(STAGE_ORDER, fill_value=0).reset_index().melt(id_vars="6th Stage", var_name="Status", value_name="Patients")
    status_stage_fig = px.bar(status_stage, x="6th Stage", y="Patients", color="Status", barmode="group", text="Patients", color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    status_stage_fig.update_traces(texttemplate="%{text:,}", textposition="outside", hovertemplate="Stage %{x}<br>%{fullData.name}: %{y:,} patients<extra></extra>")
    status_stage_fig.update_layout(xaxis_title="6th stage", yaxis_title="Patients")
    status_stage_fig = style_figure(status_stage_fig, 340)

    age_stage_fig = px.box(df, x="6th Stage", y="Age", color="Status", category_orders={"6th Stage": STAGE_ORDER}, color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    age_stage_fig.update_layout(xaxis_title="6th stage", yaxis_title="Age (years)")
    age_stage_fig = style_figure(age_stage_fig, 340)
    row3 = st.columns(2)
    with row3[0]:
        plot_card("Cancer stage vs patient status", "Grouped counts show how status is distributed within each stage.", status_stage_fig, "Grouped bar chart of patient status by cancer stage", "stage_status")
    with row3[1]:
        plot_card("Age vs 6th stage", "Age spread by stage, colored by recorded status.", age_stage_fig, "Box plot of age by cancer stage and status", "stage_age")


def render_survival(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Survival analysis", "Explore observed survival months across stage, age, hormone status, and patient status.", df, total)
    if df.empty:
        empty_state()
        return

    avg_survival = df["Survival Months"].mean()
    median_survival = df["Survival Months"].median()
    longest_stage = df.groupby("6th Stage")["Survival Months"].mean().idxmax()
    st.markdown(f'<div class="callout">Average observed survival is <strong>{avg_survival:.1f} months</strong> and median observed survival is <strong>{median_survival:.1f} months</strong>. Stage <strong>{longest_stage}</strong> has the highest average in the current view. These are descriptive associations, not causal or clinical conclusions.</div>', unsafe_allow_html=True)

    stage_survival = df.groupby("6th Stage", observed=False).agg( Average_Survival=("Survival Months", "mean"), Patients=("Survival Months", "size") ).reindex(STAGE_ORDER).dropna().reset_index()
    avg_stage_fig = px.bar(stage_survival, x="6th Stage", y="Average_Survival", text="Average_Survival", color="Average_Survival", color_continuous_scale=[COLORS["soft_blue"], COLORS["blue"], COLORS["cyan"]], hover_data={"Patients": True, "Average_Survival": ":.1f"})
    avg_stage_fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside", hovertemplate="Stage %{x}<br>Average %{y:.1f} months<br>Patients %{customdata[0]:,}<extra></extra>")
    avg_stage_fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="6th stage", yaxis_title="Average survival (months)")
    avg_stage_fig = style_figure(avg_stage_fig, 330)

    survival_dist_fig = px.histogram(df, x="Survival Months", color="Status", nbins=24, marginal="box", barmode="overlay", opacity=0.78, color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    survival_dist_fig.update_layout(xaxis_title="Survival months", yaxis_title="Patients")
    survival_dist_fig = style_figure(survival_dist_fig, 330)
    col1, col2 = st.columns(2)
    with col1:
        plot_card("Average survival by 6th stage", "Average and patient count are shown on hover.", avg_stage_fig, "Bar chart of average survival months by cancer stage", "survival_stage_average")
    with col2:
        plot_card("Survival months distribution", "Observed survival months split by status.", survival_dist_fig, "Histogram of survival months split by patient status", "survival_distribution")

    age_survival = df.groupby("Age Group", observed=False).agg(Average_Survival=("Survival Months", "mean"), Patients=("Survival Months", "size")).reindex(AGE_GROUP_ORDER).dropna().reset_index()
    age_survival_fig = px.bar(age_survival, x="Age Group", y="Average_Survival", text="Average_Survival", color="Age Group", color_discrete_sequence=[COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["coral"]], hover_data={"Patients": True})
    age_survival_fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside", hovertemplate="%{x}<br>Average %{y:.1f} months<br>Patients %{customdata[0]:,}<extra></extra>")
    age_survival_fig.update_layout(showlegend=False, xaxis_title="Age group", yaxis_title="Average survival (months)")
    age_survival_fig = style_figure(age_survival_fig, 320)

    estrogen_survival = df.groupby("Estrogen Status").agg(Average_Survival=("Survival Months", "mean"), Patients=("Survival Months", "size")).reset_index()
    estrogen_fig = px.bar(estrogen_survival, x="Estrogen Status", y="Average_Survival", text="Average_Survival", color="Estrogen Status", color_discrete_map={"Positive": COLORS["teal"], "Negative": COLORS["purple"]}, hover_data={"Patients": True})
    estrogen_fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside", hovertemplate="%{x}<br>Average %{y:.1f} months<br>Patients %{customdata[0]:,}<extra></extra>")
    estrogen_fig.update_layout(showlegend=False, xaxis_title="Estrogen status", yaxis_title="Average survival (months)")
    estrogen_fig = style_figure(estrogen_fig, 320)

    progesterone_survival = df.groupby("Progesterone Status").agg(Average_Survival=("Survival Months", "mean"), Patients=("Survival Months", "size")).reset_index()
    progesterone_fig = px.bar(progesterone_survival, x="Progesterone Status", y="Average_Survival", text="Average_Survival", color="Progesterone Status", color_discrete_map={"Positive": COLORS["blue"], "Negative": COLORS["coral"]}, hover_data={"Patients": True})
    progesterone_fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside", hovertemplate="%{x}<br>Average %{y:.1f} months<br>Patients %{customdata[0]:,}<extra></extra>")
    progesterone_fig.update_layout(showlegend=False, xaxis_title="Progesterone status", yaxis_title="Average survival (months)")
    progesterone_fig = style_figure(progesterone_fig, 320)

    row2 = st.columns(3)
    with row2[0]:
        plot_card("Survival by age group", "Average observed survival with patient count on hover.", age_survival_fig, "Bar chart of average survival by age group", "survival_age_group")
    with row2[1]:
        plot_card("Survival by estrogen status", "Descriptive comparison of average survival.", estrogen_fig, "Bar chart of average survival by estrogen status", "survival_estrogen")
    with row2[2]:
        plot_card("Survival by progesterone status", "Descriptive comparison of average survival.", progesterone_fig, "Bar chart of average survival by progesterone status", "survival_progesterone")

    stage_box_fig = px.box(df, x="6th Stage", y="Survival Months", color="Status", category_orders={"6th Stage": STAGE_ORDER}, color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    stage_box_fig.update_layout(xaxis_title="6th stage", yaxis_title="Survival months")
    stage_box_fig = style_figure(stage_box_fig, 350)
    plot_card("Survival comparison across cancer stages", "Distribution and spread of observed survival months.", stage_box_fig, "Box plot of survival months by cancer stage and status", "survival_stage_box")


def render_tumor_nodes(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Tumor & lymph node analysis", "Review tumor size and regional lymph node measures without removing valid medical outliers.", df, total)
    if df.empty:
        empty_state()
        return

    st.markdown('<div class="callout">The source values are displayed as recorded. No outlier removal or clinical interpretation is applied; use the hover details and filters to inspect the distributions responsibly.</div>', unsafe_allow_html=True)
    tumor_fig = px.histogram(df, x="Tumor Size", color="Status", nbins=24, marginal="box", barmode="overlay", opacity=0.78, color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    tumor_fig.update_layout(xaxis_title="Tumor size", yaxis_title="Patients")
    tumor_fig = style_figure(tumor_fig, 330)

    scatter_fig = px.scatter(df, x="Tumor Size", y="Survival Months", color="Status", hover_data=["Age", "6th Stage", "Reginol Node Positive"], opacity=0.72, render_mode="svg", color_discrete_map={"Alive": COLORS["teal"], "Dead": COLORS["coral"]})
    scatter_fig.update_traces(marker=dict(size=7), hovertemplate="Tumor size %{x}<br>Survival %{y} months<br>Status %{fullData.name}<extra></extra>")
    scatter_fig.update_layout(xaxis_title="Tumor size", yaxis_title="Survival months")
    scatter_fig = style_figure(scatter_fig, 330)
    col1, col2 = st.columns(2)
    with col1:
        plot_card("Tumor size distribution", "Observed tumor size values, split by patient status.", tumor_fig, "Histogram of tumor size split by patient status", "tumor_distribution")
    with col2:
        plot_card("Tumor size vs survival months", "Each point is one patient record; hover for stage and node detail.", scatter_fig, "Scatter plot of tumor size versus survival months", "tumor_survival_scatter")

    examined_fig = px.histogram(df, x="Regional Node Examined", color="6th Stage", nbins=24, barmode="stack", color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]])
    examined_fig.update_layout(xaxis_title="Regional nodes examined", yaxis_title="Patients")
    examined_fig = style_figure(examined_fig, 330)
    positive_fig = px.histogram(df, x="Reginol Node Positive", color="6th Stage", nbins=22, barmode="stack", color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"], COLORS["pink"], COLORS["blue"]])
    positive_fig.update_layout(xaxis_title="Regional nodes positive", yaxis_title="Patients")
    positive_fig = style_figure(positive_fig, 330)
    col1, col2 = st.columns(2)
    with col1:
        plot_card("Regional nodes examined", "Distribution of examined regional node counts.", examined_fig, "Histogram of regional nodes examined by cancer stage", "nodes_examined")
    with col2:
        plot_card("Regional nodes positive", "Distribution of positive regional node counts.", positive_fig, "Histogram of regional nodes positive by cancer stage", "nodes_positive")

    node_stage = df.groupby("6th Stage", observed=False).agg(Examined=("Regional Node Examined", "mean"), Positive=("Reginol Node Positive", "mean"), Positivity_Rate=("Node Positivity Rate", "mean"), Patients=("Status", "size")).reindex(STAGE_ORDER).dropna(how="all").reset_index()
    node_long = node_stage.melt(id_vars=["6th Stage", "Patients"], value_vars=["Examined", "Positive"], var_name="Measure", value_name="Average Nodes")
    node_fig = px.bar(node_long, x="6th Stage", y="Average Nodes", color="Measure", barmode="group", text="Average Nodes", color_discrete_map={"Examined": COLORS["blue"], "Positive": COLORS["coral"]}, hover_data={"Patients": True})
    node_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", hovertemplate="Stage %{x}<br>%{fullData.name}: %{y:.1f}<br>Patients %{customdata[0]:,}<extra></extra>")
    node_fig.update_layout(xaxis_title="6th stage", yaxis_title="Average node count")
    node_fig = style_figure(node_fig, 350)
    plot_card("Lymph node analysis by cancer stage", "Average nodes examined and positive; patient count is available on hover.", node_fig, "Grouped bar chart of average lymph node measures by cancer stage", "nodes_by_stage")


def numeric_feature_labels() -> dict[str, str]:
    return {
        "Age": "Age",
        "Tumor Size": "Tumor size",
        "Regional Node Examined": "Regional nodes examined",
        "Reginol Node Positive": "Regional nodes positive",
        "Survival Months": "Survival months",
        "Grade Numeric": "Grade (numeric mapping)",
    }


def render_correlation(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Correlation analysis", "Inspect numerical relationships in the selected population; correlation is descriptive, not causal.", df, total)
    if df.empty:
        empty_state()
        return

    labels = numeric_feature_labels()
    options = list(labels.keys())
    selected = st.multiselect("Numerical features", options, key="corr_features", format_func=lambda value: labels[value])
    if len(selected) < 2:
        st.info("Select at least two numerical features to build the correlation matrix.", icon=":material/insights:")
        return

    corr = df[selected].corr(numeric_only=True)
    display_names = [labels.get(column, column) for column in selected]
    corr_display = corr.copy()
    corr_display.index = display_names
    corr_display.columns = display_names
    heatmap = go.Figure(go.Heatmap(z=corr_display.values, x=corr_display.columns, y=corr_display.index, zmin=-1, zmax=1, colorscale=[[0, COLORS["coral"]], [0.5, COLORS["card"]], [1, COLORS["cyan"]]], text=np.round(corr_display.values, 2), texttemplate="%{text}", hovertemplate="%{y} x %{x}<br>Correlation %{z:.3f}<extra></extra>"))
    heatmap.update_layout(xaxis_title="", yaxis_title="", coloraxis_colorbar=dict(title="r", thickness=12))
    heatmap = style_figure(heatmap, max(350, 55 * len(selected)))
    plot_card("Interactive correlation heatmap", "Blue indicates a positive relationship; coral indicates a negative relationship.", heatmap, "Correlation heatmap for selected numerical features", "corr_heatmap")

    if "Survival Months" in selected:
        target = corr["Survival Months"].drop(labels=["Survival Months"], errors="ignore").sort_values()
        rel = target.rename("Correlation").reset_index().rename(columns={"index": "Feature"})
        rel["Feature"] = rel["Feature"].map(labels)
        rel_fig = px.bar(rel, x="Correlation", y="Feature", orientation="h", color="Correlation", color_continuous_scale=[COLORS["coral"], COLORS["card"], COLORS["blue"]], range_color=[-1, 1], text="Correlation")
        rel_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", hovertemplate="%{y}<br>Correlation %{x:.3f}<extra></extra>")
        rel_fig.update_layout(coloraxis_showscale=False, xaxis_title="Pearson correlation with survival months", yaxis_title="")
        rel_fig = style_figure(rel_fig, 330)
        positive = target[target > 0]
        negative = target[target < 0]
        positive_text = f"{labels[positive.idxmax()]} ({positive.max():.3f})" if not positive.empty else "No positive relationships in selection"
        negative_text = f"{labels[negative.idxmin()]} ({negative.min():.3f})" if not negative.empty else "No negative relationships in selection"
        col1, col2 = st.columns(2)
        with col1:
            plot_card("Correlation with survival months", "Pearson relationships for the selected numerical features.", rel_fig, "Bar chart of selected feature correlations with survival months", "corr_survival")
        with col2:
            st.markdown('<div class="section-kicker">Relationship summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-grid" style="grid-template-columns:1fr; margin-top:0;"><div class="insight-card"><div class="icon" style="background:{COLORS["soft_blue"]}; color:{COLORS["blue"]};">{material_icon("trending_up")}</div><h4>Strongest positive relationship</h4><p>{positive_text}</p></div><div class="insight-card"><div class="icon" style="background:{COLORS["soft_coral"]}; color:{COLORS["coral"]};">{material_icon("trending_down")}</div><h4>Strongest negative relationship</h4><p>{negative_text}</p></div></div>', unsafe_allow_html=True)
    else:
        st.info("Include Survival Months in the feature selection to see positive and negative relationships with survival.", icon=":material/add_chart:")


def render_data_explorer(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Data explorer", "Search, inspect, and download the exact source columns after applying the global filters.", df, total)
    query = st.text_input("Search patient records", key="patient_search", placeholder="Search by any value, such as IIA, Alive, White, or 54…", icon=":material/search:")
    display_df = df.copy()
    if query.strip():
        search_mask = display_df[ORIGINAL_COLUMNS].astype(str).apply(lambda column: column.str.contains(query.strip(), case=False, na=False)).any(axis=1)
        display_df = display_df.loc[search_mask].copy()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Records in table", f"{len(display_df):,}", border=True)
    with col2:
        st.download_button("Download table CSV", data=display_df[ORIGINAL_COLUMNS].to_csv(index=False).encode("utf-8"), file_name="Breast_Cancer_table.csv", mime="text/csv", icon=":material/download:", width="stretch")

    if display_df.empty:
        empty_state("No rows match the active filters and search term.")
        return

    visible = display_df[ORIGINAL_COLUMNS].copy()
    column_config = {
        "Age": st.column_config.NumberColumn("Age", format="%d"),
        "Tumor Size": st.column_config.NumberColumn("Tumor size", format="%d"),
        "Regional Node Examined": st.column_config.NumberColumn("Regional nodes examined", format="%d"),
        "Reginol Node Positive": st.column_config.NumberColumn("Regional nodes positive", format="%d"),
        "Survival Months": st.column_config.NumberColumn("Survival months", format="%d"),
    }
    st.dataframe(visible, column_config=column_config, hide_index=True, height=560, width="stretch", alt="Searchable breast cancer patient data table")
    st.caption(f"Showing {len(visible):,} rows. Original columns and source values are retained; the dashboard does not deduplicate or remove outliers.")


def render_insights(df: pd.DataFrame, total: int) -> None:
    add_page_heading("Insights & recommendations", "A plain-language summary generated from the currently selected patient records.", df, total)
    if df.empty:
        empty_state()
        return

    status_pct = df["Status"].value_counts(normalize=True).mul(100).reindex(STATUS_ORDER, fill_value=0)
    stage_stats = df.groupby("6th Stage", observed=False).agg(Patients=("Status", "size"), Average_Survival=("Survival Months", "mean"), Dead_Proportion=("Status", lambda values: values.eq("Dead").mean())).reindex(STAGE_ORDER).dropna(subset=["Patients"])
    most_common_stage = stage_stats["Patients"].idxmax()
    highest_survival_stage = stage_stats["Average_Survival"].idxmax()
    lowest_survival_stage = stage_stats["Average_Survival"].idxmin()
    highest_dead_stage = stage_stats["Dead_Proportion"].idxmax()
    most_common_age = int(df["Age"].mode().iloc[0])

    items = [
        ("groups", COLORS["soft_blue"], "Patients in view", f"{len(df):,} records are included after the active filters."),
        ("favorite", COLORS["soft_teal"], "Status mix", f"{status_pct['Alive']:.1f}% alive and {status_pct['Dead']:.1f}% dead in the selected records."),
        ("account_tree", COLORS["soft_purple"], "Most common stage", f"Stage {most_common_stage} has the largest patient count ({int(stage_stats.loc[most_common_stage, 'Patients']):,})."),
        ("person", COLORS["soft_blue"], "Most common age", f"Age {most_common_age} is the mode of the selected age values."),
        ("trending_up", COLORS["soft_teal"], "Highest average survival", f"Stage {highest_survival_stage} has the highest observed average at {stage_stats.loc[highest_survival_stage, 'Average_Survival']:.1f} months."),
        ("trending_down", COLORS["soft_coral"], "Lowest average survival", f"Stage {lowest_survival_stage} has the lowest observed average at {stage_stats.loc[lowest_survival_stage, 'Average_Survival']:.1f} months."),
        ("warning", COLORS["soft_coral"], "Highest dead proportion", f"Stage {highest_dead_stage} has the highest observed dead-patient proportion at {stage_stats.loc[highest_dead_stage, 'Dead_Proportion'] * 100:.1f}%."),
    ]
    cards = [f'<div class="insight-card"><div class="icon" style="background:{bg}; color:{COLORS["text"]};">{material_icon(icon)}</div><h4>{title}</h4><p>{body}</p></div>' for icon, bg, title, body in items]
    st.markdown(f'<div class="insight-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    numeric = df[["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months", "Grade Numeric"]].corr(numeric_only=True)["Survival Months"].drop("Survival Months").sort_values()
    strongest_negative = numeric.index[0]
    strongest_positive = numeric.index[-1]
    labels = numeric_feature_labels()
    st.markdown('<div class="section-kicker">Important numerical relationships</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout">Among the selected numerical features, <strong>{labels[strongest_positive]}</strong> has the strongest positive correlation with survival months ({numeric.iloc[-1]:.3f}), while <strong>{labels[strongest_negative]}</strong> has the strongest negative correlation ({numeric.iloc[0]:.3f}). These relationships describe this dataset and should not be treated as medical advice or causal evidence.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Analytical recommendations</div>', unsafe_allow_html=True)
    recommendations = [
        ("Use stage and status together", "Compare stage-level counts with status proportions so that population size and outcome mix are visible at the same time."),
        ("Inspect subgroup context", "Use the global filters to compare age, race, grade, and hormone-status subsets while keeping the same chart definitions."),
        ("Treat correlations as screening signals", "Use the Correlation Analysis page to identify relationships worth investigating further, not to infer treatment effects or individual prognosis."),
    ]
    rec_cards = [f'<div class="insight-card"><h4>{title}</h4><p>{body}</p></div>' for title, body in recommendations]
    st.markdown(f'<div class="insight-grid">{"".join(rec_cards)}</div>', unsafe_allow_html=True)


def render_page(page: str, df: pd.DataFrame, total: int) -> None:
    if page == "Overview":
        render_overview(df, total)
    elif page == "Patient Demographics":
        render_demographics(df, total)
    elif page == "Cancer Stage Analysis":
        render_stage_analysis(df, total)
    elif page == "Survival Analysis":
        render_survival(df, total)
    elif page == "Tumor & Lymph Node Analysis":
        render_tumor_nodes(df, total)
    elif page == "Correlation Analysis":
        render_correlation(df, total)
    elif page == "Data Explorer":
        render_data_explorer(df, total)
    elif page == "Insights & Recommendations":
        render_insights(df, total)


# -----------------------------------------------------------------------------
# App entry point
# -----------------------------------------------------------------------------
inject_styles()
try:
    data = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

filtered_data, context = sidebar_filters(data)
render_page(context["page"], filtered_data, len(data))
st.markdown('<div class="footer-note">CareLens Analytics · descriptive analysis of the provided source dataset</div>', unsafe_allow_html=True)
