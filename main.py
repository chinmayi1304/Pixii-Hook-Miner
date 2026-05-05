"""
Pixii Hook Mining Engine
========================
A production-ready Streamlit app that uses Tavily batch search and
Google Gemini to mine viral hook architectures for any niche.

Architecture:
  - Tavily Python SDK  → multi-query batch search for high-density signals
  - google-genai        → Gemini 2.5 Flash for pattern extraction & content gen
  - Streamlit           → dark-mode dashboard UI with live status animation

Environment variables required:
  GEMINI_API_KEY   — Google AI Studio key
  TAVILY_API_KEY   — Tavily search key
"""

import os
import re
import time
import textwrap
import traceback
from datetime import datetime

import streamlit as st
from tavily import TavilyClient
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pixii Hook Mining Engine",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark "Clean Tech" aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global dark background ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: #0d0f14 !important;
        color: #e2e8f0 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111318 !important;
        border-right: 1px solid #1e2330;
    }
    [data-testid="stSidebar"] * { color: #c9d1e0 !important; }

    /* ── Main header ── */
    .pixii-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .pixii-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }
    .pixii-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Generic card wrapper ── */
    .card {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38bdf8;
        margin-bottom: 14px;
    }

    /* ── Hook cards (Viral Pattern Library) ── */
    .hook-card {
        background: #0f1923;
        border-left: 3px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .hook-number {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #38bdf8;
        margin-bottom: 4px;
    }

    /* ── Architecture cards ── */
    .arch-card {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .arch-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #818cf8;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* ── Post cards ── */
    .post-card {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
        font-size: 0.92rem;
        line-height: 1.65;
        color: #e2e8f0;
        white-space: pre-wrap;
    }
    .post-platform {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* ── Metric badges ── */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 22px;
        flex-wrap: wrap;
    }
    .metric-badge {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 8px;
        padding: 12px 20px;
        text-align: center;
        flex: 1;
        min-width: 100px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input {
        background-color: #111827 !important;
        border: 1px solid #1e2a3a !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Primary button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0369a1, #4338ca) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 2rem !important;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #1e2a3a !important;
        color: #38bdf8 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* ── Dividers ── */
    hr { border-color: #1e2330 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key validation — fail fast with a clear message
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    st.error(
        "⚠️  **Missing API keys.**  "
        "Please set `GEMINI_API_KEY` and `TAVILY_API_KEY` as environment "
        "variables and restart the app."
    )
    st.stop()

# Initialise the new google-genai client (2026 SDK)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Sidebar — mining intensity + download placeholder
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⛏️ Mining Controls")
    st.markdown("---")

    intensity = st.selectbox(
        "Mining Intensity",
        ["Standard", "Deep", "Industrial"],
        index=1,
        help="Standard: 3 queries | Deep: 6 queries | Industrial: 9 queries",
    )

    intensity_map = {"Standard": 3, "Deep": 6, "Industrial": 9}
    num_queries = intensity_map[intensity]

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**Pixii Hook Mining Engine** crawls high-density viral signal "
        "sources — newsletters, social case studies, trending roundups — "
        "then uses Gemini AI to distil hook architectures and generate "
        "ready-to-post content for **Pixii AI**."
    )
    st.markdown("---")
    st.caption(f"Intensity: **{intensity}** · {num_queries} search queries")

    download_slot = st.empty()   # populated with download button after mining

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pixii-header">
        <p class="pixii-title">⛏️ Pixii Hook Mining Engine</p>
        <p class="pixii-subtitle">
            Crawl 1,000+ viral signals &nbsp;→&nbsp;
            Extract hook architectures &nbsp;→&nbsp;
            Generate high-converting posts for Pixii AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Niche input row
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    niche = st.text_input(
        "Target Niche",
        placeholder='e.g. "B2B SaaS", "Wellness", "Creator Economy", "E-commerce"',
        label_visibility="collapsed",
    )
with col_btn:
    mine_btn = st.button("⛏️ Mine Hooks", use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Helpers — search query generation
# ---------------------------------------------------------------------------

def build_search_queries(niche: str, n: int) -> list[str]:
    """Return n high-density search queries targeting viral hook signals."""
    pool = [
        f"{niche} viral hooks weekly roundup 2026 social media trends",
        f"{niche} trending hook newsletter 2026 high engagement posts",
        f"{niche} social media case study viral content 2026",
        f"best performing {niche} content hooks LinkedIn Twitter 2026",
        f"{niche} content marketing viral patterns psychology 2026",
        f"top viral {niche} posts hook formulas copywriting 2026",
        f"{niche} audience engagement hook strategies creators 2026",
        f"viral {niche} content breakdown what worked 2026",
        f"{niche} social proof hook templates high conversion 2026",
    ]
    return pool[:n]


# ---------------------------------------------------------------------------
# Helpers — Tavily batch search
# ---------------------------------------------------------------------------

def run_batch_search(niche: str, n: int) -> list[dict]:
    """
    Run n Tavily searches and aggregate results.
    Non-fatal errors are surfaced as warnings so partial results are kept.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    queries = build_search_queries(niche, n)
    aggregated: list[dict] = []

    for query in queries:
        try:
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )
            aggregated.extend(resp.get("results", []))
        except Exception as exc:
            st.warning(f"Search node failed (continuing): {exc}")
            time.sleep(0.5)

    return aggregated


# ---------------------------------------------------------------------------
# Helpers — Gemini prompt construction
# ---------------------------------------------------------------------------

def build_gemini_prompt(niche: str, results: list[dict]) -> str:
    """Compile search snippets into a structured analysis prompt."""
    snippets: list[str] = []
    budget = 6000
    total = 0
    for item in results:
        snippet = f"SOURCE: {item.get('url', '')}\n{item.get('content', '')}"
        if total + len(snippet) > budget:
            break
        snippets.append(snippet)
        total += len(snippet)

    corpus = "\n\n---\n\n".join(snippets) if snippets else "(no data retrieved)"

    return textwrap.dedent(f"""
        You are an elite viral-content strategist and hook architect.
        You have been given a corpus of real web data about viral content
        and social-media trends in the **{niche}** niche for 2026.

        ## CORPUS OF VIRAL SIGNALS
        {corpus}

        ---

        Based ONLY on the corpus above, produce the following analysis.
        Use EXACTLY these section headers — no extra sections.

        ## HOOK ARCHITECTURES
        Identify 3 psychological "Hook Architectures" — the underlying WHY
        that makes content go viral in this niche. For each, include:
        - ARCHITECTURE NAME (3-5 words, bold)
        - PSYCHOLOGICAL PRINCIPLE: one sentence
        - EVIDENCE FROM CORPUS: one concrete example from the data

        ## VIRAL PATTERN LIBRARY
        List 5 specific, ready-to-use hook opening lines drawn from the
        patterns in the corpus. Each hook must:
        - Be a complete opening line (not a description)
        - Be specific to the {niche} niche
        - Use [brackets] for fill-in-the-blank variables
        Number them 1-5.

        ## PIXII AI POSTS
        Write 3 complete social-media posts for a brand called "Pixii AI" —
        an AI-powered creative tool. Label each post clearly with its
        platform name on its own line. Rules:
        - LinkedIn post: open with one of the hooks above; 100-220 words;
          professional tone; hashtags at end
        - Twitter/X post: open with a hook; max 280 characters; punchy;
          1-2 hashtags
        - Instagram post: open with a hook; 150-200 words; conversational;
          hashtags at end
        All posts must feel authentic, NOT salesy.

        Write ONLY the analysis. No preamble, no closing summary.
    """).strip()


# ---------------------------------------------------------------------------
# Helpers — parse Gemini response into named sections
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> dict[str, str]:
    """Split Gemini output into three named sections."""
    markers = {
        "architectures": "## HOOK ARCHITECTURES",
        "patterns":      "## VIRAL PATTERN LIBRARY",
        "posts":         "## PIXII AI POSTS",
    }
    positions: dict[str, int] = {}
    for key, marker in markers.items():
        idx = text.find(marker)
        if idx != -1:
            positions[key] = idx

    ordered = sorted(positions, key=lambda k: positions[k])
    sections: dict[str, str] = {}
    for i, key in enumerate(ordered):
        start = positions[key] + len(markers[key])
        end = positions[ordered[i + 1]] if i + 1 < len(ordered) else len(text)
        sections[key] = text[start:end].strip()

    return sections


# ---------------------------------------------------------------------------
# Helpers — UI renderers
# ---------------------------------------------------------------------------

def render_arch_cards(text: str) -> None:
    """Render each Hook Architecture as a styled card."""
    blocks = re.split(r"\n(?=\*\*[A-Z]|\d+[\.\)])", text.strip())
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        st.markdown(f'<div class="arch-card">{text}</div>', unsafe_allow_html=True)
        return

    for block in blocks:
        lines = block.split("\n")
        title = re.sub(r"\*+", "", lines[0]).strip(" :-")
        body = "<br>".join(l.strip() for l in lines[1:] if l.strip())
        st.markdown(
            f"""
            <div class="arch-card">
                <div class="arch-label">Architecture</div>
                <strong style="color:#e2e8f0;">{title}</strong>
                <div style="color:#94a3b8;font-size:0.87rem;
                            margin-top:8px;line-height:1.6;">
                    {body}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hook_cards(text: str) -> None:
    """Render each numbered hook as a styled card."""
    items = re.split(r"\n(?=\d+[\.\)])", text.strip())
    items = [i.strip() for i in items if i.strip()]

    if len(items) <= 1:
        st.markdown(f'<div class="hook-card">{text}</div>', unsafe_allow_html=True)
        return

    for idx, item in enumerate(items, 1):
        clean = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
        st.markdown(
            f"""
            <div class="hook-card">
                <div class="hook-number">HOOK {idx:02d}</div>
                {clean}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_post_cards(text: str) -> None:
    """Render the three platform posts as styled cards."""
    platform_cfg = [
        ("linkedin",  "LinkedIn",   "#0a66c2", "💼"),
        ("twitter",   "Twitter / X","#1d9bf0", "🐦"),
        ("instagram", "Instagram",  "#e1306c", "📸"),
    ]

    lower = text.lower()
    split_pts: list[tuple[int, str]] = []
    for key, *_ in platform_cfg:
        idx = lower.find(key)
        if idx != -1:
            split_pts.append((idx, key))
    split_pts.sort()

    if len(split_pts) < 2:
        st.markdown(f'<div class="post-card">{text}</div>', unsafe_allow_html=True)
        return

    cfg_map = {k: (label, color, icon) for k, label, color, icon in platform_cfg}

    for i, (start, pkey) in enumerate(split_pts):
        end = split_pts[i + 1][0] if i + 1 < len(split_pts) else len(text)
        segment = text[start:end].strip()

        # Strip the platform header line(s)
        seg_lines = segment.split("\n")
        body_lines = [
            l for l in seg_lines
            if not re.search(
                r"^(linkedin|twitter|instagram|##\s*(linkedin|twitter|instagram))",
                l.strip(), re.IGNORECASE
            )
        ]
        body = "<br>".join(l for l in body_lines if l.strip())

        label, color, icon = cfg_map.get(pkey, ("Post", "#38bdf8", "📣"))
        st.markdown(
            f"""
            <div class="post-card">
                <div class="post-platform" style="color:{color};">
                    {icon} {label}
                </div>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main pipeline — triggered by button click
# ---------------------------------------------------------------------------

if mine_btn:
    niche = niche.strip()
    if not niche:
        st.error("Please enter a niche before mining.")
        st.stop()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_parts: list[str] = []

    try:
        # ── Live crawl animation ─────────────────────────────────────────
        with st.status("🔍 Initialising Hook Mining Engine...", expanded=True) as status:
            st.write("🌐 Connecting to high-density signal nodes...")
            time.sleep(0.3)

            st.write(f"🔎 Querying **{num_queries}** viral signal sources for **{niche}**...")
            search_results = run_batch_search(niche, num_queries)

            if not search_results:
                status.update(label="❌ No results returned.", state="error")
                st.error(
                    "Tavily returned no results. "
                    "Please check your API key or try a different niche."
                )
                st.stop()

            st.write(f"📡 Aggregating **{len(search_results)}** viral signals...")
            time.sleep(0.2)
            st.write("🧹 Normalising signal corpus...")
            time.sleep(0.2)

            st.write("🤖 Gemini analysing psychological hook patterns...")
            prompt = build_gemini_prompt(niche, search_results)
            response = gemini_client.models.generate_content(
                model="gemini-3-0-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.75,
                    max_output_tokens=3000,
                ),
            )
            gemini_text = response.text.strip()

            st.write("📐 Extracting hook architectures...")
            time.sleep(0.15)
            st.write("✍️  Generating Pixii AI posts...")
            time.sleep(0.15)

            sections = parse_sections(gemini_text)
            status.update(label="✅ Mining complete — hook data ready.", state="complete")

        # ── Metric row ───────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-badge">
                    <div class="metric-value">{len(search_results)}</div>
                    <div class="metric-label">Signals Crawled</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-value">{num_queries}</div>
                    <div class="metric-label">Search Nodes</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-value">3</div>
                    <div class="metric-label">Architectures</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-value">5</div>
                    <div class="metric-label">Hook Templates</div>
                </div>
                <div class="metric-badge">
                    <div class="metric-value">3</div>
                    <div class="metric-label">Pixii Posts</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Two-column: Architectures + Patterns ─────────────────────────
        col_arch, col_hooks = st.columns(2)

        arch_text = sections.get("architectures", gemini_text)
        pattern_text = sections.get("patterns", "")

        with col_arch:
            st.markdown(
                '<div class="card-title">🧠 Hook Architectures</div>',
                unsafe_allow_html=True,
            )
            render_arch_cards(arch_text)

        with col_hooks:
            st.markdown(
                '<div class="card-title">📚 Viral Pattern Library</div>',
                unsafe_allow_html=True,
            )
            render_hook_cards(pattern_text)

        report_parts += [
            f"PIXII HOOK MINING ENGINE — REPORT",
            f"Generated : {timestamp}",
            f"Niche     : {niche}",
            f"Intensity : {intensity} ({num_queries} queries)",
            f"Signals   : {len(search_results)}",
            "=" * 60,
            "",
            "## HOOK ARCHITECTURES",
            arch_text,
            "",
            "## VIRAL PATTERN LIBRARY",
            pattern_text,
        ]

        st.markdown("---")

        # ── Full-width: Pixii AI Posts ───────────────────────────────────
        st.markdown(
            '<div class="card-title">✨ Pixii AI — Ready-to-Post Content</div>',
            unsafe_allow_html=True,
        )
        post_text = sections.get("posts", "")
        render_post_cards(post_text)

        report_parts += ["", "## PIXII AI POSTS", post_text]

        # ── Download button (sidebar) ────────────────────────────────────
        full_report = "\n".join(report_parts)
        safe_niche = niche.replace(" ", "_").lower()
        filename = f"pixii_hooks_{safe_niche}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

        download_slot.download_button(
            label="⬇️ Download Report",
            data=full_report.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
        )

    except Exception:
        st.error(
            "**Mining failed.** An unexpected error occurred:\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        st.info(
            "Tips: verify your API keys, check your network connection, "
            "or try a simpler niche query."
        )

# ---------------------------------------------------------------------------
# Idle state — shown when no mining has been triggered yet
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:48px 24px;">
            <div style="font-size:3rem;margin-bottom:12px;">⛏️</div>
            <div style="font-size:1.15rem;font-weight:700;
                        color:#e2e8f0;margin-bottom:8px;">
                Ready to mine viral hooks
            </div>
            <div style="color:#64748b;font-size:0.9rem;
                        max-width:480px;margin:0 auto;line-height:1.6;">
                Enter your niche above and click
                <strong style="color:#38bdf8;">Mine Hooks</strong>
                to crawl high-density viral signal sources, extract
                psychological hook architectures, and generate
                ready-to-post Pixii AI content.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🌐", "Batch Web Crawl",
         "Multi-query Tavily search across viral roundups, newsletters & case studies"),
        (c2, "🧠", "Hook Architecture",
         "Gemini AI extracts the psychological WHY behind trending content"),
        (c3, "✍️", "Pixii AI Posts",
         "3 platform-ready posts crafted from real viral patterns"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="card" style="text-align:center;padding:24px 16px;">
                    <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
                    <div style="font-weight:700;color:#e2e8f0;
                                margin-bottom:6px;">{title}</div>
                    <div style="color:#64748b;font-size:0.85rem;
                                line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
