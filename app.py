import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import mapclassify
import json

st.set_page_config(page_title="CA Fatal Collisions (2011-2022)", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {background-color: #1a1a1a;}

    /* Tighten all vertical gaps */
    [data-testid="stVerticalBlock"] {gap: 0.2rem;}
    [data-testid="column"] {padding: 0.15rem;}
    section.main > div {padding-top: 1rem;}

    /* Title */
    .main-title {color: #cccccc; font-size: 34px; font-weight: bold; margin-bottom: 0; padding-top: 0; white-space: nowrap;}
    .sub-title {color: #888888; font-size: 14px; margin-top: -4px; margin-bottom: 6px;}

    /* KPI cards */
    .kpi-box-tall {
        background: #2a2a2a; border-radius: 8px; padding: 20px; text-align: center;
        height: 420px; display:flex; flex-direction:column; justify-content:center;
    }
    .kpi-box {
        background: #2a2a2a; border-radius: 8px; padding: 10px; text-align: center;
        margin-bottom: 4px; height: 132px; display:flex; flex-direction:column; justify-content:center;
    }
    .kpi-label-big {font-size: 22px; font-weight: bold; margin-bottom: 5px;}
    .kpi-label {font-size: 20px; font-weight: bold; margin-bottom: 2px;}
    .kpi-value-big {font-size: 72px; font-weight: bold; line-height: 1.1;}
    .kpi-value {font-size: 60px; font-weight: bold; line-height: 1.1;}
    .kpi-sub {font-size: 15px; color: #999;}
    .red {color: #e8465f;}
    .green {color: #21ba45;}
    .yellow {color: #fbbd08;}
    .white {color: #ffffff;}

    /* Selectbox styling */
    div[data-testid="stSelectbox"] label {color: white !important; font-size: 12px !important;}
    div[data-testid="stSelectbox"] {margin-bottom: -8px; margin-top: -8px;}

    /* Hide folium iframe border & tighten */
    iframe {border: none !important; border-radius: 6px;}

    /* Plotly chart containers */
    [data-testid="stPlotlyChart"] {margin-top: -10px;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {background-color: #1e1e1e;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {color: #ccc; font-size: 13px;}
    [data-testid="stSidebar"] a {color: #4da6ff !important;}
    [data-testid="stSidebar"] h3 {color: #fff; font-size: 16px; margin-top: 16px;}
    [data-testid="stSidebar"] hr {border-color: #444;}
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.markdown('<p class="main-title">California Fatal Collision Overview (2011–2022)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Pedestrian, Cyclist, and Other Traffic</p>', unsafe_allow_html=True)

# ── Sidebar: About & Sources ──
with st.sidebar:
    st.markdown("### ℹ️  About This Dashboard")
    st.markdown("""
This dashboard visualizes **fatal traffic collisions** across California's 58 counties
from 2011 to 2022, covering pedestrian, cyclist, and other traffic fatalities.
It helps identify where fatal crashes concentrate and how they break down by victim
type and location characteristics.
    """)

    st.markdown("---")

    st.markdown("### 🗺️  How Map Colors Work")
    st.markdown("""
Counties are grouped into 5 tiers (Low to Critical) based on fatality count. Tiers are based on patterns in the data. Each tier is created to best group counties with similar counts together.

**Colors update automatically** when you switch the victim role filter.
    """)

    st.markdown("---")

    st.markdown("### 📚  Data Sources")
    st.markdown("""
**Traffic collision data:** SWITRS (Statewide Integrated Traffic Records System) via
[TIMS — Transportation Injury Mapping System](https://tims.berkeley.edu/),
Safe Transportation Research and Education Center (SafeTREC), UC Berkeley.

**Place typology:** Based on
[*"Quantifying the Sustainability, Livability, and Equity Performance of Urban and
Suburban Places in California"*](https://journals.sagepub.com/doi/abs/10.1177/0361198118791382)
(Frost, A.R. et al., 2018, *Transportation Research Record*, 2672(3), 130–144).
    """)

# ── Load pre-aggregated data (tiny files, not 3.5M rows) ──
@st.cache_data
def load_kpis():
    with open("aggregated/kpis.json") as f:
        return json.load(f)

@st.cache_data
def load_county_stats():
    return pd.read_parquet("aggregated/county_stats.parquet")

@st.cache_data
def load_role_county():
    return pd.read_parquet("aggregated/role_county.parquet")

@st.cache_data
def load_role_counts():
    return pd.read_parquet("aggregated/role_counts.parquet")

@st.cache_data
def load_int_by_year():
    return pd.read_parquet("aggregated/int_by_year.parquet")

@st.cache_data
def load_rtor():
    return pd.read_parquet("aggregated/rtor_by_placetype.parquet")

@st.cache_data
def load_counties():
    counties = gpd.read_file("ca_counties/CA_Counties.shp")
    counties = counties.to_crs(epsg=4326)
    counties["geometry"] = counties["geometry"].simplify(0.003)
    counties["NAME_UPPER"] = counties["NAME"].str.upper()
    return counties

kpis = load_kpis()
county_stats = load_county_stats()
role_county = load_role_county()
role_counts = load_role_counts()
int_by_year = load_int_by_year()
rtor_counts = load_rtor()
counties = load_counties()

# ── KPIs ──
total_killed = kpis["total_killed"]
total_rows = kpis["total_rows"]
ped_killed = kpis["ped_killed"]
cyc_killed = kpis["cyc_killed"]
other_killed = kpis["other_killed"]

# ── Prepare merged county geodata ──
pie_colors = {"Driver": "#b07cc6", "Passenger": "#c75b7a", "Pedestrian": "#d4845e", "Bicyclist": "#5b9ec9", "Other": "#888888"}

color_map = {
    "Urban Centers": "#5a4e3a",
    "Urban Places": "#c0522e",
    "Compact Suburban Places": "#d4882b",
    "Suburban Places": "#c9a84c",
    "Rural Places": "#7ba05b",
    "Employment Centers": "#6b8a3a",
    "Special Districts": "#d1d1d1",
}

counties_merged = counties.merge(county_stats, left_on="NAME_UPPER", right_on="COUNTY", how="left").fillna(0)


def fmt_k(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


# ════════════════════════════════════════════════════════
# ROW 1 — 4 columns
# ════════════════════════════════════════════════════════
r1c1, r1c2, r1c3, r1c4 = st.columns([0.8, 2.5, 0.8, 1.2])

with r1c1:
    st.markdown(
        f"""
    <div class="kpi-box-tall">
        <p class="kpi-label-big red">Overall Fatal Collisions</p>
        <p class="kpi-value-big red">💓 {fmt_k(total_killed)}</p>
        <p class="kpi-sub">From {fmt_k(total_rows)} casualties</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with r1c2:
    selected_role = st.selectbox(
        "Filter map by victim role:",
        ["All Fatalities", "Driver", "Passenger", "Pedestrian", "Bicyclist", "Other"],
        key="role_filter",
    )
    if selected_role == "All Fatalities":
        selected_role = None

    if selected_role:
        role_data = role_county[role_county["Role"] == selected_role]
        role_data_county = role_data.rename(columns={"fatalities": "display_fatalities"})
        map_data = counties.merge(role_data_county, left_on="NAME_UPPER", right_on="COUNTY", how="left")
        map_data["display_fatalities"] = map_data["display_fatalities"].fillna(0)
        map_data = map_data.merge(
            county_stats[["COUNTY", "total_fatalities", "ped_killed", "cyc_killed", "other_killed", "Top_Crash_Zone"]],
            on="COUNTY",
            how="left",
        )
        map_data = map_data.fillna(0)
        map_data["Top_Crash_Zone"] = map_data["Top_Crash_Zone"].fillna("Unknown")
    else:
        map_data = counties_merged.copy()
        map_data["display_fatalities"] = map_data["total_fatalities"]

    max_val = max(map_data["display_fatalities"].max(), 1)
    current_vals_pos = map_data["display_fatalities"][map_data["display_fatalities"] > 0]
    if len(current_vals_pos) >= 5:
        jenks = mapclassify.NaturalBreaks(current_vals_pos, k=5)
        breaks = jenks.bins
        b1, b2, b3, b4 = breaks[0], breaks[1], breaks[2], breaks[3]
    elif len(current_vals_pos) > 1:
        b1 = np.percentile(current_vals_pos, 20)
        b2 = np.percentile(current_vals_pos, 40)
        b3 = np.percentile(current_vals_pos, 60)
        b4 = np.percentile(current_vals_pos, 80)
    else:
        b1, b2, b3, b4 = max_val * 0.2, max_val * 0.4, max_val * 0.6, max_val * 0.8

    top3 = map_data.nlargest(3, "display_fatalities")
    filter_label = selected_role if selected_role else "All Fatalities"

    if selected_role:
        map_data["tooltip_text"] = map_data.apply(
            lambda r: (
                f"<b style='font-size:15px'>{r['NAME']}</b><br><br>"
                f"<b>{selected_role} Fatalities: {int(r['display_fatalities']):,}</b><br>"
                f"<span style='color:#aaa'>Total Fatalities: {int(r['total_fatalities']):,}</span><br>"
                f"<span style='color:#aaa'>Top Crash Zone: {r.get('Top_Crash_Zone','Unknown')}</span>"
            ),
            axis=1,
        )
    else:
        map_data["tooltip_text"] = map_data.apply(
            lambda r: (
                f"<b style='font-size:15px'>{r['NAME']}</b><br><br>"
                f"<b>Total Fatalities: {int(r['total_fatalities']):,}</b><br>"
                f"Pedestrian: {int(r['ped_killed']):,}<br>"
                f"Cyclist: {int(r['cyc_killed']):,}<br>"
                f"Other: {int(r['other_killed']):,}<br><br>"
                f"<span style='color:#aaa'>Top Crash Zone: {r.get('Top_Crash_Zone','Unknown')}</span>"
            ),
            axis=1,
        )

    def style_function(feature):
        val = feature["properties"]["display_fatalities"]
        if val == 0:
            fill = "#1a1a1a"
        elif val <= b1:
            fill = "#6b7fb5"
        elif val <= b2:
            fill = "#3bb3a0"
        elif val <= b3:
            fill = "#e8c840"
        elif val <= b4:
            fill = "#ff9922"
        else:
            fill = "#e03020"
        return {"fillColor": fill, "color": "#666", "weight": 1.5, "fillOpacity": 0.5}

    m = folium.Map(location=[37.2, -119.5], zoom_start=6, tiles="CartoDB dark_matter")

    geojson_data = json.loads(map_data.to_json())
    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"],
            aliases=[""],
            style="background-color:#2a2a2a;color:white;font-size:13px;padding:12px;border-radius:6px;border:1px solid #555;",
        ),
        highlight_function=lambda x: {"weight": 3, "color": "white", "fillOpacity": 0.95},
    ).add_to(m)

    top3_rows = ""
    for idx, (_, row) in enumerate(top3.iterrows()):
        top3_rows += (
            f"<div style='display:flex;justify-content:space-between;margin:3px 0;'>"
            f"<span>{idx+1}. {row['NAME']}</span>"
            f"<span style='color:#e8465f;font-weight:bold;'>{int(row['display_fatalities']):,}</span></div>"
        )
    top3_html = f"""
    <div style="position:absolute;top:10px;left:50px;z-index:1000;">
    <details style="background:rgba(26,26,26,0.88);padding:10px 14px;border-radius:6px;border:1px solid #444;color:white;font-size:12px;min-width:200px;cursor:pointer;">
    <summary style="font-weight:bold;font-size:12px;">Top 3 Counties — {filter_label}</summary>
    <div style="margin-top:6px;">{top3_rows}</div>
    </details>
    </div>
    """
    m.get_root().html.add_child(folium.Element(top3_html))

    legend_html = f"""
    <div style="position:absolute;top:10px;right:10px;z-index:1000;">
    <details style="background:rgba(26,26,26,0.88);padding:10px 14px;border-radius:6px;border:1px solid #444;color:white;font-size:11px;cursor:pointer;">
    <summary style="font-weight:bold;font-size:12px;">Legend</summary>
    <div style="margin-top:6px;">
    <i style="background:#e03020;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;"></i>Critical (&gt; {int(b4):,})<br>
    <i style="background:#ff9922;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;"></i>Very High ({int(b3)+1:,}–{int(b4):,})<br>
    <i style="background:#e8c840;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;"></i>High ({int(b2)+1:,}–{int(b3):,})<br>
    <i style="background:#3bb3a0;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;"></i>Moderate ({int(b1)+1:,}–{int(b2):,})<br>
    <i style="background:#6b7fb5;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;"></i>Low (1–{int(b1):,})<br>
    <i style="background:#1a1a1a;width:12px;height:10px;display:inline-block;margin:2px 4px;border-radius:2px;border:1px solid #444;"></i>None (0)
    </div>
    </details>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=380, returned_objects=[])

with r1c3:
    st.markdown(
        f"""
    <div class="kpi-box">
        <p class="kpi-label red">Pedestrian Fatalities</p>
        <p class="kpi-value white">🚶 {fmt_k(ped_killed)}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div class="kpi-box">
        <p class="kpi-label green">Cyclist Fatalities</p>
        <p class="kpi-value white">🚴 {fmt_k(cyc_killed)}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div class="kpi-box">
        <p class="kpi-label yellow">Other Traffic Fatalities</p>
        <p class="kpi-value white">🚗 {fmt_k(other_killed)}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with r1c4:
    # Build legend HTML
    total_count = role_counts["Count"].sum()
    legend_items = ""
    for _, row in role_counts.iterrows():
        pct = row["Count"] / total_count * 100
        c = pie_colors.get(row["Role"], "#888")
        legend_items += (
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{c};flex-shrink:0;"></span>'
            f'<span style="color:#ccc;font-size:12px;">{row["Role"]} — {pct:.1f}%</span>'
            f'</div>'
        )

    fig_pie = go.Figure(
        go.Pie(
            labels=role_counts["Role"],
            values=role_counts["Count"],
            marker=dict(colors=[pie_colors.get(r, "#888") for r in role_counts["Role"]]),
            textinfo="none",
            hole=0.45,
            hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>",
            pull=[0.02] * len(role_counts),
        )
    )
    fig_pie.update_layout(
        title=dict(text="Intersection Level Fatalities<br>by Victim Roles", font=dict(color="#e8465f", size=13)),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="white"),
        showlegend=False,
        height=300,
        margin=dict(l=30, r=30, t=55, b=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f'<div style="padding:0 10px;">{legend_items}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ROW 2 — Line chart + Bar chart
# ════════════════════════════════════════════════════════
r2c1, r2c2 = st.columns(2)

with r2c1:
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=int_by_year["YEAR"],
            y=int_by_year["NUMBER_KILLED"],
            mode="lines+markers",
            line=dict(color="#e8465f", width=2),
            marker=dict(size=6, color="#e8465f"),
            fill="tozeroy",
            fillcolor="rgba(232,70,95,0.15)",
            hovertemplate="Year: %{x}<br>Fatalities: %{customdata:,}<extra></extra>",
            customdata=int_by_year["NUMBER_KILLED"],
        )
    )
    fig_line.update_layout(
        title=dict(text="Overall Fatality Counts at Intersection by Year", font=dict(color="white", size=14)),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, title="Years", dtick=1),
        yaxis=dict(showgrid=True, gridcolor="#333", title="Fatality Counts"),
        height=340,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    st.plotly_chart(fig_line, use_container_width=True)

with r2c2:
    rtor_sorted = rtor_counts.sort_values("Count", ascending=True)
    fig_bar = go.Figure(
        go.Bar(
            x=rtor_sorted["Count"],
            y=rtor_sorted["PlaceType"],
            orientation="h",
            marker_color="#f0ead6",
            text=rtor_sorted["Count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color="white", size=12),
            hoverinfo="none",
        )
    )
    fig_bar.update_layout(
        title=dict(text="RTOR Fatality Counts by Place Typology", font=dict(color="white", size=14)),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="#333", range=[0, rtor_sorted["Count"].max() * 1.15]),
        yaxis=dict(showgrid=False),
        height=340,
        margin=dict(l=10, r=70, t=50, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
