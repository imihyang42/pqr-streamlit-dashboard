from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics import (
    QUALITY_COLUMNS, analyze_code, candidate_factors, default_paths,
    eligible_codes, load_data, simulate_capa,
)


from case_review import verified_sources, find_cases, case_html, executive_html
import case_review as cr
import process_capa as pcap

BASE = Path(__file__).resolve().parent
ACCENT, GOOD, WARNING, CRITICAL, MUTED = "#0e6d6a", "#2f8f5b", "#b8801a", "#bb4436", "#7b8982"

FEATURE_LABELS = {
    "api_water": "API 수분", "api_total_impurities": "API 총불순물",
    "api_l_impurity": "API 불순물 L", "api_content": "API 함량",
    "api_ps01": "API 입도 10%", "api_ps05": "API 입도 50%", "api_ps09": "API 입도 90%",
    "lactose_water": "유당 수분", "lactose_sieve0045": "유당 0.045 mm 체분",
    "lactose_sieve015": "유당 0.15 mm 체분", "lactose_sieve025": "유당 0.25 mm 체분",
    "smcc_water": "SMCC 수분", "smcc_td": "SMCC 탭밀도", "smcc_bd": "SMCC 벌크밀도",
    "smcc_ps01": "SMCC 입도 10%", "smcc_ps05": "SMCC 입도 50%", "smcc_ps09": "SMCC 입도 90%",
    "starch_ph": "전분 pH", "starch_water": "전분 수분",
    "tbl_min_thickness": "정제 최소 두께", "tbl_max_thickness": "정제 최대 두께",
    "tbl_rsd_weight": "정제 중량 RSD", "tbl_min_hardness": "정제 최소 경도",
    "tbl_max_hardness": "정제 최대 경도", "tbl_av_hardness": "정제 평균 경도",
    "tbl_yield": "정제 수율", "dissolution_av": "용출 평균", "dissolution_min": "용출 최솟값",
    "resodual_solvent": "잔류용매", "impurities_total": "총불순물",
    "impurity_o": "불순물 O", "impurity_l": "불순물 L", "batch_yield": "배치 수율",
}


def feature_label(name):
    return FEATURE_LABELS.get(name, str(name).replace("_", " "))


# 후보변수별로 실제 확인해야 할 기록 종류를 구분한다.
# (예전에는 모든 변수에 "배치기록·시험 원자료·설비로그"라는 동일 문구를 씁는데,
#  변수 성격에 맞는 기록으로 구분해 CAPA 후보 표 자체가 정보를 담도록 한다.)
RECORD_HINTS = {
    # API(원료의약품)
    "api_water": "API 입고시험 원자료·수분(건조감량) 시험기록",
    "api_total_impurities": "API 불순물 시험 원자료·크로마토그램, 시험법 밸리데이션",
    "api_l_impurity": "API 불순물 L 시험 원자료·크로마토그램, 시험법 밸리데이션",
    "api_content": "API 함량 시험 원자료·정량법 계산기록",
    "api_ps01": "API 입도분석 원자료·레이저회절 장비 적격성",
    "api_ps05": "API 입도분석 원자료·레이저회절 장비 적격성",
    "api_ps09": "API 입도분석 원자료·레이저회절 장비 적격성",
    # 유당(부형제)
    "lactose_water": "유당 입고시험 원자료·수분(건조감량) 시험기록",
    "lactose_sieve0045": "유당 체분(입도) 시험 원자료·체분석 장비 기록",
    "lactose_sieve015": "유당 체분(입도) 시험 원자료·체분석 장비 기록",
    "lactose_sieve025": "유당 체분(입도) 시험 원자료·체분석 장비 기록",
    # SMCC(부형제)
    "smcc_water": "SMCC 입고시험 원자료·수분(건조감량) 시험기록",
    "smcc_td": "SMCC 분체물성 시험 원자료·탭밀도 측정기록",
    "smcc_bd": "SMCC 분체물성 시험 원자료·벌크밀도 측정기록",
    "smcc_ps01": "SMCC 입도분석 원자료·레이저회절 장비 적격성",
    "smcc_ps05": "SMCC 입도분석 원자료·레이저회절 장비 적격성",
    "smcc_ps09": "SMCC 입도분석 원자료·레이저회절 장비 적격성",
    # 전분(붕해제)
    "starch_ph": "전분 입고시험 원자료·pH 측정기록",
    "starch_water": "전분 입고시험 원자료·수분(건조감량) 시험기록",
    # 정제 물성 / 타정 공정
    "tbl_min_thickness": "타정 배치기록·정제 두께 측정 원자료(두께 측정기 적격성)",
    "tbl_max_thickness": "타정 배치기록·정제 두께 측정 원자료(두께 측정기 적격성)",
    "tbl_rsd_weight": "타정 배치기록·정제 중량편차 측정 원자료(중량선별기 기록)",
    "tbl_min_hardness": "타정 배치기록·정제 경도 측정 원자료(타정압 설정값 포함)",
    "tbl_max_hardness": "타정 배치기록·정제 경도 측정 원자료(타정압 설정값 포함)",
    "tbl_av_hardness": "타정 배치기록·정제 경도 측정 원자료(타정압 설정값 포함)",
    "tbl_yield": "배합·타정 수율 집계 기록·폐기량 기록",
    # 품질 시험 결과
    "dissolution_av": "용출 시험 원자료·용출시험기 적격성, 시료채취 기록",
    "dissolution_min": "용출 시험 원자료·용출시험기 적격성, 시료채취 기록",
    "resodual_solvent": "잔류용매 시험 원자료·GC 시험법 밸리데이션",
    "impurities_total": "총불순물 시험 원자료·크로마토그램, 시험법 밸리데이션",
    "impurity_o": "불순물 O 시험 원자료·크로마토그램, 시험법 밸리데이션",
    "impurity_l": "불순물 L 시험 원자료·크로마토그램, 시험법 밸리데이션",
    "batch_yield": "배치기록상 수율 계산·폐기량 기록",
}


def record_hint(name):
    # 목록에 없는 변수는 예전과 같은 일반 문구로 대체(안전한 기본값).
    return RECORD_HINTS.get(name, "배치기록·시험 원자료·설비로그")


# 후보변수별 실제 "조치사항"(무엇을 확인할지가 아니라 무엇을 조치할지).
# RECORD_HINTS가 "확인할 기록"이라면, 이건 그 기록 확인 이후 취할 조치 방향이다.
ACTION_HINTS = {
    # API(원료의약품) — 원료 규격·공급업체 쪽 조치
    "api_water": "API 공급업체 수분(건조감량) 규격·입고검사 기준 재검토",
    "api_total_impurities": "API 불순물 규격·공급업체 품질협약 재검토",
    "api_l_impurity": "API 불순물 L 규격·공급업체 품질협약 재검토",
    "api_content": "API 함량 규격·공급업체 품질협약 재검토",
    "api_ps01": "API 입도 규격·공급업체 관리기준 재검토",
    "api_ps05": "API 입도 규격·공급업체 관리기준 재검토",
    "api_ps09": "API 입도 규격·공급업체 관리기준 재검토",
    # 유당/SMCC/전분(부형제) — 부형제 규격·공급업체 쪽 조치
    "lactose_water": "유당 공급업체 수분 규격 재검토",
    "lactose_sieve0045": "유당 체분(입도) 규격·공급업체 관리기준 재검토",
    "lactose_sieve015": "유당 체분(입도) 규격·공급업체 관리기준 재검토",
    "lactose_sieve025": "유당 체분(입도) 규격·공급업체 관리기준 재검토",
    "smcc_water": "SMCC 공급업체 수분 규격 재검토",
    "smcc_td": "SMCC 분체물성(탭밀도) 규격·공급업체 관리기준 재검토",
    "smcc_bd": "SMCC 분체물성(벌크밀도) 규격·공급업체 관리기준 재검토",
    "smcc_ps01": "SMCC 입도 규격·공급업체 관리기준 재검토",
    "smcc_ps05": "SMCC 입도 규격·공급업체 관리기준 재검토",
    "smcc_ps09": "SMCC 입도 규격·공급업체 관리기준 재검토",
    "starch_ph": "전분 공급업체 pH 규격 재검토",
    "starch_water": "전분 공급업체 수분 규격 재검토",
    # 정제 물성/타정 공정 — 공정 파라미터 쪽 조치
    "tbl_min_thickness": "타정공정 압축력 설정값·관리한계 재검토",
    "tbl_max_thickness": "타정공정 압축력 설정값·관리한계 재검토",
    "tbl_rsd_weight": "타정공정 중량관리(충전량 균일성)·중량선별기 점검",
    "tbl_min_hardness": "타정공정 압축력·타정속도 설정값 재검토",
    "tbl_max_hardness": "타정공정 압축력·타정속도 설정값 재검토",
    "tbl_av_hardness": "타정공정 압축력·타정속도 설정값 재검토",
    "tbl_yield": "배합·타정 공정 수율 관리기준·폐기량 관리 절차 재검토",
    # 품질 시험 결과 — 시험방법 쪽 조치
    "dissolution_av": "용출시험법 재현성 확인·시험법 밸리데이션 재검토",
    "dissolution_min": "용출시험법 재현성 확인·시험법 밸리데이션 재검토",
    "resodual_solvent": "잔류용매 시험법(GC) 재현성·밸리데이션 재검토",
    "impurities_total": "총불순물 시험법 재현성·밸리데이션 재검토",
    "impurity_o": "불순물 O 시험법 재현성·밸리데이션 재검토",
    "impurity_l": "불순물 L 시험법 재현성·밸리데이션 재검토",
    "batch_yield": "배합·타정 공정 수율 관리기준·폐기량 관리 절차 재검토",
}


def action_hint(name):
    return ACTION_HINTS.get(name, "관련 공정·원료 기준 문서화 및 재검토 필요")


def display_value(value, digits=3):
    return "-" if pd.isna(value) else f"{float(value):.{digits}f}"

st.set_page_config(page_title="정제 OOS 조사·CAPA 대시보드", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.35rem;max-width:1500px}.small{font-size:.82rem;color:#64736c}div[data-testid="stSidebarNav"]{display:none}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid #d8e0dc}.stTabs [data-baseweb="tab"]{color:#65756d;background:#f4f7f5;border-radius:7px 7px 0 0;padding:.55rem 1rem}.stTabs [aria-selected="true"]{color:#0e6d6a;background:#e4f1ee;font-weight:700}.stMetric{background:#f6f8f7;border:1px solid #dce5e0;border-radius:8px;padding:.35rem .6rem}.stAlert{border-radius:8px}
.stApp{background:#f7f9f8;color:#18231f}.main .block-container{padding-left:2.2rem;padding-right:2.2rem}.stSidebar{background:#edf4f1;border-right:1px solid #d6e3de}.stSidebar h1{font-size:1.25rem;color:#123d3a}.stSidebar [data-testid="stMarkdownContainer"] p{color:#53665f}.stRadio>div{gap:.35rem}.stRadio label{border-radius:7px;padding:.28rem .45rem}.stRadio label:hover{background:#dfeee9}.stSelectbox label,.stNumberInput label,.stTextInput label,.stTextArea label{color:#40574f;font-weight:600;font-size:.84rem}.stSelectbox [data-baseweb="select"]>div{border-color:#cbdad4;border-radius:7px;background:#fff}.stButton>button,.stDownloadButton>button{border-radius:7px;border:1px solid #b8d0c8;background:#fff;color:#0e6d6a;font-weight:600}.stButton>button:hover,.stDownloadButton>button:hover{border-color:#0e6d6a;background:#e4f1ee;color:#0b5552}.stDataFrame{border:1px solid #d5e1dc;border-radius:8px;overflow:hidden}.stPlotlyChart{border:1px solid #e3ebe7;border-radius:8px;background:#fff;padding:.25rem}.stExpander{border:1px solid #d5e1dc;border-radius:8px;background:#fff}.stDivider{border-color:#d8e3de}.stCaption{color:#72827a}
.kpi-strip{display:flex;flex-wrap:wrap;border:1px solid #d8e0dc;border-radius:8px;overflow:hidden;background:#fff;margin:.3rem 0 1rem;box-shadow:0 1px 2px rgba(20,33,28,.05)}
.kpi-cell{flex:1 1 0;min-width:155px;padding:.65rem .9rem;border-right:1px solid #e6ebe8;background:color-mix(in srgb, var(--kpi-accent,#0e6d6a) 6%, #ffffff)}.kpi-cell:last-child{border-right:0}.kpi-label{display:flex;align-items:center;gap:.4rem;font-size:.72rem;color:#5b6b64}.kpi-dot{width:7px;height:7px;border-radius:50%;background:var(--kpi-accent,#0e6d6a);flex:0 0 auto}.kpi-value{font-size:1.2rem;font-weight:700;color:#14211c;font-variant-numeric:tabular-nums;margin-top:2px}.kpi-help{font-size:.68rem;color:#8a978f;margin-top:2px}
.kpi-strip--compact{background:transparent;border:none;box-shadow:none;border-radius:0;gap:.5rem;margin:.2rem 0 1.1rem}
.kpi-pill{display:flex;flex-direction:column;flex:1 1 0;min-width:170px;align-items:center;justify-content:center;gap:.2rem;text-align:center;background:#eef3f0;border:1px solid #dde6e1;border-radius:16px;padding:.65rem 1rem}
.kpi-pill-value{font-weight:700;color:#14211c;font-variant-numeric:tabular-nums;font-size:1.35rem}
.kpi-pill-label{font-size:.88rem;color:#5b6b64}
table details{margin-top:.4rem;font-size:.85rem}table details summary{cursor:pointer;color:#0e6d6a;font-weight:600;list-style:revert}table details summary:hover{color:#0b5552}
ul.candidate-list{list-style:none;margin:.5rem 0 0;padding:0;display:flex;flex-direction:column;gap:.35rem}
li.candidate-chip{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;background:#f3f7f5;border:1px solid #dfe9e4;border-radius:6px;padding:.35rem .6rem;font-size:.82rem;text-align:left}
li.candidate-chip .cc-dev-badge{background:#dcebe5;color:#0e4d47;border-radius:5px;padding:1px 7px;font-size:.72rem;font-weight:700;font-variant-numeric:tabular-nums}
li.candidate-chip .cc-var{font-weight:700;color:#14211c}li.candidate-chip .cc-arrow{color:#9aa8a2}li.candidate-chip .cc-action{color:#2f4a44}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_load(process_bytes=None, lab_bytes=None, norm_bytes=None):
    if process_bytes is None:
        return load_data(*default_paths(BASE))
    import io
    return load_data(io.BytesIO(process_bytes), io.BytesIO(lab_bytes), io.BytesIO(norm_bytes))


@st.cache_data(show_spinner=False)
def cached_analysis(df: pd.DataFrame, code: int):
    return analyze_code(df, code)


PROCESS_FILE = BASE / "data" / "Process_full.csv"

# PPT 17쪽 표의 '확인 후 조치 방향' · '판정' (대상군 기준)
GROUP_ACTION = {
    "정제경도": ("캠페인별 기준 분리 확인", "기준확인"),
    "타정속도": ("설정 범위·정속 운전 검토", "운전범위"),
    "주압축력": ("정제 품질 확인 후 조건 보완", "조건보완"),
    "배출력": ("배출 조건·펀치 상태 점검", "설비점검"),
    "예압축력": ("설정 단계 변경 사유 확인", "통합조사"),
    "실린더높이": ("설정 변경 기록 확인", "기록확인"),
    "충전깊이": ("가동 초기 충전 안정화 확인", "초기관리"),
    "공급속도": ("공급 속도 변경 기록 확인", "사후감시"),
}


@st.cache_data(show_spinner=False)
def cached_process_analysis(code: int = 17):
    """공정변수(CPP) 26개 MSPC·FMEA — PPT 17~18쪽과 같은 방법."""
    process_full, lab = pcap.load_process(BASE)
    analysis = pcap.analyze_process(process_full, lab, code)
    table = pcap.fmea_table(analysis)
    return analysis, table, pcap.fmea_groups(table)


def fmea_top8_view(groups: pd.DataFrame) -> pd.DataFrame:
    top = groups.head(8).copy()
    top.insert(0, "순위", range(1, len(top) + 1))
    top["확인 후 조치 방향"] = top["대상군"].map(lambda g: GROUP_ACTION.get(g, ("", ""))[0])
    top["판정"] = top["대상군"].map(lambda g: GROUP_ACTION.get(g, ("", ""))[1])
    return top[["순위", "대상군", "대표 변수", "S", "O", "D", "RPN", "용출 ρ", "한계 밖 배치", "확인 후 조치 방향", "판정"]]


def metrics(items, accent=None, compact=False):
    """Render a row of KPI numbers. `accent` sets the left-bar/tint color for the
    whole row (defaults to the teal brand accent); pass a 4th element per item
    to color individual cells differently within one row (e.g. 정상 vs 신호).

    `compact=True` swaps the tall alert-style card for a slim inline pill —
    use it for plain dataset-scope counts (전체 배치 수 등) that carry no
    signal/severity of their own, so they read as background context rather
    than repeating the same card shape every KPI row on the page uses for an
    actual finding."""
    row_accent = accent or ACCENT
    cells = []
    for item in items:
        label, value, help_text = item[0], item[1], item[2]
        cell_accent = item[3] if len(item) > 3 else row_accent
        if compact:
            cells.append(
                f"<div class='kpi-pill'><span class='kpi-pill-value'>{html.escape(str(value))}</span>"
                f"<span class='kpi-pill-label'>{html.escape(str(label))}</span></div>"
            )
        else:
            cells.append(
                f"<div class='kpi-cell' style='--kpi-accent:{cell_accent}'>"
                f"<div class='kpi-label'><span class='kpi-dot'></span>{html.escape(str(label))}</div>"
                f"<div class='kpi-value'>{html.escape(str(value))}</div>"
                f"<div class='kpi-help'>{html.escape(str(help_text))}</div></div>"
            )
    css_class = "kpi-strip kpi-strip--compact" if compact else "kpi-strip"
    st.markdown(f"<div class='{css_class}'>{''.join(cells)}</div>", unsafe_allow_html=True)


def signal_summary(analysis):
    frame = analysis.frame
    return {
        "code": analysis.code,
        "배치수": analysis.n,
        "SPC": int(frame["spc_signal"].sum()),
        "MSPC": int(frame["mspc_signal"].sum()),
        "통합": int(frame["integrated_signal"].sum()),
        "중복": int((frame["spc_signal"] & frame["mspc_signal"]).sum()),
        "PCA수": analysis.n_components,
        "설명분산": analysis.explained_ratio,
    }


def priority_table(analyses):
    frames = []
    for analysis in analyses:
        temp = analysis.frame[analysis.frame["integrated_signal"]].copy()
        temp["제품코드"] = analysis.code
        temp["T² 한계 대비"] = temp["t2"] / analysis.t2_limit
        temp["SPE 한계 대비"] = temp["spe"] / analysis.spe_limit
        frames.append(temp)
    if not frames:
        return pd.DataFrame()
    flagged = pd.concat(frames, ignore_index=True)
    if flagged.empty:
        return flagged
    flagged["우선순위"] = np.select(
        [flagged["spc_signal"] & flagged["mspc_signal"], flagged["spc_signal"], flagged[["T² 한계 대비", "SPE 한계 대비"]].max(axis=1) >= 1.5],
        ["1 · 복합신호", "1 · 품질 신호", "2 · 큰 MSPC 신호"], default="3 · MSPC 신호",
    )
    return flagged.sort_values(["우선순위", "제품코드", "start_date"])


def quarterly_panels(frame, quality, label):
    period = frame.set_index("start_date").groupby(pd.Grouper(freq="QS")).agg(
        batches=("batch", "count"), signals=("mspc_signal", "sum"),
        quality_mean=(quality, "mean"), measured=(quality, "count"),
    ).reset_index()
    period["quarter"] = period["start_date"].dt.to_period("Q").astype(str)
    period["rate"] = 100 * period["signals"] / period["batches"].replace(0, np.nan)
    unit = "%" if quality in ("dissolution_av", "dissolution_min", "batch_yield") else "원자료 값"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.19,
                        subplot_titles=("MSPC 신호율 · 전체 분석 배치 기준", f"{label} 평균 ({unit})"))
    fig.add_trace(go.Bar(
        x=period["quarter"], y=period["rate"], marker_color=CRITICAL,
        text=[f"{s} / {n}배치" if n else "자료 없음" for s, n in zip(period["signals"], period["batches"])],
        textposition="outside", cliponaxis=False,
        customdata=period[["signals", "batches"]],
        hovertemplate="%{x}<br>MSPC 신호율 %{y:.1f}%<br>신호 %{customdata[0]} / 전체 %{customdata[1]}배치<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=period["quarter"], y=period["quality_mean"], mode="lines+markers", connectgaps=False,
        line=dict(color=ACCENT, width=2), customdata=period[["measured", "batches"]],
        hovertemplate=f"%{{x}}<br>{label} 평균 %{{y:.3f}} ({unit})<br>측정 %{{customdata[0]}} / 전체 %{{customdata[1]}}배치<extra></extra>",
    ), row=2, col=1)
    maximum = period["rate"].max()
    fig.update_yaxes(title_text="MSPC 신호율 (%)", range=[0, min(100, max(10, maximum * 1.25))] if pd.notna(maximum) else [0, 10], row=1, col=1)
    values = frame[quality].dropna()
    if len(values):
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * .08, abs(high) * .005, .001)
        fig.update_yaxes(range=[max(0, low - padding) if low >= 0 else low - padding, high + padding], row=2, col=1)
    fig.update_yaxes(title_text=f"{label} ({unit})", row=2, col=1)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=period["quarter"].tolist())
    fig.update_xaxes(title_text="생산 분기", row=2, col=1)
    fig.update_layout(height=530, showlegend=False, margin=dict(l=15, r=15, t=45, b=70),
                      annotations=list(fig.layout.annotations) + [dict(
                          text="막대 위: 신호 / 전체 배치 수 · 평균은 측정값이 있는 배치 기준 · 두 그래프의 높낮이로 연관성을 판단하지 않습니다.",
                          x=0, y=-.17, xref="paper", yref="paper", showarrow=False, xanchor="left", font=dict(size=11, color=MUTED))])
    return fig


def html_report(selected_label, analyses, qa_conclusion, reviewer, review_date, whatif_summary=None, whatif_details=None, cases=None, process_html=""):
    summaries = pd.DataFrame([signal_summary(a) for a in analyses])
    rows = "".join(
        f"<tr><td>{int(r.code)}</td><td>{int(r.배치수)}</td><td>{int(r.SPC)}</td><td>{int(r.MSPC)}</td><td>{int(r.통합)}</td><td>{int(r.중복)}</td></tr>"
        for r in summaries.itertuples()
    )
    factor_sections, chart_sections = [], []
    include_js = True
    for a in analyses:
        factors = candidate_factors(a, 5)
        factors_report = factors[["변수", "차이"]].copy()
        factors_report["확인할 기록"] = factors_report["변수"].map(record_hint)
        factors_report["후보변수"] = factors_report.pop("변수").map(feature_label)
        factors_report["반복 편차"] = pd.to_numeric(factors_report.pop("차이"), errors="coerce").round(2)
        factor_sections.append(f"<h3>코드 {a.code}</h3>{factors_report.to_html(index=False, border=0)}")
        if len(analyses) == 1:
            frame = a.frame
            spc = go.Figure(go.Scatter(x=frame["start_date"], y=frame["dissolution_av"], mode="lines+markers", marker=dict(color=np.where(frame["spc_signal"], CRITICAL, ACCENT)), name="용출 평균"))
            spc.add_hline(y=a.spc_center, line_color=MUTED)
            spc.add_hline(y=a.spc_ucl, line_dash="dash", line_color=CRITICAL)
            spc.add_hline(y=a.spc_lcl, line_dash="dash", line_color=CRITICAL)
            spc.update_layout(title="용출 SPC 경향", height=360, yaxis_title="용출 평균(%)")
            mspc = px.scatter(frame, x="t2", y="spe", color="signal_type", hover_data=["batch", "start"], title="MSPC T² / SPE", color_discrete_map={"정상":"#b5c0bb", "MSPC":ACCENT, "SPC":WARNING, "SPC+MSPC":CRITICAL})
            mspc.add_vline(x=a.t2_limit, line_dash="dash", line_color=CRITICAL)
            mspc.add_hline(y=a.spe_limit, line_dash="dash", line_color=CRITICAL)
            charts = [spc, mspc]
            for column, label in [("impurities_total", "총 불순물 경향"), ("batch_yield", "배치 수율 경향")]:
                q = frame[["start_date", "batch", column, "mspc_signal"]].dropna(subset=[column]).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=q["start_date"], y=q[column], mode="lines+markers", name="전체 배치", line=dict(color="#91a7a0"), marker=dict(size=5)))
                signal_q = q[q["mspc_signal"]]
                fig.add_trace(go.Scatter(x=signal_q["start_date"], y=signal_q[column], mode="markers", name="MSPC 신호배치", marker=dict(color=CRITICAL, size=10, symbol="diamond")))
                fig.update_layout(title=label, height=320)
                charts.append(fig)
            charts.append(quarterly_panels(frame, "dissolution_av", "용출"))
            for fig in charts:
                chart_sections.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
                include_js = False
    priority = priority_table(analyses)
    priority_html = "<p>조사대상 없음</p>" if priority.empty else priority[["우선순위", "제품코드", "batch", "start", "signal_type"]].rename(columns={"batch":"배치", "start":"생산월", "signal_type":"신호"}).to_html(index=False, border=0)
    effect_html = "<p>시뮬레이션 미실행</p>"
    if whatif_summary is not None and not whatif_summary.empty:
        effect_html = whatif_summary[["제품코드", "정상값 반영률", "조정 전 MSPC 신호", "조정 후 MSPC 신호", "해소율(%)"]].to_html(index=False, border=0)
        if whatif_details is not None and not whatif_details.empty:
            report_details = whatif_details[["batch", "start", "조정후 판정", "해소"]].rename(columns={"batch":"배치", "start":"생산월", "조정후 판정":"조정 후", "해소":"신호 해소"})
            report_details["신호 해소"] = report_details["신호 해소"].map({True:"예", False:"아니오"})
            effect_html += report_details.to_html(index=False, border=0)
    codes = [a.code for a in analyses]
    signals_table = (f"<table><tr><th>제품코드</th><th>배치</th><th>SPC</th><th>MSPC</th><th>통합</th><th>중복</th></tr>"
                      f"{rows}</table>")
    charts_html = (''.join(chart_sections) if chart_sections
                   else '<p class="na">전체 코드 비교에서는 요약표를 제공합니다. 개별 코드를 선택하면 상세 그래프가 포함됩니다.</p>')
    cases_body = (case_html(cases) if cases
                  else '<p class="na">이번 리뷰 범위에서 동시 신호(SPC+MSPC) 배치가 없어 조사사례가 없습니다.</p>')
    cases_body += f"<h3>조사 우선순위 전체 목록</h3>{priority_html}"
    actions_body = process_html + cr.action_items_html(analyses, feature_label, record_hint, expanded=True) + f"<h3>CAPA 가정 조정 결과(시뮬레이션)</h3>{effect_html}"

    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>PQR 경향분석 보고서</title>
<style>body{{font-family:Arial,'Noto Sans KR',sans-serif;color:#18231f;max-width:1100px;margin:40px auto;line-height:1.55}}
h1,h2{{color:{ACCENT}}}table{{width:100%;border-collapse:collapse;margin:10px 0 24px}}
th,td{{border:1px solid #ccd7d2;padding:7px;text-align:center;font-size:12px}}th{{background:#edf5f2}}
@media print{{button{{display:none}}}}
.na{{color:{MUTED};font-style:italic}}
.cover h1{{margin-bottom:0}}.doc-sub{{color:{MUTED};margin-top:0}}
.doc-meta th,.doc-meta td,.sign-table th,.sign-table td,.revision th,.revision td{{text-align:left}}
.toc ul{{columns:2;column-gap:28px;list-style:none;padding-left:0}}.toc li{{margin:4px 0}}.toc a{{color:{ACCENT};text-decoration:none}}
.scroll-table{{max-height:380px;overflow:auto;border:1px solid #ccd7d2;margin:8px 0}}.scroll-table table{{margin:0}}
section{{margin:26px 0}}
@media print{{.cover{{break-after:page}}.toc{{break-after:page}}
.scroll-table{{max-height:none;overflow:visible;border:none}}}}
.executive{{background:#edf5f2;padding:20px;border-radius:10px}}.case{{margin-top:30px}}.case li{{margin:8px 0}}
code{{overflow-wrap:anywhere}}@media print{{.executive{{break-after:page}}.case section{{break-inside:avoid}}}}
table details{{margin-top:6px;text-align:left;font-size:12px}}table details summary{{cursor:pointer;color:{ACCENT};font-weight:700}}
ul.candidate-list{{list-style:none;margin:6px 0 0;padding:0;text-align:left}}
li.candidate-chip{{border:1px solid #ccd7d2;border-radius:5px;padding:5px 8px;margin:4px 0;font-size:12px}}
li.candidate-chip .cc-dev-badge{{background:#edf5f2;color:{ACCENT};border-radius:4px;padding:1px 6px;font-weight:700;margin-right:6px}}
li.candidate-chip .cc-var{{font-weight:700}}li.candidate-chip .cc-arrow{{color:{MUTED};margin:0 4px}}
</style></head><body>
<button onclick='window.print()'>인쇄 / PDF 저장</button>
{cr.cover_html(selected_label, reviewer, review_date, codes)}
{cr.toc_html()}
{executive_html(analyses, cases)}
{cr.section('purpose', '1. 목적', cr.purpose_html(selected_label))}
{cr.section('product', '2. 제품의 개요', cr.product_overview_html(analyses))}
{cr.section('batches', '3. 제조내역', cr.batch_history_html(analyses))}
{cr.section('signals', '4. 통계적 신호 현황', cr.signals_overview_note() + signals_table + f"<h3>세부 경향 그래프</h3>{charts_html}")}
{cr.na_section('change', '5. 변경관리 기록', '변경관리 이력 데이터가 없습니다.')}
{cr.section('cqa', '6. CQA Review', cr.cqa_review_html(analyses))}
{cr.section('cpp', '7. CPP Review', cr.cpp_review_note() + ''.join(factor_sections))}
{cr.section('yield', '8. 공정수율 결과 분석', cr.yield_review_html(analyses))}
{cr.na_section('validation', '9. 밸리데이션 및 생산장비 적격성평가 현황', '밸리데이션·장비 적격성평가 이력 데이터가 없습니다.')}
{cr.na_section('stability', '10. 안정성시험', '안정성시험 데이터가 없습니다.')}
{cr.na_section('complaints', '11. 반품·불만 및 회수에 대한 기록', '반품·불만·회수 기록 데이터가 없습니다.')}
{cr.na_section('audit', '12. 전년도 실사에 대한 기록', '실사 기록 데이터가 없습니다.')}
{cr.na_section('capa_prev', '13. 전년도 시정조치에 대한 기록', '전년도 CAPA 완료 기록 데이터가 없습니다.')}
{cr.section('cases', '14. 조사사례', cases_body)}
{cr.section('conclusion', '15. 결론', cr.conclusion_html(analyses, cases))}
<section><h2>QA 검토자 종합의견</h2><p>{html.escape(qa_conclusion) if qa_conclusion else '미입력'}</p></section>
{cr.section('actions', '16. 필요 조치사항', actions_body)}
{cr.section('references', '17. 관련규정 및 참고문서', cr.references_html())}
{cr.section('attachments', '18. 첨부문서', cr.attachments_html())}
</body></html>"""


# --- 데이터 입력 및 결합 -------------------------------------------------
st.sidebar.title("정제 OOS 조사·CAPA 대시보드")
with st.sidebar.expander("데이터 설정", expanded=False):
    use_uploaded = st.checkbox("다른 CSV 데이터 불러오기", value=False)
use_builtin = not use_uploaded
if use_builtin:
    df, process, laboratory, normalization = cached_load()
else:
    p_file = st.sidebar.file_uploader("Process CSV", type="csv")
    l_file = st.sidebar.file_uploader("Laboratory CSV", type="csv")
    n_file = st.sidebar.file_uploader("Normalization CSV", type="csv")
    if not all([p_file, l_file, n_file]):
        st.info("왼쪽에서 CSV 3개를 모두 선택하세요.")
        st.stop()
    df, process, laboratory, normalization = cached_load(p_file.getvalue(), l_file.getvalue(), n_file.getvalue())

counts = eligible_codes(df, 27)
code_values = [int(c) for c in counts.index]
analyses_by_code = {code: cached_analysis(df, code) for code in code_values}
sources_ok = use_builtin and verified_sources(BASE)

def _close_report():
    st.session_state["show_report"] = False


def _open_report():
    st.session_state["show_report"] = True


selected_step = st.sidebar.radio(
    "검토 단계",
    ["1. 공정·품질 경향", "2. 조사대상 배치", "3. CAPA 검토"],
    key="review_page",
    on_change=_close_report,
)
st.sidebar.divider()
st.sidebar.button("보고서 생성", on_click=_open_report)
page = "5. 보고서 생성" if st.session_state.get("show_report") else selected_step

st.title("정제 OOS 조사·CAPA 대시보드")

options = ["전체 코드 비교"] + [f"코드 {c} · {counts.loc[c]}배치" for c in code_values]
default_option = next((i for i, value in enumerate(options) if value.startswith("코드 17 ")), 0)
selected = st.selectbox("분석 코드", options, index=default_option)
if selected == "전체 코드 비교":
    selected_codes = code_values
    selected_label = f"전체 코드 {len(code_values)}개"
else:
    selected_codes = [int(selected.split()[1])]
    selected_label = selected
selected_analyses = [analyses_by_code[c] for c in selected_codes]
active_cases = find_cases(selected_analyses, process, feature_label) if sources_ok else []


# --- 1. 공정·품질 경향 (분석 개요 병합) -----------------------------------
if page.startswith("1."):
    st.subheader("제품코드별 공정·품질 경향")
    summary = pd.DataFrame([signal_summary(a) for a in selected_analyses])
    summary["통합 신호율(%)"] = (summary["통합"] / summary["배치수"] * 100).round(1)
    summary["설명분산(%)"] = (summary["설명분산"] * 100).round(1)
    metrics([
        ("분석 배치", f"{summary['배치수'].sum():,}", "선택 코드", MUTED),
        ("SPC 신호", f"{summary['SPC'].sum():,}", "용출 평균 단변량 신호", WARNING),
        ("MSPC 신호", f"{summary['MSPC'].sum():,}", "원료·정제 물성 26개의 다변량 신호", WARNING),
        ("통합 조사후보", f"{summary['통합'].sum():,}", "SPC 또는 MSPC", CRITICAL),
    ])
    if 17 in selected_codes and PROCESS_FILE.exists():
        st.caption("이 화면의 MSPC는 원료·정제 물성 기준입니다. 공정변수(타정·압축·공급) 기준 MSPC와 FMEA 조치 대상은 '3. CAPA 검토'에서 확인합니다. 두 계산은 서로 다른 분석이라 건수를 합치거나 비교하지 않습니다.")
    if len(selected_codes) > 1:
        fig = px.bar(summary, x="code", y=["SPC", "MSPC"], barmode="group", color_discrete_sequence=[CRITICAL, ACCENT], labels={"code":"제품코드", "value":"신호 배치수", "variable":"신호"})
        fig.update_layout(height=350, xaxis_title="제품코드", yaxis_title="신호 배치수", legend_title_text="신호", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        overview = summary[["code", "배치수", "SPC", "MSPC", "통합 신호율(%)"]].rename(columns={"code":"제품코드", "통합 신호율(%)":"조사대상률(%)"})
        st.dataframe(overview, hide_index=True, width="stretch")
    else:
        a = selected_analyses[0]
        frame = a.frame
        quality_labels = {
            "dissolution_av": "용출 평균", "dissolution_min": "용출 최솟값",
            "resodual_solvent": "잔류용매", "impurities_total": "총 불순물",
            "impurity_o": "불순물 O", "impurity_l": "불순물 L", "batch_yield": "배치 수율",
        }
        tab_spc, tab_mspc, tab_cqa = st.tabs(["용출 SPC", "다변량 MSPC", "품질 특성"])
        with tab_spc:
            spc_values = frame["dissolution_av"].dropna()
            recent = frame.dropna(subset=["dissolution_av"]).sort_values("start_date").tail(25)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frame["start_date"], y=frame["dissolution_av"], mode="lines+markers", name="용출 평균", marker=dict(color=np.where(frame["spc_signal"], CRITICAL, ACCENT)), customdata=frame["batch"], hovertemplate="배치 %{customdata}<br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>"))
            fig.add_hline(y=a.spc_center, line_color=MUTED, annotation_text=f"중심선 {a.spc_center:.2f}")
            fig.add_hline(y=a.spc_ucl, line_dash="dash", line_color=CRITICAL, annotation_text=f"상한 {a.spc_ucl:.2f}")
            fig.add_hline(y=a.spc_lcl, line_dash="dash", line_color=CRITICAL, annotation_text=f"하한 {a.spc_lcl:.2f}")
            fig.update_layout(height=460, yaxis_title="용출 평균(%)", margin=dict(l=10, r=10, t=20, b=10), showlegend=False, plot_bgcolor="#fbfcfb", paper_bgcolor="#fbfcfb")
            st.plotly_chart(fig, width="stretch")
            below80 = int((spc_values < 80).sum()) if len(spc_values) else 0
            st.markdown("#### CQA 요약")
            # Plain descriptive stats, not a signal/finding — compact pills instead
            # of the boxed alert-style card used for actual SPC/MSPC counts above,
            # so the two don't read as the same kind of information at a glance.
            metrics([
                ("평균", f"{display_value(spc_values.mean(), 2)}%", ""),
                ("표준편차", f"{display_value(spc_values.std(), 2)}%", ""),
                ("최솟값 / 최댓값", f"{spc_values.min():.2f} / {spc_values.max():.2f}%" if len(spc_values) else "-", ""),
                ("가정 하한(80%) 미만", f"{below80}/{len(spc_values)}배치", ""),
            ], compact=True)
            sigma_within = (a.spc_ucl - a.spc_center) / 3
            assumed_lsl = 80.0
            cpl = (a.spc_center - assumed_lsl) / (3 * sigma_within) if sigma_within > 0 else np.nan
            st.markdown("#### 공정 능력")
            st.caption("공식 규격이 없어 하한 80%를 가정한 참고 지표입니다 (하한 미만 배치 수는 위 CQA 요약 참고).")
            cap_left, cap_right = st.columns(2)
            with cap_left:
                st.metric("가정 Cpl", display_value(cpl, 2))
            with cap_right:
                st.metric("관리한계", f"{a.spc_lcl:.2f} ~ {a.spc_ucl:.2f}")
        with tab_mspc:
            fig = px.scatter(frame, x="t2", y="spe", color="signal_type", hover_data=["batch", "start"], labels={"t2":"T²", "spe":"SPE", "signal_type":"신호", "batch":"배치", "start":"생산월"}, color_discrete_map={"정상": "#b5c0bb", "MSPC": ACCENT, "SPC": WARNING, "SPC+MSPC": CRITICAL})
            fig.add_vline(x=a.t2_limit, line_dash="dash", line_color=CRITICAL)
            fig.add_hline(y=a.spe_limit, line_dash="dash", line_color=CRITICAL)
            fig.update_layout(height=430, xaxis_title="Hotelling T²", yaxis_title="SPE", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"PCA {a.n_components}개 성분 · 누적 설명분산 {a.explained_ratio:.1%} · 코드 내부 99% 한계")
        with tab_cqa:
            available_quality = [c for c in QUALITY_COLUMNS if c in frame and frame[c].notna().any()]
            selected_quality = st.selectbox(
                "품질 특성 선택", available_quality,
                format_func=lambda x: quality_labels.get(x, x), key="quality_metric",
            )
            q = frame[["start_date", "batch", selected_quality, "mspc_signal", "spc_signal", "signal_type"]].dropna(subset=[selected_quality]).copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=q["start_date"], y=q[selected_quality], mode="lines+markers", name="전체 배치",
                line=dict(color="#91a7a0", width=1.5), marker=dict(color="#91a7a0", size=5),
                customdata=q[["batch", "signal_type"]],
                hovertemplate="배치 %{customdata[0]}<br>%{x|%Y-%m}<br>값=%{y:.3f}<br>신호=%{customdata[1]}<extra></extra>",
            ))
            mspc_q = q[q["mspc_signal"]]
            fig.add_trace(go.Scatter(
                x=mspc_q["start_date"], y=mspc_q[selected_quality], mode="markers", name="MSPC 신호배치",
                marker=dict(color=CRITICAL, size=11, symbol="diamond", line=dict(color="white", width=1)),
                customdata=mspc_q[["batch", "signal_type"]],
                hovertemplate="MSPC 신호 · 배치 %{customdata[0]}<br>%{x|%Y-%m}<br>값=%{y:.3f}<extra></extra>",
            ))
            fig.update_layout(height=430, xaxis_title="생산 기간", yaxis_title=quality_labels.get(selected_quality, selected_quality), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

            normal_values = q.loc[~q["mspc_signal"], selected_quality]
            signal_values = q.loc[q["mspc_signal"], selected_quality]
            if len(signal_values) and len(normal_values):
                mean_delta = signal_values.mean() - normal_values.mean()
                pooled_sd = q[selected_quality].std()
                standardized_delta = mean_delta / pooled_sd if pd.notna(pooled_sd) and pooled_sd > 0 else np.nan
            else:
                mean_delta, standardized_delta = np.nan, np.nan
            metrics([
                (f"정상배치 평균 ({len(normal_values)}배치)", display_value(normal_values.mean()), ""),
                (f"신호배치 평균 ({len(signal_values)}배치)", display_value(signal_values.mean()), ""),
                ("평균 차이 (신호-정상)", display_value(mean_delta), ""),
                ("표준화 차이 (크기 비교)", display_value(standardized_delta, 2), ""),
            ], compact=True)


# --- 2. 조사대상 배치 ----------------------------------------------------
elif page.startswith("2."):
    st.subheader("조사대상 배치")
    flagged = priority_table(selected_analyses)

    signal_labels = {"정상": "정상", "SPC": "용출 SPC", "MSPC": "다변량 MSPC", "SPC+MSPC": "동시 신호"}

    if flagged.empty:
        st.success("선택한 코드에서 조사대상 배치가 확인되지 않았습니다.")
    else:
        st.caption(f"조사후보 {len(flagged)}배치")
        selected_signal = st.selectbox("확인할 신호 유형", ["전체", "SPC", "MSPC", "SPC+MSPC"], format_func=lambda x: "전체" if x == "전체" else signal_labels[x])
        visible = flagged if selected_signal == "전체" else flagged[flagged["signal_type"] == selected_signal]
        batch_options = [f"코드 {int(r['제품코드'])} / 배치 {int(r['batch'])} · {signal_labels[r['signal_type']]}" for _, r in visible.iterrows()]
        if not batch_options:
            st.info("이 신호 유형에 해당하는 조사대상 배치가 없습니다.")
            st.stop()
        chosen = st.selectbox("상세 확인할 배치", batch_options)
        chosen_code = int(chosen.split()[1]); chosen_batch = int(chosen.split(" / 배치 ")[1].split()[0])
        a = analyses_by_code[chosen_code]
        row = a.frame[a.frame["batch"] == chosen_batch].iloc[0]

        # Every batch gets the same spine — subheader, summary metrics, top-variable
        # chart — regardless of signal type. A concurrent-signal batch then gets the
        # detailed investigation appended below, instead of switching to a whole
        # different layout that only 804 used to get.
        st.markdown(f"#### 코드 {chosen_code} · 배치 {chosen_batch}")
        # Show this batch's own values for the SAME variables CAPA acts on for this
        # product (candidate_factors — product-wide 이상/정상 평균 |z| 차이 상위),
        # instead of this batch's own independently-ranked top-|z| variables. The
        # two used to be different rankings (batch-local extremes vs product-wide
        # discriminators), which meant a variable could be flagged here to "check
        # first" with no corresponding action on the CAPA table, or vice versa.
        # Falls back to the batch's own top-7 when the product has no MSPC
        # candidates at all (candidate_factors empty).
        candidate_vars = candidate_factors(a, 5)["변수"].tolist()
        feature_pool = candidate_vars or a.feature_names
        z_values = pd.Series({c: row.get(f"z__{c}", np.nan) for c in feature_pool}).dropna().abs().sort_values(ascending=False)
        if not candidate_vars:
            z_values = z_values.head(7)
        left, right = st.columns([1, 1.35])
        with left:
            st.markdown("##### 배치 요약")
            signal_accent = {"정상": MUTED, "SPC": WARNING, "MSPC": WARNING, "SPC+MSPC": CRITICAL}.get(row["signal_type"], MUTED)
            metrics([("신호", row["signal_type"], "구분", signal_accent), ("생산월", str(row["start"]), "배치 시점", MUTED)])
            metrics([
                ("용출 평균", display_value(row.get("dissolution_av"), 2), "%"),
                ("용출 최솟값", display_value(row.get("dissolution_min"), 2), "%"),
                ("총불순물", display_value(row.get("impurities_total")), "시험결과"),
                ("배치 수율", display_value(row.get("batch_yield"), 2), "%"),
            ], accent=MUTED)
        with right:
            st.markdown("##### 먼저 확인할 변수")
            z_df = z_values.rename("표준화 편차").rename_axis("변수").reset_index()
            z_df["변수"] = z_df["변수"].map(feature_label)
            fig = px.bar(z_df.sort_values("표준화 편차"), x="표준화 편차", y="변수", orientation="h", color_discrete_sequence=[ACCENT])
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")


# --- 3. CAPA 검토 --------------------------------------------------------
elif page.startswith("3."):
    st.subheader("CAPA 검토")
    process_ready = 17 in selected_codes and use_builtin and PROCESS_FILE.exists()
    if active_cases:
        names = ", ".join(f"코드 {c['code']} 배치 {c['batch']}" for c in active_cases)
        st.caption(f"동시 신호 배치({names})의 조치는 원인 확인 후 적용하는 제안입니다.")

    if process_ready:
        p_analysis, p_table, p_groups = cached_process_analysis(17)
        pf = p_analysis.frame
        st.markdown("#### 코드 17 · 공정변수 기준 신호")
        metrics([
            ("분석 배치", f"{len(pf):,}", "코드 17 전체", MUTED),
            ("SPC 신호", f"{int(pf['spc_signal'].sum())}", "용출 평균 I-MR", WARNING),
            ("MSPC 신호", f"{int(pf['mspc_signal'].sum())}", "공정변수 26개 · T²/SPE 99%", WARNING),
            ("통합 조사후보", f"{int((pf['spc_signal'] | pf['mspc_signal']).sum())}", "SPC 또는 MSPC", CRITICAL),
        ])
        st.caption("타정속도·압축력·공급·충전·경도·배출력 등 공정변수 26개로 계산한 신호입니다(15~18쪽과 같은 방법). 원료·정제 물성 기준 신호(8건)와는 다른 계산입니다.")

        st.markdown("#### 필요 조치사항 · FMEA 상위 8개 대상군")
        top8 = fmea_top8_view(p_groups)
        st.dataframe(top8, hide_index=True, width="stretch")
        st.caption(
            "RPN = 심각도(S) × 발생도(O) × 검출 어려움(D). S·O는 207배치 데이터와 규칙으로 계산하고, "
            "D는 일반 타정 공정 지식에 따른 가정이라 현장 확인이 필요합니다. 같은 종류 변수는 묶어 최고 RPN으로 표시했습니다. "
            "순위는 우선 확인할 대상이라는 뜻이며 원인이 확정된 것은 아닙니다. 정제경도·예압축력·실린더높이는 캠페인별 단계 변화로 O가 크고 용출 상관은 약합니다."
        )
        with st.expander("공정변수 26개 FMEA 전체 보기"):
            full = p_table[["변수", "용출 ρ", "용출 p", "이상 기여 순위", "한계 밖 배치", "S", "O", "D", "RPN"]].copy()
            full["용출 p"] = full["용출 p"].map(lambda v: f"{v:.4f}")
            st.dataframe(full, hide_index=True, width="stretch")
            st.caption("S: 용출 상관이 26개 동시검정 보정(p<0.0019) 통과 9 · 보정 전 p<0.05 7 · MSPC 이탈 배치 기여 상위 10위 6 · 그 외 4. O: 한계 밖 배치 비율 구간(1~10).")

        st.markdown("#### 가정 조정 효과 확인 (가상 시뮬레이션)")
        mode = st.radio(
            "조치 대상",
            ["FMEA 8개", "FMEA 8개 + 이상 기여 4개 (18쪽)", "직접 선택"],
            index=1, horizontal=True,
        )
        fmea8, extra4 = pcap.FMEA_8, pcap.EXTRA_4
        if mode == "FMEA 8개":
            chosen = list(fmea8)
        elif mode.startswith("FMEA 8개 +"):
            chosen = list(fmea8) + list(extra4)
        else:
            chosen = st.multiselect("조정할 변수", pcap.CPP_COLS, default=list(fmea8), format_func=lambda c: pcap.CPP[c])
        st.caption("조정 변수: " + (", ".join(pcap.CPP[c] for c in chosen) if chosen else "없음") + " · 정상 배치 중앙값 쪽으로 이동 후 다시 계산(판정 기준인 PCA·관리한계는 조치 전 값으로 고정)")
        improvement = st.select_slider("정상 수준 반영률", options=[0, 25, 50, 75, 100], value=100, format_func=lambda x: f"{x}%", key="proc_improvement")
        result = pcap.whatif(p_analysis, chosen, improvement)
        before, after = result["before"], result["after"]
        recovered = before - after
        metrics([
            ("조치 전 이탈", f"{before}건", "MSPC 이탈 배치", MUTED),
            ("조치 후 이탈", f"{after}건", f"{recovered}건 정상 복귀", GOOD if recovered > 0 else MUTED),
            ("복귀 비율", f"{recovered / before * 100:.0f}%" if before else "-", "이탈 배치 중", GOOD),
            ("새 신호", f"{result['new']}건", "정상이던 배치에서 새로 이탈", GOOD if result["new"] == 0 else WARNING),
        ])
        stage_rows = []
        for pct in [0, 25, 50, 75, 100]:
            r = pcap.whatif(p_analysis, chosen, pct)
            stage_rows.append({"정상 수준 반영률": f"{pct}%", "이탈 배치": r["after"], "정상 복귀": r["before"] - r["after"], "새 신호": r["new"], "경도·배출력 순서 불일치": r["order_violation"]})
        st.dataframe(pd.DataFrame(stage_rows), hide_index=True, width="stretch")
        st.caption(
            "100%는 가정 시나리오이며 실제 효과는 조치 시행 후 배치 관찰로 확인합니다. 이상 기여 4개는 MSPC 기여도로 골라 같은 MSPC 기준으로 평가했습니다. "
            "값을 한꺼번에 옮기면 최소·평균·최대(경도·배출력)의 순서가 맞지 않는 배치가 생기므로 그 건수를 함께 표시했습니다. "
            "설비 이력·원료 로트·작업일지로 남은 배치의 원인을 확인합니다."
        )
        with st.expander("배치별 결과 보기"):
            d = result["detail"].rename(columns={"batch": "배치", "start": "생산월"})
            d["조치 전 신호 크기"] = d["조치 전 신호 크기"].round(2)
            d["조치 후 신호 크기"] = d["조치 후 신호 크기"].round(2)
            st.dataframe(d, hide_index=True, width="stretch")
            st.caption("신호 크기 = max(T²/한계, SPE/한계). 1 아래면 정상입니다.")
        # 보고서에서 쓰는 형식으로 저장
        detail_for_report = result["detail"].rename(columns={"조치 후": "조정후 판정"}).copy()
        detail_for_report["해소"] = detail_for_report["조정후 판정"] == "정상 복귀"
        st.session_state["whatif_summary"] = pd.DataFrame([{
            "제품코드": 17, "조정 변수": ", ".join(pcap.CPP[c] for c in chosen), "정상값 반영률": f"{improvement:.0f}%",
            "조치 전 MSPC 신호": before, "조치 후 MSPC 신호": after, "해소 배치": recovered,
            "해소율(%)": round(recovered / before * 100, 1) if before else 0.0,
        }]).rename(columns={"조치 전 MSPC 신호": "조정 전 MSPC 신호", "조치 후 MSPC 신호": "조정 후 MSPC 신호"})
        st.session_state["whatif_details"] = detail_for_report
        st.session_state["whatif_code"] = 17
        legacy_title = "참고 · 원료·정제 물성 MSPC 기반 후보 (별도 분석)"
    else:
        legacy_title = "필요 조치사항 · 가정 조정 효과"
        if 17 not in selected_codes:
            st.caption("FMEA·공정변수 기준 시뮬레이션은 코드 17에서만 제공합니다(분석 범위).")

    with st.expander(legacy_title, expanded=not process_ready):
        st.markdown("##### 필요 조치사항")
        st.markdown(cr.capa_dashboard_html(selected_analyses, feature_label, action_hint), unsafe_allow_html=True)

        st.markdown("##### 가정 조정 효과 확인")
        simulation_code = selected_codes[0] if len(selected_codes) == 1 else st.selectbox("효과를 확인할 코드", selected_codes, format_func=lambda x: f"코드 {x}")
        simulation_analysis = analyses_by_code[int(simulation_code)]
        factor_options = candidate_factors(simulation_analysis, 10)["변수"].tolist()
        selected_factors = st.multiselect("조정할 후보변수", factor_options, default=factor_options[:5], format_func=feature_label)
        improvement = st.select_slider("정상 수준 반영률", options=[0, 25, 50, 75, 100], value=100, format_func=lambda x: f"{x}%")
        whatif_summary, whatif_details = simulate_capa(simulation_analysis, selected_factors, improvement)
        if whatif_summary.empty:
            st.info("선택한 코드에는 시뮬레이션 가능한 MSPC 신호가 없습니다.")
        else:
            before = int(whatif_summary.iloc[0]["조정 전 MSPC 신호"])
            after = int(whatif_summary.iloc[0]["조정 후 MSPC 신호"])
            reduced = before - after
            metrics([
                ("조정 전 신호", f"{before}배치", "정상 수준 반영 전", MUTED),
                ("조정 후 신호", f"{after}배치", f"-{reduced}배치", GOOD if reduced > 0 else MUTED),
                ("해소율", f"{whatif_summary.iloc[0]['해소율(%)']:.1f}%", "신호 감소 비율", GOOD),
            ])
            with st.expander("배치별 결과 보기"):
                detail_view = whatif_details[["batch", "start", "조정후 판정", "해소"]].rename(columns={"batch":"배치", "start":"생산월", "조정후 판정":"조정 후", "해소":"신호 해소"})
                detail_view["신호 해소"] = detail_view["신호 해소"].map({True: "예", False: "아니오"})
                st.dataframe(detail_view, hide_index=True, width="stretch")
            if not (process_ready and int(simulation_code) == 17):
                st.session_state["whatif_summary"] = whatif_summary
                st.session_state["whatif_details"] = whatif_details
                st.session_state["whatif_code"] = int(simulation_code)


# --- 5. 보고서 생성 ------------------------------------------------------
else:
    st.subheader("PQR 경향분석 보고서 생성")
    reviewer = st.text_input("작성/검토자", placeholder="예: 홍길동 / QA")
    review_date = st.date_input("검토일")
    qa_conclusion = st.text_area("QA 종합결론", height=150, placeholder="예: 코드 17의 반복 MSPC 신호 배치에 대해 원자료와 설비로그 추가 검토가 필요함.")
    stored_whatif_code = st.session_state.get("whatif_code")
    stored_summary = st.session_state.get("whatif_summary") if stored_whatif_code in selected_codes else None
    stored_details = st.session_state.get("whatif_details") if stored_whatif_code in selected_codes else None
    if active_cases:
        names = ", ".join(f"코드 {c['code']} 배치 {c['batch']}" for c in active_cases)
        st.caption(f"{names} 조사사례, 조건부 CAPA, 효과 확인 계획을 포함합니다.")
    process_html = ""
    if 17 in selected_codes and use_builtin and PROCESS_FILE.exists():
        _pa, _pt, _pg = cached_process_analysis(17)
        _pf = _pa.frame
        process_html = (
            "<h3>코드 17 · FMEA 기반 조치 대상(공정변수 기준)</h3>"
            f"<p>공정변수 26개 기준 신호: SPC {int(_pf['spc_signal'].sum())}건, MSPC {int(_pf['mspc_signal'].sum())}건 "
            "(원료·정제 물성 기준 신호와는 다른 계산입니다.)</p>"
            + fmea_top8_view(_pg).to_html(index=False, border=0)
            + "<p class='na'>RPN = S × O × D. S·O는 207배치 데이터 규칙, D는 일반 공정 지식에 따른 가정이며 현장 확인이 필요합니다. "
              "순위는 우선 확인 대상이며 원인 확정이 아닙니다.</p>"
        )
    report = html_report(
        selected_label, selected_analyses, qa_conclusion, reviewer, review_date,
        stored_summary, stored_details, active_cases, process_html,
    )
    filename = f"PQR_trend_review_{'all' if len(selected_codes)>1 else selected_codes[0]}_{review_date}.html"
    st.download_button("HTML 보고서 내려받기", report.encode("utf-8"), file_name=filename, mime="text/html", type="primary")
    st.download_button("조사후보 CSV 내려받기", pd.concat([a.frame[a.frame['integrated_signal']].assign(code=a.code) for a in selected_analyses]).to_csv(index=False).encode("utf-8-sig"), file_name="investigation_candidates.csv", mime="text/csv")
    st.markdown("#### 보고서에 포함되는 내용")
    st.write(
        "표지·개정이력·목차 · 검토 요약 · 1~18장(목적/제품개요/제조내역/통계적 신호 현황/변경관리/"
        "CQA·CPP Review/공정수율/밸리데이션·안정성·반품불만·실사·전년도CAPA(해당없음 명시)/"
        "조사사례/결론/필요 조치사항/관련규정/첨부문서) · 서명란"
    )
    st.caption(f"생성 시각(UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}")
