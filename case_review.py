"""Evidence-backed teaching case. No clinical, release, or CAPA approval decision."""
from __future__ import annotations

import hashlib
import html
from pathlib import Path

import pandas as pd

from analytics import candidate_factors, QUALITY_COLUMNS

QUALITY_LABELS = {
    "dissolution_av": "용출 평균(%)", "dissolution_min": "용출 최솟값(%)",
    "resodual_solvent": "잔류용매", "impurities_total": "총 불순물",
    "impurity_o": "불순물 O", "impurity_l": "불순물 L", "batch_yield": "배치 수율(%)",
}

SOURCE_URL = "https://app.notion.com/p/3de1d771fe6c81b4a5c3de37a22ac037"
SOURCE_DATE = "2026-09-17"
# 노션 "품질동향 리뷰" 문서가 실제로 존재하고 비교 가능한 제품코드만 여기 적는다.
# 다른 제품코드로 사례가 생겨도, 노션에 없는 비교를 지어내지 않기 위한 목록.
NOTION_CODES = {17}
SOURCE_HASHES = {
    "Laboratory.csv": "a3b0f16c7c3fa8052b38d767e38fda731a751acc4ec139b52d01577b32dc6483",
    "Normalization.csv": "957bae7af3266b5317eb8eda49a8a65f587d33f04794cda25f1062f12265eaa6",
    "Process.csv": "27f5dd9067ab20b5557380effa5e3db75abc4ce889767675a638b494ad5b2f81",
}


def verified_sources(base: Path) -> bool:
    return all((base / "data" / name).is_file() and
               hashlib.sha256((base / "data" / name).read_bytes()).hexdigest() == digest
               for name, digest in SOURCE_HASHES.items())


def build_case(analysis, process, batch, label_fn=None):
    """Calculate every numeric claim from the active, verified analysis for one batch.

    Works for any product code / batch, not just a hardcoded example — the only
    requirement is that the batch actually has a concurrent SPC+MSPC signal.
    """
    frame = analysis.frame
    matches = frame[frame["batch"] == batch]
    if len(matches) != 1:
        return None
    row = matches.iloc[0]
    if not (row["spc_signal"] and row["mspc_signal"]):
        return None
    label_fn = label_fn or (lambda name: name)
    reference = frame[~frame["integrated_signal"]]
    selected = process[process["code"] == analysis.code]
    matched = set(selected["batch"].dropna().astype(int)) & set(frame["batch"].astype(int))
    z_values = (pd.Series({key: row.get(f"z__{key}", float("nan")) for key in analysis.feature_names})
                .dropna().abs().sort_values(ascending=False))
    evidence = []
    for key in z_values.head(2).index:
        # Some raw columns come through as text for certain product codes (source data
        # quirk, not specific to any one batch) — coerce so evidence values don't crash
        # or silently compare a string batch value against a numeric reference median.
        ref_numeric = pd.to_numeric(reference[key], errors="coerce")
        evidence.append({"key": key, "label": label_fn(key), "value": float(pd.to_numeric(row[key], errors="coerce")),
                         "median": float(ref_numeric.median()),
                         "reference_n": int(ref_numeric.notna().sum()),
                         "z": float(row[f"z__{key}"])})
    return {"version": f"C{int(analysis.code)}-{int(batch)} / 1.0", "batch": int(batch), "code": int(analysis.code),
            "period": row["start_date"].strftime("%Y-%m"),
            "n": len(frame), "spc": int(frame["spc_signal"].sum()),
            "mspc": int(frame["mspc_signal"].sum()),
            "both": int((frame["spc_signal"] & frame["mspc_signal"]).sum()),
            "union": int(frame["integrated_signal"].sum()), "reference_n": len(reference),
            "features": len(analysis.feature_names), "components": analysis.n_components,
            "variance": analysis.explained_ratio, "process_n": len(matched),
            "has_process": int(batch) in matched, "dissolution": float(row["dissolution_av"]),
            "ucl": analysis.spc_ucl, "lcl": analysis.spc_lcl,
            "t2": float(row["t2"]), "t2_limit": analysis.t2_limit,
            "spe": float(row["spe"]), "spe_limit": analysis.spe_limit,
            "evidence": evidence}


def find_cases(analyses, process, label_fn=None):
    """Every batch with a concurrent SPC+MSPC signal across the given analyses, as a case.

    This is what makes the case study apply to any qualifying batch instead of
    a single hardcoded one — every product is scanned the same way.
    """
    cases = []
    for a in analyses:
        concurrent = a.frame[a.frame["spc_signal"] & a.frame["mspc_signal"]]
        for batch in concurrent["batch"].astype(int):
            case = build_case(a, process, batch, label_fn)
            if case:
                cases.append(case)
    return cases


def _dissolution_fact(c):
    if c["dissolution"] > c["ucl"]:
        return (f"용출 평균 {c['dissolution']:.2f}%로 관리상한 {c['ucl']:.2f}%를 초과했습니다. "
                "상방 신호이며 저용출 사례는 아닙니다.")
    return (f"용출 평균 {c['dissolution']:.2f}%로 관리하한 {c['lcl']:.2f}%에 미달했습니다. "
            "하방 신호이며 저용출 사례에 해당합니다.")


def _mspc_fact(c):
    over_t2, over_spe = c["t2"] > c["t2_limit"], c["spe"] > c["spe_limit"]
    if over_t2 and over_spe:
        source = "T²와 SPE 모두에서"
    elif over_t2:
        source = "T²에서"
    else:
        source = "SPE에서"
    return (f"SPE {c['spe']:.2f} {'>' if over_spe else '<'} 한계 {c['spe_limit']:.2f}. "
            f"T² {c['t2']:.2f} {'>' if over_t2 else '<'} 한계 {c['t2_limit']:.2f}. "
            f"다변량 신호는 {source} 발생했습니다.")


def _process_fact(c):
    if c["has_process"]:
        return "해당 배치의 공정행이 제공 파일에 있어 압축력·타정속도 등 공정변수 가설도 함께 확인할 수 있습니다."
    return "해당 배치의 공정행이 제공 파일에 없어 압축력·타정속도 가설은 아직 검증할 수 없습니다."


def case_sections(c):
    ev1, ev2 = c["evidence"]
    return [
        ("확인된 사실", [
            _dissolution_fact(c),
            _mspc_fact(c),
            f"{ev1['label']} {ev1['value']:.2f} / 무신호군 중앙값 {ev1['median']:.2f}; "
            f"{ev2['label']} {ev2['value']:.2f} / {ev2['median']:.2f}입니다. 단위는 원자료 확인이 필요합니다.",
        ]),
        ("조사 가설과 추가 확인", [
            "가설: 정제 물성·중량 변동 또는 시험·기록 조건 차이가 함께 나타났을 수 있습니다. "
            "이 편차는 조사 순서를 정하는 근거이며, 원인은 원기록 대조로 확인합니다.",
            "QC 확인: 용출 시험 원자료·장비 적합성·시료 준비·계산과 전사 기록. "
            "기록·측정 문제가 확인되면 그 영향을 먼저 재평가합니다.",
            f"생산·QA 확인: 중량·두께 측정 원자료, 제조기록, 원료 로트, 타정·정비 이력. {_process_fact(c)}",
        ]),
        ("조건부 CAPA 제안", [
            "측정·기록 원인이 확인되면: 승인 절차에 따라 해당 기록과 영향 범위를 검토하고, "
            "확인된 오류에 맞춰 점검·이중 확인 절차를 보완합니다. 담당 제안: QC·QA.",
            "제조 원인이 확인되면: 해당 설비·조건의 원인을 시정하고, 중량·두께 변동의 주기적 검토와 "
            "재발 시 조사 절차를 설계합니다. 담당 제안: 생산·설비·QA.",
            "현재는 조치 제안 단계입니다. 원인 증빙·영향 범위 확인 후 담당자와 기한을 정하며, "
            "조치 실행·완료 사실은 없습니다.",
        ]),
        ("효과 확인 계획 · 제안", [
            f"실행 전 기준선·모델·관리한계·판정 규칙을 고정합니다. 예시 관찰 범위는 시행 후 코드 {c['code']}의 "
            "연속 30배치이며, 실제 범위와 목표는 QA 검토 후 정합니다.",
            "30배치의 원자료와 조치 이행 증빙을 모두 확인하고, 조사로 확정한 동일 원인의 재발 0건을 "
            "제안 목표로 둡니다. 재발 시 원인조사를 다시 엽니다.",
            f"SPC·MSPC·동시 신호율을 과거 {c['n']}배치 기준선과 비교하고, 용출 하방 신호·중량 RSD·두께도 "
            "함께 검토합니다. 모델 재학습으로 신호를 없애는 것을 개선으로 보지 않습니다.",
            "표본·원인·시행 기록이 갖춰진 뒤에 효과를 판정합니다. 가정 조정으로 줄어든 신호는 "
            "실제 CAPA 효과로 보고하지 않습니다.",
        ]),
    ]


def basis_items(c):
    return [
        ("확인된 차이", f"노션: 공정 CPP 26종·MSPC 20배치·SPC와 중복 0건으로 기록. 현재 모델: 원료·정제 물성 "
         f"{c['features']}종·MSPC {c['mspc']}배치·중복 {c['both']}건. 변수 수는 같아도 입력 변수가 다릅니다."),
        ("자료 범위", f"현재 코드 {c['code']}의 시험 배치는 {c['n']}개, 공정 파일 매칭은 {c['process_n']}개입니다. "
         f"배치 {c['batch']}의 공정행은 {'있습니다' if c['has_process'] else '없습니다'}. "
         "노션의 압축력·타정속도 분석을 이 파일만으로 같은 조건에서 재현할 수 없습니다."),
        ("현재 계산", f"결측은 제품 내 중앙값 대체, 표준화 후 PCA {c['components']}성분({c['variance']:.1%}) 유지. "
         "T² 또는 SPE의 99% 한계 초과로 분류합니다. SPC는 생산월·배치 순서의 MR/1.128 기반 3σ 한계입니다."),
        ("해석 범위", "동일 이력으로 모델 적합·신호 탐지를 수행한 탐색 분석입니다. 독립 검증 성능은 아닙니다. "
         "생산일 대신 월과 배치 순서를 사용하며, SPC의 연속·추세 규칙은 적용하지 않습니다."),
        ("남은 확인", "20건→현재 건수의 차이를 완전히 분해하려면 노션 원본 학습 데이터·배치 목록·코드·전처리·"
         "관리한계가 필요합니다. 입력 변수 차이는 확인했지만, 전체 차이의 재현 검증은 미완료입니다."),
    ]


def executive_html(analyses, cases=None):
    cases = cases or []
    summaries = []
    for a in analyses:
        f = a.frame
        summaries.append(f"코드 {a.code}: {a.n}배치 중 SPC {int(f.spc_signal.sum())}, MSPC {int(f.mspc_signal.sum())}, "
                          f"동시 {int((f.spc_signal & f.mspc_signal).sum())}, 통합 조사후보 {int(f.integrated_signal.sum())}배치")
    finding = "; ".join(summaries)
    if cases:
        named = ", ".join(f"코드 {c['code']} 배치 {c['batch']}" for c in cases)
        priority = f"{named}의 용출·다변량 동시 신호(SPC+MSPC)를 대표 사례로 검토합니다."
    else:
        priority = "선택 제품의 통계적 신호 배치를 우선 검토합니다."
    return f"""<section class='executive'><h2>검토 요약 · 자동 초안</h2>
<p><b>주요 발견</b><br>{html.escape(finding)}</p>
<p><b>우선 조사</b><br>{html.escape(priority)}</p>
<p><b>다음 확인</b><br>시험 원자료와 제조·설비 이력으로 원인 가설을 확인하고, 근거가 확보된 항목에 대해 CAPA를 구체화합니다.</p>
<p><b>현재 결론</b><br>추가조사 필요 · 원인과 CAPA 효과는 조사와 시행 후 관찰로 확인합니다. 통계적 신호는 공식 규격 부적합(OOS) 또는 출하 판정이 아닙니다.</p></section>"""


def _case_article(c):
    sections = "".join(f"<section><h3>{html.escape(title)}</h3><ul>" +
                       "".join(f"<li>{html.escape(text)}</li>" for text in lines) + "</ul></section>"
                       for title, lines in case_sections(c))
    notion_block = ""
    if c["code"] in NOTION_CODES:
        basis = "".join(f"<p><b>{html.escape(title)}</b> — {html.escape(text)}</p>" for title, text in basis_items(c))
        notion_block = (f"<h3>분석 기준과 노션 결과의 차이</h3>{basis}"
                         f"<p>비교 출처: <a href='{SOURCE_URL}'>노션 제품 17·23 품질동향 리뷰</a> ({SOURCE_DATE} 수정본). "
                         "노션 수치는 문서 기재값이며 현재 모델의 재현 결과가 아닙니다.</p>")
    hashes = "".join(f"<li>{name}: <code>{digest}</code></li>" for name, digest in SOURCE_HASHES.items())
    return f"""<article class='case'><h2>조사사례 · 코드 {c['code']} / 배치 {c['batch']}</h2>
<p>{html.escape(c['version'])} · 생산월 {c['period']} · 공개 데이터 기반 사례연구 · 검토 상태: 추가조사 필요</p>
<p>선정 이유: 용출 SPC와 다변량 신호가 함께 발생해 품질 결과와 정제 물성을 연결해 검토할 수 있습니다.</p>
{sections}{notion_block}
<p>비교군: 두 신호가 모두 없는 {c['reference_n']}배치. 표준화 편차 z는 제품 전체 기준이며 인과 기여도가 아닙니다.</p>
<details><summary>재현용 데이터 식별값 · SHA-256</summary><ul>{hashes}</ul></details></article>"""


def case_html(cases):
    cases = cases or []
    return "".join(_case_article(c) for c in cases)


# --- Formal PQR-document sections (used only by the downloadable report on
# "5. 보고서 생성", not by the interactive dashboard pages) ------------------
#
# Structure and section order are adapted from a real, filed PQR report the
# user shared for comparison. Sections that report needs but this dataset
# cannot honestly support (validation history, stability testing, complaint
# records, prior-year audit/CAPA) are kept in the table of contents but
# rendered as explicit "해당 없음" placeholders instead of invented content.


def _feature_category(key: str) -> str:
    """Rough classification used only for the action-items checklist below."""
    if key.startswith("tbl_") or key in ("dissolution_av", "dissolution_min"):
        return "process"
    if key.startswith(("api_", "lactose_", "smcc_", "starch_")):
        return "material"
    return "other"


def _has_dtype_issue(analysis) -> list[str]:
    """Feature columns that came through as non-numeric (a real, found data-quality gap)."""
    frame = analysis.frame
    return [f for f in analysis.feature_names
            if f in frame.columns and not pd.api.types.is_numeric_dtype(frame[f])]


def _yield_low_outliers(analysis):
    y = analysis.frame.get("batch_yield")
    if y is None:
        return 0, 0
    y = y.dropna()
    if y.empty:
        return 0, 0
    q1, q3 = y.quantile(.25), y.quantile(.75)
    low_cut = q1 - 1.5 * (q3 - q1)
    return int((y < low_cut).sum()), len(y)


TOC_ITEMS = [
    ("purpose", "1. 목적"), ("product", "2. 제품의 개요"), ("batches", "3. 제조내역"),
    ("signals", "4. 통계적 신호 현황"), ("change", "5. 변경관리 기록"),
    ("cqa", "6. CQA Review"), ("cpp", "7. CPP Review"), ("yield", "8. 공정수율 결과 분석"),
    ("validation", "9. 밸리데이션 및 생산장비 적격성평가 현황"), ("stability", "10. 안정성시험"),
    ("complaints", "11. 반품·불만 및 회수에 대한 기록"), ("audit", "12. 전년도 실사에 대한 기록"),
    ("capa_prev", "13. 전년도 시정조치에 대한 기록"), ("cases", "14. 조사사례"),
    ("conclusion", "15. 결론"), ("actions", "16. 필요 조치사항"),
    ("references", "17. 관련규정 및 참고문서"), ("attachments", "18. 첨부문서"),
]


def toc_html():
    items = "".join(f"<li><a href='#{anchor}'>{html.escape(title)}</a></li>" for anchor, title in TOC_ITEMS)
    return f"<section class='toc'><h2>목차</h2><ul>{items}</ul></section>"


def section(anchor, title, body):
    return f"<section id='{anchor}'><h2>{html.escape(title)}</h2>{body}</section>"


def na_section(anchor, title, note):
    return section(anchor, title, f"<p class='na'>해당 없음 — {html.escape(note)}</p>")


def cover_html(selected_label, reviewer, review_date, codes):
    doc_no = "PQR-C" + "-".join(str(c) for c in codes[:3]) + ("-외" if len(codes) > 3 else "")
    reviewer_name = html.escape(reviewer) if reviewer else "________________"
    return f"""<section class='cover'>
<h1>제품 품질 검토 보고서</h1><p class='doc-sub'>(Product Quality Review Report)</p>
<table class='doc-meta'><tr><th>문서번호</th><td>{html.escape(doc_no)}</td><th>개정번호</th><td>00</td></tr>
<tr><th>분석 대상</th><td colspan='3'>{html.escape(selected_label)}</td></tr>
<tr><th>검토일</th><td colspan='3'>{html.escape(str(review_date))}</td></tr></table>
<table class='sign-table'><tr><th>구분</th><th>성명</th><th>서명</th><th>서명일자</th></tr>
<tr><td>작성/검토 (QA)</td><td>{reviewer_name}</td><td></td><td></td></tr>
<tr><td>승인 (QA 책임자)</td><td>________________</td><td></td><td></td></tr></table></section>
<section class='revision'><h2>개정 이력</h2><table><tr><th>개정번호</th><th>개정일자</th><th>제정/개정자</th>
<th>개정(검토) 사유</th><th>개정(검토) 내역</th></tr>
<tr><td>00</td><td>{html.escape(str(review_date))}</td><td>{reviewer_name}</td><td>최초 제정</td><td>최초 제정</td></tr></table></section>"""


def purpose_html(selected_label):
    return (f"<p>{html.escape(selected_label)}에 대해 확정된 배치의 공정·품질 기록을 SPC(단변량) 및 MSPC(다변량) 기준으로 "
            "검토하여 통계적 이상신호를 확인하고, 근거가 확보된 항목에 한해 조사·조치 방향을 제시한다.</p>"
            "<p>본 검토는 공식 규격·설비이력 등 일부 데이터가 없는 상태에서 수행한 통계적 신호 스크리닝이며, "
            "공식 일탈(Deviation)·기준일탈(OOS) 판정이나 출하 적합성 판정을 대체하지 않는다.</p>")


def product_overview_html(analyses):
    rows = []
    for a in analyses:
        dates = a.frame["start_date"]
        period = f"{dates.min():%Y-%m} ~ {dates.max():%Y-%m}" if len(dates) else "-"
        rows.append(f"<tr><td>{a.code}</td><td>{a.n}</td><td>{period}</td><td>{len(a.feature_names)}</td>"
                    f"<td>{a.n_components}개 ({a.explained_ratio:.1%})</td></tr>")
    return ("<table><tr><th>제품코드</th><th>분석 배치수</th><th>생산기간</th><th>공정·원료 변수 수</th>"
            "<th>MSPC 주성분(설명분산)</th></tr>" + "".join(rows) + "</table>"
            "<p class='na'>제품명·조성·포장단위 등 제품 마스터 정보는 본 데이터셋에 포함되어 있지 않아 표시하지 않습니다.</p>")


def batch_history_html(analyses):
    blocks = []
    for a in analyses:
        f = a.frame.sort_values("start_date")
        rows = "".join(
            f"<tr><td>{int(r.batch)}</td><td>{r.start}</td><td>{html.escape(str(r.signal_type))}</td>"
            f"<td>{'신호 있음 — 조사후보' if r.integrated_signal else '정상'}</td></tr>"
            for r in f.itertuples()
        )
        blocks.append(f"<h3>코드 {a.code} · {len(f)}배치</h3>"
                       f"<div class='scroll-table'><table><tr><th>배치</th><th>생산월</th><th>신호구분</th>"
                       f"<th>비고</th></tr>{rows}</table></div>")
    return "".join(blocks)


def signals_overview_note():
    return ("<p class='na'>아래는 통계적 관리한계(SPC 관리도, MSPC T²/SPE) 기준의 신호이며, "
            "공식 규격 기준의 일탈(Deviation)이나 기준일탈(OOS) 판정이 아닙니다.</p>")


def cqa_review_html(analyses):
    """Multi-metric CQA summary table — every quality column this dataset actually
    has (not just dissolution), one row per product×metric, mirroring the
    reference PQR's CQA table shape (시험항목별 평균/편차/최댓값/최솟값/판정)."""
    rows = []
    for a in analyses:
        for col in QUALITY_COLUMNS:
            if col not in a.frame or a.frame[col].dropna().empty:
                continue
            d = a.frame[col].dropna()
            if col in ("dissolution_av", "dissolution_min"):
                below_txt = f"{int((d < 80).sum())} / {len(d)}"
            else:
                below_txt = "-"
            rows.append(f"<tr><td>{a.code}</td><td>{html.escape(QUALITY_LABELS.get(col, col))}</td>"
                        f"<td>{d.mean():.2f}</td><td>{d.std():.2f}</td><td>{d.min():.2f}</td><td>{d.max():.2f}</td>"
                        f"<td>{below_txt}</td></tr>")
    table = ("<table><tr><th>제품코드</th><th>품질 특성</th><th>평균</th><th>표준편차</th><th>최솟값</th><th>최댓값</th>"
             "<th>가정 하한(80%) 미만 배치</th></tr>" + "".join(rows) + "</table>")
    note = ("<p class='na'>공식 규격(관리기준)이 문서화되어 있지 않아 Cp/Cpk는 계산하지 않습니다. "
            "'가정 하한 80%'는 용출 특성에 한해 검토 편의를 위해 사용한 참고 기준이며 공식 규격이 아닙니다.</p>")
    return table + note


def cpp_review_note():
    return ("<p class='na'>공식 CPP 관리기준(규격)이 문서화되어 있지 않아 Cp/Cpk 대신, "
            "이상배치와 정상배치 간 표준화 편차(z-score) 차이로 우선 확인할 변수 순위를 제공합니다.</p>")


def yield_review_html(analyses):
    rows = []
    for a in analyses:
        y = a.frame.get("batch_yield")
        if y is None or y.dropna().empty:
            continue
        y = y.dropna()
        low_n, total_n = _yield_low_outliers(a)
        rows.append((a.code, y.mean(), y.std(), y.min(), low_n, total_n))
    if not rows:
        return "<p class='na'>배치 수율 데이터가 없습니다.</p>"
    body = "".join(f"<tr><td>{c}</td><td>{m:.2f}</td><td>{s:.2f}</td><td>{mn:.2f}</td><td>{ln} / {n}</td></tr>"
                   for c, m, s, mn, ln, n in rows)
    return ("<table><tr><th>제품코드</th><th>평균 수율(%)</th><th>표준편차</th><th>최솟값</th>"
            "<th>IQR 기준 저조 배치</th></tr>" + body + "</table>"
            "<p class='na'>공식 수율 기준(스펙)이 없어 IQR(사분위범위) 기준 저조 배치 건수를 참고 지표로 제공합니다.</p>")


def _category_candidates(analyses, category, feature_label, record_hint):
    """Per-product candidate variables that belong to one checklist category, each
    with its own repeat-deviation value and the record that actually needs
    checking — this is the exact data that used to sit in a second, separate CAPA
    table; nesting it under the checklist row it actually supports removes that
    duplication instead of just recoloring it.

    Shows record_hint (확인할 기록), not action_hint (조치사항): the row above
    already states the action decision ('Yes'/'No'); repeating a per-variable
    action here would just restate that same decision N times. What each
    candidate variable actually adds is the specific evidence trail to go
    verify before anyone acts on the row's judgment.
    """
    rows = []
    for a in analyses:
        factors = candidate_factors(a, 5)
        for _, r in factors.iterrows():
            if _feature_category(r["변수"]) != category:
                continue
            rows.append((a.code, feature_label(r["변수"]), float(r["차이"]), record_hint(r["변수"])))
    return rows


def _candidate_detail_html(rows, multi_product, open_attr):
    """Drill-down for a flagged checklist row, deliberately shaped as a chip list
    rather than another <table> — the point is a different structural format, not
    a different color, so this reads as inline evidence for the row above it
    instead of a second grid competing with the outer table."""
    if not rows:
        return ""
    chips = []
    for code, label, deviation, record in rows:
        prefix = f"코드{code} · " if multi_product else ""
        chips.append(
            "<li class='candidate-chip'>"
            f"<span class='cc-dev-badge'>편차 {deviation:.2f}</span>"
            f"<span class='cc-var'>{html.escape(prefix)}{html.escape(label)}</span>"
            "<span class='cc-arrow'>→</span>"
            f"<span class='cc-action'>{html.escape(record)}</span>"
            "</li>"
        )
    return (f"<details{open_attr}><summary>근거가 된 후보변수 {len(rows)}건 — 변수별 확인할 기록 보기</summary>"
            f"<ul class='candidate-list'>{''.join(chips)}</ul></details>")


def action_items_html(analyses, feature_label, record_hint, expanded=False):
    """The 7-category follow-up checklist from the reference PQR, but every
    Yes/No/판단보류 is computed from this review's actual data instead of
    copied — see the discussion this was designed from.

    공정방법/제품처방·원료 두 항목은 그 판단의 근거가 된 실제 후보변수를 각 행 안에
    접이식으로 붙인다(이전에는 이 후보변수표가 별도 표/expander로 아래에 따로 있었는데,
    같은 근거를 두 번 다른 모양으로 보여주는 중복이었다 — 합쳐서 한 덩어리로 만듦).
    펼치면 나오는 건 변수별 조치사항이 아니라 확인할 기록이다 — 행 자체가 이미
    Yes/No 조치 판단이라, 후보변수마다 조치사항을 또 적으면 같은 판단을 N번
    반복하는 셈이라 대신 그 판단의 근거가 되는 원자료를 보여준다."""
    top_categories = set()
    for a in analyses:
        for key in candidate_factors(a, 3)["변수"]:
            top_categories.add(_feature_category(key))
    dtype_issues = [(a.code, bad) for a in analyses if (bad := _has_dtype_issue(a))]
    yield_flag = any(_yield_low_outliers(a)[0] > 0 for a in analyses)

    process_flag = "process" in top_categories
    material_flag = "material" in top_categories
    dtype_flag = bool(dtype_issues)
    dtype_codes = ", ".join(str(c) for c, _ in dtype_issues)
    multi_product = len(analyses) > 1
    open_attr = " open" if expanded else ""

    def _note(text, category=None):
        escaped = html.escape(text)
        if category is None:
            return escaped
        detail = _candidate_detail_html(
            _category_candidates(analyses, category, feature_label, record_hint), multi_product, open_attr)
        return escaped + detail

    rows = [
        ("공정방법 개선 필요", process_flag,
         _note("이상배치 판별 상위 변수에 타정·정제 물성 관련 항목이 반복적으로 포함됨" if process_flag
               else "상위 후보 변수에 공정 관련 항목이 뚜렷하게 나타나지 않음",
               category="process" if process_flag else None)),
        ("제품처방/원료 개선 필요", material_flag,
         _note("이상배치 판별 상위 변수에 원료(API/부형제) 관련 항목이 포함됨 — "
               "제품처방보다는 원료 규격·공급업체 관리 검토가 더 적합할 수 있음" if material_flag
               else "상위 후보 변수에 원료 관련 항목이 뚜렷하게 나타나지 않음",
               category="material" if material_flag else None)),
        ("공정검사(IPC) 기준 개선 필요", None,
         _note("공식 IPC 관리기준 자체가 데이터에 문서화되어 있지 않아 Yes/No 판단 근거가 없음 — 기준 문서화가 먼저 필요")),
        ("공정수율 기준 개선 필요", yield_flag,
         _note("IQR 기준 저조 배치가 확인됨" if yield_flag else "배치수율이 전반적으로 안정적임")),
        ("시험방법 개선 필요", dtype_flag,
         _note((f"원자료 일부가 숫자가 아닌 문자열로 기록된 항목 확인(코드 {dtype_codes}) — 원자료 입력/기록 방식 점검 필요")
               if dtype_flag else "원자료 형식에서 특이사항이 확인되지 않음")),
        ("재밸리데이션 필요", None, _note("밸리데이션 이력 데이터가 없어 판단 불가")),
        ("기타", True, _note("대부분 CQA/CPP에 공식 규격(관리기준)이 문서화되어 있지 않음 — "
         "규격 확정이 선행되어야 Cp/Cpk 기반 판정이 가능함")),
    ]
    body = "".join(
        f"<tr><td>{html.escape(item)}</td><td>{'Yes' if flag is True else ('No' if flag is False else '판단보류')}</td>"
        f"<td>{note}</td></tr>"
        for item, flag, note in rows
    )
    return (f"<table><tr><th>항목별 조치사항</th><th>판단</th><th>근거</th></tr>{body}</table>"
            "<p class='na'>위 판단은 현재 보유한 데이터 범위 내 통계적 근거이며, "
            "실제 조치 여부는 원인조사 결과에 따라 QA가 최종 결정합니다.</p>")


def capa_dashboard_html(analyses, feature_label, action_hint):
    """CAPA 검토 page's main table on the interactive dashboard: candidate
    variable first, then the action item that follows from it — action_items_html()'s
    7-category checklist (공정방법 개선 필요: Yes/No/... 근거) is the reference
    PQR's own structure and stays in the downloadable report, but three of its
    rows (IPC 기준, 재밸리데이션, 기타) never change — same text regardless of
    which product or batch is selected — which is exactly the kind of thing
    that doesn't belong on an interactive screen. This shows only what this
    review's actual data drives: the real MSPC candidate variables and what to
    do about each one, plus the two checks (수율, 시험방법) that do vary by
    product.

    The 수율/시험방법 checks used to render as a separate bullet list under the
    candidate-variable table — same underlying idea (evidence -> action) in a
    different shape. They're now rows in the same table: '근거' carries the
    per-variable deviation number for MSPC candidates and the plain-text
    finding for these two checks, so every action item on this page shares one
    format instead of two.
    """
    multi_product = len(analyses) > 1
    rows = []
    for a in analyses:
        factors = candidate_factors(a, 5)
        for _, r in factors.iterrows():
            rows.append((a.code, feature_label(r["변수"]), f"{float(r['차이']):.2f} (Z-score 차이)", action_hint(r["변수"])))

    yield_flag = any(_yield_low_outliers(a)[0] > 0 for a in analyses)
    dtype_issues = [(a.code, bad) for a in analyses if (bad := _has_dtype_issue(a))]
    if yield_flag:
        rows.append((None, "공정수율", "IQR 기준 저조 배치 확인됨", "배합·타정 공정 수율 관리기준·폐기량 관리 절차 재검토"))
    if dtype_issues:
        codes = ", ".join(str(c) for c, _ in dtype_issues)
        rows.append((None, "시험방법", f"원자료 일부가 숫자가 아닌 문자열로 기록됨(코드 {codes})", "원자료 입력/기록 방식 점검"))

    if not rows:
        table = "<p class='na'>MSPC 신호가 없어 후보요인을 산출하지 못했습니다.</p>"
    else:
        header = "<th>제품코드</th>" if multi_product else ""
        header += "<th>후보변수</th><th>근거</th><th>조치사항</th>"
        body = "".join(
            (f"<tr><td>{code if code is not None else '-'}</td>" if multi_product else "<tr>") +
            f"<td>{html.escape(label)}</td><td>{html.escape(str(evidence))}</td><td>{html.escape(action)}</td></tr>"
            for code, label, evidence, action in rows
        )
        table = f"<table><tr>{header}</tr>{body}</table>"

    return (table +
            "<p class='na'>위 조치사항은 현재 보유한 데이터 범위 내 통계적 근거이며, "
            "실제 조치 여부는 원인조사 결과에 따라 QA가 최종 결정합니다.</p>")


def references_html():
    items = [
        "MFDS, 완제의약품 제조 및 품질관리기준 가이던스",
        "EU Guidelines, Eudralex Volume 4 Part 1",
        "PIC/S, PE 009 Part 1",
        "ICH Q10, Pharmaceutical Quality System",
        "cGMP Guidelines, 21 CFR Part 211",
    ]
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def attachments_html():
    items = ["조사후보 배치 CSV (본 화면에서 별도 다운로드)"] + \
            [f"{name} (원자료, SHA-256 확인됨)" for name in SOURCE_HASHES]
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def conclusion_html(analyses, cases):
    cqa_lines = []
    for a in analyses:
        d = a.frame["dissolution_av"].dropna()
        below = int((d < 80).sum())
        if below == 0:
            cqa_lines.append(f"코드 {a.code}: 용출평균 {d.mean():.2f}%(표준편차 {d.std():.2f}), 가정 하한(80%) 미만 배치 없음")
        else:
            cqa_lines.append(f"코드 {a.code}: 용출평균 {d.mean():.2f}%, 가정 하한(80%) 미만 {below}배치 확인")
    cpp_lines = []
    for a in analyses:
        factors = candidate_factors(a, 1)
        if factors.empty:
            cpp_lines.append(f"코드 {a.code}: MSPC 신호 없음")
        else:
            cpp_lines.append(f"코드 {a.code}: 이상배치 판별 최상위 변수는 '{factors.iloc[0]['변수']}'")
    if cases:
        names = ", ".join(f"코드{c['code']}/배치{c['batch']}" for c in cases)
        case_line = f"동시 신호(SPC+MSPC) 배치 {len(cases)}건({names})을 14장에서 상세 조사함"
    else:
        case_line = "이번 리뷰 범위에서 동시 신호(SPC+MSPC) 배치는 없었습니다."
    method_note = ("본 검토는 개별 CQA/CPP를 독립적으로 보는 단변량 관리도(SPC)뿐 아니라, 여러 변수의 조합 이상을 함께 "
                   "탐지하는 다변량(MSPC, PCA 기반 T²/SPE) 분석을 병행합니다. 이를 통해 개별 변수는 관리상태여도 "
                   "여러 변수가 함께 이상 패턴을 보이는 배치를 우선순위로 선별하고, 배치별 근거변수를 표준화 편차 기준으로 "
                   "자동 랭킹해 조사 시작점을 제시합니다. CAPA 검토 화면의 조건부 시뮬레이션으로 후보변수 개선 시 "
                   "신호 감소 효과도 사전 확인할 수 있습니다.")
    return (f"<p><b>CQA 결과</b></p><ul>{''.join(f'<li>{html.escape(l)}</li>' for l in cqa_lines)}</ul>"
            f"<p><b>CPP/공정 결과</b></p><ul>{''.join(f'<li>{html.escape(l)}</li>' for l in cpp_lines)}</ul>"
            f"<p><b>조사사례</b><br>{html.escape(case_line)}</p>"
            f"<p><b>분석 방법론</b><br>{html.escape(method_note)}</p>"
            "<p><b>종합 판정</b><br>추가조사 필요 · 원인과 CAPA 효과는 조사와 시행 후 관찰로 확인합니다. "
            "통계적 신호는 공식 규격 부적합(OOS) 또는 출하 판정이 아닙니다.</p>")
