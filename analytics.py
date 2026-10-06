from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6,
    "jul": 7, "avg": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12,
}

# 공개 데이터에서 배치 전반에 비교적 일관되게 기록된 26개 원료·공정/제품 물성 변수.
# 완제품 이상신호를 만들 때 사용하는 dissolution/impurity 결과는 의도적으로 제외한다.
MODEL_FEATURES = [
    "api_water", "api_total_impurities", "api_l_impurity", "api_content",
    "api_ps01", "api_ps05", "api_ps09", "lactose_water",
    "lactose_sieve0045", "lactose_sieve015", "lactose_sieve025",
    "smcc_water", "smcc_td", "smcc_bd", "smcc_ps01", "smcc_ps05",
    "smcc_ps09", "starch_ph", "starch_water", "tbl_min_thickness",
    "tbl_max_thickness", "tbl_rsd_weight", "tbl_min_hardness",
    "tbl_max_hardness", "tbl_av_hardness", "tbl_yield",
]

QUALITY_COLUMNS = [
    "dissolution_av", "dissolution_min", "resodual_solvent",
    "impurities_total", "impurity_o", "impurity_l", "batch_yield",
]


def _read_csv(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
        return pd.read_csv(io.BytesIO(raw), sep=";")
    return pd.read_csv(source, sep=";")


def parse_start(value) -> pd.Timestamp:
    try:
        month, year = str(value).strip().lower().split(".")
        return pd.Timestamp(year=2000 + int(year), month=MONTH_MAP[month], day=1)
    except Exception:
        return pd.NaT


def load_data(process_source, laboratory_source, normalization_source):
    process = _read_csv(process_source).rename(
        columns={"main_CompForce mean": "main_CompForce_mean"}
    )
    laboratory = _read_csv(laboratory_source)
    normalization = _read_csv(normalization_source).rename(
        columns={"Product code": "code"}
    )

    # Laboratory가 1,005개 완제품 시험 배치의 기준 테이블이다. Process는 일부 배치만
    # 존재하므로 left join으로 보존하고, 동일 이름의 완제품 결과는 Laboratory 값을 쓴다.
    merged = laboratory.merge(
        process, on=["batch", "code"], how="left", suffixes=("", "_process")
    )
    merged = merged.merge(normalization, on="code", how="left")
    merged["start_date"] = merged["start"].map(parse_start)
    merged["code"] = pd.to_numeric(merged["code"], errors="coerce").astype("Int64")
    merged["batch"] = pd.to_numeric(merged["batch"], errors="coerce").astype("Int64")
    return merged, process, laboratory, normalization


def eligible_codes(df: pd.DataFrame, min_batches: int = 27) -> pd.Series:
    counts = df.groupby("code", observed=True)["batch"].nunique().sort_values(ascending=False)
    return counts[counts >= min_batches]


def _jackson_spe_limit(eigenvalues: np.ndarray, retained: int, alpha: float) -> float:
    discarded = np.asarray(eigenvalues[retained:], dtype=float)
    if not len(discarded) or np.allclose(discarded, 0):
        return 0.0
    theta1 = discarded.sum()
    theta2 = np.square(discarded).sum()
    theta3 = np.power(discarded, 3).sum()
    if theta1 <= 0 or theta2 <= 0:
        return 0.0
    h0 = max(0.001, 1 - (2 * theta1 * theta3) / (3 * theta2**2))
    inside = (
        norm.ppf(alpha) * np.sqrt(2 * theta2 * h0**2) / theta1
        + 1
        + theta2 * h0 * (h0 - 1) / theta1**2
    )
    return float(theta1 * max(inside, 1e-9) ** (1 / h0))


@dataclass
class CodeAnalysis:
    frame: pd.DataFrame
    code: int
    n: int
    feature_names: list[str]
    n_components: int
    explained_ratio: float
    spc_center: float
    spc_ucl: float
    spc_lcl: float
    t2_limit: float
    spe_limit: float
    pca_components: np.ndarray
    eigenvalues: np.ndarray


def analyze_code(df: pd.DataFrame, code: int, alpha: float = 0.99) -> CodeAnalysis:
    sub = df[df["code"] == int(code)].copy().sort_values(["start_date", "batch"])
    sub = sub.reset_index(drop=True)
    n = len(sub)

    dissolution = pd.to_numeric(sub["dissolution_av"], errors="coerce")
    mr_bar = dissolution.diff().abs().mean()
    sigma = mr_bar / 1.128 if pd.notna(mr_bar) and mr_bar > 0 else dissolution.std(ddof=1)
    center = dissolution.mean()
    ucl = center + 3 * sigma
    lcl = center - 3 * sigma
    sub["spc_signal"] = (dissolution > ucl) | (dissolution < lcl)

    available = [c for c in MODEL_FEATURES if c in sub.columns and sub[c].notna().sum() >= max(5, int(n * .7))]
    numeric = sub[available].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    variable = [c for c in numeric.columns if numeric[c].nunique(dropna=True) > 1]
    numeric = numeric[variable]

    if n >= 4 and len(variable) >= 2:
        scaled = StandardScaler().fit_transform(numeric)
        full = PCA().fit(scaled)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        retained = int(np.searchsorted(cumulative, 0.80) + 1)
        retained = min(retained, len(variable), n - 2)
        scores = full.transform(scaled)[:, :retained]
        eigen = full.explained_variance_[:retained]
        t2 = np.sum(np.square(scores) / eigen, axis=1)
        reconstruction = scores @ full.components_[:retained, :]
        spe = np.square(scaled - reconstruction).sum(axis=1)
        if n > retained:
            t2_limit = float(
                retained * (n - 1) * (n + 1) / (n * (n - retained))
                * f.ppf(alpha, retained, n - retained)
            )
        else:
            t2_limit = float(np.quantile(t2, alpha))
        spe_limit = _jackson_spe_limit(full.explained_variance_, retained, alpha)
        if not np.isfinite(spe_limit) or spe_limit <= 0:
            spe_limit = float(np.quantile(spe, alpha))
        sub["t2"] = t2
        sub["spe"] = spe
        sub["mspc_signal"] = (t2 > t2_limit) | (spe > spe_limit)
        explained = float(cumulative[retained - 1])
        # 표준화된 절대 편차: 배치 상세와 원인 후보 순위에 사용한다.
        z = pd.DataFrame(scaled, columns=variable, index=sub.index)
        for col in variable:
            sub[f"z__{col}"] = z[col]
    else:
        retained, explained, t2_limit, spe_limit = 0, np.nan, np.nan, np.nan
        full = None
        sub["t2"] = np.nan
        sub["spe"] = np.nan
        sub["mspc_signal"] = False

    sub["integrated_signal"] = sub["spc_signal"] | sub["mspc_signal"]
    sub["signal_type"] = np.select(
        [sub["spc_signal"] & sub["mspc_signal"], sub["spc_signal"], sub["mspc_signal"]],
        ["SPC+MSPC", "SPC", "MSPC"], default="정상"
    )
    return CodeAnalysis(
        frame=sub, code=int(code), n=n, feature_names=variable,
        n_components=retained, explained_ratio=explained,
        spc_center=float(center), spc_ucl=float(ucl), spc_lcl=float(lcl),
        t2_limit=float(t2_limit), spe_limit=float(spe_limit),
        pca_components=(full.components_.copy() if full is not None else np.empty((0, 0))),
        eigenvalues=(full.explained_variance_.copy() if full is not None else np.array([])),
    )


def simulate_capa(analysis: CodeAnalysis, features: list[str], improvement: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move selected standardized variables toward the normal-batch median.

    The original PCA model and control limits stay fixed. Only batches that were
    originally MSPC signals are modified and rescored.
    """
    frame = analysis.frame
    flagged = frame["mspc_signal"].to_numpy()
    z_cols = [f"z__{name}" for name in analysis.feature_names]
    if not flagged.any() or not z_cols or analysis.pca_components.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    z = frame[z_cols].to_numpy(dtype=float)
    normal = ~flagged
    normal_target = np.nanmedian(z[normal], axis=0) if normal.any() else np.zeros(z.shape[1])
    feature_index = {name: i for i, name in enumerate(analysis.feature_names)}
    selected_idx = [feature_index[name] for name in features if name in feature_index]
    adjusted = z.copy()
    fraction = min(max(float(improvement), 0.0), 100.0) / 100.0
    if selected_idx:
        flagged_rows = np.where(flagged)[0]
        adjusted[np.ix_(flagged_rows, selected_idx)] += fraction * (
            normal_target[selected_idx] - adjusted[np.ix_(flagged_rows, selected_idx)]
        )

    scores = adjusted @ analysis.pca_components.T
    retained = analysis.n_components
    retained_scores = scores[:, :retained]
    t2_after = np.sum(np.square(retained_scores) / analysis.eigenvalues[:retained], axis=1)
    reconstructed = retained_scores @ analysis.pca_components[:retained, :]
    spe_after = np.square(adjusted - reconstructed).sum(axis=1)
    after_flag = (t2_after > analysis.t2_limit) | (spe_after > analysis.spe_limit)

    details = frame.loc[flagged, ["batch", "start", "t2", "spe"]].copy()
    details["T² 조정후"] = t2_after[flagged]
    details["SPE 조정후"] = spe_after[flagged]
    details["조정후 판정"] = np.where(after_flag[flagged], "이상 유지", "정상범위")
    details["해소"] = ~after_flag[flagged]

    before_count = int(flagged.sum())
    after_count = int(after_flag[flagged].sum())
    summary = pd.DataFrame([{
        "제품코드": analysis.code,
        "조정 변수": ", ".join(features),
        "정상값 반영률": f"{improvement:.0f}%",
        "조정 전 MSPC 신호": before_count,
        "조정 후 MSPC 신호": after_count,
        "해소 배치": before_count - after_count,
        "해소율(%)": round((before_count - after_count) / before_count * 100, 1) if before_count else 0.0,
        "평균 T² 변화": f"{frame.loc[flagged, 't2'].mean():.2f} → {t2_after[flagged].mean():.2f}",
        "평균 SPE 변화": f"{frame.loc[flagged, 'spe'].mean():.2f} → {spe_after[flagged].mean():.2f}",
    }])
    return summary, details


def candidate_factors(analysis: CodeAnalysis, top_n: int = 8) -> pd.DataFrame:
    frame = analysis.frame
    flagged = frame[frame["mspc_signal"]]
    normal_rows = frame[~frame["mspc_signal"]]
    rows = []
    for feature in analysis.feature_names:
        z_col = f"z__{feature}"
        if z_col not in frame or flagged.empty:
            continue
        flagged_abs = flagged[z_col].abs().mean()
        normal_abs = normal_rows[z_col].abs().mean() if not normal_rows.empty else 0.0
        rows.append({
            "변수": feature,
            "이상배치 평균 |z|": float(flagged_abs),
            "정상배치 평균 |z|": float(normal_abs),
            "차이": float(flagged_abs - normal_abs),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["변수", "이상배치 평균 |z|", "정상배치 평균 |z|", "차이"])
    return result.sort_values(["차이", "이상배치 평균 |z|"], ascending=False).head(top_n).reset_index(drop=True)


def source_quality(merged, process, laboratory, normalization) -> pd.DataFrame:
    process_keys = set(zip(process["batch"], process["code"]))
    lab_keys = set(zip(laboratory["batch"], laboratory["code"]))
    return pd.DataFrame([
        {"데이터": "Laboratory", "행": len(laboratory), "중복키": int(laboratory.duplicated(["batch", "code"]).sum()), "결측셀": int(laboratory.isna().sum().sum()), "용도": "기준 배치·원료·완제품 시험"},
        {"데이터": "Process", "행": len(process), "중복키": int(process.duplicated(["batch", "code"]).sum()), "결측셀": int(process.isna().sum().sum()), "용도": "존재 배치의 공정 보조정보"},
        {"데이터": "Normalization", "행": len(normalization), "중복키": int(normalization.duplicated(["Product code"] if "Product code" in normalization else ["code"]).sum()), "결측셀": int(normalization.isna().sum().sum()), "용도": "제품코드별 배치규모"},
        {"데이터": "결합 결과", "행": len(merged), "중복키": int(merged.duplicated(["batch", "code"]).sum()), "결측셀": int(merged.isna().sum().sum()), "용도": f"Laboratory 보존·Process 매칭 {len(process_keys & lab_keys)}건"},
    ])


def default_paths(base: Path):
    return (
        base / "data" / "Process.csv",
        base / "data" / "Laboratory.csv",
        base / "data" / "Normalization.csv",
    )
