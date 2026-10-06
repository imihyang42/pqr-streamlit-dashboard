"""공정변수(CPP) 기반 SPC·MSPC → FMEA → What-if.

PPT 17~18쪽과 같은 방법을 대시보드에서 다시 계산한다.
- 이상 신호: 용출 평균 I-MR(SPC) + 공정변수 26개 MSPC(PCA, T²·SPE 99%)
- 조치 대상: FMEA 점수 RPN = S × O × D
- 효과 확인: 선택 변수를 정상 배치 중앙값 쪽으로 p% 옮기고 PCA를 다시 맞춰 MSPC 재계산
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 영문 컬럼 → 한글 이름(PPT와 같은 표기, 순서 = MSPC 변수 순서)
CPP = {
    "tbl_speed_mean": "타정속도 평균", "tbl_speed_change": "타정속도 변화량",
    "tbl_speed_0_duration": "타정 정지시간", "total_waste": "총폐기량",
    "startup_waste": "가동초기 폐기량", "fom_mean": "공급기 유량 평균",
    "fom_change": "공급기 유량 변화량", "SREL_startup_mean": "초기 공급속도 평균",
    "SREL_production_mean": "생산 중 공급속도 평균", "SREL_production_max": "생산 중 공급속도 최대",
    "main_CompForce mean": "주압축력 평균", "main_CompForce_sd": "주압축력 표준편차",
    "main_CompForce_median": "주압축력 중앙값", "pre_CompForce_mean": "예압축력 평균",
    "tbl_fill_mean": "정제 충전깊이 평균", "tbl_fill_sd": "정제 충전깊이 표준편차",
    "cyl_height_mean": "실린더높이 평균", "stiffness_mean": "정제경도 평균",
    "stiffness_max": "정제경도 최대", "stiffness_min": "정제경도 최소",
    "ejection_mean": "배출력 평균", "ejection_max": "배출력 최대", "ejection_min": "배출력 최소",
    "Startup_tbl_fill_maxDifference": "가동초기 충전깊이 최대편차",
    "Startup_main_CompForce_mean": "가동초기 주압축력 평균",
    "Startup_tbl_fill_mean": "가동초기 충전깊이 평균",
}
CPP_COLS = list(CPP)

# D(검출 어려움, 1~10): 현장 검출 체계 자료가 없어 일반 타정 공정 지식으로 둔 가정값.
# 압축력·충전깊이·실린더높이는 타정기가 상시 감시(낮게), 시료시험·별도 추적 항목은 높게.
D_ASSUMED = {
    "tbl_speed_mean": 5, "tbl_speed_change": 6, "tbl_speed_0_duration": 5, "total_waste": 3,
    "startup_waste": 4, "fom_mean": 4, "fom_change": 6, "SREL_startup_mean": 5,
    "SREL_production_mean": 4, "SREL_production_max": 5, "main_CompForce mean": 4,
    "main_CompForce_sd": 6, "main_CompForce_median": 4, "pre_CompForce_mean": 5,
    "tbl_fill_mean": 4, "tbl_fill_sd": 6, "cyl_height_mean": 3, "stiffness_mean": 5,
    "stiffness_max": 6, "stiffness_min": 6, "ejection_mean": 5, "ejection_max": 5,
    "ejection_min": 6, "Startup_tbl_fill_maxDifference": 6, "Startup_main_CompForce_mean": 5,
    "Startup_tbl_fill_mean": 5,
}

# 같은 종류 변수는 묶어서 최고 RPN을 대표값으로 쓴다(PPT 17쪽).
GROUPS = {
    "정제경도": ["stiffness_max", "stiffness_mean", "stiffness_min"],
    "타정속도": ["tbl_speed_mean"],
    "주압축력": ["main_CompForce mean", "main_CompForce_median", "main_CompForce_sd", "Startup_main_CompForce_mean"],
    "배출력": ["ejection_min", "ejection_mean", "ejection_max"],
    "예압축력": ["pre_CompForce_mean"],
    "실린더높이": ["cyl_height_mean"],
    "충전깊이": ["Startup_tbl_fill_mean", "tbl_fill_mean", "tbl_fill_sd", "Startup_tbl_fill_maxDifference"],
    "공급속도": ["SREL_production_mean", "SREL_production_max", "SREL_startup_mean", "fom_mean", "fom_change"],
    "타정속도 변화": ["tbl_speed_change", "tbl_speed_0_duration"],
    "폐기량": ["total_waste", "startup_waste"],
}
# 대표 변수로 표시할 이름(PPT 표기). 점수가 같으면 앞에 있는 변수를 쓴다.
GROUP_LABEL = {
    "정제경도": "정제경도 최대", "타정속도": "타정속도 평균", "주압축력": "주압축력 평균",
    "배출력": "배출력 최소", "예압축력": "예압축력 평균", "실린더높이": "실린더높이 평균",
    "충전깊이": "초기 충전깊이", "공급속도": "생산 중 공급속도",
    "타정속도 변화": "타정속도 변화·정지", "폐기량": "총폐기량",
}

# PPT 18쪽: FMEA 상위 8개 + 이상 기여 큰 4개
EXTRA_4 = ["tbl_speed_0_duration", "tbl_speed_change", "main_CompForce_sd", "total_waste"]
FMEA_8 = [
    "stiffness_max", "tbl_speed_mean", "main_CompForce mean", "ejection_min",
    "pre_CompForce_mean", "cyl_height_mean", "Startup_tbl_fill_mean", "SREL_production_mean",
]

ALPHA = 0.01
N_PC = 9


@dataclass
class ProcessAnalysis:
    code: int
    frame: pd.DataFrame            # batch, start, dissolution_av, SPC/MSPC flag, t2, spe + CPP 26
    n_pc: int
    spc_center: float
    spc_ucl: float
    spc_lcl: float
    t2_limit: float
    spe_limit: float
    contrib_rank: pd.Series        # 변수별 이상 배치 평균 SPE 기여 순위(1이 가장 큼)


def load_process(base: Path):
    process = pd.read_csv(base / "data" / "Process_full.csv", sep=";")
    lab = pd.read_csv(base / "data" / "Laboratory.csv", sep=";")
    return process, lab


def _mspc(X: np.ndarray, n_pc: int = N_PC, alpha: float = ALPHA):
    n = len(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    full = PCA().fit(Xs)
    pca = PCA(n_components=n_pc).fit(Xs)
    scores = pca.transform(Xs)
    lam = pca.explained_variance_
    t2 = np.sum(scores**2 / lam, axis=1)
    t2_limit = (n_pc * (n - 1) * (n + 1)) / (n * (n - n_pc)) * f_dist.ppf(1 - alpha, n_pc, n - n_pc)
    resid = Xs - pca.inverse_transform(scores)
    spe = np.sum(resid**2, axis=1)
    d = full.explained_variance_[n_pc:]
    th1, th2, th3 = d.sum(), (d**2).sum(), (d**3).sum()
    h0 = 1 - 2 * th1 * th3 / (3 * th2**2)
    ca = norm.ppf(1 - alpha)
    spe_limit = th1 * ((ca * np.sqrt(2 * th2 * h0**2) / th1) + 1 + (th2 * h0 * (h0 - 1)) / th1**2) ** (1 / h0)
    return t2, t2_limit, spe, spe_limit, resid**2


def analyze_process(process: pd.DataFrame, lab: pd.DataFrame, code: int = 17) -> ProcessAnalysis:
    merged = process.merge(lab[["batch", "code", "start", "dissolution_av"]], on=["batch", "code"], how="inner")
    sub = merged[merged["code"] == code].dropna(subset=["dissolution_av"]).copy()
    sub = sub.sort_values("batch").reset_index(drop=True)
    x = sub["dissolution_av"].to_numpy(float)
    mr = np.abs(np.diff(x)).mean()
    sigma = mr / 1.128
    center, ucl, lcl = x.mean(), x.mean() + 3 * sigma, x.mean() - 3 * sigma
    sub["spc_signal"] = (x > ucl) | (x < lcl)

    X = sub[CPP_COLS].to_numpy(float)
    t2, t2l, spe, spel, resid_sq = _mspc(X)
    sub["t2"], sub["spe"] = t2, spe
    sub["mspc_signal"] = (t2 > t2l) | (spe > spel)
    flagged = sub["mspc_signal"].to_numpy()
    contrib = pd.Series(resid_sq[flagged].mean(axis=0) if flagged.any() else np.zeros(len(CPP_COLS)), index=CPP_COLS)
    return ProcessAnalysis(
        code=code, frame=sub, n_pc=N_PC, spc_center=float(center), spc_ucl=float(ucl), spc_lcl=float(lcl),
        t2_limit=float(t2l), spe_limit=float(spel),
        contrib_rank=contrib.rank(ascending=False).astype(int),
    )


def _o_score(pct: float) -> int:
    if pct <= 0:
        return 1
    for th, s in [(2, 2), (4, 3), (6, 4), (8, 5), (10, 6), (15, 7), (20, 8), (30, 9)]:
        if pct < th:
            return s
    return 10


def fmea_table(a: ProcessAnalysis) -> pd.DataFrame:
    """변수 26개 FMEA. S: 용출 상관·이상 기여, O: I-MR 한계 밖 비율, D: 가정."""
    f = a.frame
    y = f["dissolution_av"].to_numpy(float)
    rows = []
    for c in CPP_COLS:
        v = f[c].to_numpy(float)
        rho, p = spearmanr(v, y)
        mr = np.abs(np.diff(v)).mean()
        xb = v.mean()
        n_out = int(((v > xb + 2.66 * mr) | (v < xb - 2.66 * mr)).sum())
        s = 9 if p < 0.05 / len(CPP_COLS) else 7 if p < 0.05 else 6 if a.contrib_rank[c] <= 10 else 4
        o = _o_score(n_out / len(f) * 100)
        d = D_ASSUMED[c]
        rows.append({
            "var": c, "변수": CPP[c], "용출 ρ": round(float(rho), 3), "용출 p": float(p),
            "이상 기여 순위": int(a.contrib_rank[c]), "한계 밖 배치": n_out,
            "S": s, "O": o, "D": d, "RPN": s * o * d,
        })
    return pd.DataFrame(rows).sort_values("RPN", ascending=False).reset_index(drop=True)


def fmea_groups(table: pd.DataFrame) -> pd.DataFrame:
    """같은 종류 변수를 묶어 최고 RPN을 대표값으로 → 상위 대상군."""
    idx = table.set_index("var")
    out = []
    for g, members in GROUPS.items():
        sub = idx.loc[[m for m in members if m in idx.index]]
        best = sub["RPN"].max()
        top = sub[sub["RPN"] == best].iloc[0]
        out.append({
            "대상군": g, "대표 변수": top["변수"], "S": int(top["S"]), "O": int(top["O"]),
            "D": int(top["D"]), "RPN": int(best), "용출 ρ": top["용출 ρ"],
            "한계 밖 배치": int(top["한계 밖 배치"]),
        })
    return pd.DataFrame(out).sort_values(["RPN", "S"], ascending=[False, False], kind="stable").reset_index(drop=True)


# 값이 서로 맞아야 하는 짝(최소 ≤ 평균 ≤ 최대). 조치 값 이동 후 순서가 깨지는 배치 수를 함께 센다.
ORDER_TRIPLES = [
    ("stiffness_min", "stiffness_mean", "stiffness_max"),
    ("ejection_min", "ejection_mean", "ejection_max"),
]


def _order_violation(X: np.ndarray) -> np.ndarray:
    v = np.zeros(len(X), bool)
    for lo, mid, hi in ORDER_TRIPLES:
        a, b, c = (CPP_COLS.index(k) for k in (lo, mid, hi))
        v |= (X[:, a] > X[:, b]) | (X[:, b] > X[:, c])
    return v


def whatif(a: ProcessAnalysis, variables: list[str], pct: float):
    """선택 변수를 정상 배치 중앙값 쪽으로 pct% 옮기고, 조치 전 PCA·관리한계(고정)로 MSPC를 다시 계산."""
    f = a.frame
    flag = f["mspc_signal"].to_numpy()
    X0 = f[CPP_COLS].to_numpy(float)
    scaler = StandardScaler().fit(X0)
    pca = PCA(n_components=a.n_pc).fit(scaler.transform(X0))
    lam = pca.explained_variance_
    X = X0.copy()
    for c in variables:
        j = CPP_COLS.index(c)
        med = np.median(X0[~flag, j])
        X[flag, j] = X[flag, j] + (med - X[flag, j]) * pct / 100
    Z = scaler.transform(X)
    sc = pca.transform(Z)
    t2 = np.sum(sc**2 / lam, axis=1)
    spe = np.sum((Z - pca.inverse_transform(sc)) ** 2, axis=1)
    out = (t2 > a.t2_limit) | (spe > a.spe_limit)
    ratio = np.maximum(t2 / a.t2_limit, spe / a.spe_limit)
    viol = _order_violation(X)
    detail = f.loc[flag, ["batch", "start"]].copy()
    detail["조치 전 신호 크기"] = np.maximum(f["t2"].to_numpy() / a.t2_limit, f["spe"].to_numpy() / a.spe_limit)[flag]
    detail["조치 후 신호 크기"] = ratio[flag]
    detail["조치 후"] = np.where(out[flag], "이탈 유지", "정상 복귀")
    return {
        "before": int(flag.sum()), "after": int(out[flag].sum()),
        "new": int((out & ~flag).sum()), "order_violation": int(viol.sum()),
        "total_after": int(out.sum()), "detail": detail.reset_index(drop=True),
    }
