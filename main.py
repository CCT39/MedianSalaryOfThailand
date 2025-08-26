import sys

# 第三方套件
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

# 非必要：設定終端機輸出為 UTF-8（避免 Windows cmd 中文亂碼）
sys.stdout.reconfigure(encoding='utf-8')

# 設定 Matplotlib 中文字型
matplotlib.rc('font', family='Microsoft JhengHei')


# 泰國分組薪資資料
data = [
    {"lower": 1, "upper": 2500, "count": 95700},
    {"lower": 2501, "upper": 5500, "count": 674200},
    {"lower": 5501, "upper": 10000, "count": 6230400},
    {"lower": 10001, "upper": 15000, "count": 6016100},
    {"lower": 15001, "upper": np.inf, "count": 5657700},
]

# 數值穩定用的常數
SAFETY_LOWER_BOUND = 1e-15 # 防止underflow的安全下限：保持數值穩定性（numerical stability）用
LOG_OF_TINY_NUM = -39 # 規定ln(極小數)會變成這個值。ln(1e-15)約等於-34.5387764……，這邊設定成比這個小一點
INIT_PARAMS = [8.0, 1.0]  # 初始猜測 μ=8, σ=1

# 圖表用常數
X_AXIS_UPPER_LIMIT = 25000 # X軸的最右邊界限（單位：THB）
PDF_SCALING = 1.2  # 無因次的縮放因子：單純是為了圖表的視覺效果，讓曲線更貼近直方圖。可以根據需要調整或移除


def interval_probability(l, u, mu, sigma):
    """計算在 LogNormal(mu, sigma) 下，薪資落在 [l, u] 的機率。"""
    dist = lognorm(sigma, scale=np.exp(mu))
    cdf_u = 1.0 if np.isinf(u) else dist.cdf(u)
    cdf_l = 0.0 if l < 0 else dist.cdf(l)
    return max(SAFETY_LOWER_BOUND, cdf_u - cdf_l)


def neg_log_likelihood(params, data):
    """計算對數概似函數（取負號，方便做最小化）。"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    ll = 0.0
    for g in data:
        p = interval_probability(g["lower"], g["upper"], mu, sigma)
        ln_val = np.log(p) if p > SAFETY_LOWER_BOUND else LOG_OF_TINY_NUM
        ll += g["count"] * ln_val
    return -ll


def fit_lognormal(data):
    """利用MLE擬合對數常態分布，找到最佳的μ、σ與中位數。"""
    result = minimize(
        neg_log_likelihood,
        INIT_PARAMS,
        args=(data,),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8},
    )
    mu, sigma = result.x
    median = np.exp(mu)  # 對數常態分布的中位數 = exp(mu)
    return mu, sigma, median


def format_million(x, pos):
    """數字格式化：追加千分位之分隔符號。"""
    return f"{int(x):,}"


def plot_distribution(data, mu, sigma, median):
    """繪製分組薪資資料與擬合曲線。"""
    counts = [d["count"] for d in data]

    # 畫圖
    plt.figure(figsize=(10, 6))
    # 畫bar
    midpoints = [
        (d["lower"] + (d["upper"] if not np.isinf(d["upper"]) else X_AXIS_UPPER_LIMIT)) / 2
        for d in data
    ]
    widths = [
        d["upper"] - d["lower"] if not np.isinf(d["upper"]) else X_AXIS_UPPER_LIMIT - d["lower"]
        for d in data
    ]
    plt.bar(midpoints, counts, width=widths, alpha=0.6, label="每月薪資分組資料")

    # 機率密度函數（PDF）
    x = np.linspace(min(d["lower"] for d in data), X_AXIS_UPPER_LIMIT, 1000)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    # 縮放PDF，使其和直方圖高度對齊
    bin_width = np.mean([d["upper"] - d["lower"] for d in data if not np.isinf(d["upper"])])  # 平均組距
    pdf_scaled = pdf * bin_width * sum(d["count"] for d in data) * PDF_SCALING
    plt.plot(x, pdf_scaled, "r-", label="最佳擬合之對數常態分布之機率密度函數")

    # 標記中位數
    plt.axvline(median, color="g", linestyle="--", label=f"中位數（ {median:,.0f} THB）")

    # 格式化x和y軸數字分位
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(format_million))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(format_million))

    # 把圖表的薪資範圍限縮在0到20,000以方便檢視
    plt.xlim(0, X_AXIS_UPPER_LIMIT)

    # 設定標題及各種標籤並輸出
    plt.xlabel("每月薪資 (泰銖 THB)")
    plt.ylabel("人數（人）")
    plt.title("以泰國2023年第3季之薪資分布分組資料擬合的對數常態分布圖")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    mu, sigma, median = fit_lognormal(data)
    print(f"最佳 μ = {mu:.5f}")
    print(f"最佳 σ = {sigma:.5f}")
    print(f"估計中位數 = {median:.2f} THB/月")

    plot_distribution(data, mu, sigma, median)