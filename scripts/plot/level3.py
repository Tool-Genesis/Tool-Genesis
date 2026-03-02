import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# ==========================================
# 1. 暴力美化设置 (大字体 + 加粗)
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 18,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "lines.linewidth": 3,
    "lines.markersize": 10,
})

# ==========================================
# 2. 准备数据 (模拟 Scaling Law 数据)
# ==========================================
# X轴: 参数量 (Billion)
# 必须覆盖几个数量级: 1.5B -> 7B -> 14B -> 32B -> 70B -> 235B
model_sizes = np.array([1.5, 4, 7, 14, 32, 72, 235]) 

# Y轴: Success Rate (SR)
# Direct: 增长较慢 (线性增长)
sr_direct = np.array([5, 8, 12, 18, 22, 28, 35]) 

# Agent: 增长较快 (斜率更大)，在小模型处 Gap 小，大模型处 Gap 大
sr_agent =  np.array([6, 12, 20, 35, 48, 62, 78]) 

# ==========================================
# 3. 核心计算：对数线性回归拟合
# ==========================================
def fit_log_linear(x, y):
    # 在 log(x) 空间进行线性拟合: y = m * log10(x) + c
    # 这样在半对数坐标系(semilogx)下就是一条直线
    log_x = np.log10(x)
    m, c = np.polyfit(log_x, y, 1)
    return m, c

# 拟合 Direct
m_d, c_d = fit_log_linear(model_sizes, sr_direct)
# 拟合 Agent
m_a, c_a = fit_log_linear(model_sizes, sr_agent)

# 生成平滑的绘图线 (用于画直线和阴影)
x_smooth = np.logspace(np.log10(1.5), np.log10(300), 100) # 延伸一点到 300B
y_direct_fit = m_d * np.log10(x_smooth) + c_d
y_agent_fit = m_a * np.log10(x_smooth) + c_a

# ==========================================
# 4. 绘图 (鳄鱼嘴曲线)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7))

# --- 4.1 画阴影区域 (The Widening Gap) ---
# 这是整张图的灵魂！
ax.fill_between(
    x_smooth,
    y_direct_fit,
    y_agent_fit,
    color="#E3F2FD",
    alpha=0.6,
    label="Repair Gain ($\\Delta$)",
)

# --- 4.2 画散点 (真实数据) ---
# Direct: 灰色圆点
ax.scatter(model_sizes, sr_direct, color='gray', marker='o', s=100, label='_nolegend_', zorder=3)
# Agent: 蓝色三角
ax.scatter(model_sizes, sr_agent, color='#1976D2', marker='^', s=120, label='_nolegend_', zorder=3)

# --- 4.3 画拟合线 ---
# Direct: 灰色虚线
ax.plot(x_smooth, y_direct_fit, color='gray', linestyle='--', label=f'Direct (Slope={m_d:.1f})')
# Agent: 蓝色实线
ax.plot(x_smooth, y_agent_fit, color='#1565C0', linestyle='-', label=f'Code-Agent (Slope={m_a:.1f})')

# ==========================================
# 5. 美化与标注
# ==========================================

# --- 设置 X 轴为 Log Scale ---
ax.set_xscale('log')

# 自定义 X 轴刻度显示 (显示 1B, 10B, 100B 而不是 10^0)
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}B' if x >= 1 else f'{x}B'))
ax.set_xticks([1, 10, 100, 300]) # 强制显示主要刻度

# 标题与标签
ax.set_xlabel('Model Parameters (Log Scale)', fontsize=18, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
ax.set_title('Scaling Law: The Multiplier Effect', fontsize=22, fontweight='bold', pad=15)

# 限制 Y 轴范围
ax.set_ylim(0, 90)
ax.set_xlim(1.2, 350)

# --- 核心标注：箭头与文字 ---
# 在最右侧画一个双头箭头，标注 Gap
last_x = 235
last_y_d = m_d * np.log10(last_x) + c_d
last_y_a = m_a * np.log10(last_x) + c_a
gap_center_x = 235
gap_center_y = (last_y_d + last_y_a) / 2

ax.annotate('', xy=(last_x, last_y_a), xytext=(last_x, last_y_d),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(last_x * 1.1, gap_center_y, 'Widening\nGap', 
        color='red', va='center', fontsize=16, fontweight='bold')

# --- 图例 ---
ax.legend(loc='upper left', frameon=False)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('scaling_multiplier_effect.pdf', bbox_inches='tight')
plt.show()
