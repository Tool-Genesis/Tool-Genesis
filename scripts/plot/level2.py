import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 暴力美化设置
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "bold",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

# ==========================================
# 2. 准备数据 (Storytelling Data)
# ==========================================
# 假设我们有 4 个代表性模型，从小到大
models = ['Qwen-4B', 'Qwen-32B', 'Llama-70B', 'GPT-4']
x = np.arange(len(models))
width = 0.35  # 柱子的宽度

# 定义错误类型 (从底到顶)
# L1/L2: Surface Errors (编译失败、JSON 错) -> "低级错误"
# L3: Semantic Errors (跑通了但结果错) -> "高级错误"
# L4: Success (全对)

# --- Direct 模式数据 (Direct 通常死在低级错误上) ---
# 数据加起来必须等于 100
direct_surface = [80, 60, 40, 20]  # 小模型大部分是低级错误
direct_semantic = [15, 25, 30, 30] # 逻辑错误被掩盖了
direct_success  = [ 5, 15, 30, 50]

# --- Agent 模式数据 (Agent 修复了低级错误，暴露了逻辑错误) ---
agent_surface = [10,  5,  2,  0]   # 视觉冲击：低级错误几乎消失！
agent_semantic = [60, 55, 40, 20]  # 视觉冲击：逻辑错误变多了 (这是好事，说明跑起来了)
agent_success  = [30, 40, 58, 80]  # 成功率大幅提升

# ==========================================
# 3. 绘图 (Side-by-Side Stacked Bars)
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

# 定义颜色：灰/红(低级)，橙(逻辑)，蓝/绿(成功)
colors = ['#E0E0E0', '#FFB347', '#77DD77'] # 浅灰，橙色，粉绿 (比较柔和)
labels = ['Surface Error (L1/L2)', 'Semantic Error (L3)', 'Success (L4)']

# --- 画左柱子 (Direct) ---
p1 = ax.bar(x - width/2 - 0.02, direct_surface, width, label=labels[0], color=colors[0], edgecolor='white')
p2 = ax.bar(x - width/2 - 0.02, direct_semantic, width, bottom=direct_surface, label=labels[1], color=colors[1], edgecolor='white')
p3 = ax.bar(x - width/2 - 0.02, direct_success, width, bottom=np.array(direct_surface)+np.array(direct_semantic), label=labels[2], color=colors[2], edgecolor='white')

# --- 画右柱子 (Agent) ---
# 给右边的柱子加斜线纹理 (hatch='//') 以示区别
# 注意：bottom 的计算是基于 agent 数据的累加
agent_bottom_1 = np.array(agent_surface)
agent_bottom_2 = np.array(agent_surface) + np.array(agent_semantic)

p4 = ax.bar(x + width/2 + 0.02, agent_surface, width, color=colors[0], edgecolor='black', hatch='//', alpha=0.9)
p5 = ax.bar(x + width/2 + 0.02, agent_semantic, width, bottom=agent_bottom_1, color=colors[1], edgecolor='black', hatch='//', alpha=0.9)
p6 = ax.bar(x + width/2 + 0.02, agent_success, width, bottom=agent_bottom_2, color=colors[2], edgecolor='black', hatch='//', alpha=0.9)

# ==========================================
# 4. 美化与标注
# ==========================================
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Failure Mode Shift: From Surface to Semantics', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 100)

# 在图上标出 "Direct" 和 "Agent"
# 这种标注比 Legend 更直观
for i in range(len(models)):
    ax.text(x[i] - width/2, 102, 'Direct', ha='center', va='bottom', fontsize=14, color='gray')
    ax.text(x[i] + width/2, 102, 'Agent', ha='center', va='bottom', fontsize=14, fontweight='bold')

# 自定义 Legend
# 创建两个“空”的图例项来解释实心 vs 斜线
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=colors[0], label='Surface Error (L1/L2)'),
    Patch(facecolor=colors[1], label='Semantic Error (L3)'),
    Patch(facecolor=colors[2], label='Success (L4)'),
    # 分隔线
    Line2D([0], [0], color='white', label=' '), 
    Patch(facecolor='white', edgecolor='black', label='Direct Prompting'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Code-Agent Framework'),
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15) # 给顶部文字和底部图例留空
plt.savefig('failure_shift_bar.pdf', bbox_inches='tight')
plt.show()