import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 暴力美化设置 (字体极大化 + 加粗)
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 22,                
    "axes.titlesize": 26,           
    "xtick.labelsize": 22,          
    "ytick.labelsize": 22,          
})

# ==========================================
# 2. 数据准备 (保持不变)
# ==========================================
display_models = [
    'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 
    'qwen3-30b', 'qwen3-32b', 'qwen3-235b', # 注意：这里顺序要从小到大，配合X轴趋势
]

def format_model_name(name):
    parts = name.split('-')
    # 处理 Qwen3 235B 这种格式
    return f"{parts[0].capitalize()} {parts[1].upper()}"

display_labels = [format_model_name(m) for m in display_models]

models = [
    'deepseek_deepseek-v3-2', 'google_gemini-3-flash-preview', 'moonshotai_kimi-k2',
    'openai_gpt-4-1', 'openai_gpt-4-1-mini', 'openai_gpt-4o', 'openai_gpt-5-1',
    'qwen3-14b', 'qwen3-235b-a22b-instruct-2507', 'qwen3-30b-a3b-instruct-2507',
    'qwen3-32b', 'qwen3-4b', 'qwen3-8b',
]

data_direct = {
    'Schema': [0.2143, 0.0859, 0.2020, 0.1637, 0.4016, 0.3663, 0.1332, 0.0957, 0.1851, 0.1338, 0.15, 0.0000, 0.0111],
    'FVR(Soft)': [0.1634, 0.0792, 0.1205, 0.1014, 0.2974, 0.3046, 0.0933, 0.0707, 0.1224, 0.0751, 0.10, 0.0000, 0.0013],
    'FVR(Hard)': [0.1931, 0.0937, 0.1550, 0.1236, 0.3324, 0.3490, 0.1080, 0.0924, 0.1447, 0.1041, 0.11, 0.0000, 0.0014],
}

data_coder_agent = {
    'Schema': [0.4372, 0.5733, 0.5881, 0.5727, 0.4993, 0.0000, 0.4230, 0.5116, 0.5834, 0.5028, 0.5159, 0.1910, 0.3249],
    'FVR(Soft)': [0.3499, 0.5442, 0.5672, 0.5477, 0.4043, 0.0000, 0.3292, 0.4659, 0.5823, 0.4522, 0.4317, 0.1280, 0.2250],
    'FVR(Hard)': [0.3879, 0.5399, 0.5677, 0.5570, 0.4247, 0.0000, 0.3719, 0.4794, 0.5767, 0.4673, 0.4526, 0.1546, 0.2645],
}

display_to_full = {
    'gemini-3-flash': 'google_gemini-3-flash-preview', 'openai_gpt-5-1': 'openai_gpt-5-1',
    'deepseek-v3.2': 'deepseek_deepseek-v3-2', 'kimi-k2': 'moonshotai_kimi-k2',
    'qwen3-235b': 'qwen3-235b-a22b-instruct-2507', 'qwen3-32b': 'qwen3-32b',
    'qwen3-30b': 'qwen3-30b-a3b-instruct-2507', 'qwen3-14b': 'qwen3-14b',
    'qwen3-8b': 'qwen3-8b', 'qwen3-4b': 'qwen3-4b',
}
full_index = {m: i for i, m in enumerate(models)}
filtered_models = [m for m in display_models if display_to_full.get(m) in full_index]

def _filter_data(data: dict) -> dict:
    out = {}
    for k, vals in data.items():
        out[k] = [vals[full_index[display_to_full[m]]] for m in filtered_models]
    return out

data_direct = _filter_data(data_direct)
data_coder_agent = _filter_data(data_coder_agent)

# 关键变化：这里使用了 .T 进行转置
df_direct = pd.DataFrame(data_direct, index=display_labels).T
df_agent = pd.DataFrame(data_coder_agent, index=display_labels).T

# ==========================================
# 3. 绘图 (改为 2行 1列 布局)
# ==========================================
# figsize 改为宽短型 (16, 10)，适应上下布局
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# 调整子图间距
plt.subplots_adjust(hspace=0.15) 

cmap = "Blues" 
vmin, vmax = 0, 1 

# --- 上图：Direct Prompting ---
sns.heatmap(df_direct, annot=True, fmt=".2f", cmap=cmap, ax=ax[0],
            annot_kws={"size": 22, "weight": "bold"}, 
            vmin=vmin, vmax=vmax, cbar=False,  # 上图不画 Colorbar，省空间
            linewidths=2, linecolor='white')

# 标题放在右侧或者左上
ax[0].set_title('Direct Prompting (Weak Alignment)', fontsize=26, fontweight='bold', pad=15)
ax[0].tick_params(axis='y', rotation=0) # Y轴标签水平放置，易读

# --- 下图：Code-Agent Framework ---
cbar_kws = {
    "shrink": 1.0,           
    "aspect": 15,            
    "pad": 0.02,             
}

sns.heatmap(df_agent, annot=True, fmt=".2f", cmap=cmap, ax=ax[1],
            annot_kws={"size": 22, "weight": "bold"},
            vmin=vmin, vmax=vmax, cbar=True, cbar_kws=cbar_kws,
            linewidths=2, linecolor='white')

ax[1].set_title('Code-Agent Framework (Strong Alignment)', fontsize=26, fontweight='bold', pad=15)
ax[1].tick_params(axis='y', rotation=0)
ax[1].tick_params(axis='x', rotation=0) # X轴模型名水平放置 (如果太长可以设为 15)

# 美化 Colorbar
cbar = ax[1].collections[0].colorbar
cbar.ax.tick_params(labelsize=20, width=2)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight("bold")

# ==========================================
# 4. 最终标签美化
# ==========================================
# 在底部添加 Model Scale 说明箭头
fig.text(0.5, 0.02, 'Model Scale (Small $\\rightarrow$ Large)', 
         ha='center', fontsize=24, fontweight='bold')

# 左侧添加 Intermediate Signals 说明
fig.text(0.02, 0.5, 'Intermediate Signals', 
         va='center', rotation='vertical', fontsize=24, fontweight='bold')

# 调整边距，给标签留位置
plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.12)

plt.savefig('heatmap_transposed_bold.pdf', dpi=300, bbox_inches='tight')
plt.show()