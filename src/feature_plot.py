import matplotlib.pyplot as plt
import pandas as pd

def analyze_zero_variance_feature(df, feature, target):
    """
    Analyzes a near-zero-variance feature against a binary target.
    
    - Frequency table: % of each target class when feature is 0 vs > 0
    - Boxplot: distribution of feature > 0 values split by target class
    """
    
    # ── 1. Frequency table ───────────────────────────────────────────────
    df_temp = df[[feature, target]].copy()
    df_temp["has_value"] = (df_temp[feature] > 0).map({True: f"{feature} > 0",
                                                        False: f"{feature} = 0"})
    freq_table = (
        df_temp.groupby(["has_value", target])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda x: x.div(x.sum(axis=1), axis=0) * 100)
        .round(2)
    )
    freq_table.columns = [f"{target} = {c}" for c in freq_table.columns]
    freq_table["pct_rows"] = (df_temp.groupby("has_value").size() / len(df) * 100).round(2)
    print("── Frequency table (% within each group) ──────────────────")
    print(freq_table.to_string())
    print(f"\nBase rates  →  {target}=0: {(df[target]==0).mean()*100:.2f}%  |  "
          f"{target}=1: {(df[target]==1).mean()*100:.2f}%")

    # ── 2. Boxplot on non-zero values only ───────────────────────────────
    df_nonzero = df_temp[df_temp[feature] > 0]
    n_nonzero  = len(df_nonzero)
    pct_nonzero = n_nonzero / len(df) * 100

    groups = [
        df_nonzero.loc[df_nonzero[target] == cls, feature].dropna()
        for cls in sorted(df_nonzero[target].unique())
    ]
    labels = [f"{target} = {cls}\n(n={len(g):,})"
              for cls, g in zip(sorted(df_nonzero[target].unique()), groups)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#AED6F1", color="#2E86C1"),
               medianprops=dict(color="#E74C3C", linewidth=2),
               flierprops=dict(marker="o", markersize=3,
                               alpha=0.3, color="#7F8C8D"))

    ax.set_title(f"{feature} distribution (non-zero values only)\n"
                 f"{n_nonzero:,} rows ({pct_nonzero:.2f}% of dataset)",
                 fontsize=12)
    ax.set_ylabel(feature)
    ax.set_xlabel(target)
    plt.tight_layout()
    plt.show()

def analyze_categorical_feature(df, feature, target):
    """
    Analyzes a categorical feature against a binary target.
    Shows the default rate per category vs the overall base rate.
    """

    # ── 1. Frequency table ───────────────────────────────────────────────
    freq_table = (
        df.groupby([feature, target])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda x: x.div(x.sum(axis=1), axis=0) * 100)
        .round(2)
    )
    freq_table.columns = [f"{target} = {c}" for c in freq_table.columns]
    freq_table["pct_rows"] = (df.groupby(feature).size() / len(df) * 100).round(2)

    base_rate = (df[target] == 1).mean() * 100
    print("── Frequency table (% within each category) ───────────────")
    print(freq_table.to_string())
    base_rate_0 = (df[target] == 0).mean() * 100
    base_rate_1 = (df[target] == 1).mean() * 100
    print(f"\nBase rates  →  {target}=0: {base_rate_0:.2f}%  |  {target}=1: {base_rate_1:.2f}%")

def plot_continuous_variable_categoric_feature(df, feature, target, figsize=(8, 5)):
    
    grupos = df[feature].dropna().unique()
    data = [df.loc[df[feature] == g, target] for g in grupos]

    colores = ['#B5D4F4', '#F7C1C1', '#C0DD97', '#FAC775', '#F4C0D1']
    colores_borde = ['#185FA5', '#A32D2D', '#3B6D11', '#BA7517', '#993556']

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        vert=True,
        widths=0.4,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', color='black', alpha=0.2, markersize=3)
    )

    for patch, color, border in zip(bp['boxes'], colores, colores_borde):
        patch.set_facecolor(color)
        patch.set_edgecolor(border)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colores[i], edgecolor=colores_borde[i])
        for i in range(len(grupos))
    ]
    ax.legend(handles, grupos, fontsize=11)

    ax.set_xticks(range(1, len(grupos) + 1))
    ax.set_xticklabels(grupos, fontsize=12)
    ax.set_ylabel(target, fontsize=12)
    ax.set_xlabel(feature, fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()

def plot_boxplots_list(df, features, cols=4):
    
    n      = len(features)
    rows = (n + cols - 1) // cols
    actual_cols = min(n, cols)
    
    fig, axes = plt.subplots(rows, actual_cols, figsize=(actual_cols * 4, rows * 4))
    
    # Normalize to 2D array
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif actual_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, feature in enumerate(features):
        row = idx // cols
        col = idx  % cols
        ax  = axes[row, col]
        
        data = df[feature].dropna()
        
        ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            widths=0.4,
            boxprops=dict(facecolor='#B5D4F4', color='#185FA5'),
            medianprops=dict(color='#185FA5', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2),
            flierprops=dict(marker='o', color='black', alpha=0.2, markersize=3)
        )
        
        ax.set_title(feature, fontsize=10)
        ax.set_xticks([])
        ax.grid(True, alpha=0.2, axis='y')
    
    # Hide excess axes if n is not a multiple of cols
    for idx in range(n, rows * actual_cols):
        axes[idx // cols, idx % cols].set_visible(False)
    
    plt.tight_layout(h_pad=4)
    plt.show()

def plot_scatter_list(df, features, target, figsize=(16, 5)):
    
    fig, axes = plt.subplots(1, len(features), figsize=figsize)
    
    if len(features) == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, features):
        ax.scatter(
            df[feature],
            df[target],
            alpha=0.15,
            s=8,
            color='#3266ad'
        )
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(target, fontsize=10)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def plot_categoricals(df, features, figsize_per_plot=(6, 4)):
    
    n           = len(features)
    cols        = min(n, 3)
    rows        = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
    
    # Normalize to 2D array
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, feature in enumerate(features):
        row = idx // cols
        col = idx  % cols
        ax  = axes[row, col]
        
        counts = df[feature].value_counts().sort_values(ascending=False)
        
        ax.bar(counts.index, counts.values, color='#B5D4F4', edgecolor='#185FA5', linewidth=0.8)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel('Category', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.2, axis='y')
    
    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    
    plt.tight_layout(h_pad=4)
    plt.show()