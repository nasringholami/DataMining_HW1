import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from src.boxplot2d import boxplot2d

# Load iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add species column
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


df.rename(columns={
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width'
}, inplace=True)


def calculate_statistics(series):
    stats = {}

    stats['missing'] = series.isna().sum()
    stats['min'] = series.min()
    stats['q1'] = series.quantile(0.25)
    stats['med'] = series.median()
    stats['q3'] = series.quantile(0.75)
    stats['p95'] = series.quantile(0.95)
    stats['max'] = series.max()
    stats['mean'] = series.mean()
    stats['range'] = series.max() - series.min()
    stats['iqr'] = stats['q3'] - stats['q1']
    stats['std'] = series.std(ddof=1)
    stats['std_pop'] = series.std(ddof=0)
    stats['mad'] = np.median(np.abs(series - stats['med']))

    return stats


results = []

for species in df['species'].unique():
    sepal_width_data = df[df['species'] == species]['sepal_width']
    stats = calculate_statistics(sepal_width_data)
    stats['label'] = species
    results.append(stats)

# Create final DataFrame with required column order
results_df = pd.DataFrame(results)[
    ['label', 'missing', 'min', 'q1', 'med', 'q3', 'p95',
     'max', 'mean', 'range', 'iqr', 'std', 'std_pop', 'mad']
]

# Save to CSV
results_df.to_csv('statistics.csv', index=False)

print("statistics.csv file has been created successfully.")

# -----------------------------
# Part 2: Correlation Analysis
# -----------------------------

# Compute correlation matrix using Pearson's correlation
corr_matrix = df.drop(columns='species').corr(method='pearson')

# Save correlation matrix to CSV
corr_matrix.to_csv('correlation_matrix.csv')

# Take absolute values of correlation matrix
abs_corr = corr_matrix.abs()

# Remove self-correlation by setting diagonal to NaN
np.fill_diagonal(abs_corr.values, np.nan)

# Find maximum absolute correlation
max_corr_value = abs_corr.max().max()
max_corr_pair = abs_corr.stack().idxmax()

# Find minimum absolute correlation
min_corr_value = abs_corr.min().min()
min_corr_pair = abs_corr.stack().idxmin()

print(f"Maximum absolute correlation: {max_corr_value:.4f}")
print(f"Feature pair: {max_corr_pair[0]} and {max_corr_pair[1]}")

print(f"\nMinimum absolute correlation: {min_corr_value:.4f}")
print(f"Feature pair: {min_corr_pair[0]} and {min_corr_pair[1]}")

# -----------------------------
#  Part 3: Visualization
# -----------------------------

label_counts = df['species'].value_counts()

# Create a bar plot
plt.figure(figsize=(6, 4))
plt.bar(label_counts.index, label_counts.values,
        color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save figure
plt.savefig('../dist/label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# Part 3.2: Histograms
# -----------------------------

features = ['petal_length', 'sepal_width']

plt.figure(figsize=(12, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 2, i)
    plt.hist(df[feature], bins=15, color='skyblue',
             edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('../dist/histograms_separate.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# # 3D Histogram: Petal Length vs Sepal Width
# -----------------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Compute 2D histogram
hist, xedges, yedges = np.histogram2d(
    df['petal_length'], df['sepal_width'], bins=10)

# Flatten coordinates for bar3d
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos, ypos = xpos.ravel(), ypos.ravel()
dz = hist.ravel()
dx = dy = (xedges[1]-xedges[0]) * 0.9

# Plot 3D bars
ax.bar3d(xpos, ypos, np.zeros_like(dz), dx, dy, dz,
         color='lightgreen', edgecolor='black', alpha=0.8)

# Axis labels and title
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Count')
ax.set_title('3D Histogram: Petal Length vs Sepal Width')

plt.savefig('../dist/histogram_3d.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# Box Plots per Species
# -----------------------------


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feature in zip(axes, features):
    df.boxplot(column=feature, by='species', ax=ax)
    ax.set_title(f'{feature} by Species')
    ax.set_xlabel('Species')
    ax.set_ylabel(feature)

# Remove automatic pandas title
fig.suptitle('')

plt.tight_layout()
plt.savefig('../dist/boxplots_by_species.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 2D Box Plot: Petal Length vs Sepal Width
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 6))

boxplot2d(
    df,
    x='petal_length',
    y='sepal_width',
    ax=ax
)

ax.set_title('A 2D Boxplot For "petal length (cm)" And "sepal width (cm)"')

plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('../dist/boxplot_2d.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# Quantile Plot Function
# -----------------------------
def quantile_plot(data, feature, ax):
    sorted_data = np.sort(data[feature])
    n = len(sorted_data)
    quantiles = np.arange(1, n + 1) / (n + 1)

    ax.plot(quantiles, sorted_data, marker='o', linestyle='none')
    ax.set_xlabel('Quantiles')
    ax.set_ylabel(feature)
    ax.set_title(f'Quantile Plot of {feature}')
    ax.grid(True, linestyle='--', alpha=0.6)


# -----------------------------
# Create Quantile Plots
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

quantile_plot(df, 'petal_length', axes[0])
quantile_plot(df, 'sepal_width', axes[1])

plt.tight_layout()
plt.savefig('../dist/quantile_plots.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 2D Scatter Plots (All Pairs)
# -----------------------------
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
pairs = list(itertools.combinations(features, 2))
species = df['species'].unique()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (x_feat, y_feat) in zip(axes, pairs):
    for sp in species:
        subset = df[df['species'] == sp]
        ax.scatter(subset[x_feat], subset[y_feat], label=sp, alpha=0.7)

    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.set_title(f'{x_feat} vs {y_feat}')

# Show legend only once
axes[0].legend()

plt.tight_layout()
plt.savefig('../dist/scatterplots_2d_all_pairs.png',
            dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 3D Scatter Plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for sp in species:
    subset = df[df['species'] == sp]
    ax.scatter(
        subset['sepal_length'],
        subset['sepal_width'],
        subset['petal_length'],
        label=sp,
        alpha=0.7
    )

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('3D Scatter Plot: Sepal & Petal Features')


ax.view_init(elev=20, azim=45)

ax.legend()
plt.tight_layout()
plt.savefig('../dist/scatterplot_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------
# Probability Distributions
# ------------------------------

sns.kdeplot(data=df, x='petal_length', hue='species',
            fill=True, common_norm=False)

plt.tight_layout()
plt.savefig('../dist/petal_length_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()
