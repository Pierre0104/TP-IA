import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency
from IPython.display import display

plt.style.use('ggplot')
sns.set(style="whitegrid")

df = pd.read_csv("data.csv")

print("Aperçu des premières lignes du dataframe :")
display(df.head())

print("\nInformations sur le dataframe :")
display(df.info())

print("\nStatistiques descriptives :")
display(df.describe())

print("\nNombre de valeurs manquantes par colonne :")
display(df.isnull().sum())

print(f"\nLe dataframe contient {df.shape[0]} lignes et {df.shape[1]} colonnes")

num_vars = ['hp', 'attack', 'defense', 'speed']
print("Statistiques descriptives pour les variables numériques sélectionnées :")
display(df[num_vars].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['hp'], kde=True, bins=30)
plt.title('Distribution des points de vie (HP) des Pokémon')
plt.xlabel('HP')
plt.ylabel('Nombre de Pokémon')
plt.savefig('figure1.png')

plt.figure(figsize=(12, 7))
sns.boxplot(data=df[['attack', 'defense', 'sp_attack', 'sp_defense', 'speed']])
plt.title('Distribution des statistiques de combat des Pokémon')
plt.ylabel('Valeur')
plt.grid(True, alpha=0.3)
plt.savefig('figure2.png')

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='attack', y='defense', alpha=0.7)
plt.title('Relation entre Attaque et Défense des Pokémon')
plt.xlabel('Attaque')
plt.ylabel('Défense')
plt.grid(True, alpha=0.3)
plt.savefig('figure3.png')

plt.figure(figsize=(10, 8))
sns.jointplot(data=df, x='sp_attack', y='speed', kind='hex')
plt.suptitle('Relation entre Attaque Spéciale et Vitesse', y=1.02)
plt.tight_layout()
plt.savefig('figure4.png')

fig = px.histogram(df, x='height_m', title='Distribution de la taille des Pokémon',
                  labels={'height_m': 'Taille (m)'}, opacity=0.7)
fig.update_layout(bargap=0.1)
fig.show()

plt.figure(figsize=(14, 10))
cross_tab = pd.crosstab(df['type1'], df['generation'])
sns.heatmap(cross_tab, annot=True, cmap='viridis', fmt='d')
plt.title('Répartition des Types par Génération')
plt.xlabel('Génération')
plt.ylabel('Type Primaire')
plt.savefig('figure5.png')

plt.figure(figsize=(14, 8))
legendary_by_type = pd.crosstab(df['type1'], df['is_legendary'], normalize='index') * 100
legendary_by_type.plot(kind='bar', stacked=True)
plt.title('Pourcentage de Pokémon Légendaires par Type Primaire')
plt.xlabel('Type Primaire')
plt.ylabel('Pourcentage')
plt.legend(title='Légendaire')
plt.xticks(rotation=45)
plt.savefig('figure6.png')

plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='type1', y='attack')
plt.title('Distribution de l\'Attaque par Type Primaire')
plt.xlabel('Type Primaire')
plt.ylabel('Attaque')
plt.xticks(rotation=45)
plt.savefig('figure7.png')

plt.figure(figsize=(12, 8))
stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
leg_stats = df.groupby('is_legendary')[stats].mean().reset_index()
leg_stats_melted = pd.melt(leg_stats, id_vars='is_legendary', value_vars=stats)
sns.barplot(data=leg_stats_melted, x='variable', y='value', hue='is_legendary')
plt.title('Moyenne des Statistiques par Statut Légendaire')
plt.xlabel('Statistique')
plt.ylabel('Valeur Moyenne')
plt.legend(title='Légendaire')
plt.savefig('figure8.png')

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df, x='attack', y='sp_attack', hue='type1', size='is_legendary',
               sizes=(50, 200), alpha=0.7)
plt.title('Relation entre Attaque, Attaque Spéciale, Type et Statut Légendaire')
plt.xlabel('Attaque')
plt.ylabel('Attaque Spéciale')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('figure9.png')

fig = px.scatter_3d(df, x='attack', y='defense', z='speed', 
                   color='type1', symbol='is_legendary', opacity=0.7,
                   title="Relation entre Attaque, Défense et Vitesse par Type et Statut Légendaire")
fig.show()

num_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 
           'height_m', 'weight_kg', 'capture_rate', 'base_egg_steps', 'base_happiness']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

corr_matrix = df[num_cols].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot=True, fmt='.2f')
plt.title('Matrice de Corrélation des Variables Numériques')
plt.tight_layout()
plt.savefig('figure10.png')

print("Les paires de variables les plus corrélées sont :")
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
high_corr = corr_pairs[(corr_pairs < 0.99) & (corr_pairs > 0.5)]
display(high_corr)

plt.figure(figsize=(10, 8))
sns.regplot(data=df, x='height_m', y='weight_kg', scatter_kws={'alpha':0.5})
plt.title('Relation entre Taille et Poids des Pokémon')
plt.xlabel('Taille (m)')
plt.ylabel('Poids (kg)')
plt.grid(True, alpha=0.3)
plt.savefig('figure11.png')

plt.figure(figsize=(10, 8))
sns.regplot(data=df, x='attack', y='sp_attack', scatter_kws={'alpha':0.5})
plt.title('Relation entre Attaque et Attaque Spéciale')
plt.xlabel('Attaque')
plt.ylabel('Attaque Spéciale')
plt.grid(True, alpha=0.3)
plt.savefig('figure12.png')

plt.figure(figsize=(14, 10))
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

for i, stat in enumerate(stats):
    row, col = i // 3, i % 3
    sns.kdeplot(data=df, x=stat, hue='is_legendary', ax=axes[row, col], fill=True, common_norm=False)
    axes[row, col].set_title(f'Distribution de {stat} par Statut Légendaire')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure13.png')

contingency_table = pd.crosstab(df['type1'], df['generation'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Résultat du test Chi2 entre type1 et generation:")
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}, degrés de liberté = {dof}")
if p < 0.05:
    print("Il existe une relation significative entre le type primaire et la génération (p < 0.05)")
else:
    print("Il n'y a pas de relation significative entre le type primaire et la génération (p > 0.05)")

stats_by_type = df.groupby('type1')[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean()
stats_by_generation = df.groupby('generation')[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean()

with pd.ExcelWriter('pokemon_analysis_results.xlsx') as writer:
    stats_by_type.to_excel(writer, sheet_name='Stats par Type')
    stats_by_generation.to_excel(writer, sheet_name='Stats par Génération')
    corr_matrix.to_excel(writer, sheet_name='Matrice de Corrélation')
    
print("Résultats exportés avec succès dans 'pokemon_analysis_results.xlsx'")