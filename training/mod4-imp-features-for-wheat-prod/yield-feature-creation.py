import pandas as pd

area_df = pd.read_csv(
    r'C:\Users\hp\Desktop\projects\agro-scan\mod4-imp-features-for-wheat-prod\new-data\area_harvested.csv')
prod_df = pd.read_csv(
    r'C:\Users\hp\Desktop\projects\agro-scan\mod4-imp-features-for-wheat-prod\new-data\wheat_production.csv')

area_df.rename(columns={'Market Year': 'Year',
               'Area Harvested': 'Area Harvested (1000 HA)'}, inplace=True)
prod_df.rename(columns={'Market Year': 'Year',
               'Production': 'Production (1000 MT)'}, inplace=True)

df = pd.merge(prod_df[['Year', 'Production (1000 MT)']],
              area_df[['Year', 'Area Harvested (1000 HA)']], on='Year')

# Calculate Yield (tons per hectare)
df['Yield_ton_per_HA'] = df['Production (1000 MT)'] / \
    df['Area Harvested (1000 HA)']

df.to_csv("wheat_yield_data.csv", index=False)
