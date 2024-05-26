import pandas as pd
pd.set_option('display.max_rows', None)

# Read the csv file into pandas dataframe
importance_df = pd.read_csv('/Users/okiverny/workspace/Kaggle/results_CreditRisk/feature_importance_s1_last3mean_cpu.csv')

# Less important features
mask = importance_df['importance_split']<10

# print dataframe with all rows to see results
print(importance_df)

print(list(importance_df[mask]['feature_name'].values))