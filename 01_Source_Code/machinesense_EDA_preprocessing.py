import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_name = BASE_DIR.parent / "02_Data" / "predictive_maintenance_v3.csv"

print("Uploaded file:", file_name)

df = pd.read_csv(file_name)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print('Dataset shape:', df.shape)

# Save information needed later for temporal evaluation.
evaluation_data = df[
    ['machine_id', 'timestamp', 'rul_hours']
].copy()

print('Evaluation data shape:', evaluation_data.shape)
print(
    'Duplicate machine-time rows:',
    evaluation_data.duplicated(['machine_id', 'timestamp']).sum()
)
print('Missing RUL values:', evaluation_data['rul_hours'].isnull().sum())

print('Dataset shape:', df.shape)
print('Number of machines:', df['machine_id'].nunique())

df.head()

df.info()

df.describe()

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

missing_percent = (df.isnull().sum() / len(df)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values()

missing_percent.plot.barh(color='steelblue')
plt.xlabel('Missing Values (%)')
plt.title('Missing Values by Column')
plt.show()

print('Duplicate rows:', df.duplicated().sum())
print('Duplicate machine-time rows:', df.duplicated(['machine_id', 'timestamp']).sum())

target_count = df['failure_within_24h'].value_counts().sort_index()
target_percent = df['failure_within_24h'].value_counts(normalize=True).sort_index() * 100

target_table = pd.DataFrame({
    'Count': target_count,
    'Percent': target_percent
})
target_table

sns.countplot(x='failure_within_24h', data=df, palette='coolwarm')
plt.xlabel('Failure Within 24 Hours')
plt.ylabel('Count')
plt.title('Target Distribution')
plt.show()

numeric_features = [
    'vibration_rms',
    'temperature_motor',
    'current_phase_avg',
    'pressure_level',
    'rpm',
    'hours_since_maintenance',
    'ambient_temp'
]

df[numeric_features].hist(figsize=(14, 10), bins=30, color='steelblue')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.boxplot(x='failure_within_24h', y='temperature_motor', data=df, ax=axes[0])
sns.boxplot(x='failure_within_24h', y='vibration_rms', data=df, ax=axes[1])
sns.boxplot(x='failure_within_24h', y='current_phase_avg', data=df, ax=axes[2])

axes[0].set_title('Motor Temperature')
axes[1].set_title('Vibration')
axes[2].set_title('Phase Current')

plt.tight_layout()
plt.show()

correlation_data = df[numeric_features + ['failure_within_24h']].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

correlation_data['failure_within_24h'].sort_values(ascending=False)

mode_failure_rate = df.groupby('operating_mode')['failure_within_24h'].mean() * 100
mode_failure_rate.plot.bar(color='orange')
plt.xlabel('Operating Mode')
plt.ylabel('Failure Rate (%)')
plt.title('Failure Rate by Operating Mode')
plt.xticks(rotation=0)
plt.show()

machine_failure_rate = df.groupby('machine_id')['failure_within_24h'].mean() * 100
machine_failure_rate.plot.bar(figsize=(10, 4), color='steelblue')
plt.xlabel('Machine ID')
plt.ylabel('Failure Rate (%)')
plt.title('Failure Rate by Machine')
plt.show()

daily_failure_rate = df.groupby(df['timestamp'].dt.date)['failure_within_24h'].mean() * 100
daily_failure_rate.plot(figsize=(10, 4), marker='o', color='red')
plt.xlabel('Date')
plt.ylabel('Failure Rate (%)')
plt.title('Daily Failure Rate')
plt.xticks(rotation=30)
plt.show()

leakage_columns = [
    'failure_type',
    'rul_hours',
    'estimated_repair_cost'
]

data_without_leakage = df.drop(columns=leakage_columns)

print('Removed columns:', leakage_columns)
print('Remaining columns:', list(data_without_leakage.columns))

last_time = data_without_leakage['timestamp'].max()
cutoff_time = last_time - pd.Timedelta(hours=24)

right_censored_data = data_without_leakage[data_without_leakage['timestamp'] > cutoff_time].copy()
model_data = data_without_leakage[data_without_leakage['timestamp'] <= cutoff_time].copy()

print('Last timestamp:', last_time)
print('Cutoff timestamp:', cutoff_time)
print('Rows saved for review:', len(right_censored_data))
print('Rows used for modeling:', len(model_data))

categorical_features = [
    'machine_type',
    'operating_mode'
]

features = numeric_features + categorical_features
target = 'failure_within_24h'

print(features)

model_data = model_data.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)

group_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_val_index, test_index = next(group_split.split(
    model_data[features],
    model_data[target],
    groups=model_data['machine_id']
))

train_val_data = model_data.iloc[train_val_index].copy()
test_data = model_data.iloc[test_index].copy()

train_list = []
validation_list = []

for machine_id, machine_data in train_val_data.groupby('machine_id'):
    machine_data = machine_data.sort_values('timestamp')
    split_point = int(len(machine_data) * 0.80)

    train_list.append(machine_data.iloc[:split_point])
    validation_list.append(machine_data.iloc[split_point:])

train_data = pd.concat(train_list).reset_index(drop=True)
validation_data = pd.concat(validation_list).reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Create metadata for each data split.
train_metadata = train_data[
    ['machine_id', 'timestamp', target]
].merge(
    evaluation_data,
    on=['machine_id', 'timestamp'],
    how='left'
)

validation_metadata = validation_data[
    ['machine_id', 'timestamp', target]
].merge(
    evaluation_data,
    on=['machine_id', 'timestamp'],
    how='left'
)

test_metadata = test_data[
    ['machine_id', 'timestamp', target]
].merge(
    evaluation_data,
    on=['machine_id', 'timestamp'],
    how='left'
)

print('Train metadata:', train_metadata.shape)
print('Validation metadata:', validation_metadata.shape)
print('Test metadata:', test_metadata.shape)

print(
    'Test machines:',
    sorted(test_metadata['machine_id'].unique())
)

print(
    'Missing metadata values:',
    train_metadata.isnull().sum().sum(),
    validation_metadata.isnull().sum().sum(),
    test_metadata.isnull().sum().sum()
)

split_summary = pd.DataFrame({
    'Data': ['Train', 'Validation', 'Test'],
    'Rows': [len(train_data), len(validation_data), len(test_data)],
    'Machines': [
        train_data['machine_id'].nunique(),
        validation_data['machine_id'].nunique(),
        test_data['machine_id'].nunique()
    ],
    'Failure Rate (%)': [
        train_data[target].mean() * 100,
        validation_data[target].mean() * 100,
        test_data[target].mean() * 100
    ]
})

split_summary

numeric_median = train_data[numeric_features].median()
categorical_mode = train_data[categorical_features].mode().iloc[0]

for data in [train_data, validation_data, test_data]:
    data[numeric_features] = data[numeric_features].fillna(numeric_median)
    data[categorical_features] = data[categorical_features].fillna(categorical_mode)

print(train_data[features].isnull().sum().sum())
print(validation_data[features].isnull().sum().sum())
print(test_data[features].isnull().sum().sum())

X_train = pd.get_dummies(
    train_data[features],
    columns=categorical_features,
    dtype=int
)

X_validation = pd.get_dummies(
    validation_data[features],
    columns=categorical_features,
    dtype=int
)

X_test = pd.get_dummies(
    test_data[features],
    columns=categorical_features,
    dtype=int
)

X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)

train_min = X_train[numeric_features].min()
train_max = X_train[numeric_features].max()
train_range = (train_max - train_min).replace(0, 1)

X_train[numeric_features] = (X_train[numeric_features] - train_min) / train_range
X_validation[numeric_features] = (X_validation[numeric_features] - train_min) / train_range
X_test[numeric_features] = (X_test[numeric_features] - train_min) / train_range

y_train = train_data[target].reset_index(drop=True)
y_validation = validation_data[target].reset_index(drop=True)
y_test = test_data[target].reset_index(drop=True)

print('Missing values after preprocessing:')
print(X_train.isnull().sum().sum())
print(X_validation.isnull().sum().sum())
print(X_test.isnull().sum().sum())

# Check that metadata matches the processed target files.
print('Target alignment:')
print(
    'Train:',
    np.array_equal(train_metadata[target].values, y_train.values)
)
print(
    'Validation:',
    np.array_equal(validation_metadata[target].values, y_validation.values)
)
print(
    'Test:',
    np.array_equal(test_metadata[target].values, y_test.values)
)

# Check the chronological order within each machine.
print('\nChronological order:')
print(
    'Train:',
    train_metadata.groupby('machine_id')['timestamp']
    .apply(lambda values: values.is_monotonic_increasing).all()
)
print(
    'Validation:',
    validation_metadata.groupby('machine_id')['timestamp']
    .apply(lambda values: values.is_monotonic_increasing).all()
)
print(
    'Test:',
    test_metadata.groupby('machine_id')['timestamp']
    .apply(lambda values: values.is_monotonic_increasing).all()
)




original_sensor_data = test_data[
    [
        'machine_id',
        'timestamp',
        'vibration_rms',
        'temperature_motor',
        'current_phase_avg',
        'pressure_level',
        'rpm',
        'hours_since_maintenance',
        'ambient_temp'
    ]
].copy()

# Save one complete result file for the data used in the final test set.
final_result = test_data.copy()
final_result.to_csv('MachineSense_Final_Result.csv', index=False)

print('Final result file saved: MachineSense_Final_Result.csv')
original_sensor_data = test_data[
    [
        'machine_id',
        'timestamp',
        'vibration_rms',
        'temperature_motor',
        'current_phase_avg',
        'pressure_level',
        'rpm',
        'hours_since_maintenance',
        'ambient_temp'
    ]
].copy()

original_sensor_data.to_csv(
    os.path.join(processed_folder, 'test_sensor_data_original.csv'),
    index=False
)

print('Original test sensor data saved.')