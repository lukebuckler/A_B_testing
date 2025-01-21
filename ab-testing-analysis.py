# Marketing Campaign A/B Testing Analysis

# Import required libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set styling for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Data Loading and Preprocessing
# Load both datasets
control_df = pd.read_csv('control_group.csv', sep=';')
test_df = pd.read_csv('test_group.csv', sep=';')

# Add group identifier
control_df['Group'] = 'Control'
test_df['Group'] = 'Test'

# Convert date format
control_df['Date'] = pd.to_datetime(control_df['Date'], format='%d.%m.%Y')
test_df['Date'] = pd.to_datetime(test_df['Date'], format='%d.%m.%Y')

# Combine datasets for easier analysis
combined_df = pd.concat([control_df, test_df], ignore_index=True)

# Display basic information about the datasets
print("Control Group Summary:")
print(control_df.describe())
print("\nTest Group Summary:")
print(test_df.describe())

# Calculate key metrics
def calculate_metrics(df):
    metrics = {
        'Total Spend': df['Spend [USD]'].sum(),
        'Total Impressions': df['# of Impressions'].sum(),
        'Total Reach': df['Reach'].sum(),
        'Total Purchases': df['# of Purchase'].sum(),
        'CTR': (df['# of Website Clicks'].sum() / df['# of Impressions'].sum()) * 100,
        'CVR': (df['# of Purchase'].sum() / df['# of Website Clicks'].sum()) * 100,
        'CPA': df['Spend [USD]'].sum() / df['# of Purchase'].sum(),
        'ROI': ((df['# of Purchase'].sum() * 50) - df['Spend [USD]'].sum()) / df['Spend [USD]'].sum() * 100  # Assuming $50 per purchase
    }
    return pd.Series(metrics)

control_metrics = calculate_metrics(control_df)
test_metrics = calculate_metrics(test_df)

# Display key metrics comparison
metrics_comparison = pd.DataFrame({
    'Control': control_metrics,
    'Test': test_metrics,
    'Difference %': ((test_metrics - control_metrics) / control_metrics * 100).round(2)
})

print("\nKey Metrics Comparison:")
print(metrics_comparison)

# Visualization Section

# 1. Daily Trends
plt.figure(figsize=(15, 6))
sns.lineplot(data=combined_df, x='Date', y='# of Purchase', hue='Group')
plt.title('Daily Purchases: Control vs Test')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Conversion Funnel
def plot_funnel(control_df, test_df):
    metrics = ['# of Impressions', 'Reach', '# of Website Clicks', 
              '# of Searches', '# of View Content', '# of Add to Cart', '# of Purchase']
    
    control_values = [control_df[metric].mean() for metric in metrics]
    test_values = [test_df[metric].mean() for metric in metrics]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, control_values, width, label='Control')
    plt.bar(x + width/2, test_values, width, label='Test')
    
    plt.xlabel('Funnel Stage')
    plt.ylabel('Average Count')
    plt.title('Marketing Funnel Comparison')
    plt.xticks(x, [m.replace('# of ', '') for m in metrics], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_funnel(control_df, test_df)

# Statistical Testing Section

def perform_ttest(metric):
    t_stat, p_value = stats.ttest_ind(
        control_df[metric],
        test_df[metric]
    )
    return t_stat, p_value

# Test key metrics
metrics_to_test = ['# of Purchase', '# of Website Clicks', '# of Add to Cart']
statistical_results = {}

for metric in metrics_to_test:
    t_stat, p_value = perform_ttest(metric)
    statistical_results[metric] = {
        'Control Mean': control_df[metric].mean(),
        'Test Mean': test_df[metric].mean(),
        't-statistic': t_stat,
        'p-value': p_value,
        'Significant': p_value < 0.05
    }

statistical_df = pd.DataFrame(statistical_results).T
print("\nStatistical Test Results:")
print(statistical_df)

# ROI Analysis
def calculate_roi_metrics(df, cost_per_purchase=50):
    total_revenue = df['# of Purchase'].sum() * cost_per_purchase
    total_cost = df['Spend [USD]'].sum()
    roi = (total_revenue - total_cost) / total_cost * 100
    
    return {
        'Total Revenue': total_revenue,
        'Total Cost': total_cost,
        'ROI %': roi,
        'Cost per Conversion': total_cost / df['# of Purchase'].sum()
    }

control_roi = calculate_roi_metrics(control_df)
test_roi = calculate_roi_metrics(test_df)

roi_comparison = pd.DataFrame({
    'Control': control_roi,
    'Test': test_roi,
    'Difference %': {k: ((test_roi[k] - control_roi[k]) / control_roi[k] * 100) 
                    for k in control_roi.keys()}
})

print("\nROI Analysis:")
print(roi_comparison)

# Confidence Intervals
def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, mean, std_err)
    return mean, ci

metrics_for_ci = ['# of Purchase', '# of Website Clicks', 'Spend [USD]']
ci_results = {}

for metric in metrics_for_ci:
    control_mean, control_ci = calculate_confidence_interval(control_df[metric])
    test_mean, test_ci = calculate_confidence_interval(test_df[metric])
    
    ci_results[metric] = {
        'Control Mean': control_mean,
        'Control CI': control_ci,
        'Test Mean': test_mean,
        'Test CI': test_ci
    }

print("\nConfidence Intervals:")
for metric, results in ci_results.items():
    print(f"\n{metric}:")
    print(f"Control: {results['Control Mean']:.2f} [{results['Control CI'][0]:.2f}, {results['Control CI'][1]:.2f}]")
    print(f"Test: {results['Test Mean']:.2f} [{results['Test CI'][0]:.2f}, {results['Test CI'][1]:.2f}]")

# Final Recommendations
print("\nKey Findings and Recommendations:")
significant_metrics = statistical_df[statistical_df['Significant']].index.tolist()
print(f"1. Statistically significant differences found in: {', '.join(significant_metrics)}")
print(f"2. Overall ROI difference: {roi_comparison['Difference %']['ROI %']:.2f}%")

if roi_comparison['Difference %']['ROI %'] > 0:
    print("Recommendation: Consider implementing the test variant due to improved ROI")
else:
    print("Recommendation: Stick with the control variant or conduct further testing")
