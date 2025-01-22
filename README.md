# Marketing Campaign A/B Testing Analysis

Statistical analysis of marketing campaign performance comparing control and test groups using Python.

## Overview

Analyzed marketing campaign effectiveness through:
- Conversion metrics comparison
- Statistical significance testing
- ROI analysis
- Interactive visualization dashboard

## Data Structure

Two CSV files containing daily metrics:
- Campaign Name
- Date
- Spend [USD]
- Impressions
- Reach
- Website Clicks
- Searches
- View Content
- Add to Cart
- Purchase

## Key Metrics Analyzed

- Conversion Rate (CR)
- Click-Through Rate (CTR)
- Cost per Acquisition (CPA)
- Return on Investment (ROI)
- View-to-Cart Rate
- Cart-to-Purchase Rate

## Statistical Methods

1. Hypothesis Testing
   - Null Hypothesis: No difference between control and test groups
   - Alternative Hypothesis: Significant difference exists
   - Methods: t-tests, chi-square tests

2. Confidence Intervals
   - 95% confidence level for key metrics
   - Effect size calculations

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ab-testing-analysis.git

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook ab_testing_analysis.ipynb
```

## Project Structure

```
ab-testing-analysis/
├── notebooks/
│   └── ab_testing_analysis.ipynb
├── data/
│   ├── control_group.csv
│   └── test_group.csv
├── requirements.txt
└── README.md
```

## Key Findings

1. Conversion Metrics
   - CTR: Test group showed X% improvement
   - Conversion Rate: Y% increase in test group
   - Statistical significance: p-value < 0.05

2. Cost Efficiency
   - CPA reduction: Z%
   - ROI improvement: W%

3. User Behavior
   - Higher engagement metrics in test group
   - Improved funnel progression rates

## Visualization Types

- Time series comparisons
- Conversion funnel analysis
- Statistical distribution plots
- ROI comparison charts

## Dependencies

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- plotly
- jupyter

## Future Improvements

1. Additional Analysis
   - Segmentation analysis
   - Long-term impact assessment
   - Multi-variant testing

2. Technical Enhancements
   - Automated reporting
   - Real-time monitoring
   - A/B test size calculator

## Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## License

MIT License

## Contact

Your Name
[your.email@example.com]
Project Link: [https://github.com/yourusername/ab-testing-analysis]