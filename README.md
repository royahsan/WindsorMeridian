<h1>📘 Wiki: Using Windsor.ai Data with Google Meridian</h1>

</br>

🔍 **Overview**

Google Meridian is a Bayesian Marketing Mix Modeling (MMM) library developed by Google. It helps quantify how much different advertising channels contribute to business outcomes like conversions or revenue. This is done through probabilistic modeling that estimates ROI and performance attribution.

Windsor.ai makes it easy to fetch marketing performance data from platforms like Google Ads, Facebook, and others. It saves time by eliminating the need to manually consolidate data, making it ideal for feeding into Meridian for analysis.

In our guide, we use Google, Facebook, Reddit and Bing ads data from Windsor.ai where:
- The KPI is Conversions
- We model how Spend influences these Conversions over time as per our media channels

🐍 **Installation Requirements**

To use the google-meridian package with your data:

✅ Python Version

Make sure you are using Python 3.11 or higher.

📆 Required Libraries

Install the required Python libraries:
```python
pip install google-meridian tensorflow tensorflow-probability pandas requests psutil
```

🔁 **Data Integration Summary**

To use Windsor.ai data with Google Meridian:

1. Fetch the Data: Use Windsor.ai's API to pull ad performance data with fields like date, spend, impressions, clicks, and conversions.

2. Format the Data: Aggregate daily performance metrics, fill missing values, and structure it using CsvDataLoader from the Meridian library.

3. Configure the Model: Define ROI priors based on your expected conversions per dollar spend.

4. Run the Model: Sample from prior and posterior distributions to fit the Bayesian model.

5. Summarize Output: Generate a report that provides ROI estimates and media channel performance.

📌 **Reference**

📚 Google Meridian GitHub: https://github.com/google/meridian

📧 Contact: support@windsor.ai
