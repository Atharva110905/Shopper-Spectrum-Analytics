# 🛒 Shopper Spectrum
**Customer Segmentation & Product Recommendation Engine**
*Final Year Project — E-Commerce Analytics*

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (if re-training needed)
python utils/pipeline.py

# 3. Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
shopper_spectrum/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   ├── rfm.csv             # RFM scores + cluster labels
│   ├── monthly.csv         # Monthly revenue/orders/customers
│   ├── country.csv         # Country-wise revenue
│   ├── top_products.csv    # Top 20 products by revenue
│   ├── heatmap.csv         # Day × Hour purchase heatmap
│   ├── kpis.json           # Overall KPI metrics
│   ├── elbow.json          # KMeans elbow curve data
│   ├── products.json       # Full product list
│   └── cluster_labels.json # Cluster ID → Segment name mapping
├── models/
│   ├── scaler.pkl          # StandardScaler (fitted)
│   ├── kmeans.pkl          # KMeans model (k=4)
│   └── cosine_sim.pkl      # Item-item cosine similarity matrix
└── utils/
    └── pipeline.py         # Full ETL + ML training pipeline
```

---

## 🧠 ML Techniques

| Module | Algorithm | Purpose |
|---|---|---|
| Customer Segmentation | KMeans (k=4) | Group customers by RFM |
| Recommendation | Item-based Collaborative Filtering (Cosine Similarity) | Similar products |
| Preprocessing | StandardScaler | Normalize RFM features |
| Evaluation | Elbow Method + Silhouette Score | Optimal cluster selection |

---

## 📊 Dataset

- **Source**: UCI Online Retail Dataset  
- **Records**: 541,909 transactions  
- **Period**: Dec 2022 – Dec 2023  
- **Countries**: 38  
- **Customers**: 4,372 (with ID)  
- **Products**: 3,866 unique  

### Cleaning Steps
- Removed cancelled orders (InvoiceNo starting with 'C')
- Removed rows with missing CustomerID
- Removed negative/zero Quantity and UnitPrice
- Deduplicated rows
- **Final clean records**: 392,692

---

## 🎯 Features

### 📊 Dashboard
- Revenue trend (area chart)
- Customer segment donut chart
- Monthly orders & customers
- Country-wise revenue bar chart
- Top 10 products
- Purchase heatmap (Day × Hour)
- Monetary distribution by segment (box plots)

### 👥 Customer Segmentation
- 4 behavioral segments: High-Value, Regular, Occasional, At-Risk
- RFM scatter plot (Recency vs Monetary, sized by Frequency)
- 3D RFM cluster visualization
- Interactive RFM table with segment filter

### 🎯 Product Recommendations
- Item-based collaborative filtering via cosine similarity
- Fuzzy product name matching
- Top-N recommendations (3–10, configurable)
- Product similarity heatmap (top 15 products)

### 📈 EDA Explorer
- Time analysis: Revenue, Orders, Customers over time
- Geo analysis: Country revenue pie + table
- Product analysis: Revenue vs Quantity scatter
- RFM violin distributions by segment
- Elbow curve + silhouette score chart

### 🤖 RFM Predictor
- Real-time segment prediction from R, F, M inputs
- Gauge charts showing customer position vs max values
- Comparison bar chart: customer vs all 4 segment averages
- Business advice per segment

---

## 🎨 Design Philosophy
- Dark-mode first glassmorphic design
- Inter font, custom CSS variables
- Plotly charts with transparent backgrounds
- Premium KPI cards with gradient top-borders
- No Bootstrap, no default Streamlit styling
