"""
About page — project methodology, tech stack, and honest limitations.
Useful reference during your viva.
"""

import streamlit as st

st.set_page_config(page_title="Hotspot Map", page_icon="dashboard/assets/logo.png", layout="wide")

st.markdown("""
<img src="x" style="display:none" onerror="
    const doc = (window.parent.document || document);
    if (!doc.getElementById(&quot;global-sidebar-style&quot;)) {
        const style = doc.createElement(&quot;style&quot;);
        style.id = &quot;global-sidebar-style&quot;;
        style.textContent = &quot;[data-testid=stSidebarNav] span { text-transform: uppercase !important; letter-spacing: 0.5px !important; }&quot;;
        doc.head.appendChild(style);
    }
">
<style>
[data-testid="stSidebarNav"] span {
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)



st.title("ℹ️ About This Project")

st.markdown("""

## Road-Safe Ranchi - Smart Traffic Accident Analytics and Safety recommendation System 

### Project Summary
This project demonstrates a complete data analytics and machine learning pipeline
applied to road safety, focused on **44 major locations in Ranchi, Jharkhand**.

### Methodology
1. **Location Data (Real):** 44 major roads/intersections in Ranchi were geocoded
   using OpenStreetMap's Nominatim API to obtain real latitude/longitude coordinates.
2. **Accident Data (Synthetic):** Since detailed accident-level datasets are not publicly
   available for Ranchi, a Monte Carlo simulation was used to generate ~8,000 accident records, using probability distributions informed by general road
   safety patterns (e.g., higher accident rates during rush hours, elevated severity risk
   in adverse weather, heavier vehicles associated with more severe outcomes).
3. **Database:** All data is stored and queried using MySQL.
4. **Preprocessing:** Data was cleaned, missing values checked, duplicates removed, and
   new features engineered (time-of-day buckets, rush-hour flags, weekend flags, etc.)
5. **EDA:** Statistical analysis and visualizations were used to explore patterns in the data.
6. **Machine Learning:** Three classifiers (Decision Tree, Random Forest, XGBoost) were
   trained to predict accident severity, evaluated using Accuracy, Precision, Recall, and
   F1-Score, with the best model selected for deployment.
7. **Dashboard:** This Streamlit application presents the results interactively.

### Tech Stack
| Layer | Tool |
|---|---|
| Language | Python |
| Geocoding | OpenStreetMap Nominatim (via geopy) |
| Database | MySQL |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Mapping | Folium |
| Dashboard | Streamlit |

### Important Limitations
- **The accident dataset is synthetic**, not sourced from real police/government records.
  It was designed to be statistically realistic, but predictions from this dashboard
  should **not** be treated as reflecting actual real-world accident risk in Ranchi.
- The architecture is intentionally modular: the synthetic dataset generator
  (`simulation/`) can be swapped for a real accident dataset (e.g. from open government
  data portals) without changing the database schema, preprocessing, ML, or dashboard code.
- Some hyperlocal location names may not have resolved via OpenStreetMap and were
  added manually where necessary.

### Future Enhancements
- Integrate real accident records if/when made publicly available
- Add live traffic data via a traffic API
- Extend the model to include weather forecast integration for proactive risk alerts
""")