# 🌾 Crop Yield Predictor — Uttar Pradesh

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end Machine Learning web application that predicts **district-level crop yield across Uttar Pradesh** using satellite NDVI data, live weather API, and soil quality inputs — built with XGBoost and deployed with Streamlit.

> 🛰️ Inspired by ISRO Geospatial Technology & Applications certification in Remote Sensing and ML.

---

## 🚀 Live Demo

🔗 **[crop-yield-predictor.onrender.com](https://crop-yield-predictor.onrender.com)**

---

## 📸 Screenshots

| Dashboard | Prediction Result |
|---|---|
| Interactive UP district map with all 75 districts | Real-time yield prediction with NDVI gauge |

---

## 🎯 What Does This App Do?

Traditional crop yield estimation happens **after harvest** — too late for planning. This app predicts yield **before harvest** using:

- 🛰️ **Satellite NDVI** — measures crop health from Sentinel-2 imagery
- 🌧️ **Live Weather** — real-time temperature & rainfall via OpenWeatherMap
- 🌱 **Soil Quality** — district-level soil index
- 🤖 **XGBoost ML Model** — trained on 3000 samples, 93% R² accuracy

---

## ✨ Features

- 🗺️ **Interactive UP Map** — all 75 districts plotted with Folium
- ⛅ **Live Weather Fetch** — auto-fills temperature & rainfall for selected district
- 🔮 **Yield Prediction** — predict crop yield in tons/hectare instantly
- 📊 **Analytics Dashboard** — top districts, NDVI vs yield scatter, crop distribution
- 📈 **Historical Trend** — 10-year yield trend per district
- 🌡️ **NDVI Gauge** — visual crop health indicator

---

## 🧠 ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (with StandardScaler pipeline) |
| Training Samples | 3,000 |
| Features | NDVI, Rainfall, Temperature, Soil Quality, Crop Type, Season |
| Target | Yield (tons/hectare) |
| R² Score | **0.93** |
| MAE | **0.21 t/ha** |

---

## 📁 Project Structure

```
crop-yield-predictor/
├── app.py                  # Main Streamlit UI
├── model.py                # XGBoost model training & prediction
├── data_processor.py       # 75 UP districts data + NDVI utilities
├── weather_fetcher.py      # OpenWeatherMap API integration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML | XGBoost, Scikit-learn, NumPy, Pandas |
| Web App | Streamlit |
| Maps | Folium, GeoPandas |
| Charts | Plotly |
| Weather API | OpenWeatherMap |
| Deployment | Render |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ANUSHKASHUKLA09/crop-yield-predictor.git
cd crop-yield-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenWeatherMap API key in app.py
# API_KEY = "your_key_here"

# 5. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501** 🎉

---

## 🌍 Data Sources

| Data | Source |
|---|---|
| Satellite NDVI | Sentinel-2 via Google Earth Engine |
| Crop Yield Stats | data.gov.in / agrimarket.gov.in |
| District Boundaries | GADM / Bhuvan (ISRO) |
| Live Weather | OpenWeatherMap API |
| Rainfall History | IMD (India Meteorological Department) |

---

## 🚀 Deploy on Render (Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Deploy** ✅

---

## 🔮 Future Improvements

- [ ] Real NDVI fetch via Google Earth Engine API (fully automated)
- [ ] Soil moisture index integration
- [ ] District-wise crop recommendation system
- [ ] Mobile responsive UI
- [ ] Multi-state support beyond UP

---

## 👩‍💻 Author

**Anushka Shukla**  
B.Tech CSE | ISRO Certified — Geospatial Technology & ML  
[GitHub](https://github.com/ANUSHKASHUKLA09)

---

## 📝 Resume Description

> Built and deployed an end-to-end ML web app predicting crop yield across 75 UP districts using satellite NDVI data, live weather API, and XGBoost (93% R² accuracy). Features interactive maps, real-time weather integration, and analytics dashboard.
> `Python · XGBoost · Scikit-learn · GeoPandas · Folium · Plotly · Streamlit · Render`

---

