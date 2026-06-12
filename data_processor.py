import pandas as pd
import numpy as np

# Real UP district coordinates (lat, lon)
UP_DISTRICTS = {
    "Agra": (27.18, 78.01),
    "Aligarh": (27.88, 78.08),
    "Allahabad": (25.45, 81.84),
    "Ambedkar Nagar": (26.45, 82.54),
    "Amethi": (26.15, 81.82),
    "Amroha": (28.90, 78.47),
    "Auraiya": (26.47, 79.51),
    "Azamgarh": (26.07, 83.19),
    "Badaun": (28.03, 79.12),
    "Baghpat": (28.95, 77.22),
    "Bahraich": (27.57, 81.59),
    "Ballia": (25.76, 84.14),
    "Balrampur": (27.43, 82.18),
    "Banda": (25.48, 80.34),
    "Barabanki": (26.92, 81.18),
    "Bareilly": (28.36, 79.41),
    "Basti": (26.80, 82.73),
    "Bijnor": (29.37, 78.13),
    "Budaun": (28.03, 79.12),
    "Bulandshahr": (28.40, 77.85),
    "Chandauli": (25.27, 83.27),
    "Chitrakoot": (25.18, 80.89),
    "Deoria": (26.51, 83.78),
    "Etah": (27.56, 78.67),
    "Etawah": (26.78, 79.01),
    "Faizabad": (26.77, 82.14),
    "Farrukhabad": (27.39, 79.58),
    "Fatehpur": (25.93, 80.81),
    "Firozabad": (27.15, 78.39),
    "Gautam Buddha Nagar": (28.52, 77.39),
    "Ghaziabad": (28.67, 77.45),
    "Ghazipur": (25.58, 83.57),
    "Gonda": (27.13, 81.96),
    "Gorakhpur": (26.76, 83.37),
    "Hamirpur": (25.95, 80.14),
    "Hapur": (28.73, 77.78),
    "Hardoi": (27.39, 80.13),
    "Hathras": (27.60, 78.05),
    "Jalaun": (26.14, 79.34),
    "Jaunpur": (25.74, 82.68),
    "Jhansi": (25.45, 78.57),
    "Kannauj": (27.06, 79.91),
    "Kanpur Dehat": (26.41, 79.74),
    "Kanpur Nagar": (26.46, 80.33),
    "Kaushambi": (25.54, 81.38),
    "Kushinagar": (26.74, 83.89),
    "Lakhimpur Kheri": (27.94, 80.77),
    "Lalitpur": (24.69, 78.41),
    "Lucknow": (26.85, 80.95),
    "Maharajganj": (27.13, 83.56),
    "Mahoba": (25.29, 79.87),
    "Mainpuri": (27.23, 79.02),
    "Mathura": (27.49, 77.67),
    "Mau": (25.94, 83.56),
    "Meerut": (28.98, 77.71),
    "Mirzapur": (25.14, 82.56),
    "Moradabad": (28.84, 78.77),
    "Muzaffarnagar": (29.47, 77.69),
    "Pilibhit": (28.63, 79.80),
    "Pratapgarh": (25.90, 81.99),
    "Raebareli": (26.23, 81.24),
    "Rampur": (28.80, 79.02),
    "Saharanpur": (29.97, 77.55),
    "Sambhal": (28.59, 78.57),
    "Sant Kabir Nagar": (26.78, 82.98),
    "Shahjahanpur": (27.88, 79.91),
    "Shamli": (29.45, 77.31),
    "Shravasti": (27.52, 81.84),
    "Siddharthnagar": (27.29, 83.07),
    "Sitapur": (27.56, 80.68),
    "Sonbhadra": (24.68, 82.79),
    "Sultanpur": (26.26, 82.07),
    "Unnao": (26.55, 80.49),
    "Varanasi": (25.32, 83.00),
}


def get_district_data() -> pd.DataFrame:
    """
    Generate realistic synthetic district-level data for UP.
    In a real project you'd load this from CSV / GEE exports.
    """
    np.random.seed(42)
    records = []
    for district, (lat, lon) in UP_DISTRICTS.items():
        # Western UP (higher latitude) tends to have better wheat yields
        base_yield = 2.5 + (lat - 24) * 0.08 + np.random.uniform(-0.3, 0.4)
        avg_ndvi   = 0.35 + (lat - 24) * 0.012 + np.random.uniform(-0.05, 0.08)
        avg_rain   = 700 + np.random.uniform(-200, 300)
        records.append({
            "district":   district,
            "lat":        lat,
            "lon":        lon,
            "avg_yield":  round(max(1.2, base_yield), 2),   # t/ha
            "avg_ndvi":   round(min(0.85, max(0.25, avg_ndvi)), 3),
            "avg_rainfall": round(avg_rain, 1),
        })
    return pd.DataFrame(records)


def calculate_ndvi_stats(ndvi_array: np.ndarray) -> dict:
    """
    Given a 2-D NDVI raster array, return summary statistics.
    """
    valid = ndvi_array[~np.isnan(ndvi_array)]
    return {
        "mean":   float(np.mean(valid)),
        "max":    float(np.max(valid)),
        "min":    float(np.min(valid)),
        "std":    float(np.std(valid)),
        "p25":    float(np.percentile(valid, 25)),
        "p75":    float(np.percentile(valid, 75)),
    }


def generate_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Simulate a labelled training dataset for the ML model.
    Features: ndvi, rainfall, temperature, soil_quality, crop_enc, season_enc
    Target:   yield (t/ha)
    """
    np.random.seed(0)
    crop_map   = {"Wheat": 0, "Rice": 1, "Sugarcane": 2, "Mustard": 3}
    season_map = {"rabi": 0, "kharif": 1, "zaid": 2}

    ndvi        = np.random.uniform(0.15, 0.85, n_samples)
    rainfall    = np.random.uniform(200, 1500, n_samples)
    temperature = np.random.uniform(15, 45, n_samples)
    soil        = np.random.randint(1, 11, n_samples)
    crop        = np.random.choice(list(crop_map.values()), n_samples)
    season      = np.random.choice(list(season_map.values()), n_samples)

    # Synthetic yield formula with noise
    crop_bonus  = np.array([0.3, 0.1, 0.8, 0.0])[crop]
    season_mult = np.array([1.1, 0.95, 0.85])[season]

    yield_val = (
        1.0
        + 3.5 * ndvi
        + 0.001 * rainfall
        - 0.02 * np.abs(temperature - 28)
        + 0.15 * soil
        + crop_bonus
    ) * season_mult + np.random.normal(0, 0.25, n_samples)

    return pd.DataFrame({
        "ndvi":        ndvi,
        "rainfall":    rainfall,
        "temperature": temperature,
        "soil_quality":soil.astype(float),
        "crop":        crop.astype(float),
        "season":      season.astype(float),
        "yield":       np.clip(yield_val, 0.5, 8.0),
    })