import requests

# District coordinates (lat, lon) for weather fetching
DISTRICT_COORDS = {
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
    "Bulandshahr": (28.40, 77.85),
    "Chandauli": (25.27, 83.27),
    "Deoria": (26.51, 83.78),
    "Etah": (27.56, 78.67),
    "Etawah": (26.78, 79.01),
    "Faizabad": (26.77, 82.14),
    "Farrukhabad": (27.39, 79.58),
    "Fatehpur": (25.93, 80.81),
    "Firozabad": (27.15, 78.39),
    "Ghaziabad": (28.67, 77.45),
    "Ghazipur": (25.58, 83.57),
    "Gonda": (27.13, 81.96),
    "Gorakhpur": (26.76, 83.37),
    "Hamirpur": (25.95, 80.14),
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
    "Shahjahanpur": (27.88, 79.91),
    "Sitapur": (27.56, 80.68),
    "Sonbhadra": (24.68, 82.79),
    "Sultanpur": (26.26, 82.07),
    "Unnao": (26.55, 80.49),
    "Varanasi": (25.32, 83.00),
}


def get_weather(district: str, api_key: str) -> dict:
    """
    Fetch real-time weather for a UP district using OpenWeatherMap API.
    Returns dict with temperature, humidity, rainfall, description.
    """
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None

    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            return None

        # Extract rainfall (last 1 hour if available, else 0)
        rainfall_1h = data.get("rain", {}).get("1h", 0)

        return {
            "temperature": round(data["main"]["temp"], 1),
            "feels_like":  round(data["main"]["feels_like"], 1),
            "humidity":    data["main"]["humidity"],
            "rainfall_1h": rainfall_1h,
            "description": data["weather"][0]["description"].title(),
            "wind_speed":  data["wind"]["speed"],
            "icon":        data["weather"][0]["icon"],
        }

    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None


def get_weather_emoji(description: str) -> str:
    """Return an emoji based on weather description."""
    desc = description.lower()
    if "clear" in desc:       return "☀️"
    if "cloud" in desc:       return "☁️"
    if "rain" in desc:        return "🌧️"
    if "storm" in desc:       return "⛈️"
    if "snow" in desc:        return "❄️"
    if "mist" in desc or "fog" in desc: return "🌫️"
    if "haze" in desc:        return "😶‍🌫️"
    return "🌤️"


def estimate_seasonal_rainfall(temperature: float, humidity: float, season: str) -> float:
    """
    Estimate seasonal rainfall (mm) from current weather + season.
    Used as a proxy when monthly rainfall data isn't available.
    """
    base = humidity * 3.5  # rough proxy

    season_multiplier = {
        "kharif": 2.2,   # monsoon season — high rainfall
        "rabi":   0.6,   # winter — low rainfall
        "zaid":   0.4,   # summer — very low
    }

    mult = season_multiplier.get(season, 1.0)

    # Temperature adjustment: very high temp = more evaporation = less effective rain
    temp_adj = max(0.5, 1 - (max(0, temperature - 35) * 0.02))

    estimated = base * mult * temp_adj
    return round(min(1500, max(100, estimated)), 1)