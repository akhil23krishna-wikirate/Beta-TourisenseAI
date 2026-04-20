# Bamberg Crowd Prediction — Flask API

## Folder structure

```
flask/
├── app.py               ← main API (this is the only file you need to edit)
├── requirements.txt     ← Python dependencies
├── gunicorn.conf.py     ← production server config
├── models/
│   ├── model_gabelman.pkl
│   ├── model_maxplatz.pkl
│   ├── model_mustrasse.pkl
│   ├── model_newRathawsOst.pkl
│   ├── model_newRathawsWest.pkl
│   ├── model_oldRathaws.pkl
│   ├── model_sandstrasse.pkl
│   ├── model_touristinformation.pkl
│   └── model_domkranz.pkl
└── data/
    ├── df_model_gabelman.csv
    ├── df_model_maxplatz.csv
    └── ...  (one CSV per zone, same names as the models)
```

## Quick start (development)

```bash
cd flask
pip install -r requirements.txt
python app.py          # runs on http://localhost:5000
```

## Production

```bash
gunicorn -c gunicorn.conf.py app:app
```

## API

### POST /predict

**Request**
```json
{ "target_time": "2026-04-02T10:00:00" }
```

**Response** — full day (06:00–22:00, every 15 min) for all zones
```json
{
  "date": "2026-04-02",
  "data": [
    {
      "time": "2026-04-02 06:00:00",
      "zones": [
        {
          "zone": "gabelman",
          "predicted_people": 42,
          "density_per_1000m2": 1.34,
          "crowd_level": "Medium"
        }
      ]
    }
  ]
}
```

Crowd levels: `Low` (< 1 / 1000 m²) · `Medium` (1–3) · `High` (> 3)

### GET /health

Returns `{ "status": "ok", "zones": [...] }` — useful for container health checks.
