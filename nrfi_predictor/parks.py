from __future__ import annotations


PARKS = {
    "Angel Stadium": {"lat": 33.8003, "lon": -117.8827, "run_factor": 1.00},
    "Busch Stadium": {"lat": 38.6226, "lon": -90.1928, "run_factor": 0.98},
    "Chase Field": {"lat": 33.4455, "lon": -112.0667, "run_factor": 1.03},
    "Citi Field": {"lat": 40.7571, "lon": -73.8458, "run_factor": 0.96},
    "Citizens Bank Park": {"lat": 39.9061, "lon": -75.1665, "run_factor": 1.04},
    "Comerica Park": {"lat": 42.3390, "lon": -83.0485, "run_factor": 0.99},
    "Coors Field": {"lat": 39.7559, "lon": -104.9942, "run_factor": 1.18},
    "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "run_factor": 1.00},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "run_factor": 1.05},
    "Globe Life Field": {"lat": 32.7473, "lon": -97.0842, "run_factor": 1.01},
    "Great American Ball Park": {"lat": 39.0974, "lon": -84.5066, "run_factor": 1.06},
    "Guaranteed Rate Field": {"lat": 41.8300, "lon": -87.6338, "run_factor": 1.01},
    "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803, "run_factor": 1.00},
    "loanDepot park": {"lat": 25.7781, "lon": -80.2197, "run_factor": 0.96},
    "Minute Maid Park": {"lat": 29.7573, "lon": -95.3555, "run_factor": 1.00},
    "Nationals Park": {"lat": 38.8730, "lon": -77.0074, "run_factor": 0.99},
    "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "run_factor": 0.93},
    "Oriole Park at Camden Yards": {"lat": 39.2840, "lon": -76.6217, "run_factor": 1.00},
    "PNC Park": {"lat": 40.4469, "lon": -80.0057, "run_factor": 0.97},
    "Petco Park": {"lat": 32.7073, "lon": -117.1566, "run_factor": 0.95},
    "Progressive Field": {"lat": 41.4962, "lon": -81.6852, "run_factor": 1.00},
    "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "run_factor": 1.02},
    "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "run_factor": 0.95},
    "Target Field": {"lat": 44.9817, "lon": -93.2776, "run_factor": 0.98},
    "Tropicana Field": {"lat": 27.7682, "lon": -82.6534, "run_factor": 0.97},
    "Truist Park": {"lat": 33.8907, "lon": -84.4677, "run_factor": 1.02},
    "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "run_factor": 1.03},
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "run_factor": 1.03},
}

DEFAULT_PARK = {"lat": 39.5, "lon": -98.35, "run_factor": 1.0}


def park_info(name: str | None) -> dict[str, float]:
    if not name:
        return DEFAULT_PARK.copy()
    return PARKS.get(name, DEFAULT_PARK).copy()
