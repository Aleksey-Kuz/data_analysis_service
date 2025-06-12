""" Generating data for the example of cars """
import numpy as np
import pandas as pd
from pathlib import Path


def generate_car_data(filepath: Path, num_samples: int):
    """ Generating data for machines with output to a csv file """
    years = np.random.randint(1990, 2025, num_samples)
    kilometers = np.random.randint(0, 500001, num_samples)
    engine_types = np.random.choice(["petrol", "diesel", "electric", "hybrid"], num_samples)
    transmission_types = np.random.choice(["manual", "automatic", "robotic"], num_samples)
    services = np.random.randint(0, 16, num_samples)
    brands = np.random.choice(["Toyota", "BMW", "Mercedes", "Ford", "Hyundai"], num_samples)
    regions = np.random.choice(["North", "South", "Central"], num_samples)
    damage_status = np.random.choice(["none", "minor", "medium", "severe"], num_samples)
    engine_size = np.round(np.random.uniform(1.0, 6.0, num_samples), 1)
    owners_count = np.random.randint(1, 6, num_samples)
    car_colors = np.random.choice(["black", "white", "red", "silver"], num_samples)
    tire_condition = np.random.choice(["new", "average", "worn"], num_samples)

    base_value = 2000000
    value_depreciation = 20000

    costs = base_value - (years - 2000) * value_depreciation - (kilometers // 10000) * 5000

    brand_price_adjustment = {
        "Toyota": 0.1,
        "BMW": -0.2,
        "Mercedes": -0.3,
        "Ford": 0.05,
        "Hyundai": 0
    }
    costs = costs * (1 + np.array([brand_price_adjustment[brand] for brand in brands]))

    costs = costs + services * 20000

    damage_adjustment = {
        "none": 0,
        "minor": -0.05,
        "medium": -0.1,
        "severe": -0.2
    }
    costs = costs * (1 + np.array([damage_adjustment[damage] for damage in damage_status]))

    engine_price_adjustment = {
        "petrol": 0,
        "diesel": -0.05,
        "electric": 0.15,
        "hybrid": 0.1
    }
    costs = costs * (1 + np.array([engine_price_adjustment[engine] for engine in engine_types]))

    failure_prob = 0.1 * (kilometers // 100000) + 0.05 * (owners_count - 1) - 0.03 * services
    failure_prob = np.clip(failure_prob, 0, 1)

    damage_prob_adjustment = {
        "none": 0,
        "minor": 0.05,
        "medium": 0.1,
        "severe": 0.2
    }
    failure_prob += np.array([damage_prob_adjustment[damage] for damage in damage_status])
    failure_prob = np.clip(failure_prob, 0, 1)

    data = pd.DataFrame({
        "year_of_manufacture": years,
        "kilometers": kilometers,
        "engine_type": engine_types,
        "transmission_type": transmission_types,
        "service_visits": services,
        "car_brand": brands,
        "region": regions,
        "body_damage": damage_status,
        "engine_size": engine_size,
        "number_of_owners": owners_count,
        "car_color": car_colors,
        "tire_condition": tire_condition,
        "market_value": costs,
        "failure_probability": failure_prob
    })

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    data["failure_category"] = pd.cut(
        data["failure_probability"],
        bins=bins,
        labels=labels,
        right=False
    )

    data.to_csv(filepath, index=False)


if __name__ == "__main__":
    path_to_file = Path().cwd().parent / "data" / "datasets" / "test_car_csv.csv"
    generate_car_data(path_to_file, 100_000)
