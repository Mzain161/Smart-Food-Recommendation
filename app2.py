from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2
import random
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB Connection
mongo_uri = "mongodb+srv://Muhammad:Muhammad123@cluster0.9lnu2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["FoodRecommendation"]
collection = db["AgricultureData"]

# 🔹 Load Data from MongoDB
df = pd.DataFrame(list(collection.find()))

# Ensure numeric ratings
if "Supplier Rating" not in df.columns:
    df["Supplier Rating"] = np.random.uniform(3.5, 5.0, len(df)).round(1)
else:
    df["Supplier Rating"] = pd.to_numeric(df["Supplier Rating"], errors="coerce").fillna(3.5)

if "Buyer Rating" not in df.columns:
    df["Buyer Rating"] = np.random.uniform(3.5, 5.0, len(df)).round(1)
else:
    df["Buyer Rating"] = pd.to_numeric(df["Buyer Rating"], errors="coerce").fillna(3.5)

# Prepare Encoders & AI model setup
label_encoder = LabelEncoder()
df["category"] = df["category"].fillna("Unknown")
df["Category Encoded"] = label_encoder.fit_transform(df["category"])

# 🔹 Train logistic regression to predict good suppliers
y = (df["Supplier Rating"] >= 4.0).astype(int)
X = df[["Category Encoded", "price_per_kg", "units_sold_kg", "units_on_hand_kg"]]
model = LogisticRegression()
model.fit(X, y)

# 🔹 Train KMeans clustering model
cluster_features = df[["Category Encoded", "price_per_kg", "units_sold_kg", "units_on_hand_kg"]]
kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(cluster_features)

# 🔹 Expanded postcode coordinates for all of West Yorkshire
postcode_coords = {
    "BD1": (53.793, -1.752), "BD2": (53.806, -1.740), "BD3": (53.792, -1.725), "BD4": (53.779, -1.712), "BD5": (53.777, -1.768),
    "BD6": (53.764, -1.790), "BD7": (53.777, -1.784), "BD8": (53.803, -1.778), "BD9": (53.816, -1.796), "BD10": (53.828, -1.716),
    "BD11": (53.748, -1.668), "BD12": (53.748, -1.768), "BD13": (53.771, -1.841), "BD14": (53.790, -1.832), "BD15": (53.805, -1.848),
    "BD16": (53.846, -1.838), "BD17": (53.849, -1.757), "BD18": (53.831, -1.788), "BD19": (53.729, -1.712), "BD20": (53.902, -1.988),
    "BD21": (53.867, -1.906), "BD22": (53.841, -1.945), "BD23": (53.962, -2.017), "BD24": (54.067, -2.277),
    "LS1": (53.7997, -1.5492), "LS2": (53.801, -1.547), "LS3": (53.799, -1.555), "LS4": (53.812, -1.584), "LS5": (53.821, -1.599),
    "LS6": (53.820, -1.572), "LS7": (53.822, -1.532), "LS8": (53.827, -1.512), "LS9": (53.801, -1.509), "LS10": (53.774, -1.515),
    "LS11": (53.783, -1.558), "LS12": (53.791, -1.590), "LS13": (53.802, -1.635), "LS14": (53.827, -1.464), "LS15": (53.807, -1.453),
    "LS16": (53.847, -1.591), "LS17": (53.865, -1.526), "LS18": (53.847, -1.644), "LS19": (53.850, -1.677), "LS20": (53.865, -1.712),
    "LS21": (53.905, -1.693), "LS22": (53.924, -1.383), "LS23": (53.897, -1.325), "LS24": (53.890, -1.234), "LS25": (53.786, -1.231),
    "LS26": (53.756, -1.445), "LS27": (53.750, -1.600), "LS28": (53.808, -1.667), "LS29": (53.916, -1.823),
    "WF1": (53.6833, -1.4977), "WF2": (53.670, -1.520), "WF3": (53.740, -1.510), "WF4": (53.628, -1.532), "WF5": (53.676, -1.590),
    "WF6": (53.707, -1.424), "WF7": (53.678, -1.358), "WF8": (53.686, -1.289), "WF9": (53.612, -1.325), "WF10": (53.715, -1.343),
    "WF11": (53.699, -1.259), "WF12": (53.6900, -1.6300), "WF13": (53.692, -1.641), "WF14": (53.684, -1.659), "WF15": (53.706, -1.712),
    "WF16": (53.712, -1.725), "WF17": (53.715, -1.650),
    "HX1": (53.7200, -1.8600), "HX2": (53.749, -1.904), "HX3": (53.728, -1.835), "HX4": (53.690, -1.875), "HX5": (53.685, -1.841),
    "HX6": (53.707, -1.945), "HX7": (53.743, -2.008),
    "HD1": (53.6458, -1.7850), "HD2": (53.667, -1.778), "HD3": (53.655, -1.816), "HD4": (53.631, -1.810), "HD5": (53.642, -1.754),
    "HD6": (53.700, -1.785), "HD7": (53.636, -1.905), "HD8": (53.602, -1.682), "HD9": (53.590, -1.789)
}

def get_age_group(age):
    if age < 20:
        return "16-20"
    elif age < 26:
        return "21-25"
    elif age < 31:
        return "26-30"
    elif age < 36:
        return "31-35"
    else:
        return "36+"

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@app.route("/", methods=["GET"])
def form():
    return render_template("input_form.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.form
        print("Received data:", data)
        name = data["name"]
        age = int(data["age"])
        occupation = data["occupation"]
        education = data["education"]
        postcode = data["postcode"][:4].upper()  # Normalize postcode
        item = data["item"].lower().strip()
        quantity = float(data["quantity"])  # convert from string to float
        preference = data.get("preference", "rating").lower()
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({"status": "error", "message": str(e)})
    
    age_group = get_age_group(age)

    if postcode not in postcode_coords:
        return jsonify({"error": "Invalid or unsupported postcode in West Yorkshire."})

    user_lat, user_lon = postcode_coords[postcode]

    # Filter by item name (partial match)
    item_df = df[df["product_name"].str.lower().str.strip().str.contains(item)]
    item_df = item_df[item_df["units_on_hand_kg"] >= quantity]
    if item_df.empty:
        return jsonify({"message": "No suppliers have enough stock for this quantity."})

    item_df["Category Encoded"] = label_encoder.transform(item_df["category"])

    # Predict cluster for the current item
    user_item_features = item_df[["Category Encoded", "price_per_kg", "units_sold_kg", "units_on_hand_kg"]]
    predicted_clusters = kmeans.predict(user_item_features)
    cluster_id = predicted_clusters[0]
    item_df = item_df[item_df["cluster"] == cluster_id]
  
    # Score suppliers using Logistic Regression
    item_df["Relevance Score"] = model.predict_proba(
        item_df[["Category Encoded", "price_per_kg", "units_sold_kg", "units_on_hand_kg"]]
    )[:, 1]

    # Calculate distance to farm
    def extract_postcode_code(loc):
        try:
            return loc.split(",")[-1].strip()[:4].upper()
        except:
            return "BD1"

    item_df["farm_postcode"] = item_df["farm_location"].apply(extract_postcode_code)
    item_df["distance_km"] = item_df["farm_postcode"].apply(lambda pc: calculate_distance(
        user_lat, user_lon, *postcode_coords.get(pc, (53.793, -1.752))
    ))

    #  Sort by user preference
    if preference == "price":
        item_df = item_df.sort_values(by=["price_per_kg", "Supplier Rating", "distance_km"], ascending=[True, False, True])
    elif preference == "distance":
        item_df = item_df.sort_values(by=["distance_km", "Supplier Rating", "price_per_kg"], ascending=[True, False, True])
    elif preference == "quality":
        item_df = item_df.sort_values(by=["Relevance Score", "Supplier Rating", "price_per_kg"], ascending=[False, False, True])
    else:  # Default: Rating
        item_df = item_df.sort_values(by=["Supplier Rating", "Relevance Score", "distance_km"], ascending=[False, False, True])

    # Sort by relevance and distance
    top_matches = item_df.sort_values(
        by=["Relevance Score", "Supplier Rating", "distance_km"],
        ascending=[False, False, True]
    ).head(5)

    # Final recommendation logic
    def compute_final_score(row):
        score = (
            row["Supplier Rating"] * 0.4 +
            (1 / (1 + row["distance_km"])) * 0.3 +
            (row["units_on_hand_kg"] >= quantity) * 0.3
        )
        return score

    # Top 5 matches

    top_matches["Final Score"] = top_matches.apply(compute_final_score, axis=1)
    final_recommendation = top_matches.sort_values(by="Final Score", ascending=False).iloc[0]

    results = top_matches[[
    "supplier", "farm_location", "product_name", "price_per_kg", 
    "distance_km", "Supplier Rating", "units_on_hand_kg"
    ]].rename(columns={"units_on_hand_kg": "available_stock_kg"}).to_dict(orient="records")

    # Final recommendation from top 5
    final_result = {
    "supplier": str(final_recommendation["supplier"]),
    "farm_location": str(final_recommendation["farm_location"]),
    "product_name": str(final_recommendation["product_name"]),
    "price_per_kg": float(final_recommendation["price_per_kg"]),
    "distance_km": float(final_recommendation["distance_km"]),
    "Supplier Rating": float(final_recommendation["Supplier Rating"]),
    "available_stock_kg": int(final_recommendation["units_on_hand_kg"])
    }


    print("FINAL RESULT:", final_result)


    return jsonify({
        "status": "success",
        "results": results,
        "final": final_result
    })

if __name__ == "__main__":
    app.run(debug=True)
