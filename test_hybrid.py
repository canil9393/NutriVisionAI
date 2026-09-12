from hybrid_detector import hybrid_decision


# ============================================================
# Test 1 - Poha
# ============================================================

yolo = [
    {
        "food": "poha",
        "confidence": 0.7328
    }
]

api1 = [
    {
        "food": "Poha",
        "confidence": 0.9987
    }
]

api2 = [
    {
        "food": "Egg",
        "confidence": 0.8897
    }
]


result = hybrid_decision(yolo, api1, api2)


print("=" * 60)
print("NUTRIVISION AI - HYBRID DECISION TEST")
print("=" * 60)

print("\nYOLO11:")
print(yolo)

print("\nAPI 1:")
print(api1)

print("\nAPI 2:")
print(api2)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Food:", result["food"])
print("Confidence:", round(result["confidence"], 2), "%")
print("Reason:", result["reason"])

print("\nScores:")

for food, score in result["scores"].items():
    print(f"  {food}: {round(score, 4)}")