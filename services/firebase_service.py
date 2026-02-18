from firebase_config import db
from datetime import datetime


# -------------------------------
# Save / Update User Profile
# -------------------------------
def save_user_profile(uid: str, profile: dict):
    db.collection("users").document(uid).set(
        {
            "profile": profile,
            "created_at": datetime.utcnow()
        },
        merge=True
    )


# -------------------------------
# Save Meal Plan History
# -------------------------------
def save_meal_plan(uid: str, meal_plan_data: dict):
    """
    Stores each generated meal plan as history
    Path: users/{uid}/meal_plans/{auto_id}
    """
    db.collection("users") \
      .document(uid) \
      .collection("meal_plans") \
      .add(
          {
              **meal_plan_data,
              "created_at": datetime.utcnow()
          }
      )


