from firebase_config import db

db.collection("test").add({"hello": "firebase"})
print("Firebase connected!")
