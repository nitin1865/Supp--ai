from services.firebase_service import save_user_profile, save_meal_plan
from auth.firebase_auth import get_current_user
from fastapi import Depends
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import os
import openai
import re
import logging
from dotenv import load_dotenv
from enum import Enum

from fastapi.responses import JSONResponse


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="🥗 Personalized Meal Plan Generator API",
    description="Generate customized meal plans based on user health goals and preferences",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIGURATION ----------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = openai.OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o-mini"

# ---------- ENUMS ----------
class DietType(str, Enum):
    OMNIVORE = "Omnivore"
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"
    PESCATARIAN = "Pescatarian"
    LOW_CARB = "Low-carb/Keto"
    GLUTEN_FREE = "Gluten-free"
    JAIN = "Jain/Sattvic"
    OTHER = "Other"

class ActivityLevel(str, Enum):
    LITTLE = "Little movement"
    WALKING = "Walking, stretching everyday"
    MODERATE = "Exercise 2–3 times a week"
    DAILY = "Daily training"

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

# ---------- PYDANTIC MODELS ----------
class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=5, le=120)
    gender: Gender
    height: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=20, le=300)
    diet: DietType
    activity_level: ActivityLevel
    food_allergies: List[str] = Field(default=[])
    health_goals: List[str] = Field(default=[])
    disease: List[str] = Field(default=[])
    supplement_preferences: List[str] = Field(default=[])
    food_type: Optional[str] = None

class MealPlanRequest(BaseModel):
    user_profile: UserProfile

class MealModificationRequest(BaseModel):
    meal_plan: str
    modification: str = Field(..., min_length=1)

class MealAdjustmentRequest(BaseModel):
    meal_plan: str
    food_name: str = Field(..., min_length=1)
    quantity_grams: int = Field(..., ge=1, le=1000)
    meal_replaced: str

class CalorieInfo(BaseModel):
    found: bool
    calories: Optional[int] = None
    calories_per_100g: Optional[float] = None
    food_name: str
    quantity: int
    fat: Optional[float] = None
    carbs: Optional[float] = None
    protein: Optional[float] = None

class MealPlanResponse(BaseModel):
    meal_plan: str
    bmi: float
    bmr: float
    calorie_range: str
    daily_goal: str

class CalorieInfo(BaseModel):
    found: bool
    calories: Optional[int] = None
    food_name: str
    quantity: int

# ---------- DATA LOADING ----------
@app.on_event("startup")
async def load_data():
    """Load CSV data on startup"""
    try:
        global meals_df, nutrition_df
        
        # Load meals data
        if os.path.exists("meal.csv"):
            meals_df = pd.read_csv("meal.csv")
            if 'Unnamed: 0' in meals_df.columns:
                meals_df = meals_df.drop(columns=['Unnamed: 0'])
            logger.info(f"Loaded {len(meals_df)} meals from meal.csv")
        else:
            logger.warning("meal.csv not found")
            meals_df = pd.DataFrame()
        
        # Load nutrition data
        if os.path.exists("nutrition_data.csv"):
            nutrition_df = pd.read_csv("nutrition_data.csv")
            logger.info(f"Loaded {len(nutrition_df)} food items from nutrition_data.csv")
        else:
            logger.warning("nutrition_data.csv not found")
            nutrition_df = pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# ---------- HELPER FUNCTIONS ----------
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
    if gender.lower().startswith('m'):
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, 1)
    else:
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161, 1)

def get_calories_for_food(food_name: str, quantity_grams: int) -> Dict:
    """Search for calorie information for a specific food"""
    try:
        if nutrition_df.empty:
            return estimate_calories_with_ai(food_name, quantity_grams)
        
        # Search for food (case-insensitive)
        food_matches = nutrition_df[
            nutrition_df['Food Item'].astype(str).str.contains(food_name, case=False, na=False)
        ]
        
        if not food_matches.empty:
            food_row = food_matches.iloc[0]
            
            try:
                calories_per_100g = float(food_row['Calories'])
                fat_per_100g = float(food_row['Fat(g)'])
                carbs_per_100g = float(food_row['Carbs(g)'])
                protein_per_100g = float(food_row['Protein(g)'])
            except (ValueError, TypeError):
                return estimate_calories_with_ai(food_name, quantity_grams)
            
            # Calculate for given quantity
            total_calories = int((calories_per_100g * quantity_grams) / 100)
            total_fat = round((fat_per_100g * quantity_grams) / 100, 2)
            total_carbs = round((carbs_per_100g * quantity_grams) / 100, 2)
            total_protein = round((protein_per_100g * quantity_grams) / 100, 2)
            
            return {
                'found': True,
                'calories': total_calories,
                'calories_per_100g': calories_per_100g,
                'food_name': food_row['Food Item'],
                'quantity': quantity_grams,
                'fat': total_fat,
                'carbs': total_carbs,
                'protein': total_protein
            }
        else:
            return estimate_calories_with_ai(food_name, quantity_grams)
    
    except Exception as e:
        logger.error(f"Error searching for food: {str(e)}")
        return {
            'found': False,
            'calories': None,
            'food_name': food_name,
            'quantity': quantity_grams
        }

def estimate_calories_with_ai(food_name: str, quantity_grams: int) -> Dict:
    """Use OpenAI to estimate calories"""
    try:
        prompt = f"""Estimate the nutritional information for {quantity_grams}g of {food_name}.

Provide ONLY the response in this exact format, no other text:
Calories: [number]
Fat: [number]g
Carbs: [number]g  
Protein: [number]g"""
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response
        calories_match = re.search(r'Calories:\s*(\d+)', response_text)
        fat_match = re.search(r'Fat:\s*([\d.]+)', response_text)
        carbs_match = re.search(r'Carbs:\s*([\d.]+)', response_text)
        protein_match = re.search(r'Protein:\s*([\d.]+)', response_text)
        
        if calories_match:
            return {
                'found': True,
                'calories': int(calories_match.group(1)),
                'calories_per_100g': int((int(calories_match.group(1)) * 100) / quantity_grams),
                'food_name': food_name,
                'quantity': quantity_grams,
                'fat': float(fat_match.group(1)) if fat_match else 0,
                'carbs': float(carbs_match.group(1)) if carbs_match else 0,
                'protein': float(protein_match.group(1)) if protein_match else 0
            }
    except Exception as e:
        logger.error(f"AI estimation failed: {str(e)}")
    
    return {
        'found': False,
        'calories': None,
        'food_name': food_name,
        'quantity': quantity_grams
    }

def filter_meals_by_preferences(meals_data: pd.DataFrame, user_profile: UserProfile) -> pd.DataFrame:
    """Filter meals based on user preferences"""
    if meals_data.empty:
        return meals_data
    
    filtered_df = meals_data.copy()
    
    try:
        name_col = 'Dish Name'
        time_col = 'Meal Time'
        diet_col = 'Diet Type'
        allergen_col = 'Allergens'
        
        # Filter by diet
        diet_type = user_profile.diet.lower()
        if diet_type == 'vegetarian':
            filtered_df = filtered_df[
                filtered_df[diet_col].astype(str).str.lower().str.contains('vegetarian|vegan', na=False)
            ]
        elif diet_type == 'vegan':
            filtered_df = filtered_df[
                filtered_df[diet_col].astype(str).str.lower().str.contains('vegan', na=False)
            ]
        elif diet_type == 'pescatarian':
            filtered_df = filtered_df[
                ~filtered_df[diet_col].astype(str).str.lower().str.contains('non-veg|meat|chicken', na=False)
            ]
        
        # Filter out allergens
        for allergen in user_profile.food_allergies:
            allergen_lower = allergen.lower().strip()
            if allergen_lower and allergen_lower != 'none':
                filtered_df = filtered_df[
                    ~filtered_df[allergen_col].astype(str).str.lower().str.contains(allergen_lower, na=False)
                ]
        
        # Group by meal time and limit
        meal_times = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
        limited_meals = []
        
        for meal_time in meal_times:
            meals_of_time = filtered_df[
                filtered_df[time_col].astype(str).str.contains(meal_time, case=False, na=False)
            ]
            if not meals_of_time.empty:
                limited_meals.append(meals_of_time.head(10))
        
        if not limited_meals:
            result_df = filtered_df.head(50)
        else:
            result_df = pd.concat(limited_meals, ignore_index=True)
        
        result_df = result_df.drop_duplicates(subset=[name_col])
        return result_df if not result_df.empty else filtered_df.head(50)
    
    except Exception as e:
        logger.error(f"Error filtering meals: {str(e)}")
        return filtered_df.head(50)

def generate_meal_plan(user_profile: UserProfile) -> Dict:
    """Generate personalized meal plan"""
    bmi_value = calculate_bmi(user_profile.weight, user_profile.height)
    bmr = calculate_bmr(user_profile.weight, user_profile.height, user_profile.age, user_profile.gender.value)
    
    # Determine calorie goal
    if bmi_value < 18.5:
        calorie_range = f"{int(bmr * 1.15)}-{int(bmr * 1.30)}"
        calorie_goal = "ABOVE BMR"
    elif bmi_value >= 25:
        calorie_range = f"{int(bmr * 0.75)}-{int(bmr * 0.90)}"
        calorie_goal = "BELOW BMR"
    else:
        calorie_range = f"{int(bmr * 0.95)}-{int(bmr * 1.05)}"
        calorie_goal = "AROUND BMR"
    
    # Filter and prepare meals
    filtered_meals = filter_meals_by_preferences(meals_df, user_profile)
    meals_list = filtered_meals.to_string(index=False)
    
    prompt = f"""Expert dietitian: Create 5-meal plan.

PROFILE: {user_profile.name}, {user_profile.age}y, {user_profile.gender.value}
BMI: {bmi_value} ({calorie_goal}) | Calories: {calorie_range}kcal
Diet: {user_profile.diet.value} | Activity: {user_profile.activity_level.value}
Health Goals: {', '.join(user_profile.health_goals) if user_profile.health_goals else 'General wellness'}

MEALS (pick 5):
{meals_list}

OUTPUT (one line each):
Breakfast: [Name] (cal, P, C, F) - reason
Mid-Morning Snack: [Name] (cal, P, C, F) - reason
Lunch: [Name] (cal, P, C, F) - reason
Afternoon Snack: [Name] (cal, P, C, F) - reason
Dinner: [Name] (cal, P, C, F) - reason"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        
        meal_plan_text = response.choices[0].message.content.strip()
        
        return {
            'meal_plan': meal_plan_text,
            'bmi': bmi_value,
            'bmr': bmr,
            'calorie_range': calorie_range,
            'daily_goal': calorie_goal
        }
    
    except Exception as e:
        logger.error(f"Error generating meal plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating meal plan: {str(e)}")

def modify_meal_plan(meal_plan: str, modification: str, meals_data: pd.DataFrame) -> str:
    """Modify existing meal plan"""
    try:
        limited_meals = meals_data.groupby('Meal Time').head(15).to_string(index=False) if 'Meal Time' in meals_data.columns else meals_data.head(50).to_string(index=False)
    except:
        limited_meals = meals_data.head(50).to_string(index=False)
    
    prompt = f"""Dietitian: User wants: "{modification}"

Current plan:
{meal_plan}

Available meals:
{limited_meals}

Replace ONE meal only. Keep calories ±30kcal. Don't repeat.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: [name] (cal) - reason"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error modifying meal plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error modifying meal plan: {str(e)}")

def adjust_meal_plan(current_meal_plan: str, consumed_food: Dict, meal_replaced: str) -> str:
    """Adjust meal plan based on consumed food"""
    try:
        limited_meals = meals_df.groupby('Meal Time').head(12).to_string(index=False) if 'Meal Time' in meals_df.columns else meals_df.head(40).to_string(index=False)
    except:
        limited_meals = meals_df.head(40).to_string(index=False)
    
    prompt = f"""Dietitian: Adjust plan.

Current:
{current_meal_plan}

User ate: {consumed_food['food_name']} ({consumed_food['quantity']}g) = {consumed_food['calories']}kcal
Replaces: {meal_replaced}

Available meals:
{limited_meals}

Adjust other meals to balance daily calories.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: {consumed_food['food_name']} ({consumed_food['calories']}kcal) - consumed"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error adjusting meal plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adjusting meal plan: {str(e)}")

def parse_meal_calories(meal_text: str) -> Dict[str, int]:
    """Parse meal plan to extract calories"""
    calories = {}
    meal_types = ['Breakfast', 'Mid-Morning Snack', 'Lunch', 'Afternoon Snack', 'Dinner']
    
    for line in meal_text.splitlines():
        line = line.strip()
        for meal_type in meal_types:
            if line.startswith(meal_type + ':'):
                match = re.search(r'\((\d+)', line)
                if match:
                    calories[meal_type] = int(match.group(1))
                    break
    
    return calories

# ---------- API ENDPOINTS ----------

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "🥗 Personalized Meal Plan Generator API",
        "status": "running",
        "version": "1.0.0"
    }
@app.post("/generate-meal-plan", response_model=MealPlanResponse)
async def create_meal_plan(
    request: MealPlanRequest,
    uid: str = Depends(get_current_user)
):
    result = generate_meal_plan(request.user_profile)

    # ✅ FIXED HERE
    save_user_profile(uid, request.user_profile.model_dump())

    save_meal_plan(uid, result)

    return MealPlanResponse(**result)


@app.post("/modify-meal-plan", tags=["Meal Planning"])
async def modify_plan(request: MealModificationRequest):
    """
    Modify an existing meal plan based on user preferences
    
    Request body:
    - meal_plan: Current meal plan text
    - modification: Description of changes (e.g., "Replace chicken with tofu")
    
    Returns:
    - updated_meal_plan: Modified meal plan
    """
    if meals_df.empty:
        raise HTTPException(status_code=500, detail="Meal data not loaded")
    
    updated_plan = modify_meal_plan(request.meal_plan, request.modification, meals_df)
    return {"updated_meal_plan": updated_plan}

@app.post("/get-food-calories", response_model=CalorieInfo, tags=["Nutrition"])
async def get_food_calories(food_name: str, quantity_grams: int):
    """
    Get calorie information for a specific food
    
    Query parameters:
    - food_name: Name of the food item
    - quantity_grams: Quantity in grams (1-1000)
    
    Returns:
    - found: Whether food was found in database
    - calories: Total calories for given quantity
    - calories_per_100g: Calories per 100g
    - food_name: Name of the food found
    - fat, carbs, protein: Macronutrients
    """
    return get_calories_for_food(food_name, quantity_grams)

@app.post("/adjust-meal-plan", tags=["Meal Planning"])
async def adjust_plan(request: MealAdjustmentRequest):
    """
    Adjust meal plan based on food actually consumed
    
    Request body:
    - meal_plan: Current meal plan
    - food_name: Name of food eaten
    - quantity_grams: Amount in grams
    - meal_replaced: Which meal was replaced (e.g., "Dinner")
    
    Returns:
    - adjusted_meal_plan: Updated meal plan
    """
    if meals_df.empty:
        raise HTTPException(status_code=500, detail="Meal data not loaded")
    
    food_info = get_calories_for_food(request.food_name, request.quantity_grams)
    
    if not food_info['found']:
        raise HTTPException(status_code=404, detail="Could not find calorie information for this food")
    
    adjusted_plan = adjust_meal_plan(
        request.meal_plan,
        food_info,
        request.meal_replaced
    )
    
    return {"adjusted_meal_plan": adjusted_plan}

@app.post("/parse-calories", tags=["Nutrition"])
async def extract_meal_calories(meal_plan: str):
    """
    Parse meal plan text and extract calorie information
    
    Request body:
    - meal_plan: Meal plan text
    
    Returns:
    - calories: Dictionary with calories per meal type
    - total_calories: Sum of all meal calories
    """
    calories = parse_meal_calories(meal_plan)
    return {
        "calories": calories,
        "total_calories": sum(calories.values())
    }

@app.get("/calculate-bmi", tags=["Health Metrics"])
async def calculate_bmi_endpoint(weight_kg: float, height_cm: float):
    """
    Calculate BMI from weight and height
    
    Query parameters:
    - weight_kg: Weight in kilograms
    - height_cm: Height in centimeters
    
    Returns:
    - bmi: Calculated BMI value
    - category: BMI category (underweight, normal, overweight, obese)
    """
    if weight_kg <= 0 or height_cm <= 0:
        raise HTTPException(status_code=400, detail="Invalid weight or height values")
    
    bmi = calculate_bmi(weight_kg, height_cm)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return {"bmi": bmi, "category": category}

@app.get("/calculate-bmr", tags=["Health Metrics"])
async def calculate_bmr_endpoint(weight_kg: float, height_cm: float, age: int, gender: Gender):
    """
    Calculate Basal Metabolic Rate
    
    Query parameters:
    - weight_kg: Weight in kilograms
    - height_cm: Height in centimeters
    - age: Age in years
    - gender: Gender (Male/Female/Other)
    
    Returns:
    - bmr: Basal Metabolic Rate (calories burned at rest)
    """
    if weight_kg <= 0 or height_cm <= 0 or age <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")
    
    bmr = calculate_bmr(weight_kg, height_cm, age, gender.value)
    
    return {"bmr": bmr}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with data status"""
    return {
        "status": "healthy",
        "meals_loaded": not meals_df.empty if 'meals_df' in globals() else False,
        "nutrition_data_loaded": not nutrition_df.empty if 'nutrition_df' in globals() else False,
        "openai_configured": bool(openai_api_key)
    }

# ---------- ERROR HANDLERS ----------


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)