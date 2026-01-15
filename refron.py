import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os
import random
import time
import urllib.parse 
import re 
import webbrowser 
from google import genai 


#  Gemini API Key
GEMINI_API_KEY = "AIzaSyBd8ZWTUQq_AQ7RiN5uOxN3KKAX85pkySU" 
GEMINI_MODEL = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are Agri-Bot, an expert assistant for farmers. Provide clear, short, and practical "
    "advice about crop diseases, pest control, and Indian agriculture. "
    "Be polite, simple, and professional."
)

GEMINI_LOADED = False

def get_or_create_gemini():
    global GEMINI_LOADED
    try:
        if not GEMINI_API_KEY:
            raise ValueError("API Key is missing.")
            
        # 1. Create client once
        if "gemini_client" not in st.session_state or st.session_state.gemini_client is None:
            st.session_state.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        # 2. Create chat session once
        if "gemini_chat" not in st.session_state or st.session_state.gemini_chat is None:
            st.session_state.gemini_chat = st.session_state.gemini_client.chats.create(
                model=GEMINI_MODEL,
                config={"system_instruction": SYSTEM_INSTRUCTION},
            )
            # Add initial welcome message to chat history on first run
            if not st.session_state.messages:
                 st.session_state.messages.append({
                    "role": "assistant",
                    "content": "👋 Hello! I'm Agri-Bot. Ask me about any crop disease or pest issue."
                 })
                 
        GEMINI_LOADED = True
        return st.session_state.gemini_client, st.session_state.gemini_chat

    except Exception as e:
        st.warning(f"⚠️ Gemini initialization failed — demo mode active. Error: {e}")
        GEMINI_LOADED = False
        return None, None

# Attempt initialization at start
get_or_create_gemini()


st.set_page_config(
    page_title="🌿 Agri Scan AI", 
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#STYLING 

st.markdown("""
    <style>
        /* All previous styles retained */
        .stApp {
            background: linear-gradient(135deg, #000 0%, #1b1b1b 100%) !important; 
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4 { text-align: center; }
        
        /* FIX: Professional Title Formatting */
        h1.title-main { 
            color: #76ff03;
            font-size: 52px !important;
            font-weight: 700;
            letter-spacing: 2px;
            text-shadow: 0 0 10px rgba(0,255,128,0.7);
        }
        .subhead {
            color: #a5d6a7;
            font-size: 20px;
            margin-bottom: 30px;
            text-align: center;
        }

        .content-box {
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.15);
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,255,128,.2);
            backdrop-filter: blur(10px);
            padding: 30px;
            margin: 20px 0;
        }
        
        .disease-name {
            color: #00e676; 
            font-weight: bold; 
            text-shadow: 0 0 10px #00ff88; 
            font-size: 24px;
            text-align: center;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #00c853, #76ff03); 
            color: black;
            font-weight: bold;
        }
        
        .stLinkButton>a {
            background: #f44336; 
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 15px;
            text-align: center;
            display: block;
        }
        
        @keyframes fadein {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .splash-title {
            color: #c5e1a5;
            font-size: 90px !important;
            font-weight: 900;
            letter-spacing: 5px;
            animation: fadein 2.5s ease-out forwards;
            text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.7);
        }
        
        [data-testid="stMetricValue"] { color: #00e676; }
        [data-testid="stMetricLabel"] { color: #a5d6a7; }
        
        /* New styles to mimic chat interface look */
        /* Custom styling for chat messages to mimic bubble placement */
        /* User messages (right side) */
        [data-testid="stChatMessage"][data-user="true"] {
             text-align: right; 
             margin-left: 20%; 
             background-color: #004d40; 
        }
        /* Assistant messages (left side) */
        [data-testid="stChatMessage"][data-assistant="true"] {
            text-align: left; 
            margin-right: 20%; 
            background-color: #1f1f1f; 
            border-left: 3px solid #76ff03; 
        }
    </style>
""", unsafe_allow_html=True)

#  CLASSES & PESTICIDE DATA

classes = [
    "Bacterial Blight", "Corn__Common_Rust", "Corn_Gray_Leaf_Spot", "Corn_Healthy",
    "Corn_Northern_Leaf_Blight", "Healthy", "Potato_Early_Blight", "Potato_Healthy",
    "Potato__Late_Blight", "Red_Rot", "Rice__Leaf_Blast", "Rice__Brown_Spot",
    "Rice__Neck_Blast", "Wheat__Yellow_Rust", "Wheat__Brown_Rust", "Wheat__Healthy",
    "Sugarcane__Red_Rot"
]

pesticides = {
    "Bacterial Blight": ["Use Copper oxychloride", "Remove affected leaves."],
    "Corn__Common_Rust": ["Apply Mancozeb or Propiconazole", "Use resistant varieties."],
    "Corn_Gray_Leaf_Spot": ["Use Strobilurin fungicides", "Rotate crops."],
    "Corn_Northern_Leaf_Blight": ["Use Azoxystrobin + Propiconazole", "Maintain air circulation."],
    "Potato_Early_Blight": ["Use Chlorothalonil or Mancozanil", "Avoid overhead irrigation."],
    "Potato__Late_Blight": ["Use Metalaxyl or Cymoxanil-based sprays", "Remove infected leaves."],
    "Rice__Leaf_Blast": ["Spray Tricyclazole or Isoprothiolane", "Avoid excess nitrogen."],
    "Rice__Brown_Spot": ["Use Mancozeb or Thiophanate-methyl", "Apply balanced fertilizer."],
    "Rice__Neck_Blast": ["Apply Tricyclazole", "Maintain proper spacing."],
    "Wheat__Yellow_Rust": ["Spray Tebuconazole", "Use resistant varieties."],
    "Wheat__Brown_Rust": ["Apply Propiconazole", "Remove volunteer plants."],
    "Sugarcane__Red_Rot": ["Use Carbendazim", "Avoid ratooning diseased crops."],
    "Healthy": ["No pesticide needed", "Maintain good field hygiene."],
    "Corn_Healthy": ["No pesticide needed", "Maintain good field hygiene."],
    "Potato_Healthy": ["No pesticide needed", "Maintain good field hygiene."],
    "Wheat__Healthy": ["No pesticide needed", "Maintain good field hygiene."],
    "Red_Rot": ["Use Carbendazim (for Sugarcane Red Rot)", "Avoid ratooning diseased crops."]
}


# MODEL LOAD & TRANSFORMS 

MODEL_PATH = r"C:\Users\ashwath\OneDrive - Dr.MAHALINGAM COLLEGE OF ENGINEERING AND TECHNOLOGY\Desktop\re project\c_disease_model.pt"

def load_efficientnet_model(classes):
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(classes))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, True
    except Exception:
        return None, False

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# SMART CROP DETECTION 

def is_crop_image(img: Image.Image, green_thresh=0.12, gray_thresh=0.25):
    arr = np.array(img.resize((224, 224)))
    if arr.ndim != 3:
        return False
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    green_ratio = np.sum((g > r + 20) & (g > b + 20)) / arr.size * 3
    color_std = np.std(arr / 255.0)
    gray_ratio = np.sum((abs(r - g) < 15) & (abs(g - b) < 15)) / arr.size * 3

    return green_ratio > green_thresh and gray_ratio < gray_thresh and color_std > 0.05

#login page

if 'app_state' not in st.session_state:
    st.session_state.app_state = 'splash'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if "messages" not in st.session_state:
    st.session_state.messages = []


if 'model_data' not in st.session_state:
    st.session_state.model, st.session_state.model_loaded = load_efficientnet_model(classes)
    st.session_state.classes = classes
    st.session_state.pesticides = pesticides
    st.session_state.transform = transform


#  CHATBOT

def get_relevant_disease(query, classes):
    """Searches for a known disease in the user's query for DEMO fallback."""
    clean_query = re.sub(r'[^a-z0-9]', ' ', query.lower())
    for full_class_name in classes:
        parts = re.split(r'[_]+', full_class_name.lower())
        disease_term = ' '.join(parts[-2:]) 
        if disease_term in clean_query or full_class_name.lower() in clean_query:
            return full_class_name
        for part in parts:
             if len(part) > 3 and part in clean_query:
                return full_class_name
    return None

#frontend

def splash_screen():
    st.empty()
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='splash-title'>AgriScan AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #a5d6a7; text-align: center;'>AI-Powered Agricultural Intelligence</h3>", unsafe_allow_html=True)
        st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)
    
    time.sleep(1.5) 
    st.session_state.app_state = 'login'
    st.rerun()

def login_page():
    st.empty()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.markdown("<h2>🔑 Secure Access Portal</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='content-box' style='padding: 20px 40px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #76ff03; text-align: center;'>System Login</h4>", unsafe_allow_html=True)
        
        st.session_state.username = st.text_input("Username", placeholder="Enter your ID or Name")
        password = st.text_input("Password", type="password", placeholder="Enter any password")
        
        if st.button("Access Dashboard", use_container_width=True):
            if st.session_state.username and password:
                st.session_state.logged_in = True
                st.session_state.app_state = 'menu'
                st.rerun()
            else:
                st.error("Please enter a Username and Password (dummy).")
        
        st.markdown("</div>", unsafe_allow_html=True)

def main_menu():
    st.empty()
    # TITLE CHANGED HERE
    st.markdown("<h1 class='title-main'>👋 Welcome to Agri Scan AI!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #c5e1a5;'>What would you like to do today?</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # --- MAIN ACTIONS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<h3>🔬 Analyze Crop Disease</h3>", unsafe_allow_html=True)
        st.write("Upload an image of a crop leaf to instantly diagnose the disease and get treatment recommendations.")
        if st.button("Start Diagnosis", key="btn_analyze", use_container_width=True):
            st.session_state.app_state = 'analyze'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<h3>💬 Ask the Chatbot</h3>", unsafe_allow_html=True)
        st.write("Ask our AI assistant for advice on diseases, treatments, and general farming practices.")
        if st.button("Launch Chatbot", key="btn_chatbot", use_container_width=True):
            st.session_state.app_state = 'chatbot'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<h3>📞 Kisan Call Centre</h3>", unsafe_allow_html=True)
        st.write("Connect with a government agricultural expert immediately for direct human advice.")
        
        st.link_button(
            "Call 1800-180-1551 📱", 
            url="tel:1800-180-1551",
            help="Opens your phone's dialer application with the number pre-filled.",
            type="secondary",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # --- LOGOUT ---
    st.markdown("---")

    if st.button("↩️ Logout", key="btn_logout"):
        st.session_state.logged_in = False
        st.session_state.app_state = 'login'
        st.rerun()


# DISEASE ANALYSIS PAGE (Core Logic)

def analyze_page():
    st.empty()
    st.markdown("<h2>🔬 Crop Disease Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a5d6a7;'>Upload a clear image of the diseased leaf for diagnosis.</p>", unsafe_allow_html=True)
    st.markdown("---")
        
    if st.button("⬅️ Back to Menu"):
        st.session_state.app_state = 'menu'
        st.rerun()
    
    # --- Local Farm Advisory ---
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    st.markdown("<h3>☁️ Local Farm Advisory</h3>", unsafe_allow_html=True)
    
    st.caption("Simulated data for Bangalore, current season:")

    weather_col, advisory_col, risk_col = st.columns(3)

    with weather_col:
        st.metric(label="Current Temp / Rain", value="28°C / Low", delta="-2°C from normal")
    with advisory_col:
        st.metric(label="Leaf Wetness (Sim)", value="10 Hours", delta="High for Rust")
    with risk_col:
        st.metric(label="Disease Risk Index", value="HIGH", delta="Increase scouting frequency")
        
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    input_col, display_col = st.columns([1, 1.2])

    file = None
    with input_col:
        st.subheader("1. Input Image")
        option = st.radio("Choose Input Method", ["Upload from Device", "Use Camera"], horizontal=True)

        uploaded_file = None
        captured_image = None

        if option == "Upload from Device":
            uploaded_file = st.file_uploader("Upload your crop leaf image", type=["jpg", "jpeg", "png"])
        elif option == "Use Camera":
            captured_image = st.camera_input("Capture a photo using your camera")
        file = uploaded_file or captured_image

    
    if file:
        img = Image.open(file).convert("RGB")
        
        with display_col:
            st.subheader("2. Image Preview")
            st.image(img, caption="Uploaded/Captured Image", width=300) 

        # --- Smart Crop Check ---
        if not is_crop_image(img):
            st.error("⚠ This doesn’t look like a crop leaf. Please upload a proper crop image.", icon="🚫")
            return

        # --- Analysis Button ---
        if st.button("🔍 Analyze Image", use_container_width=True):
            st.markdown("---")
            st.subheader("📊 Diagnosis Results")
            
            model = st.session_state.model
            classes = st.session_state.classes
            model_loaded = st.session_state.model_loaded
            
            pred = "Unknown"
            conf = 0.0
            pesticide_search_term = ""

            if not model_loaded:
                # DEMO Mode Prediction
                pred = random.choice(classes)
                conf = random.uniform(0.70, 0.95)
                st.warning("Model not found on this system — running in **DEMO mode** (randomized prediction).")
            else:
                # Real Prediction Logic
                x = st.session_state.transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(x)
                    probs = F.softmax(out, dim=1).squeeze()
                
                top3 = torch.topk(probs, 3)

                raw_pred_idx, raw_conf = top3.indices.tolist()[0], top3.values.tolist()[0]
                raw_pred_name = classes[raw_pred_idx]
                MIN_CONFIDENCE_FOR_DISEASE = 0.40

                is_healthy_prediction = "healthy" in raw_pred_name.lower()
                
                if not is_healthy_prediction and raw_conf < MIN_CONFIDENCE_FOR_DISEASE:
                    healthy_idx = classes.index("Healthy") if "Healthy" in classes else -1
                    if healthy_idx != -1 and probs[healthy_idx].item() > raw_conf:
                        pred = "Healthy"
                        conf = probs[healthy_idx].item()
                    else:
                        pred = raw_pred_name
                        conf = raw_conf
                else:
                    pred = raw_pred_name
                    conf = raw_conf


            # --- Display Final Result ---
            st.markdown(f"<p class='disease-name'>Detected: {pred}</p>", unsafe_allow_html=True)
            st.caption(f"Confidence: {conf * 100:.1f}%")

            # --- Remedies & Maps ---
            if "healthy" not in pred.lower():
                rec = st.session_state.pesticides.get(pred, ["Data Not Found", "Check local extension services."])
                
                pesticide_full_text = rec[0]
                pesticide_search_term = pesticide_full_text.split(' or ')[0].replace("Apply ", "").replace("Use ", "").strip()
                
                st.markdown("---")
                st.subheader("💊 Recommended Pesticide")
                st.warning(pesticide_full_text, icon="💉")
                
                st.subheader("🌿 Field Remedy & Tips")
                st.info(rec[1], icon="🌱")
                
                # --- Shop Buttons and Maps ---
                st.markdown("---")
                st.subheader("🛒 Purchase Options")

                col_online, col_offline = st.columns(2)
                
                # 1. Online Purchase Button (General Google Shopping Search)
                online_query = f"{pesticide_search_term} buy pesticide online India"
                online_url = f"https://www.google.com/search?tbm=shop&q={urllib.parse.quote_plus(online_query)}"
                
                with col_online:
                    st.link_button(
                        f"Buy {pesticide_search_term} Online 🛒",
                        url=online_url,
                        help=f"Opens Google Shopping and searches for {pesticide_search_term}.",
                        type="secondary",
                        use_container_width=True
                    )

                # 2. Offline Purchase Maps (Opens New Tab)
                map_search_query = "nearby pesticide shop"
                maps_url_external = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(map_search_query)}"
                
                with col_offline:
                    st.link_button(
                        "Find Nearby Shops (Open Map) 🗺️",
                        url=maps_url_external,
                        help="Opens Google Maps in a new tab to find nearby pesticide shops.",
                        use_container_width=True
                    )
                
                st.info("""
                **Note on Nearby Shops:** Click the button to open Google Maps in a new tab for the best-ranked local shop results.
                """)
                
            else:
                st.success("🎉 Your crop looks **Healthy**! 🌱", icon="✅")
                st.info("Maintain good irrigation and field hygiene to keep your crops safe.")
    
    else:
        with display_col:
            st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #a5d6a7;'>Upload an image to see the preview.</div>", unsafe_allow_html=True)
        
# ----------------------------------
# 💬 CHATBOT POPUP PAGE (GEMINI INTEGRATED)
# ----------------------------------
def chatbot_page():
    """Fully functional Gemini-based Agri-Bot chat UI."""
    st.empty()

    if st.button("⬅️ Back to Menu", key="chat_back"):
        st.session_state.app_state = 'menu'
        st.session_state.messages = []
        st.session_state.gemini_chat = None  # reset chat
        st.rerun()

    # 🧠 Chat Interface
    st.markdown("<h2>💬 Agri-Bot: AI Farming Assistant</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#a5d6a7;'>Ask any query related to crop diseases, treatments, or general farming advice.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Use a clean container for consistent styling
    st.markdown("<div style='padding: 0 10px 0 10px;'>", unsafe_allow_html=True) 
    
    # 🧾 Display chat history using native Streamlit chat elements
    for message in st.session_state.messages:
        # User role gets the 'user' style, pushing the bubble right
        avatar = "🧑‍🌾" if message["role"] == "user" else "🤖"
        
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Input field
    user_query = st.chat_input("Type your query here...", key="chat_input")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        ai_response = "⚠️ Gemini service unavailable (demo mode)."

        # --- Safe Gemini Call ---
        client, chat = get_or_create_gemini()
        if client and chat:
            try:
                with st.spinner("Agri-Bot is thinking..."):
                    response = chat.send_message(user_query)
                    ai_response = response.text
            except Exception as e:
                # 💡 FIX: Auto-reconnect attempt
                st.warning("Gemini communication error. Attempting reconnect...")
                st.session_state.gemini_chat = None
                client, chat = get_or_create_gemini()
                try:
                    response = chat.send_message(user_query)
                    ai_response = response.text
                except Exception as inner_e:
                    ai_response = f"Gemini failed again. Running in demo mode. Error: {inner_e}"
        
        # --- If Gemini Failed, use Local Fallback ---
        if not ai_response or ai_response.startswith("⚠️"):
            disease_found = get_relevant_disease(user_query, st.session_state.classes)
            if disease_found:
                rec = st.session_state.pesticides.get(disease_found, ["No pesticide data.", "No remedy data."])
                ai_response = (
                    f"I found information on **{disease_found.replace('__',' ').replace('_',' ')}** (Offline Lookup) 🌿\n\n"
                    f"**Recommended Treatment:** {rec[0]}\n\n"
                    f"**Field Tips:** {rec[1]}"
                )
            elif "hello" in user_query.lower() or "hi" in user_query.lower():
                ai_response = "Hello! I'm the Agri-Bot, ready to assist with your crop health questions. Please ask me about a disease or treatment!"
            else:
                ai_response = "I'm currently in offline mode. Please ask a specific crop disease name or check your internet/API connection."


        # Add assistant message and rerun
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.rerun() 

    st.markdown("</div>", unsafe_allow_html=True) # Closing the chat padding container

# ==================================
# MAIN APP FLOW EXECUTION
# ==================================

if st.session_state.app_state == 'splash':
    splash_screen()
elif st.session_state.app_state == 'login':
    login_page()
elif st.session_state.app_state == 'menu' and st.session_state.logged_in:
    main_menu()
elif st.session_state.app_state == 'analyze' and st.session_state.logged_in:
    analyze_page()
elif st.session_state.app_state == 'chatbot' and st.session_state.logged_in:
    chatbot_page()
else:
    st.session_state.app_state = 'login'
    st.rerun()


# ----------------------------------
# 🔟 FOOTER
# ----------------------------------
st.markdown("---")
st.caption("Developed by **Team SmartAgro** 🌾 | Presented for Project Review | Designed for Professional Agricultural Use")