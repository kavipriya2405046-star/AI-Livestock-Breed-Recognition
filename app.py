# ============================================================
# LIVESTOCK AI - GRADIO APP
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
import os
import torch
import torch.nn as nn
import timm
import gradio as gr
import traceback
from PIL import Image
from torchvision import transforms

# ============================================================
# MODEL FOLDER
# ============================================================
FOLDER_PATH = "/app/models"

SHEEP_CLASSES = [
    "Deccani", "Garole", "Madgyal", "Mandya",
    "Marwari", "Nellore", "Sawakni"
]

COW_CLASSES = [
    "Gir", "Hallikar", "Hariana", "Kankrej", "Kangayam",
    "Murrah", "Ongole", "Rathi", "Sahiwal", "Tharparkar",
    "Holstein Friesian"
]

BUFFALO_CLASSES = sorted([
    "Chhattisgarhi", "Jaffarabadi", "banni", "bargur", "bhadwari",
    "chilika", "gojri", "kalahandi", "luit_(swamp)", "marathwadi",
    "mehsana", "murrah", "nagpuri", "nili-ravi", "pandharpuri",
    "surti", "toda"
])

# ============================================================
# BREED DATA — BUFFALO
# ============================================================
BUFFALO_DATA = {
    "Chhattisgarhi": {
        "Origin": "Chhattisgarh, India",
        "Coat Color": "Black or dark grey",
        "Body Weight": "350–450 kg",
        "Horns": "Short, curved upward",
        "Primary Use": "Draught work & milk",
        "Milk Yield": "3–5 litres/day",
        "Milk Fat %": "6.5–7.0%",
        "Lactation Period": "270–300 days",
        "Adaptability": "Humid tropical climate",
        "Temperament": "Docile and easy to manage",
        "Conservation Status": "Endangered — population declining",
        "Native to India": "Yes",
        "Economic Value": "Low-cost maintenance, suitable for small farmers",
        "Special Note": "Hardy breed, well adapted to humid tropical climate of Chhattisgarh",
    },
    "Jaffarabadi": {
        "Origin": "Gir Forest, Gujarat, India",
        "Coat Color": "Jet black",
        "Body Weight": "500–800 kg",
        "Horns": "Massive, drooping down and backward",
        "Primary Use": "Heavy milk producer",
        "Milk Yield": "10–20 litres/day",
        "Milk Fat %": "7.0–8.0%",
        "Lactation Period": "290–320 days",
        "Adaptability": "Tropical, humid",
        "Temperament": "Calm but needs space",
        "Conservation Status": "Vulnerable",
        "Native to India": "Yes",
        "Economic Value": "High — premium milk and breeding value",
        "Special Note": "Heaviest buffalo breed in India; excellent milk fat content",
    },
    "banni": {
        "Origin": "Kutch, Gujarat, India",
        "Coat Color": "Black",
        "Body Weight": "450–550 kg",
        "Horns": "Flat, curved outward and upward",
        "Primary Use": "Milk",
        "Milk Yield": "8–12 litres/day",
        "Milk Fat %": "7.5–8.5%",
        "Lactation Period": "280–310 days",
        "Adaptability": "Arid and semi-arid regions",
        "Temperament": "Hardy, independent",
        "Conservation Status": "Registered GI Tag Breed",
        "Native to India": "Yes",
        "Economic Value": "High — premium dairy in Gujarat",
        "Special Note": "Thrives in harsh Kutch desert conditions — unique salt-tolerant breed",
    },
    "bargur": {
        "Origin": "Bargur Hills, Tamil Nadu, India",
        "Coat Color": "Brown with white markings",
        "Body Weight": "300–400 kg",
        "Horns": "Short, turned backward",
        "Primary Use": "Draught (hill terrain)",
        "Milk Yield": "2–4 litres/day",
        "Milk Fat %": "6.0–7.0%",
        "Lactation Period": "240–270 days",
        "Adaptability": "Steep hilly terrain",
        "Temperament": "Agile, energetic",
        "Conservation Status": "Endangered",
        "Native to India": "Yes",
        "Economic Value": "Draught value in hill farming",
        "Special Note": "Extremely agile — ideal for steep hilly terrain in Erode district",
    },
    "bhadwari": {
        "Origin": "Agra & Etawah, Uttar Pradesh, India",
        "Coat Color": "Copper-brown",
        "Body Weight": "350–450 kg",
        "Horns": "Short, tightly curved",
        "Primary Use": "Milk (high fat)",
        "Milk Yield": "4–6 litres/day",
        "Milk Fat %": "13–14% (World Record!)",
        "Lactation Period": "270–300 days",
        "Adaptability": "Semi-arid UP conditions",
        "Temperament": "Calm, easy to manage",
        "Conservation Status": "Endangered",
        "Native to India": "Yes",
        "Economic Value": "Highest fat milk — premium ghee production",
        "Special Note": "Highest fat % (13–14%) among ALL buffalo breeds in the world",
    },
    "chilika": {
        "Origin": "Chilika Lake region, Odisha, India",
        "Coat Color": "Black or dark grey",
        "Body Weight": "300–400 kg",
        "Horns": "Curved upward",
        "Primary Use": "Draught & milk",
        "Milk Yield": "3–5 litres/day",
        "Milk Fat %": "6.5–7.5%",
        "Lactation Period": "250–280 days",
        "Adaptability": "Marshy wetlands, amphibious",
        "Temperament": "Calm near water, strong swimmer",
        "Conservation Status": "Endangered",
        "Native to India": "Yes",
        "Economic Value": "Fishing communities — draught in wetlands",
        "Special Note": "Amphibious breed — excellent swimmer, used in marshy Chilika wetlands",
    },
    "gojri": {
        "Origin": "Punjab & Himachal Pradesh, India",
        "Coat Color": "Black or grey",
        "Body Weight": "400–500 kg",
        "Horns": "Medium, curved backward",
        "Primary Use": "Milk",
        "Milk Yield": "6–8 litres/day",
        "Milk Fat %": "7.0–8.0%",
        "Lactation Period": "270–300 days",
        "Adaptability": "Seasonal migration — hills to plains",
        "Temperament": "Docile, nomadic",
        "Conservation Status": "Vulnerable",
        "Native to India": "Yes",
        "Economic Value": "Dairy value for Gujjar nomadic community",
        "Special Note": "Nomadic breed — migrates with Gujjar tribe seasonally",
    },
    "kalahandi": {
        "Origin": "Kalahandi, Odisha, India",
        "Coat Color": "Black",
        "Body Weight": "350–420 kg",
        "Horns": "Medium, curved outward",
        "Primary Use": "Draught & milk",
        "Milk Yield": "2–4 litres/day",
        "Milk Fat %": "6.0–7.0%",
        "Lactation Period": "240–270 days",
        "Adaptability": "Tribal regions of Odisha",
        "Temperament": "Strong, active",
        "Conservation Status": "Endangered",
        "Native to India": "Yes",
        "Economic Value": "Crucial draught animal in tribal farming",
        "Special Note": "Strong draught animal crucial for tribal farming in Odisha",
    },
    "luit_(swamp)": {
        "Origin": "Assam & Northeast India",
        "Coat Color": "Grey or slate",
        "Body Weight": "250–350 kg",
        "Horns": "Long, swept back flat",
        "Primary Use": "Draught (paddy fields)",
        "Milk Yield": "1–3 litres/day",
        "Milk Fat %": "5.5–6.5%",
        "Lactation Period": "220–260 days",
        "Adaptability": "Waterlogged paddy fields",
        "Temperament": "Calm in water, strong",
        "Conservation Status": "Vulnerable",
        "Native to India": "Yes",
        "Economic Value": "Essential for Assam paddy cultivation",
        "Special Note": "Swamp buffalo — uniquely suited for waterlogged paddy cultivation in Assam",
    },
    "marathwadi": {
        "Origin": "Marathwada, Maharashtra, India",
        "Coat Color": "Black",
        "Body Weight": "400–500 kg",
        "Horns": "Curved inward and upward",
        "Primary Use": "Milk & draught",
        "Milk Yield": "4–7 litres/day",
        "Milk Fat %": "6.5–7.5%",
        "Lactation Period": "270–300 days",
        "Adaptability": "Drought-prone Marathwada region",
        "Temperament": "Docile, drought-hardy",
        "Conservation Status": "Vulnerable",
        "Native to India": "Yes",
        "Economic Value": "Moderate dairy and draught value",
        "Special Note": "Well adapted to drought-prone Marathwada region of Maharashtra",
    },
    "mehsana": {
        "Origin": "Mehsana, Gujarat, India",
        "Coat Color": "Black or grey-black",
        "Body Weight": "450–550 kg",
        "Horns": "Irregular, sickle-shaped",
        "Primary Use": "Milk",
        "Milk Yield": "8–12 litres/day",
        "Milk Fat %": "7.0–8.0%",
        "Lactation Period": "300+ days",
        "Adaptability": "Gujarat plains",
        "Temperament": "Calm, manageable",
        "Conservation Status": "Stable",
        "Native to India": "Yes",
        "Economic Value": "Commercial dairy farms in Gujarat",
        "Special Note": "Cross between Surti & Murrah — long lactation period of 300+ days",
    },
    "murrah": {
        "Origin": "Rohtak & Hisar, Haryana, India",
        "Coat Color": "Jet black",
        "Body Weight": "500–600 kg",
        "Horns": "Tightly curled like a watch spring",
        "Primary Use": "Milk (best dairy buffalo globally)",
        "Milk Yield": "15–25 litres/day",
        "Milk Fat %": "6.5–7.5%",
        "Lactation Period": "300–320 days",
        "Adaptability": "North Indian plains",
        "Temperament": "Docile, high-maintenance",
        "Conservation Status": "Stable — widely bred",
        "Native to India": "Yes",
        "Economic Value": "Highest commercial dairy value globally",
        "Special Note": "World's highest milk-yielding buffalo — exported to Italy for Mozzarella!",
    },
    "nagpuri": {
        "Origin": "Vidarbha, Maharashtra, India",
        "Coat Color": "Black with white spots on face & legs",
        "Body Weight": "450–550 kg",
        "Horns": "Long, flat, sweeping backward",
        "Primary Use": "Milk & draught",
        "Milk Yield": "5–8 litres/day",
        "Milk Fat %": "6.5–7.5%",
        "Lactation Period": "270–300 days",
        "Adaptability": "Central India",
        "Temperament": "Active, strong",
        "Conservation Status": "Stable",
        "Native to India": "Yes",
        "Economic Value": "Good dual-purpose value",
        "Special Note": "Distinct white markings on face & legs — easily identifiable breed",
    },
    "nili-ravi": {
        "Origin": "Punjab, Pakistan & India",
        "Coat Color": "Black with white markings",
        "Body Weight": "500–650 kg",
        "Horns": "Small, tightly curved",
        "Primary Use": "Heavy milk producer",
        "Milk Yield": "12–18 litres/day",
        "Milk Fat %": "6.5–7.0%",
        "Lactation Period": "290–320 days",
        "Adaptability": "Punjab plains",
        "Temperament": "Docile, large-framed",
        "Conservation Status": "Stable",
        "Native to India": "Yes",
        "Economic Value": "High — second best dairy buffalo globally",
        "Special Note": "Wall-eye (blue eye) is a characteristic feature — second best dairy buffalo",
    },
    "pandharpuri": {
        "Origin": "Solapur, Maharashtra, India",
        "Coat Color": "Black",
        "Body Weight": "450–550 kg",
        "Horns": "Very long, drooping down then curving up",
        "Primary Use": "Milk",
        "Milk Yield": "6–10 litres/day",
        "Milk Fat %": "7.0–8.0%",
        "Lactation Period": "270–300 days",
        "Adaptability": "Maharashtra plateau",
        "Temperament": "Calm, manageable",
        "Conservation Status": "Vulnerable",
        "Native to India": "Yes",
        "Economic Value": "Premium dairy breed of Maharashtra",
        "Special Note": "Longest horns among all buffalo breeds in the world",
    },
    "surti": {
        "Origin": "Surat & Vadodara, Gujarat, India",
        "Coat Color": "Rust-brown or silver-grey",
        "Body Weight": "400–500 kg",
        "Horns": "Sickle-shaped, medium length",
        "Primary Use": "Milk",
        "Milk Yield": "8–10 litres/day",
        "Milk Fat %": "8.0–9.0%",
        "Lactation Period": "290–310 days",
        "Adaptability": "Gujarat coastal conditions",
        "Temperament": "Gentle, easy to handle",
        "Conservation Status": "Stable",
        "Native to India": "Yes",
        "Economic Value": "Premium dairy — highest fat % among Gujarat breeds",
        "Special Note": "Two white collars on neck — premium dairy breed of Gujarat",
    },
    "toda": {
        "Origin": "Nilgiri Hills, Tamil Nadu, India",
        "Coat Color": "Dark grey or black",
        "Body Weight": "300–400 kg",
        "Horns": "Short, curved forward",
        "Primary Use": "Sacred / milk for Toda tribe",
        "Milk Yield": "1–2 litres/day",
        "Milk Fat %": "6.0–7.0%",
        "Lactation Period": "220–260 days",
        "Adaptability": "High altitude Nilgiri Hills",
        "Temperament": "Sacred — not worked",
        "Conservation Status": "Critically Endangered",
        "Native to India": "Yes",
        "Economic Value": "Cultural & religious — not commercial",
        "Special Note": "Sacred animal of the Toda tribe — used for religious rituals only",
    },
}

# ============================================================
# BREED DATA — SHEEP
# ============================================================
SHEEP_DATA = {
    "Sawakni": {
        "Origin": "Sudan — Darfur region", "Type": "Meat-type sheep",
        "Avg Weight (Male)": "50–70 kg", "Avg Weight (Female)": "35–50 kg",
        "Wool Quality": "Coarse — minimal economic value",
        "Adaptability": "Semi-arid Sahelian climate, drought resistant",
        "Meat Quality": "Good quality mutton with characteristic fat tail meat",
        "Advantages": "Excellent heat tolerance | Disease resistant | Survives on poor forage",
        "Disadvantages": "Requires extensive grazing | Slower growth vs improved breeds",
        "Primary Use": "Meat production", "Native to India": "No",
        "Special Note": "Critical breed for nomadic herders in Sudan",
    },
    "Deccani": {
        "Origin": "Deccan Plateau, Maharashtra, India", "Type": "Carpet-wool / Meat",
        "Avg Weight (Male)": "30–40 kg", "Avg Weight (Female)": "22–30 kg",
        "Wool Quality": "Coarse wool — carpet making",
        "Adaptability": "Semi-arid Deccan conditions", "Meat Quality": "Lean mutton",
        "Advantages": "Drought tolerant | Low-input management | Carpet wool",
        "Disadvantages": "Low production per animal",
        "Primary Use": "Wool and meat", "Native to India": "Yes",
        "Special Note": "Important for smallholder farmers in Maharashtra",
    },
    "Garole": {
        "Origin": "Sundarbans, West Bengal, India", "Type": "Meat / Prolific breed",
        "Avg Weight (Male)": "18–22 kg", "Avg Weight (Female)": "12–16 kg",
        "Wool Quality": "Hair-type — no wool value",
        "Adaptability": "Humid deltaic mangrove environment", "Meat Quality": "Good lean mutton",
        "Advantages": "Exceptional prolificacy | Disease resistant | Early maturity",
        "Disadvantages": "Very small frame | Limited meat per animal",
        "Primary Use": "Meat production in coastal wetlands", "Native to India": "Yes",
        "Special Note": "World's most prolific sheep — carries FecB (Booroola) gene!",
    },
    "Madgyal": {
        "Origin": "Sangli & Kolhapur, Maharashtra, India", "Type": "Meat / Wool",
        "Avg Weight (Male)": "35–50 kg", "Avg Weight (Female)": "25–35 kg",
        "Wool Quality": "Semi-fine carpet value", "Adaptability": "Semi-arid Maharashtra",
        "Meat Quality": "Good quality mutton",
        "Advantages": "Good meat + moderate wool | Adaptable | Disease resistant",
        "Disadvantages": "Limited to specific districts",
        "Primary Use": "Meat and wool", "Native to India": "Yes",
        "Special Note": "Dual-purpose value for Maharashtra farmers",
    },
    "Mandya": {
        "Origin": "Mandya & Mysore, Karnataka, India", "Type": "Meat breed",
        "Avg Weight (Male)": "35–45 kg", "Avg Weight (Female)": "25–32 kg",
        "Wool Quality": "Hair-type coat — no wool value",
        "Adaptability": "South Karnataka conditions", "Meat Quality": "Very good quality mutton",
        "Advantages": "Good meat quality | Disease resistant | Low maintenance",
        "Disadvantages": "Limited distribution outside Karnataka",
        "Primary Use": "Mutton production", "Native to India": "Yes",
        "Special Note": "Culturally important for Karnataka festivals (Dasara)",
    },
    "Marwari": {
        "Origin": "Jodhpur & Barmer, Rajasthan, India", "Type": "Dual-purpose (Wool & Meat)",
        "Avg Weight (Male)": "35–50 kg", "Avg Weight (Female)": "25–38 kg",
        "Wool Quality": "Carpet-grade wool — 400–600g per shearing",
        "Adaptability": "Extreme desert, drought hardy", "Meat Quality": "Moderate mutton quality",
        "Advantages": "Extremely drought resistant | Good wool yield | Long trekking endurance",
        "Disadvantages": "Lower meat yield | Poor in humid climates",
        "Primary Use": "Wool and meat — nomadic pastoralism", "Native to India": "Yes",
        "Special Note": "Backbone of Rajasthan's nomadic shepherding communities",
    },
    "Nellore": {
        "Origin": "Nellore district, Andhra Pradesh, India", "Type": "Meat breed",
        "Avg Weight (Male)": "45–60 kg", "Avg Weight (Female)": "30–40 kg",
        "Wool Quality": "Very coarse — floor mats only",
        "Adaptability": "Hot humid coastal Andhra", "Meat Quality": "Excellent lean mutton",
        "Advantages": "Fast growth | Disease resistant | Good for commercial mutton",
        "Disadvantages": "Poor wool quality | Not for cooler climates",
        "Primary Use": "Mutton production", "Native to India": "Yes",
        "Special Note": "Most popular mutton breed in Andhra Pradesh",
    },
}

# ============================================================
# BREED DATA — COW
# ============================================================
COW_DATA = {
    "Gir": {
        "Origin": "Gir Forest, Gujarat, India", "Type": "Dual-purpose",
        "Milk Yield (L/day)": "6–12 L", "Avg Weight (Male)": "550–650 kg",
        "Avg Weight (Female)": "350–450 kg", "Milk Fat %": "4.5–5.0%",
        "Coat Color": "Red with white patches", "Horns": "Curved outward and backward",
        "Advantages": "High-fat milk | Disease resistance | Long productive life",
        "Disadvantages": "Lower milk yield than exotic breeds",
        "Primary Use": "Milk and draught", "Native to India": "Yes",
        "Special Note": "Exported to Brazil — forms 'Guzerat' breed used globally",
    },
    "Hallikar": {
        "Origin": "Tumkur & Hassan, Karnataka, India", "Type": "Draught",
        "Milk Yield (L/day)": "2–4 L", "Avg Weight (Male)": "400–550 kg",
        "Avg Weight (Female)": "280–380 kg", "Milk Fat %": "4.5–5.0%",
        "Coat Color": "Grey to white", "Horns": "Long, lyre-shaped",
        "Advantages": "Outstanding draught endurance | Tough hooves",
        "Disadvantages": "Low milk yield",
        "Primary Use": "Agricultural draught work", "Native to India": "Yes",
        "Special Note": "Famous for Kambala cattle racing festivals in Karnataka",
    },
    "Hariana": {
        "Origin": "Rohtak, Hisar & Karnal, Haryana, India", "Type": "Dual-purpose",
        "Milk Yield (L/day)": "8–12 L", "Avg Weight (Male)": "500–600 kg",
        "Avg Weight (Female)": "350–430 kg", "Milk Fat %": "4.0–4.5%",
        "Coat Color": "White or light grey", "Horns": "Short, stumpy",
        "Advantages": "Good milk yield + powerful draught | Economical",
        "Disadvantages": "Needs more nutrition than smaller breeds",
        "Primary Use": "Milk and draught", "Native to India": "Yes",
        "Special Note": "Foundation breed of Haryana's agricultural economy",
    },
    "Kankrej": {
        "Origin": "Kankrej region, Gujarat & Rajasthan, India", "Type": "Dual-purpose",
        "Milk Yield (L/day)": "5–10 L", "Avg Weight (Male)": "550–700 kg",
        "Avg Weight (Female)": "350–450 kg", "Milk Fat %": "3.8–4.5%",
        "Coat Color": "Silver grey to iron grey", "Horns": "Lyre-shaped, spreading",
        "Advantages": "Enormous draught power | Heat tolerant",
        "Disadvantages": "Difficult to handle when young",
        "Primary Use": "Heavy draught and milk", "Native to India": "Yes",
        "Special Note": "Contributed genetics to 'Brahman' breed in USA",
    },
    "Kangayam": {
        "Origin": "Kangayam, Erode & Tiruppur, Tamil Nadu, India", "Type": "Draught",
        "Milk Yield (L/day)": "1–3 L", "Avg Weight (Male)": "400–550 kg",
        "Avg Weight (Female)": "250–350 kg", "Milk Fat %": "4.5–5.5%",
        "Coat Color": "Grey with black markings", "Horns": "Medium, forward-pointing",
        "Advantages": "Outstanding draught power | Extreme endurance | Frugal feeder",
        "Disadvantages": "Very low milk yield",
        "Primary Use": "Agricultural draught work", "Native to India": "Yes",
        "Special Note": "Protected GI Tag — finest draught breed of South India",
    },
    "Murrah": {
        "Origin": "Rohtak & Hisar, Haryana, India", "Type": "Dairy Buffalo",
        "Milk Yield (L/day)": "15–25 L", "Avg Weight (Male)": "500–600 kg",
        "Avg Weight (Female)": "400–500 kg", "Milk Fat %": "6.5–7.5%",
        "Coat Color": "Jet black", "Horns": "Tightly coiled",
        "Advantages": "World's best dairy buffalo | Very high fat content",
        "Disadvantages": "High water requirement",
        "Primary Use": "High-fat milk production", "Native to India": "Yes",
        "Special Note": "Exported to Italy for authentic Mozzarella di Bufala production",
    },
    "Ongole": {
        "Origin": "Prakasam district, Andhra Pradesh, India", "Type": "Dual-purpose",
        "Milk Yield (L/day)": "4–8 L", "Avg Weight (Male)": "600–750 kg",
        "Avg Weight (Female)": "350–500 kg", "Milk Fat %": "4.0–4.5%",
        "Coat Color": "White to grayish white", "Horns": "Short, thick, blunt",
        "Advantages": "Great heat & humidity tolerance | Heavy draught",
        "Disadvantages": "Slow growth | Lower milk than dairy breeds",
        "Primary Use": "Draught and milk", "Native to India": "Yes",
        "Special Note": "Exported to Brazil & USA — forms American 'Nellore' cattle",
    },
    "Rathi": {
        "Origin": "Bikaner & Ganganagar, Rajasthan, India", "Type": "Dairy",
        "Milk Yield (L/day)": "8–16 L", "Avg Weight (Male)": "450–550 kg",
        "Avg Weight (Female)": "300–400 kg", "Milk Fat %": "4.5–5.5%",
        "Coat Color": "Brown or reddish-brown with white patches", "Horns": "Short, upward-curving",
        "Advantages": "Best native dairy for Rajasthan | Very high fat milk",
        "Disadvantages": "Limited performance outside arid zones",
        "Primary Use": "Milk production", "Native to India": "Yes",
        "Special Note": "Called the 'Pride of Rajasthan' for dairy performance in desert",
    },
    "Sahiwal": {
        "Origin": "Sahiwal district, Punjab (now Pakistan)", "Type": "Dairy",
        "Milk Yield (L/day)": "10–16 L", "Avg Weight (Male)": "450–500 kg",
        "Avg Weight (Female)": "300–400 kg", "Milk Fat %": "4.0–4.5%",
        "Coat Color": "Reddish dun or pale red", "Horns": "Short, stumpy",
        "Advantages": "Highest milk among zebu breeds | Tick resistant",
        "Disadvantages": "Needs good nutrition",
        "Primary Use": "Milk production", "Native to India": "Yes",
        "Special Note": "Best dairy zebu breed in the world — FAO listed as endangered",
    },
    "Tharparkar": {
        "Origin": "Tharparkar (Sindh/Rajasthan border)", "Type": "Dual-purpose",
        "Milk Yield (L/day)": "6–10 L", "Avg Weight (Male)": "450–550 kg",
        "Avg Weight (Female)": "300–380 kg", "Milk Fat %": "4.5–5.0%",
        "Coat Color": "White to light grey", "Horns": "Moderate, upward curving",
        "Advantages": "Excellent in hot arid conditions | Good milk + draught combo",
        "Disadvantages": "Lower yield in non-desert environments",
        "Primary Use": "Milk and draught in arid zones", "Native to India": "Yes",
        "Special Note": "India's most drought-resilient cattle breed",
    },
    "Holstein Friesian": {
        "Origin": "Netherlands & North Germany", "Type": "Dairy (Exotic)",
        "Milk Yield (L/day)": "25–40 L", "Avg Weight (Male)": "900–1000 kg",
        "Avg Weight (Female)": "580–700 kg", "Milk Fat %": "3.2–3.5%",
        "Coat Color": "Black and white patches", "Horns": "Usually dehorned",
        "Advantages": "World's highest milk producer | Ideal for commercial dairying",
        "Disadvantages": "Low heat tolerance | Requires high feed | Expensive management",
        "Primary Use": "Commercial dairy", "Native to India": "No (Exotic)",
        "Special Note": "Responsible for 90%+ of commercial milk in developed countries",
    },
}

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

# ----- Sheep Model -----
class ChannelAttentionOld(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg = x.mean(dim=[2, 3], keepdim=True)
        max_ = x.amax(dim=[2, 3], keepdim=True)
        return x * torch.sigmoid(self.fc(avg) + self.fc(max_))


class SpatialAttentionOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(
            self.conv(torch.cat([avg, max_], dim=1))
        )


class CBAMOld(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttentionOld(in_channels)
        self.spatial_att = SpatialAttentionOld()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class SwinClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            global_pool=""
        )
        self.cbam1 = CBAMOld(96)
        self.cbam2 = CBAMOld(192)
        self.cbam3 = CBAMOld(384)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.swin(x)
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        x = x.transpose(1, 2)
        x = self.head[0](x).squeeze(-1)
        x = self.head[2](x)
        x = self.head[3](x)
        x = self.head[4](x)
        x = self.head[5](x)
        x = self.head[6](x)
        x = self.head[7](x)
        return x


# ----- Cow Model -----
class CowCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 12, bias=False),
            nn.ReLU(),
            nn.Linear(12, 192, bias=False),
            nn.Sigmoid()
        )
        self.sp_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x):
        att = self.mlp(
            x.mean(dim=[2, 3])
        ).unsqueeze(-1).unsqueeze(-1)

        x = x * att

        sp_avg = x.mean(dim=1, keepdim=True)
        sp_max = x.amax(dim=1, keepdim=True)

        return x * torch.sigmoid(
            self.sp_conv(torch.cat([sp_avg, sp_max], dim=1))
        )


class CowClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            global_pool=""
        )
        self.cbam = CowCBAM()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.LayerNorm(768),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.swin(x)

        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)

        x = x.transpose(1, 2)

        x = self.head[0](x).squeeze(-1)
        x = self.head[1](x)
        x = self.head[3](x)
        x = self.head[4](x)
        x = self.head[5](x)
        x = self.head[6](x)

        return x


# ----- Buffalo Model -----
class ChannelAttentionNew(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(
                in_channels,
                in_channels // reduction,
                bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_channels // reduction,
                in_channels,
                bias=False
            )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        avg = self.fc(
            self.avg_pool(x).view(b, c)
        )

        max_ = self.fc(
            self.max_pool(x).view(b, c)
        )

        return x * self.sigmoid(
            avg + max_
        ).view(b, c, 1, 1)


class SpatialAttentionNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(
            x, dim=1, keepdim=True
        )

        max_, _ = torch.max(
            x, dim=1, keepdim=True
        )

        return x * self.sigmoid(
            self.conv(
                torch.cat([avg, max_], dim=1)
            )
        )


class CBAMNew(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttentionNew(
            in_channels,
            reduction
        )
        self.sa = SpatialAttentionNew()

    def forward(self, x):
        return self.sa(self.ca(x))


class CBAMBridgeNew(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cbam = CBAMNew(
            in_channels,
            reduction
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.cbam(x)
        return x.permute(0, 2, 3, 1)


class SwinCBAM(nn.Module):
    def __init__(self, num_classes=17, pretrained=False):
        super().__init__()

        self.swin = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            global_pool=""
        )

        self.cbam0 = CBAMBridgeNew(96)
        self.cbam1 = CBAMBridgeNew(192)
        self.cbam2 = CBAMBridgeNew(384)
        self.cbam3 = CBAMBridgeNew(768)

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.use_cbam = True

    def forward(self, x):
        x = self.swin.patch_embed(x)

        if (
            hasattr(self.swin, "absolute_pos_embed")
            and self.swin.absolute_pos_embed is not None
        ):
            x += self.swin.absolute_pos_embed

        if hasattr(self.swin, "pos_drop"):
            x = self.swin.pos_drop(x)

        for layer, cbam in zip(
            self.swin.layers,
            [
                self.cbam0,
                self.cbam1,
                self.cbam2,
                self.cbam3
            ]
        ):
            x = layer(x)

            if isinstance(x, tuple):
                x = x[0]

            if self.use_cbam:
                x = cbam(x)

        return self.head(
            x.mean(dim=[1, 2])
        )


# ============================================================
# LOAD ALL 3 MODELS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Device: {device}")

# Sheep
sheep_model = SwinClassifier(
    num_classes=len(SHEEP_CLASSES)
)

sheep_model.load_state_dict(
    torch.load(
        os.path.join(
            FOLDER_PATH,
            "best_model.pth"
        ),
        map_location=device
    )
)

sheep_model = sheep_model.to(device).eval()

print("✅ Sheep model loaded!")


# Cow
cow_model = CowClassifier(
    num_classes=len(COW_CLASSES)
)

cow_model.load_state_dict(
    torch.load(
        os.path.join(
            FOLDER_PATH,
            "best_cow_model.pth"
        ),
        map_location=device
    )
)

cow_model = cow_model.to(device).eval()

print("✅ Cow model loaded!")


# Buffalo
buffalo_model = SwinCBAM(
    num_classes=len(BUFFALO_CLASSES),
    pretrained=False
)

buffalo_model.load_state_dict(
    torch.load(
        os.path.join(
            FOLDER_PATH,
            "best_buffalo_finetuned (1).pth"
        ),
        map_location=device
    )
)

buffalo_model = buffalo_model.to(device).eval()

print("✅ Buffalo model loaded!")


# ============================================================
# IMAGE TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# BUILD HTML TABLE
# ============================================================
def build_html_table(
    top_breed,
    conf,
    animal_type,
    breed_info,
    scores
):

    accent_map = {
        "Buffalo": "#3b82f6",
        "Cow": "#8b5cf6",
        "Sheep": "#6366f1"
    }

    accent2_map = {
        "Buffalo": "#1d4ed8",
        "Cow": "#6d28d9",
        "Sheep": "#4f46e5"
    }

    accent = accent_map.get(
        animal_type,
        "#3b82f6"
    )

    accent2 = accent2_map.get(
        animal_type,
        "#1d4ed8"
    )

    emoji_map = {
        "Buffalo": "🐃",
        "Cow": "🐄",
        "Sheep": "🐑"
    }

    em = emoji_map.get(
        animal_type,
        "🐾"
    )

    bg_main = "#0f1117"
    bg_card = "#1a1d2e"
    bg_row1 = "#1e2235"
    bg_row2 = "#1a1d2e"
    txt_main = "#e2e8f0"
    txt_sub = "#94a3b8"
    border = "#2d3748"

    conf_color = (
        "#22c55e"
        if conf >= 85
        else "#f59e0b"
        if conf >= 60
        else "#ef4444"
    )

    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;
                padding:14px;background:{bg_main};
                border-radius:14px;color:{txt_main};">

        <div style="background:linear-gradient(135deg,
                    {accent2},{accent});
                    padding:16px 20px;border-radius:12px;
                    margin-bottom:14px;">

            <h2 style="margin:0;font-size:22px;color:white;
                       letter-spacing:0.5px;">
                {em} {top_breed.replace('_',' ').title()}
            </h2>

            <p style="margin:4px 0 0 0;font-size:13px;
                      color:rgba(255,255,255,0.8);">
                {animal_type} Breed · AI Classification Result
            </p>
        </div>

        <div style="display:flex;gap:10px;margin-bottom:14px;
                    flex-wrap:wrap;">

            <div style="background:{bg_card};
                        border:1.5px solid {conf_color};
                        border-radius:10px;padding:10px 18px;
                        text-align:center;flex:1;min-width:80px;">

                <div style="font-size:26px;font-weight:700;
                            color:{conf_color};">
                    {conf}%
                </div>

                <div style="font-size:11px;color:{txt_sub};
                            margin-top:2px;">
                    Confidence
                </div>
            </div>

            <div style="background:{bg_card};
                        border:1.5px solid {accent};
                        border-radius:10px;padding:10px 18px;
                        text-align:center;flex:1;min-width:80px;">

                <div style="font-size:18px;font-weight:700;
                            color:{accent};">
                    {animal_type}
                </div>

                <div style="font-size:11px;color:{txt_sub};
                            margin-top:2px;">
                    Animal Type
                </div>
            </div>
        </div>

        <div style="background:{bg_card};
                    border-radius:10px;overflow:hidden;
                    margin-bottom:12px;border:1px solid {border};">

            <div style="background:linear-gradient(90deg,
                        {accent2},{accent});
                        color:white;padding:9px 14px;
                        font-weight:600;font-size:13px;">
                📋 Breed Information
            </div>

            <table style="width:100%;border-collapse:collapse;
                          font-size:13px;">
    """

    icon_map = {
        "Origin": "📍",
        "Coat Color": "🎨",
        "Body Weight": "⚖️",
        "Horns": "🦌",
        "Primary Use": "🎯",
        "Milk Yield": "🥛",
        "Milk Fat %": "💧",
        "Lactation Period": "📅",
        "Adaptability": "🌿",
        "Temperament": "😊",
        "Conservation Status": "🔰",
        "Economic Value": "💰",
        "Type": "📌",
        "Avg Weight (Male)": "♂️",
        "Avg Weight (Female)": "♀️",
        "Wool Quality": "🐑",
        "Meat Quality": "🥩",
        "Milk Yield (L/day)": "🥛",
    }

    skip_keys = {
        "Advantages",
        "Disadvantages",
        "Native to India",
        "Special Note"
    }

    row_idx = 0

    for key, val in breed_info.items():

        if key in skip_keys:
            continue

        icon = icon_map.get(
            key,
            "•"
        )

        bg = (
            bg_row1
            if row_idx % 2 == 0
            else bg_row2
        )

        html += f"""
            <tr style="background:{bg};
                       border-bottom:1px solid {border};">

                <td style="padding:8px 12px;
                           font-weight:600;color:{txt_sub};
                           width:40%;white-space:nowrap;
                           font-size:12px;">
                    {icon} {key}
                </td>

                <td style="padding:8px 12px;
                           color:{txt_main};
                           font-size:13px;">
                    {val}
                </td>

            </tr>
        """

        row_idx += 1

    html += "</table></div>"

    # Advantages
    if "Advantages" in breed_info:

        tags = "".join([
            f'<span style="background:rgba(34,197,94,0.15);'
            f'color:#4ade80;padding:4px 11px;'
            f'border-radius:20px;margin:3px;'
            f'display:inline-block;font-size:12px;'
            f'font-weight:500;'
            f'border:1px solid rgba(34,197,94,0.3);">'
            f'✅ {t.strip()}</span>'
            for t in breed_info["Advantages"].split("|")
        ])

        html += f"""
        <div style="background:{bg_card};
                    border-radius:10px;padding:12px 14px;
                    margin-bottom:10px;
                    border:1px solid {border};">

            <div style="font-weight:600;color:#4ade80;
                        margin-bottom:8px;font-size:13px;">
                ✅ Advantages
            </div>

            <div>{tags}</div>
        </div>
        """

    # Disadvantages
    if "Disadvantages" in breed_info:

        tags = "".join([
            f'<span style="background:rgba(239,68,68,0.15);'
            f'color:#f87171;padding:4px 11px;'
            f'border-radius:20px;margin:3px;'
            f'display:inline-block;font-size:12px;'
            f'font-weight:500;'
            f'border:1px solid rgba(239,68,68,0.3);">'
            f'❌ {t.strip()}</span>'
            for t in breed_info["Disadvantages"].split("|")
        ])

        html += f"""
        <div style="background:{bg_card};
                    border-radius:10px;padding:12px 14px;
                    margin-bottom:10px;
                    border:1px solid {border};">

            <div style="font-weight:600;color:#f87171;
                        margin-bottom:8px;font-size:13px;">
                ❌ Disadvantages
            </div>

            <div>{tags}</div>
        </div>
        """

    # Native to India
    if "Native to India" in breed_info:

        val = breed_info["Native to India"]

        icon = (
            "✅ Yes — Native Indian Breed"
            if val == "Yes"
            else "🌍 No — Exotic / Imported"
        )

        clr = (
            "#4ade80"
            if val == "Yes"
            else "#f87171"
        )

        bdr = (
            "rgba(74,222,128,0.4)"
            if val == "Yes"
            else "rgba(248,113,113,0.4)"
        )

        bgc = (
            "rgba(34,197,94,0.1)"
            if val == "Yes"
            else "rgba(239,68,68,0.1)"
        )

        html += f"""
        <div style="background:{bgc};
                    border:1px solid {bdr};
                    border-radius:10px;padding:10px 14px;
                    margin-bottom:10px;font-weight:600;
                    color:{clr};font-size:13px;">
            🇮🇳 {icon}
        </div>
        """

    # Special Note
    if "Special Note" in breed_info:

        html += f"""
        <div style="background:rgba(251,191,36,0.08);
                    border-left:4px solid #fbbf24;
                    border-radius:0 10px 10px 0;
                    padding:12px 14px;margin-bottom:14px;
                    border-top:1px solid rgba(251,191,36,0.2);
                    border-bottom:1px solid rgba(251,191,36,0.2);
                    border-right:1px solid rgba(251,191,36,0.2);">

            <div style="font-weight:600;color:#fbbf24;
                        margin-bottom:4px;font-size:13px;">
                ⭐ Special Note
            </div>

            <div style="color:#fde68a;font-size:13px;">
                {breed_info["Special Note"]}
            </div>

        </div>
        """

    # Top 5 Predictions
    top5 = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    html += f"""
    <div style="background:{bg_card};
                border-radius:10px;padding:14px;
                border:1px solid {border};">

        <div style="font-weight:600;color:{accent};
                    margin-bottom:12px;font-size:13px;">
            📊 Top 5 Predictions
        </div>
    """

    for breed, prob in top5:

        pct = round(
            prob * 100,
            1
        )

        is_top = breed == top_breed

        bar_clr = (
            f"linear-gradient(90deg,{accent2},{accent})"
            if is_top
            else "#374151"
        )

        txt_clr = (
            "white"
            if is_top
            else txt_sub
        )

        pct_clr = (
            accent
            if is_top
            else txt_sub
        )

        html += f"""
        <div style="margin-bottom:10px;">

            <div style="display:flex;
                        justify-content:space-between;
                        font-size:13px;margin-bottom:4px;">

                <span style="font-weight:
                    {'700' if is_top else '400'};
                    color:{txt_clr};">

                    {em + ' ' if is_top else ''}
                    {breed.replace('_',' ').title()}

                </span>

                <span style="font-weight:600;
                             color:{pct_clr};">
                    {pct}%
                </span>

            </div>

            <div style="background:#1e293b;
                        border-radius:6px;
                        overflow:hidden;height:8px;">

                <div style="background:{bar_clr};
                            height:100%;width:{pct}%;
                            border-radius:6px;">
                </div>

            </div>

        </div>
        """

    html += "</div></div>"

    return html


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(animal_type, image):

    try:

        if image is None:
            return (
                "Please upload an image first.",
                {},
                ""
            )

        img = image.convert("RGB")

        tensor = transform(
            img
        ).unsqueeze(0).to(device)

        if animal_type == "Buffalo":

            model = buffalo_model
            labels = BUFFALO_CLASSES
            data = BUFFALO_DATA

        elif animal_type == "Cow":

            model = cow_model
            labels = COW_CLASSES
            data = COW_DATA

        else:

            model = sheep_model
            labels = SHEEP_CLASSES
            data = SHEEP_DATA

        with torch.no_grad():

            probs = torch.softmax(
                model(tensor),
                dim=1
            )[0].cpu().tolist()

        scores = {
            labels[i]: round(
                probs[i],
                4
            )
            for i in range(len(labels))
        }

        top_breed = max(
            scores,
            key=scores.get
        )

        conf = round(
            scores[top_breed] * 100,
            1
        )

        breed_info = data.get(
            top_breed,
            {}
        )

        html = build_html_table(
            top_breed,
            conf,
            animal_type,
            breed_info,
            scores
        )

        result_txt = (
            f"Predicted: "
            f"{top_breed.replace('_',' ').title()} "
            f"| Confidence: {conf}% "
            f"| Animal: {animal_type}"
        )

        return (
            result_txt,
            scores,
            html
        )

    except Exception as e:

        err = traceback.format_exc()

        print(err)

        return (
            f"Error: {str(e)}",
            {},
            f"<pre style='color:red'>{err}</pre>"
        )


# ============================================================
# GRADIO UI — DARK BLUE/PURPLE THEME
# ============================================================
dark_css = """
    body, .gradio-container {
        background-color: #0f1117 !important;
        color: #e2e8f0 !important;
    }

    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }

    .block, .panel, .form, .gap, .gr-box,
    .gr-padded, .gr-panel, .gr-form,
    div[data-testid="block"] {
        background-color: #1a1d2e !important;
        border-color: #2d3748 !important;
    }

    label, .label-wrap span, .svelte-1gfkn6j {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    input, textarea, .scroll-hide {
        background-color: #1e2235 !important;
        color: #e2e8f0 !important;
        border-color: #3b4a6b !important;
    }

    .wrap.svelte-12cmxck, .wrap {
        background-color: #1e2235 !important;
        border-color: #3b4a6b !important;
        color: #e2e8f0 !important;
    }

    input[type="radio"]:checked + span {
        color: #3b82f6 !important;
        font-weight: 700 !important;
    }

    .primary, button.primary, .gr-button-primary {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    }

    .primary:hover {
        background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
    }

    .upload-container, .image-container,
    div[data-testid="image"] {
        background-color: #1e2235 !important;
        border: 2px dashed #3b4a6b !important;
        border-radius: 12px !important;
    }

    .label-container, .output-class {
        background-color: #1e2235 !important;
        color: #e2e8f0 !important;
    }

    .confidence-bar > div {
        background: linear-gradient(90deg, #1d4ed8, #8b5cf6) !important;
    }

    h1 {
        text-align: center !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    h3 {
        text-align: center !important;
        color: #64748b !important;
        font-weight: 400 !important;
    }

    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1d2e;
    }

    ::-webkit-scrollbar-thumb {
        background: #3b4a6b;
        border-radius: 3px;
    }

    footer, .built-with {
        display: none !important;
    }
"""


# ============================================================
# GRADIO APP
# ============================================================
with gr.Blocks(
    title="AI Based Livestock Breed Recognition"
) as demo:

    gr.Markdown(
        "# 🐾 AI Based Breed Recognition System"
    )

    gr.Markdown(
        "### Identify Indian Buffalo · Cow · Sheep breeds using Deep Learning"
    )

    gr.Markdown("""
    <div style="text-align:center;
                background:rgba(59,130,246,0.1);
                border:1px solid rgba(59,130,246,0.3);
                padding:8px 14px;border-radius:8px;
                font-size:13px;color:#93c5fd;
                margin-bottom:4px;">

        🐃 Buffalo:
        <strong style="color:#60a5fa;">
            Available
        </strong>
        (17 breeds)

        &nbsp;|&nbsp;

        🐄 Cow:
        <strong style="color:#a78bfa;">
            Available
        </strong>
        (11 breeds)

        &nbsp;|&nbsp;

        🐑 Sheep:
        <strong style="color:#818cf8;">
            Available
        </strong>
        (7 breeds)

    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            animal_type = gr.Radio(
                choices=[
                    "Buffalo",
                    "Cow",
                    "Sheep"
                ],
                value="Buffalo",
                label="🐾 Select Animal Type",
            )

            image_input = gr.Image(
                label="📷 Upload Animal Image",
                type="pil",
                sources=["upload"],
                height=300,
            )

            predict_btn = gr.Button(
                "🔍 Identify Breed",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):

            prediction_text = gr.Textbox(
                label="🎯 Prediction Result",
                interactive=False
            )

            confidence_bar = gr.Label(
                label="📊 Confidence Scores",
                num_top_classes=5
            )

            breed_details = gr.HTML(
                label="📋 Breed Details"
            )

    predict_btn.click(
        fn=predict,
        inputs=[
            animal_type,
            image_input
        ],
        outputs=[
            prediction_text,
            confidence_bar,
            breed_details
        ]
    )


# ============================================================
# LAUNCH FOR RENDER
# ============================================================
print("✅ Launching Gradio App...")

demo.launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get(
            "PORT",
            7860
        )
    ),
    debug=False,
    theme=gr.themes.Base(),
    css=dark_css
)
