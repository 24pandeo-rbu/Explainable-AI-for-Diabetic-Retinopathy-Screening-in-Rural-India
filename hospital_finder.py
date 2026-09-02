"""
Hospital Finder Module for EYE-CONIQ.

Provides a curated database of eye hospitals across Indian states
with map visualization using Folium.
"""

import folium
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Eye Hospital Database (Major hospitals across India)
# ──────────────────────────────────────────────

HOSPITALS = [
    # ── Delhi / NCR ──
    {"name": "AIIMS Eye Centre", "city": "New Delhi", "state": "Delhi",
     "address": "Ansari Nagar East, New Delhi 110029", "phone": "011-26588500",
     "lat": 28.5672, "lng": 77.2100, "type": "Government", "specialists": ["Retina", "Vitreoretinal Surgery"],
     "emergency": True, "ambulance": "108"},
    {"name": "Dr. Rajendra Prasad Centre for Ophthalmic Sciences (RPC)", "city": "New Delhi", "state": "Delhi",
     "address": "AIIMS Campus, New Delhi 110029", "phone": "011-26593135",
     "lat": 28.5680, "lng": 77.2105, "type": "Government", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    {"name": "Guru Nanak Eye Centre", "city": "New Delhi", "state": "Delhi",
     "address": "Maharaja Ranjit Singh Marg, New Delhi 110002", "phone": "011-23234296",
     "lat": 28.6380, "lng": 77.2218, "type": "Government", "specialists": ["Retina", "Laser Treatment"],
     "emergency": True, "ambulance": "108"},
    {"name": "Centre for Sight", "city": "New Delhi", "state": "Delhi",
     "address": "B-5/24, Safdarjung Enclave, New Delhi 110029", "phone": "011-49601919",
     "lat": 28.5612, "lng": 77.1954, "type": "Private", "specialists": ["Retina", "Vitreoretinal", "Anti-VEGF"],
     "emergency": True, "ambulance": "108"},
    
    # ── Maharashtra ──
    {"name": "Sankara Nethralaya (Mumbai)", "city": "Mumbai", "state": "Maharashtra",
     "address": "Andheri West, Mumbai 400058", "phone": "022-28260022",
     "lat": 19.1248, "lng": 72.8364, "type": "Trust", "specialists": ["Retina", "DR Screening", "Laser"],
     "emergency": True, "ambulance": "108"},
    {"name": "Aravind Eye Hospital (Mumbai)", "city": "Mumbai", "state": "Maharashtra",
     "address": "Dadar West, Mumbai 400028", "phone": "022-24300300",
     "lat": 19.0178, "lng": 72.8478, "type": "Trust", "specialists": ["Retina", "Vitreoretinal Surgery"],
     "emergency": True, "ambulance": "108"},
    {"name": "LV Prasad Eye Institute (Hyderabad Branch)", "city": "Pune", "state": "Maharashtra",
     "address": "Baner Road, Pune 411045", "phone": "020-67448899",
     "lat": 18.5570, "lng": 73.7898, "type": "Trust", "specialists": ["Retina", "DR Clinic"],
     "emergency": False, "ambulance": "108"},
    {"name": "H.V. Desai Eye Hospital", "city": "Pune", "state": "Maharashtra",
     "address": "93, Tarawade Wasti, Mohammadwadi, Pune 411060", "phone": "020-26990599",
     "lat": 18.4722, "lng": 73.9116, "type": "Trust", "specialists": ["Retina", "Laser"],
     "emergency": True, "ambulance": "108"},
    
    # ── Tamil Nadu ──
    {"name": "Sankara Nethralaya", "city": "Chennai", "state": "Tamil Nadu",
     "address": "18, College Road, Nungambakkam, Chennai 600006", "phone": "044-28271616",
     "lat": 13.0589, "lng": 80.2372, "type": "Trust", "specialists": ["Retina", "Vitreoretinal", "Anti-VEGF", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    {"name": "Aravind Eye Hospital (Madurai)", "city": "Madurai", "state": "Tamil Nadu",
     "address": "1, Anna Nagar, Madurai 625020", "phone": "0452-4356100",
     "lat": 9.9459, "lng": 78.1210, "type": "Trust", "specialists": ["Retina", "Vitreoretinal Surgery", "Laser"],
     "emergency": True, "ambulance": "108"},
    {"name": "Aravind Eye Hospital (Coimbatore)", "city": "Coimbatore", "state": "Tamil Nadu",
     "address": "Avinashi Road, Coimbatore 641014", "phone": "0422-4360400",
     "lat": 11.0236, "lng": 76.9766, "type": "Trust", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    
    # ── Telangana ──
    {"name": "LV Prasad Eye Institute", "city": "Hyderabad", "state": "Telangana",
     "address": "Kallam Anji Reddy Campus, Banjara Hills, Hyderabad 500034", "phone": "040-30612345",
     "lat": 17.4239, "lng": 78.4738, "type": "Trust", "specialists": ["Retina", "Vitreoretinal", "DR Clinic", "Anti-VEGF"],
     "emergency": True, "ambulance": "108"},
    {"name": "Maxivision Eye Hospital", "city": "Hyderabad", "state": "Telangana",
     "address": "Begumpet, Hyderabad 500016", "phone": "040-44455555",
     "lat": 17.4400, "lng": 78.4700, "type": "Private", "specialists": ["Retina", "Laser"],
     "emergency": True, "ambulance": "108"},
    
    # ── Karnataka ──
    {"name": "Narayana Nethralaya", "city": "Bangalore", "state": "Karnataka",
     "address": "121/C, Chord Road, Rajajinagar, Bangalore 560010", "phone": "080-66121600",
     "lat": 12.9887, "lng": 77.5523, "type": "Private", "specialists": ["Retina", "Vitreoretinal", "Anti-VEGF"],
     "emergency": True, "ambulance": "108"},
    {"name": "Minto Eye Hospital (BMCRI)", "city": "Bangalore", "state": "Karnataka",
     "address": "Chamrajpet, Bangalore 560002", "phone": "080-26706509",
     "lat": 12.9590, "lng": 77.5728, "type": "Government", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    
    # ── Kerala ──
    {"name": "Aravind Eye Hospital (Pondicherry)", "city": "Pondicherry", "state": "Kerala",
     "address": "Cuddalore Main Road, Pondicherry 605007", "phone": "0413-2619100",
     "lat": 11.9139, "lng": 79.8145, "type": "Trust", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    {"name": "Little Flower Hospital & Research Centre", "city": "Angamaly", "state": "Kerala",
     "address": "Angamaly, Ernakulam 683572", "phone": "0484-2452214",
     "lat": 10.1960, "lng": 76.3860, "type": "Private", "specialists": ["Retina", "Vitreoretinal"],
     "emergency": True, "ambulance": "108"},
    
    # ── West Bengal ──
    {"name": "LV Prasad Eye Institute (Bhubaneswar)", "city": "Bhubaneswar", "state": "Odisha",
     "address": "Plot No. 81, Infocity Road, Bhubaneswar 751024", "phone": "0674-3989600",
     "lat": 20.3350, "lng": 85.8138, "type": "Trust", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    {"name": "Disha Eye Hospital", "city": "Kolkata", "state": "West Bengal",
     "address": "88B, Topsia Road, Kolkata 700046", "phone": "033-40220000",
     "lat": 22.5397, "lng": 88.3818, "type": "Private", "specialists": ["Retina", "Vitreoretinal"],
     "emergency": True, "ambulance": "108"},
    {"name": "Susrut Eye Foundation", "city": "Kolkata", "state": "West Bengal",
     "address": "HB-36/A/1, Sector III, Salt Lake, Kolkata 700106", "phone": "033-23352525",
     "lat": 22.5800, "lng": 88.4140, "type": "Trust", "specialists": ["Retina", "DR Screening", "Laser"],
     "emergency": True, "ambulance": "108"},
    
    # ── Rajasthan ──
    {"name": "PBMA H.V. Desai Eye Hospital", "city": "Jaipur", "state": "Rajasthan",
     "address": "Tonk Road, Jaipur 302018", "phone": "0141-2704055",
     "lat": 26.8615, "lng": 75.8003, "type": "Trust", "specialists": ["Retina", "Laser"],
     "emergency": True, "ambulance": "108"},
    
    # ── Uttar Pradesh ──
    {"name": "King George's Medical University Eye Dept", "city": "Lucknow", "state": "Uttar Pradesh",
     "address": "Shah Mina Road, Lucknow 226003", "phone": "0522-2257540",
     "lat": 26.8567, "lng": 80.9462, "type": "Government", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    
    # ── Gujarat ──
    {"name": "M & J Western Regional Institute of Ophthalmology", "city": "Ahmedabad", "state": "Gujarat",
     "address": "Civil Hospital Campus, Ahmedabad 380016", "phone": "079-22683721",
     "lat": 23.0348, "lng": 72.5862, "type": "Government", "specialists": ["Retina", "Vitreoretinal"],
     "emergency": True, "ambulance": "108"},
    {"name": "Iladevi Cataract & IOL Research Centre", "city": "Ahmedabad", "state": "Gujarat",
     "address": "Memnagar, Ahmedabad 380052", "phone": "079-27492303",
     "lat": 23.0444, "lng": 72.5387, "type": "Private", "specialists": ["Retina", "Anti-VEGF"],
     "emergency": True, "ambulance": "108"},
    
    # ── Punjab ──
    {"name": "Advanced Eye Centre, PGIMER", "city": "Chandigarh", "state": "Punjab",
     "address": "PGIMER Campus, Sector 12, Chandigarh 160012", "phone": "0172-2747837",
     "lat": 30.7635, "lng": 76.7794, "type": "Government", "specialists": ["Retina", "Vitreoretinal", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    
    # ── Madhya Pradesh ──
    {"name": "Aravind Eye Hospital (Tirunelveli)", "city": "Tirunelveli", "state": "Tamil Nadu",
     "address": "Tirunelveli 627001", "phone": "0462-2338800",
     "lat": 8.7139, "lng": 77.7567, "type": "Trust", "specialists": ["Retina", "DR Screening"],
     "emergency": True, "ambulance": "108"},
    
    # ── Assam ──
    {"name": "Sri Sankaradeva Nethralaya", "city": "Guwahati", "state": "Assam",
     "address": "Beltola, Guwahati 781028", "phone": "0361-2302300",
     "lat": 26.1280, "lng": 91.7986, "type": "Trust", "specialists": ["Retina", "Vitreoretinal", "DR Screening"],
     "emergency": True, "ambulance": "108"},
]

# State-wise grouping
STATES = sorted(set(h["state"] for h in HOSPITALS))

# Emergency numbers
EMERGENCY_NUMBERS = {
    "ambulance": "108",
    "emergency": "112",
    "national_eye_helpline": "1800-345-4545",
}


def get_hospitals_by_state(state: str) -> List[Dict]:
    """Get all hospitals in a given state."""
    return [h for h in HOSPITALS if h["state"] == state]


def get_all_states() -> List[str]:
    """Get list of all states with hospitals."""
    return STATES


def get_nearest_hospitals(state: Optional[str] = None, 
                          emergency_only: bool = False) -> List[Dict]:
    """
    Get hospitals, optionally filtered by state and emergency availability.
    """
    results = HOSPITALS
    if state:
        results = [h for h in results if h["state"] == state]
    if emergency_only:
        results = [h for h in results if h.get("emergency", False)]
    return results


def create_hospital_map(hospitals: List[Dict], 
                        center_lat: float = 20.5937, 
                        center_lng: float = 78.9629,
                        zoom: int = 5) -> folium.Map:
    """
    Create a Folium map with hospital markers.
    
    Args:
        hospitals: List of hospital dicts to plot.
        center_lat: Map center latitude (default: center of India).
        center_lng: Map center longitude.
        zoom: Initial zoom level.
    
    Returns:
        Folium Map object.
    """
    # If hospitals provided, center on them
    if hospitals:
        center_lat = sum(h["lat"] for h in hospitals) / len(hospitals)
        center_lng = sum(h["lng"] for h in hospitals) / len(hospitals)
        if len(hospitals) <= 3:
            zoom = 10
        elif len(hospitals) <= 8:
            zoom = 7
        else:
            zoom = 5
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    
    for hospital in hospitals:
        # Color based on type
        color_map = {
            "Government": "blue",
            "Trust": "green",
            "Private": "red",
        }
        color = color_map.get(hospital.get("type", ""), "gray")
        icon_name = "hospital" if hospital.get("emergency") else "medkit"
        
        # Build popup content
        specialists = ", ".join(hospital.get("specialists", []))
        emergency_badge = "🚨 24/7 Emergency" if hospital.get("emergency") else ""
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 250px;">
            <h4 style="color: #0066cc; margin: 0 0 8px 0;">{hospital['name']}</h4>
            <p style="margin: 2px 0; font-size: 12px;">
                📍 {hospital['address']}
            </p>
            <p style="margin: 2px 0; font-size: 12px;">
                📞 <a href="tel:{hospital['phone']}">{hospital['phone']}</a>
            </p>
            <p style="margin: 2px 0; font-size: 12px;">
                👨‍⚕️ <b>Specialists:</b> {specialists}
            </p>
            <p style="margin: 2px 0; font-size: 12px;">
                🏥 <b>Type:</b> {hospital.get('type', 'N/A')}
            </p>
            {'<p style="margin: 4px 0; font-size: 13px; color: red; font-weight: bold;">' + emergency_badge + '</p>' if emergency_badge else ''}
            <hr style="margin: 8px 0;">
            <div style="display: flex; gap: 8px;">
                <a href="tel:108" style="background: #ff4444; color: white; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: bold;">
                    🚑 Call 108
                </a>
                <a href="tel:{hospital['phone']}" style="background: #0066cc; color: white; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px;">
                    📞 Call Hospital
                </a>
            </div>
        </div>
        """
        
        folium.Marker(
            location=[hospital["lat"], hospital["lng"]],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=hospital["name"],
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(m)
    
    # Add a legend
    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
                background: white; padding: 10px 15px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 12px;">
        <b>Hospital Types</b><br>
        <span style="color: blue;">●</span> Government<br>
        <span style="color: green;">●</span> Trust / NGO<br>
        <span style="color: red;">●</span> Private
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m
