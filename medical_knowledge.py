"""
Medical Knowledge Base for Diabetic Retinopathy.

Provides detailed multilingual explanations of:
- What the disease is
- Why it happened (causes)
- How to cure/manage it
- Lifestyle changes
- When to seek emergency care
"""


def get_medical_info(severity_level: int, lang: str = "en") -> dict:
    """
    Get comprehensive medical information for a given DR severity level.
    
    Returns dict with keys: what_is, causes, symptoms, treatment,
    lifestyle, urgency_message, do_list, dont_list
    """
    info = MEDICAL_INFO.get(lang, MEDICAL_INFO["en"])
    return info.get(severity_level, info[0])


# ──────────────────────────────────────────────
# Medical Information Database (Multilingual)
# ──────────────────────────────────────────────

MEDICAL_INFO = {
    "en": {
        0: {
            "what_is": "Your retinal scan shows **no signs of Diabetic Retinopathy**. The blood vessels in your retina appear healthy with no visible damage.",
            "causes": "Even though your eyes are currently healthy, having diabetes puts you at risk. High blood sugar over time can damage the tiny blood vessels in the retina.",
            "symptoms": "Currently no symptoms. However, diabetic retinopathy often has no early symptoms — damage can occur before you notice any vision changes.",
            "treatment": "No treatment needed at this time. Continue regular monitoring.",
            "lifestyle": [
                "🩸 Keep blood sugar levels under control (HbA1c < 7%)",
                "🫀 Monitor blood pressure regularly (target < 130/80 mmHg)",
                "🏃 Exercise regularly — at least 30 minutes of walking daily",
                "🥗 Eat a balanced diet rich in green vegetables and fruits",
                "🚭 Avoid smoking and limit alcohol consumption",
                "👁️ Get annual eye screening even if vision feels normal",
            ],
            "urgency": "none",
            "urgency_message": "✅ No immediate concern. Schedule routine screening in 12 months.",
            "do_list": [
                "Take diabetes medication regularly as prescribed",
                "Get your eyes checked every year",
                "Maintain a healthy weight",
                "Keep blood sugar diary",
            ],
            "dont_list": [
                "Don't skip diabetes medications",
                "Don't ignore regular eye check-ups",
                "Don't eat excessive sugar or processed foods",
                "Don't smoke",
            ],
        },
        1: {
            "what_is": "**Mild Non-Proliferative Diabetic Retinopathy (Mild NPDR)** has been detected. Small balloon-like swellings called **microaneurysms** are visible in the small blood vessels of your retina. This is the earliest stage of diabetic eye disease.",
            "causes": "Prolonged high blood sugar levels have caused the walls of tiny retinal blood vessels to weaken. These weakened vessels develop small bulges (microaneurysms) that may leak small amounts of fluid or blood into the retina.",
            "symptoms": "Usually **no noticeable symptoms** at this stage. Vision is typically still normal. This is why regular screening is so important — the disease can be caught early before vision loss occurs.",
            "treatment": "**No immediate medical treatment is usually needed**, but careful monitoring is essential:\n\n- Repeat eye screening in **9-12 months**\n- Focus on blood sugar control\n- Regular monitoring of blood pressure\n- No laser treatment or injections needed at this stage",
            "lifestyle": [
                "🩸 Strictly control blood sugar — aim for HbA1c below 7%",
                "💊 Take diabetes medications exactly as prescribed",
                "🫀 Check blood pressure weekly — keep it below 130/80 mmHg",
                "🥗 Follow a diabetic diet — low sugar, high fiber",
                "🏃 Walk for at least 30 minutes daily",
                "🚭 Stop smoking immediately — it accelerates vessel damage",
                "👁️ Return for eye screening in 9-12 months",
            ],
            "urgency": "low",
            "urgency_message": "🟡 Early stage detected. No immediate danger, but strict diabetes management is crucial to prevent progression.",
            "do_list": [
                "Control blood sugar strictly",
                "Take medications on time",
                "Get eyes rechecked in 9-12 months",
                "Exercise regularly",
                "Eat healthy foods",
            ],
            "dont_list": [
                "Don't panic — this is very early stage and manageable",
                "Don't skip medications or eye check-ups",
                "Don't eat sweets, fried foods, or white rice excessively",
                "Don't smoke or chew tobacco",
                "Don't ignore blood pressure",
            ],
        },
        2: {
            "what_is": "**Moderate Non-Proliferative Diabetic Retinopathy (Moderate NPDR)** has been detected. Beyond microaneurysms, there are now **retinal hemorrhages** (bleeding), **hard exudates** (fatty deposits), and possibly **cotton wool spots** (areas of poor blood flow). The blood vessels are becoming more damaged.",
            "causes": "Continued high blood sugar has caused significant damage to retinal blood vessels. Blood and fluid are leaking from damaged vessels, and some areas of the retina are not receiving enough oxygen. Fatty deposits (exudates) are accumulating in the retina.",
            "symptoms": "You may start noticing:\n- Slightly blurred or fluctuating vision\n- Dark spots or floaters\n- Difficulty reading or seeing fine details\n- Colors appearing faded\n\n**Even if vision seems normal, damage is occurring.**",
            "treatment": "**Referral to an ophthalmologist within 3-6 months is recommended:**\n\n- Comprehensive dilated eye examination needed\n- OCT (Optical Coherence Tomography) scan may be required\n- Intensive blood sugar and blood pressure management\n- Possible **anti-VEGF injections** if macular edema is present\n- Close monitoring every 3-6 months",
            "lifestyle": [
                "🩸 Urgent blood sugar control — aim for HbA1c below 6.5%",
                "💊 You may need adjustment in diabetes medication — consult your doctor",
                "🫀 Blood pressure must be strictly controlled — below 130/80 mmHg",
                "🍽️ Strict diabetic diet — avoid all sugary drinks, processed food",
                "🏃 Regular moderate exercise — walking, cycling, swimming",
                "🚭 Stop smoking completely — it worsens blood vessel damage",
                "💧 Stay well hydrated — drink plenty of water",
                "👁️ See an eye specialist within 3-6 months",
            ],
            "urgency": "medium",
            "urgency_message": "🟠 **Moderate damage detected. Please see an eye doctor within 3-6 months.** Blood sugar control is now critical to prevent further damage.",
            "do_list": [
                "Visit an eye specialist within 3-6 months",
                "Get blood sugar under strict control immediately",
                "Monitor blood pressure daily",
                "Take all medications as prescribed",
                "Eat diabetic-friendly meals",
                "Exercise regularly",
            ],
            "dont_list": [
                "Don't delay the eye specialist visit",
                "Don't skip any medications",
                "Don't eat sugary foods, fried snacks, or white rice",
                "Don't smoke, chew tobacco, or drink alcohol",
                "Don't lift very heavy weights (increases eye pressure)",
                "Don't ignore any changes in vision",
            ],
        },
        3: {
            "what_is": "**Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR)** has been detected. This is a serious stage where many blood vessels are blocked, starving the retina of blood supply. There are **extensive hemorrhages**, **venous beading** (distorted veins), and **IRMA** (abnormal blood vessel patterns). The retina is sending signals to grow new blood vessels, which is dangerous.",
            "causes": "Chronic uncontrolled diabetes has caused severe damage to retinal blood vessels. Many vessels are now blocked, cutting off blood supply to large areas of the retina. The retina is becoming oxygen-deprived (ischemic), which triggers the growth of abnormal new blood vessels — a dangerous process.",
            "symptoms": "You may experience:\n- **Significant vision blurring**\n- Dark spots or large floaters\n- Difficulty seeing at night\n- Sudden vision changes\n- Reading becomes very difficult\n\n⚠️ **Vision loss can become permanent if not treated urgently.**",
            "treatment": "**URGENT: See a retina specialist within 2-4 weeks!**\n\n- **Pan-retinal photocoagulation (PRP)** laser treatment may be needed to prevent progression to proliferative DR\n- **Anti-VEGF injections** (Avastin/Lucentis/Eylea) to stop abnormal vessel growth\n- **Intravitreal steroid injections** if significant edema\n- Intensive systemic management of diabetes, blood pressure, and cholesterol\n- Frequent monitoring every 2-3 months",
            "lifestyle": [
                "🚨 See a retina specialist within 2-4 weeks — this is urgent",
                "🩸 Immediate strict blood sugar control — consult your doctor TODAY",
                "💊 You likely need medication changes — see your diabetologist immediately",
                "🫀 Blood pressure must be below 130/80 — check daily",
                "🍽️ Very strict diet — no sugar, no processed food, reduce carbohydrates",
                "🏃 Gentle exercise only — avoid heavy lifting or straining",
                "🚭 Absolutely no smoking or tobacco",
                "😰 Manage stress — it raises blood sugar and blood pressure",
            ],
            "urgency": "high",
            "urgency_message": "🔴 **SEVERE DAMAGE DETECTED! Urgent referral to a retina specialist within 2-4 weeks is essential.** Without treatment, this can progress to sight-threatening proliferative DR.",
            "do_list": [
                "See a retina specialist URGENTLY within 2-4 weeks",
                "Consult your diabetologist for medication adjustment",
                "Monitor blood sugar multiple times daily",
                "Take all medications without fail",
                "Keep this report and show it to the eye doctor",
            ],
            "dont_list": [
                "Don't delay the specialist visit — this is urgent",
                "Don't miss any medications",
                "Don't lift heavy objects or strain",
                "Don't rub your eyes",
                "Don't ignore any sudden vision changes — go to hospital immediately",
                "Don't smoke or use tobacco in any form",
            ],
        },
        4: {
            "what_is": "**Proliferative Diabetic Retinopathy (PDR)** — the most advanced and dangerous stage — has been detected. **Abnormal new blood vessels (neovascularization)** are growing on the surface of the retina and into the vitreous (jelly inside the eye). These fragile new vessels can rupture and bleed, causing **vitreous hemorrhage** and potentially **tractional retinal detachment**, leading to **permanent blindness**.",
            "causes": "Prolonged severe oxygen deprivation of the retina has triggered the growth of abnormal blood vessels (neovascularization). These vessels are extremely fragile and can:\n- **Bleed into the vitreous** (vitreous hemorrhage) — causing sudden vision loss\n- **Pull on the retina** (tractional retinal detachment) — causing permanent blindness\n- **Block fluid drainage** — causing neovascular glaucoma",
            "symptoms": "Serious symptoms that require immediate attention:\n- **Sudden severe vision loss**\n- Large dark floaters or \"curtain\" across vision\n- Flashes of light\n- Severe blurring that doesn't clear\n- Pain or pressure in the eye\n\n🚨 **THIS IS A MEDICAL EMERGENCY if you experience sudden vision loss.**",
            "treatment": "**IMMEDIATE SPECIALIST REFERRAL REQUIRED!**\n\n- **Pan-retinal photocoagulation (PRP) laser** — to destroy oxygen-deprived retina and stop new vessel growth\n- **Anti-VEGF injections** — Avastin, Lucentis, or Eylea injected into the eye\n- **Vitrectomy surgery** — if vitreous hemorrhage or retinal detachment occurs\n- **Combination therapy** — often multiple treatments are needed\n- **Lifelong monitoring** — even after treatment, regular follow-up is essential",
            "lifestyle": [
                "🚨 **EMERGENCY**: See a retina specialist IMMEDIATELY — do not wait",
                "🏥 Go to the nearest eye hospital TODAY",
                "🩸 Blood sugar control is now life-and-sight critical",
                "💊 Do not miss any medication — this is an emergency",
                "🫀 Blood pressure must be controlled strictly",
                "🍽️ Follow strict diabetic diet as advised by your doctor",
                "🏃 No strenuous activity — avoid heavy lifting, bending",
                "🚭 Absolutely no smoking, tobacco, or alcohol",
                "😰 Stay calm — stress can worsen the condition",
                "🚑 If you experience sudden vision loss, call 108 immediately",
            ],
            "urgency": "critical",
            "urgency_message": "🔴🔴 **CRITICAL EMERGENCY! Proliferative DR detected — immediate specialist referral required!** Without urgent treatment (laser/injections/surgery), this can lead to PERMANENT BLINDNESS. Go to the nearest eye hospital IMMEDIATELY!",
            "do_list": [
                "Go to an eye hospital IMMEDIATELY — TODAY",
                "Call 108 for ambulance if sudden vision loss occurs",
                "Bring this report to show the eye doctor",
                "Keep taking all medications",
                "Stay calm and avoid physical strain",
                "Have someone accompany you to the hospital",
            ],
            "dont_list": [
                "🚫 Do NOT delay — every day matters",
                "🚫 Do NOT rub or press your eyes",
                "🚫 Do NOT lift heavy objects",
                "🚫 Do NOT bend forward for long periods",
                "🚫 Do NOT ignore sudden vision changes — this is an emergency",
                "🚫 Do NOT smoke, use tobacco, or drink alcohol",
                "🚫 Do NOT try home remedies — see a doctor",
            ],
        },
    },
    
    # ── Hindi ──
    "hi": {
        0: {
            "what_is": "आपकी रेटिना स्कैन में **डायबिटिक रेटिनोपैथी का कोई संकेत नहीं** दिखा। आपकी आंख की रक्त वाहिकाएं स्वस्थ दिखती हैं।",
            "causes": "भले ही आपकी आंखें अभी स्वस्थ हैं, मधुमेह होने से आपको खतरा है। लंबे समय तक उच्च रक्त शर्करा रेटिना की छोटी रक्त वाहिकाओं को नुकसान पहुंचा सकती है।",
            "symptoms": "वर्तमान में कोई लक्षण नहीं। हालांकि, डायबिटिक रेटिनोपैथी में अक्सर शुरुआती लक्षण नहीं होते — आपको दृष्टि में बदलाव महसूस होने से पहले ही नुकसान हो सकता है।",
            "treatment": "इस समय कोई उपचार की आवश्यकता नहीं। नियमित निगरानी जारी रखें।",
            "lifestyle": [
                "🩸 रक्त शर्करा को नियंत्रण में रखें (HbA1c < 7%)",
                "🫀 नियमित रूप से रक्तचाप की जांच करें (लक्ष्य < 130/80 mmHg)",
                "🏃 रोजाना कम से कम 30 मिनट पैदल चलें",
                "🥗 हरी सब्जियों और फलों से भरपूर संतुलित आहार लें",
                "🚭 धूम्रपान से बचें और शराब सीमित करें",
                "👁️ हर साल आंखों की जांच कराएं, भले ही दृष्टि सामान्य लगे",
            ],
            "urgency": "none",
            "urgency_message": "✅ कोई तत्काल चिंता नहीं। 12 महीने में नियमित जांच कराएं।",
            "do_list": ["मधुमेह की दवाई नियमित लें", "हर साल आंखों की जांच कराएं", "स्वस्थ वजन बनाए रखें", "रक्त शर्करा की डायरी रखें"],
            "dont_list": ["मधुमेह की दवाइयां न छोड़ें", "नियमित आंखों की जांच न भूलें", "अत्यधिक चीनी या प्रोसेस्ड खाद्य पदार्थ न खाएं", "धूम्रपान न करें"],
        },
        1: {
            "what_is": "**हल्की नॉन-प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (Mild NPDR)** पाई गई है। आपकी रेटिना की छोटी रक्त वाहिकाओं में **माइक्रोएन्यूरिज्म** (छोटे गुब्बारे जैसी सूजन) दिखाई दे रहे हैं। यह डायबिटिक आंख की बीमारी का सबसे शुरुआती चरण है।",
            "causes": "लंबे समय तक उच्च रक्त शर्करा के कारण रेटिना की छोटी रक्त वाहिकाओं की दीवारें कमजोर हो गई हैं। इन कमजोर वाहिकाओं में छोटे उभार (माइक्रोएन्यूरिज्म) बन जाते हैं।",
            "symptoms": "आमतौर पर इस चरण में **कोई ध्यान देने योग्य लक्षण नहीं** होते। दृष्टि आमतौर पर सामान्य रहती है। यही कारण है कि नियमित जांच इतनी महत्वपूर्ण है।",
            "treatment": "**तत्काल चिकित्सा उपचार की आवश्यकता नहीं**, लेकिन सावधानीपूर्वक निगरानी जरूरी है:\n\n- **9-12 महीने** में आंखों की दोबारा जांच\n- रक्त शर्करा नियंत्रण पर ध्यान दें\n- इस चरण में लेजर या इंजेक्शन की जरूरत नहीं",
            "lifestyle": ["🩸 रक्त शर्करा सख्ती से नियंत्रित करें", "💊 दवाइयां समय पर लें", "🫀 रक्तचाप साप्ताहिक जांचें", "🥗 डायबिटिक आहार का पालन करें", "🏃 रोजाना 30 मिनट पैदल चलें", "🚭 तुरंत धूम्रपान बंद करें"],
            "urgency": "low",
            "urgency_message": "🟡 शुरुआती चरण पाया गया। तत्काल खतरा नहीं है, लेकिन मधुमेह प्रबंधन महत्वपूर्ण है।",
            "do_list": ["रक्त शर्करा सख्ती से नियंत्रित करें", "दवाइयां समय पर लें", "9-12 महीने में आंखों की दोबारा जांच कराएं", "नियमित व्यायाम करें"],
            "dont_list": ["घबराएं नहीं — यह बहुत शुरुआती चरण है", "दवाइयां या आंखों की जांच न छोड़ें", "मिठाई, तले हुए खाद्य पदार्थ न खाएं", "धूम्रपान न करें"],
        },
        2: {
            "what_is": "**मध्यम नॉन-प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (Moderate NPDR)** पाई गई है। माइक्रोएन्यूरिज्म के अलावा, अब **रेटिनल हेमरेज** (रक्तस्राव), **हार्ड एक्सुडेट्स** (वसायुक्त जमाव) भी हैं। रक्त वाहिकाएं अधिक क्षतिग्रस्त हो रही हैं।",
            "causes": "लगातार उच्च रक्त शर्करा ने रेटिना की रक्त वाहिकाओं को महत्वपूर्ण नुकसान पहुंचाया है। क्षतिग्रस्त वाहिकाओं से रक्त और तरल पदार्थ रिस रहा है।",
            "symptoms": "आपको महसूस हो सकता है:\n- हल्की धुंधली या उतार-चढ़ाव वाली दृष्टि\n- काले धब्बे या तैरते कण\n- पढ़ने में कठिनाई\n- रंग फीके दिखना",
            "treatment": "**3-6 महीने के भीतर नेत्र विशेषज्ञ के पास जाना अनुशंसित है।**\n\n- व्यापक डाइलेटेड आंखों की जांच जरूरी\n- OCT स्कैन की आवश्यकता हो सकती है\n- गहन रक्त शर्करा और रक्तचाप प्रबंधन",
            "lifestyle": ["🩸 तत्काल रक्त शर्करा नियंत्रण", "💊 दवा में बदलाव की जरूरत हो सकती है", "🫀 रक्तचाप 130/80 से कम रखें", "🍽️ सख्त डायबिटिक आहार", "🏃 नियमित मध्यम व्यायाम", "🚭 धूम्रपान पूरी तरह बंद करें"],
            "urgency": "medium",
            "urgency_message": "🟠 **मध्यम क्षति पाई गई। कृपया 3-6 महीने के भीतर नेत्र विशेषज्ञ से मिलें।**",
            "do_list": ["3-6 महीने के भीतर नेत्र विशेषज्ञ से मिलें", "रक्त शर्करा तुरंत सख्ती से नियंत्रित करें", "सभी दवाइयां समय पर लें", "डायबिटिक-अनुकूल भोजन करें"],
            "dont_list": ["विशेषज्ञ के पास जाने में देरी न करें", "कोई भी दवाई न छोड़ें", "मीठा, तला हुआ खाना न खाएं", "धूम्रपान, तंबाकू या शराब का सेवन न करें"],
        },
        3: {
            "what_is": "**गंभीर नॉन-प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (Severe NPDR)** पाई गई है। यह एक गंभीर चरण है जहां कई रक्त वाहिकाएं अवरुद्ध हैं। **व्यापक रक्तस्राव**, **शिरापरक माला** और **असामान्य रक्त वाहिका पैटर्न** मौजूद हैं।",
            "causes": "दीर्घकालिक अनियंत्रित मधुमेह ने रेटिना की रक्त वाहिकाओं को गंभीर नुकसान पहुंचाया है। कई वाहिकाएं अब अवरुद्ध हैं, जिससे रेटिना को ऑक्सीजन की कमी हो रही है।",
            "symptoms": "आपको अनुभव हो सकता है:\n- **महत्वपूर्ण दृष्टि धुंधलापन**\n- काले धब्बे या बड़े तैरते कण\n- रात में देखने में कठिनाई\n- अचानक दृष्टि परिवर्तन\n\n⚠️ **उपचार न होने पर दृष्टि हानि स्थायी हो सकती है।**",
            "treatment": "**तत्काल: 2-4 सप्ताह के भीतर रेटिना विशेषज्ञ से मिलें!**\n\n- **पैन-रेटिनल फोटोकोएग्यूलेशन (PRP) लेजर** उपचार\n- **एंटी-VEGF इंजेक्शन**\n- हर 2-3 महीने में बार-बार निगरानी",
            "lifestyle": ["🚨 2-4 सप्ताह के भीतर रेटिना विशेषज्ञ से मिलें", "🩸 तत्काल सख्त रक्त शर्करा नियंत्रण", "💊 दवा में बदलाव की जरूरत — आज ही डॉक्टर से मिलें", "🏃 केवल हल्का व्यायाम", "🚭 बिल्कुल धूम्रपान या तंबाकू नहीं"],
            "urgency": "high",
            "urgency_message": "🔴 **गंभीर क्षति पाई गई! 2-4 सप्ताह के भीतर रेटिना विशेषज्ञ से तत्काल मिलें।** उपचार के बिना, यह अंधापन का कारण बन सकता है।",
            "do_list": ["तत्काल 2-4 सप्ताह में रेटिना विशेषज्ञ से मिलें", "दिन में कई बार रक्त शर्करा जांचें", "यह रिपोर्ट डॉक्टर को दिखाएं", "सभी दवाइयां बिना चूक लें"],
            "dont_list": ["विशेषज्ञ के पास जाने में देरी न करें — यह तत्काल है", "कोई भी दवाई न छोड़ें", "भारी वस्तुएं न उठाएं", "अचानक दृष्टि परिवर्तन को न नजरअंदाज करें"],
        },
        4: {
            "what_is": "**प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (PDR)** — सबसे उन्नत और खतरनाक चरण — पाया गया है। **असामान्य नई रक्त वाहिकाएं** रेटिना पर बढ़ रही हैं। ये नाजुक नई वाहिकाएं फट सकती हैं और रक्तस्राव कर सकती हैं, जिससे **स्थायी अंधापन** हो सकता है।",
            "causes": "लंबे समय तक गंभीर ऑक्सीजन की कमी ने असामान्य रक्त वाहिकाओं के विकास को शुरू कर दिया है। ये वाहिकाएं अत्यंत नाजुक हैं और फट कर आंख में खून भर सकती हैं।",
            "symptoms": "तत्काल ध्यान देने योग्य गंभीर लक्षण:\n- **अचानक गंभीर दृष्टि हानि**\n- बड़े काले तैरते कण या दृष्टि में \"पर्दा\"\n- प्रकाश की चमक\n- आंख में दर्द या दबाव\n\n🚨 **अचानक दृष्टि हानि होने पर यह एक चिकित्सा आपातकाल है।**",
            "treatment": "**तत्काल विशेषज्ञ रेफरल आवश्यक!**\n\n- **PRP लेजर** उपचार\n- **एंटी-VEGF इंजेक्शन**\n- **विट्रेक्टॉमी सर्जरी** — यदि विट्रियस हेमरेज या रेटिनल डिटैचमेंट हो\n- **आजीवन निगरानी** जरूरी",
            "lifestyle": ["🚨 **आपातकाल**: आज ही नजदीकी नेत्र अस्पताल जाएं", "🏥 तुरंत रेटिना विशेषज्ञ से मिलें", "🩸 रक्त शर्करा नियंत्रण अब जीवन-और-दृष्टि के लिए महत्वपूर्ण", "💊 कोई भी दवाई न छोड़ें", "🚭 बिल्कुल धूम्रपान, तंबाकू या शराब नहीं", "🚑 अचानक दृष्टि हानि पर तुरंत 108 पर कॉल करें"],
            "urgency": "critical",
            "urgency_message": "🔴🔴 **गंभीर आपातकाल! प्रोलिफेरेटिव DR पाया गया — तत्काल विशेषज्ञ रेफरल आवश्यक!** बिना तत्काल उपचार के, यह **स्थायी अंधापन** का कारण बन सकता है। आज ही नजदीकी नेत्र अस्पताल जाएं!",
            "do_list": ["आज ही नेत्र अस्पताल जाएं", "अचानक दृष्टि हानि पर 108 पर कॉल करें", "यह रिपोर्ट डॉक्टर को दिखाएं", "सभी दवाइयां लेते रहें", "किसी को साथ लेकर अस्पताल जाएं"],
            "dont_list": ["🚫 देरी न करें — हर दिन मायने रखता है", "🚫 आंखों को न रगड़ें", "🚫 भारी वस्तुएं न उठाएं", "🚫 अचानक दृष्टि परिवर्तन को न नजरअंदाज करें", "🚫 घरेलू नुस्खे न आजमाएं — डॉक्टर से मिलें"],
        },
    },

    # ── Tamil ──
    "ta": {
        0: {
            "what_is": "உங்கள் விழித்திரை ஸ்கேனில் **நீரிழிவு விழித்திரை நோயின் அறிகுறிகள் எதுவும் இல்லை**. உங்கள் கண்ணின் ரத்தக் குழாய்கள் ஆரோக்கியமாக உள்ளன.",
            "causes": "உங்கள் கண்கள் தற்போது ஆரோக்கியமாக இருந்தாலும், நீரிழிவு நோய் இருப்பது ஆபத்தை ஏற்படுத்துகிறது. நீண்ட நேரம் அதிக இரத்த சர்க்கரை விழித்திரையின் சிறிய இரத்தக் குழாய்களை சேதப்படுத்தும்.",
            "symptoms": "தற்போது எந்த அறிகுறியும் இல்லை. ஆனால், நீரிழிவு விழித்திரை நோயில் ஆரம்ப அறிகுறிகள் இருப்பதில்லை.",
            "treatment": "இப்போது சிகிச்சை தேவையில்லை. வழக்கமான கண்காணிப்பைத் தொடரவும்.",
            "lifestyle": ["🩸 இரத்த சர்க்கரையை கட்டுப்பாட்டில் வைக்கவும்", "🫀 இரத்த அழுத்தத்தை தவறாமல் கண்காணிக்கவும்", "🏃 தினமும் குறைந்தது 30 நிமிடம் நடக்கவும்", "🥗 காய்கறிகள் மற்றும் பழங்கள் நிறைந்த உணவு உண்ணவும்", "🚭 புகைபிடிப்பதைத் தவிர்க்கவும்", "👁️ ஆண்டுதோறும் கண் பரிசோதனை செய்யவும்"],
            "urgency": "none",
            "urgency_message": "✅ உடனடி கவலை இல்லை. 12 மாதங்களில் வழக்கமான பரிசோதனை செய்யவும்.",
            "do_list": ["நீரிழிவு மருந்துகளை தவறாமல் எடுக்கவும்", "ஆண்டுதோறும் கண் பரிசோதனை", "ஆரோக்கியமான எடையை பராமரிக்கவும்"],
            "dont_list": ["நீரிழிவு மருந்துகளை தவிர்க்காதீர்கள்", "கண் பரிசோதனைகளை தவிர்க்காதீர்கள்", "அதிக சர்க்கரை உணவுகளை சாப்பிடாதீர்கள்"],
        },
        1: {
            "what_is": "**லேசான NPDR** கண்டறியப்பட்டது. உங்கள் விழித்திரையின் சிறிய ரத்தக் குழாய்களில் **நுண்குமிழ்கள்** (microaneurysms) தெரிகின்றன.",
            "causes": "நீண்ட நேரம் அதிக இரத்த சர்க்கரை விழித்திரையின் சிறிய ரத்தக் குழாய்களின் சுவர்களை பலவீனமாக்கியது.",
            "symptoms": "பொதுவாக இந்த நிலையில் **எந்த அறிகுறியும் இல்லை**. பார்வை இயல்பாக இருக்கும்.",
            "treatment": "உடனடி சிகிச்சை தேவையில்லை. 9-12 மாதங்களில் மீண்டும் பரிசோதனை செய்யவும்.",
            "lifestyle": ["🩸 இரத்த சர்க்கரையை கட்டுப்படுத்தவும்", "💊 மருந்துகளை சரியான நேரத்தில் எடுக்கவும்", "🥗 நீரிழிவு உணவு முறையைப் பின்பற்றவும்", "🏃 தினமும் 30 நிமிடம் நடக்கவும்", "🚭 புகைபிடிப்பதை நிறுத்தவும்"],
            "urgency": "low",
            "urgency_message": "🟡 ஆரம்ப நிலை கண்டறியப்பட்டது. உடனடி ஆபத்து இல்லை, ஆனால் நீரிழிவு மேலாண்மை முக்கியம்.",
            "do_list": ["இரத்த சர்க்கரையை கட்டுப்படுத்தவும்", "9-12 மாதங்களில் கண் பரிசோதனை", "தொடர்ந்து உடற்பயிற்சி செய்யவும்"],
            "dont_list": ["பதற்றம் வேண்டாம்", "மருந்துகளைத் தவிர்க்காதீர்கள்", "இனிப்பு வகைகளை சாப்பிடாதீர்கள்"],
        },
        2: {
            "what_is": "**மிதமான NPDR** கண்டறியப்பட்டது. ரத்தப்போக்கு, கொழுப்பு படிவுகள் மற்றும் பஞ்சு புள்ளிகள் உள்ளன.",
            "causes": "தொடர்ந்து அதிக இரத்த சர்க்கரை விழித்திரை ரத்தக் குழாய்களுக்கு கணிசமான சேதத்தை ஏற்படுத்தியுள்ளது.",
            "symptoms": "லேசான மங்கலான பார்வை, கருப்பு புள்ளிகள், படிக்க சிரமம்.",
            "treatment": "**3-6 மாதங்களுக்குள் கண் நிபுணரை அணுகவும்.**",
            "lifestyle": ["🩸 உடனடி இரத்த சர்க்கரை கட்டுப்பாடு", "💊 மருந்து மாற்றம் தேவைப்படலாம்", "👁️ 3-6 மாதங்களுக்குள் கண் நிபுணரை பாருங்கள்"],
            "urgency": "medium",
            "urgency_message": "🟠 **மிதமான சேதம் கண்டறியப்பட்டது. 3-6 மாதங்களுக்குள் கண் நிபுணரை பாருங்கள்.**",
            "do_list": ["3-6 மாதங்களுக்குள் கண் நிபுணரை பாருங்கள்", "இரத்த சர்க்கரையை உடனடியாக கட்டுப்படுத்தவும்"],
            "dont_list": ["கண் நிபுணரை சந்திப்பதை தாமதிக்காதீர்கள்", "இனிப்பு வகைகளை சாப்பிடாதீர்கள்"],
        },
        3: {
            "what_is": "**கடுமையான NPDR** கண்டறியப்பட்டது. பல ரத்தக் குழாய்கள் தடைபட்டு, விழித்திரைக்கு ரத்த விநியோகம் குறைந்துள்ளது.",
            "causes": "நீண்டகால கட்டுப்பாடற்ற நீரிழிவு விழித்திரை ரத்தக் குழாய்களுக்கு கடுமையான சேதத்தை ஏற்படுத்தியுள்ளது.",
            "symptoms": "**குறிப்பிடத்தக்க பார்வை மங்கல்**, கருப்பு புள்ளிகள், இரவில் பார்ப்பதில் சிரமம்.",
            "treatment": "**அவசரம்: 2-4 வாரங்களுக்குள் விழித்திரை நிபுணரை பாருங்கள்!** லேசர் சிகிச்சை, எதிர்-VEGF ஊசிகள் தேவைப்படலாம்.",
            "lifestyle": ["🚨 2-4 வாரங்களுக்குள் விழித்திரை நிபுணரை பாருங்கள்", "🩸 உடனடி கடுமையான இரத்த சர்க்கரை கட்டுப்பாடு", "🚭 புகைபிடிப்பதை முற்றிலும் நிறுத்தவும்"],
            "urgency": "high",
            "urgency_message": "🔴 **கடுமையான சேதம்! 2-4 வாரங்களுக்குள் விழித்திரை நிபுணரை அவசரமாக பாருங்கள்.**",
            "do_list": ["உடனடியாக விழித்திரை நிபுணரை பாருங்கள்", "இந்த அறிக்கையை டாக்டரிடம் காட்டுங்கள்"],
            "dont_list": ["தாமதிக்காதீர்கள் — இது அவசரம்", "கனமான பொருட்களை தூக்காதீர்கள்"],
        },
        4: {
            "what_is": "**பரவும் DR (PDR)** — மிகவும் முன்னேறிய ஆபத்தான நிலை — கண்டறியப்பட்டது. **அசாதாரண புதிய ரத்தக் குழாய்கள்** விழித்திரையில் வளர்கின்றன. இவை உடைந்து ரத்தப்போக்கு ஏற்படுத்தலாம், **நிரந்தர பார்வையிழப்பு** ஏற்படலாம்.",
            "causes": "விழித்திரையில் நீண்ட கால ஆக்சிஜன் பற்றாக்குறை அசாதாரண ரத்தக் குழாய்களின் வளர்ச்சியைத் தூண்டியுள்ளது.",
            "symptoms": "**திடீர் கடுமையான பார்வையிழப்பு**, பெரிய கருப்பு புள்ளிகள், ஒளி மின்னல்கள், கண் வலி.\n\n🚨 **திடீர் பார்வையிழப்பு ஏற்பட்டால் இது மருத்துவ அவசரநிலை.**",
            "treatment": "**உடனடி நிபுணர் பரிந்துரை அவசியம்!** PRP லேசர், எதிர்-VEGF ஊசிகள், விட்ரெக்டமி அறுவை சிகிச்சை.",
            "lifestyle": ["🚨 இன்றே அருகிலுள்ள கண் மருத்துவமனைக்கு செல்லுங்கள்", "🚑 திடீர் பார்வையிழப்பு ஏற்பட்டால் 108 அழைக்கவும்", "🚭 புகைபிடிப்பதை முற்றிலும் நிறுத்தவும்"],
            "urgency": "critical",
            "urgency_message": "🔴🔴 **அவசர நிலை! PDR கண்டறியப்பட்டது — உடனடி நிபுணர் பரிந்துரை அவசியம்! சிகிச்சையின்றி நிரந்தர பார்வையிழப்பு ஏற்படலாம்!**",
            "do_list": ["இன்றே கண் மருத்துவமனைக்கு செல்லுங்கள்", "108 அழைக்கவும் — ஆம்புலன்ஸ்", "இந்த அறிக்கையை டாக்டரிடம் காட்டுங்கள்"],
            "dont_list": ["🚫 தாமதிக்காதீர்கள்", "🚫 கண்களை தேய்க்காதீர்கள்", "🚫 கனமான பொருட்களை தூக்காதீர்கள்", "🚫 வீட்டு வைத்தியம் முயற்சிக்காதீர்கள்"],
        },
    },
}

# For languages not fully translated yet, fall back to English
for lang_code in ["te", "bn", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"]:
    if lang_code not in MEDICAL_INFO:
        MEDICAL_INFO[lang_code] = MEDICAL_INFO["en"]
