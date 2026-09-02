"""
Clinical Report Generation Module.

Generates downloadable PDF screening reports with:
- Patient/session metadata
- Original & enhanced images
- Grad-CAM explainability overlay
- Vessel segmentation map
- DR severity grade with confidence
- Clinical recommendations
"""

import io
import datetime
import numpy as np
import cv2
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, 
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from typing import Dict, Optional


# Color palette matching the app's medical theme
THEME_COLORS = {
    "primary": colors.HexColor("#00D4AA"),
    "dark_bg": colors.HexColor("#0A0F1C"),
    "card_bg": colors.HexColor("#1A1F2E"),
    "text": colors.HexColor("#E8E8F0"),
    "accent_blue": colors.HexColor("#4A9EFF"),
    "danger": colors.HexColor("#FF4757"),
    "warning": colors.HexColor("#FFA502"),
    "success": colors.HexColor("#2ED573"),
}

SEVERITY_COLORS = {
    0: colors.HexColor("#2ECC71"),
    1: colors.HexColor("#F1C40F"),
    2: colors.HexColor("#F39C12"),
    3: colors.HexColor("#E74C3C"),
    4: colors.HexColor("#C0392B"),
}


def numpy_to_reportlab_image(img_array: np.ndarray, width: float = 3 * inch,
                              is_bgr: bool = True) -> Image:
    """Convert a numpy image array to a ReportLab Image object."""
    if is_bgr:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    pil_img = PILImage.fromarray(img_array)
    
    # Calculate proportional height
    aspect = pil_img.height / pil_img.width
    height = width * aspect
    
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    
    return Image(buf, width=width, height=height)


def create_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor("#00D4AA"),
        spaceAfter=6,
        fontName='Helvetica-Bold',
    ))
    
    styles.add(ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#888899"),
        spaceAfter=20,
        alignment=TA_CENTER,
    ))
    
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#4A9EFF"),
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        borderPadding=(0, 0, 4, 0),
    ))
    
    styles.add(ParagraphStyle(
        'BodyText2',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#333344"),
        spaceAfter=6,
        leading=14,
    ))
    
    styles.add(ParagraphStyle(
        'SeverityLabel',
        parent=styles['Normal'],
        fontSize=18,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        spaceAfter=4,
    ))
    
    styles.add(ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
    ))
    
    styles.add(ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#555566"),
        leading=12,
    ))
    
    return styles


def generate_pdf_report(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    gradcam_overlay: np.ndarray,
    segmentation_overlay: np.ndarray,
    clinical_evidence: Dict,
    quality_report: Dict,
    lesion_counts: Dict,
    session_id: str = None,
) -> bytes:
    """
    Generate a comprehensive PDF clinical screening report.
    
    Args:
        original_image: Original fundus image (BGR).
        enhanced_image: Enhanced fundus image (BGR).
        gradcam_overlay: Grad-CAM heatmap overlay (BGR).
        segmentation_overlay: Retinal structure overlay (BGR).
        clinical_evidence: Output from explainability.generate_clinical_evidence().
        quality_report: Quality assessment metrics dict.
        lesion_counts: Lesion count summary dict.
        session_id: Optional session/patient identifier.
    
    Returns:
        PDF file content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
    )
    
    styles = create_styles()
    story = []
    
    # ── HEADER ──
    story.append(Paragraph("👁️ EYE-CONIQ", styles['ReportTitle']))
    story.append(Paragraph(
        "AI-Powered Diabetic Retinopathy Screening Report", 
        styles['ReportSubtitle']
    ))
    
    # Horizontal rule
    story.append(HRFlowable(
        width="100%", thickness=2, 
        color=colors.HexColor("#00D4AA"), spaceAfter=12
    ))
    
    # ── SESSION INFO ──
    now = datetime.datetime.now()
    session_data = [
        ["Report ID", session_id or f"EYC-{now.strftime('%Y%m%d%H%M%S')}"],
        ["Date & Time", now.strftime("%B %d, %Y  •  %I:%M %p")],
        ["System", "EYE-CONIQ v2.0 — Explainable AI DR Screening"],
        ["Model", "ResNet-18 (APTOS 2019 trained)"],
    ]
    
    session_table = Table(session_data, colWidths=[120, 350])
    session_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#4A9EFF")),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor("#333344")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor("#DDDDEE")),
    ]))
    story.append(session_table)
    story.append(Spacer(1, 16))
    
    # ── DIAGNOSIS SUMMARY ──
    severity_level = clinical_evidence.get("severity_level", 0)
    severity_name = clinical_evidence.get("severity", "Unknown")
    confidence = clinical_evidence.get("confidence", 0) * 100
    sev_color = SEVERITY_COLORS.get(severity_level, colors.gray)
    
    story.append(Paragraph("DIAGNOSIS", styles['SectionHeader']))
    
    diag_data = [
        [
            Paragraph(f'<font color="{sev_color.hexval()}" size="20"><b>{severity_name}</b></font>', 
                      styles['SeverityLabel']),
            Paragraph(f'<font size="12">Level {severity_level}/4</font><br/>'
                      f'<font size="10" color="#888899">Confidence: {confidence:.1f}%</font>',
                      styles['BodyText2']),
        ]
    ]
    
    diag_table = Table(diag_data, colWidths=[300, 170])
    diag_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 2, sev_color),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 8))
    
    # Clinical recommendation
    recommendation = clinical_evidence.get("recommendation", "")
    urgency = clinical_evidence.get("urgency", "none")
    urgency_label = {"none": "🟢", "low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🔴🔴"}
    
    story.append(Paragraph(
        f'{urgency_label.get(urgency, "")} <b>Recommendation:</b> {recommendation}',
        styles['BodyText2']
    ))
    
    # Referable DR flag
    is_referable = clinical_evidence.get("is_referable", False)
    ref_prob = clinical_evidence.get("referable_dr_probability", 0) * 100
    if is_referable:
        story.append(Paragraph(
            f'<font color="#E74C3C"><b>⚠ REFERABLE DR DETECTED</b></font> '
            f'(Probability: {ref_prob:.1f}%)',
            styles['BodyText2']
        ))
    
    story.append(Spacer(1, 12))
    
    # ── IMAGES ──
    story.append(Paragraph("RETINAL IMAGES", styles['SectionHeader']))
    
    img_width = 3.2 * inch
    
    # Original & Enhanced side by side
    img_data = [[
        numpy_to_reportlab_image(original_image, width=img_width),
        numpy_to_reportlab_image(enhanced_image, width=img_width),
    ], [
        Paragraph('<font size="9"><b>Original Image</b></font>', 
                  ParagraphStyle('c', alignment=TA_CENTER)),
        Paragraph('<font size="9"><b>Enhanced Image</b></font>', 
                  ParagraphStyle('c', alignment=TA_CENTER)),
    ]]
    
    img_table = Table(img_data, colWidths=[img_width + 10, img_width + 10])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 12))
    
    # Grad-CAM & Segmentation side by side
    story.append(Paragraph("EXPLAINABILITY & STRUCTURE ANALYSIS", styles['SectionHeader']))
    
    analysis_data = [[
        numpy_to_reportlab_image(gradcam_overlay, width=img_width),
        numpy_to_reportlab_image(segmentation_overlay, width=img_width),
    ], [
        Paragraph('<font size="9"><b>Grad-CAM Attention Map</b><br/>'
                  '<font size="8" color="#888899">Highlights regions influencing the AI decision</font></font>',
                  ParagraphStyle('c', alignment=TA_CENTER)),
        Paragraph('<font size="9"><b>Retinal Structure Map</b><br/>'
                  '<font size="8" color="#888899">Red: Vessels | Yellow: Exudates | Blue: Hemorrhages</font></font>',
                  ParagraphStyle('c', alignment=TA_CENTER)),
    ]]
    
    analysis_table = Table(analysis_data, colWidths=[img_width + 10, img_width + 10])
    analysis_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 12))
    
    # ── IMAGE QUALITY METRICS ──
    story.append(Paragraph("IMAGE QUALITY ASSESSMENT", styles['SectionHeader']))
    
    quality_data = [
        ["Metric", "Score", "Status"],
        ["Focus", f"{quality_report.get('focus_score', 0):.0f}/100", 
         "✅" if quality_report.get('focus_score', 0) >= 50 else "⚠️"],
        ["Illumination", f"{quality_report.get('illumination_score', 0):.0f}/100",
         "✅" if quality_report.get('illumination_score', 0) >= 50 else "⚠️"],
        ["Field of View", f"{quality_report.get('fov_score', 0):.0f}/100",
         "✅" if quality_report.get('fov_score', 0) >= 50 else "⚠️"],
        ["Contrast", f"{quality_report.get('contrast_score', 0):.0f}/100",
         "✅" if quality_report.get('contrast_score', 0) >= 50 else "⚠️"],
        ["Overall", f"{quality_report.get('overall_score', 0):.0f}/100",
         quality_report.get('grade', 'N/A')],
    ]
    
    q_table = Table(quality_data, colWidths=[160, 120, 80])
    q_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4A9EFF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDEE")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 12))
    
    # ── LESION ANALYSIS ──
    story.append(Paragraph("LESION ANALYSIS", styles['SectionHeader']))
    
    lesion_data = [
        ["Structure", "Count / Value"],
        ["Microaneurysms", str(lesion_counts.get("microaneurysm_count", 0))],
        ["Exudate Regions", str(lesion_counts.get("exudate_count", 0))],
        ["Hemorrhage Regions", str(lesion_counts.get("hemorrhage_count", 0))],
        ["Vessel Density", f"{lesion_counts.get('vessel_density_pct', 0):.1f}%"],
        ["Optic Disc Detected", "Yes" if lesion_counts.get("optic_disc_detected") else "No"],
    ]
    
    l_table = Table(lesion_data, colWidths=[200, 160])
    l_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00D4AA")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDEE")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(l_table)
    story.append(Spacer(1, 12))
    
    # ── ATTENTION ANALYSIS ──
    story.append(Paragraph("SPATIAL ATTENTION ANALYSIS", styles['SectionHeader']))
    
    quadrant_data = clinical_evidence.get("quadrant_activations", {})
    most_affected = clinical_evidence.get("most_affected_quadrant", "N/A")
    
    story.append(Paragraph(
        f'<b>Most affected quadrant:</b> {most_affected}<br/>'
        f'<b>Activation coverage:</b> {clinical_evidence.get("activation_coverage", 0)*100:.1f}% of retina<br/>'
        f'<b>Peak activation:</b> {clinical_evidence.get("peak_activation", 0):.2f}<br/>'
        f'<b>Model uncertainty:</b> {clinical_evidence.get("uncertainty", 0)*100:.1f}%',
        styles['BodyText2']
    ))
    
    if quadrant_data:
        quad_rows = [["Quadrant", "Activation Level"]]
        for quadrant, value in quadrant_data.items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            quad_rows.append([quadrant, f"{bar}  {value:.3f}"])
        
        quad_table = Table(quad_rows, colWidths=[160, 300])
        quad_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (1, 1), (1, -1), 'Courier'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDEE")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(Spacer(1, 6))
        story.append(quad_table)
    
    story.append(Spacer(1, 12))
    
    # ── CLASS PROBABILITIES ──
    story.append(Paragraph("CLASS PROBABILITY DISTRIBUTION", styles['SectionHeader']))
    
    class_probs = clinical_evidence.get("class_probabilities", {})
    prob_rows = [["DR Level", "Probability"]]
    for cls_name, prob in class_probs.items():
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        prob_rows.append([cls_name, f"{bar}  {prob*100:.1f}%"])
    
    prob_table = Table(prob_rows, colWidths=[160, 300])
    prob_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (1, 1), (1, -1), 'Courier'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00D4AA")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDEE")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # ── FOOTER / DISCLAIMER ──
    story.append(HRFlowable(
        width="100%", thickness=1, 
        color=colors.HexColor("#CCCCDD"), spaceAfter=8
    ))
    
    story.append(Paragraph(
        '<font size="8" color="#999999">'
        '<b>DISCLAIMER:</b> This report is generated by an AI-assisted screening system '
        'and is intended to support — not replace — clinical judgment. All findings must be '
        'reviewed and confirmed by a qualified ophthalmologist before any treatment decisions. '
        'This system has been designed for primary screening in resource-limited settings and '
        'should be used as part of a human-in-the-loop diagnostic workflow.'
        '</font>',
        styles['SmallText']
    ))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f'<font size="7" color="#AAAAAA">'
        f'Generated by EYE-CONIQ v2.0 | {now.strftime("%Y-%m-%d %H:%M:%S")} | '
        f'Report ID: {session_id or "N/A"}'
        f'</font>',
        styles['FooterStyle']
    ))
    
    # Build PDF
    doc.build(story)
    
    return buffer.getvalue()
