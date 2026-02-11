from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import datetime

def generate_pdf_report(diagnosis_data: dict, image_bytes: bytes, patient_name: str = "Anonymous", patient_age: str = "N/A") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6
    )
    normal_style = styles['Normal']
    
    # Content Helper
    story = []
    
    # --- HEADER ---
    story.append(Paragraph("RADIOLOGY AI ANALYSIS REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
    story.append(Spacer(1, 12))
    
    # --- METADATA TABLE ---
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_id = f"RPT-{int(datetime.datetime.now().timestamp())}"
    
    meta_data = [
        ["Report ID:", report_id, "Date:", date_str],
        ["Patient Name:", patient_name, "Age/Gender:", f"{patient_age} / Unknown"],
        ["Referring Dr:", "N/A (AI Direct)", "AI Model:", "Microsoft Rad-DINO"]
    ]
    
    meta_table = Table(meta_data, colWidths=[1.2*inch, 2*inch, 1*inch, 2*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), # Labels bold
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'), # Labels bold
        ('TEXTCOLOR', (0,0), (-1,-1), colors.darkslategray),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.lightgrey, spaceBefore=0, spaceAfter=0))
    story.append(Spacer(1, 20))

    # --- DIAGNOSIS SECTION ---
    story.append(Paragraph("PRIMARY DIAGNOSIS", heading_style))
    
    label = diagnosis_data.get('label', 'Unknown')
    confidence = diagnosis_data.get('confidence', 0)
    note = diagnosis_data.get('note', '')
    
    # Color logic
    diag_color = colors.darkgreen if label.upper() == "NORMAL" else colors.firebrick
    
    diag_style = ParagraphStyle(
        'Diagnosis',
        parent=styles['BodyText'],
        fontSize=28,
        leading=32,
        alignment=TA_CENTER,
        textColor=diag_color,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph(label.upper(), diag_style))
    story.append(Spacer(1, 10))
    
    conf_text = f"<b>Confidence Level:</b> {confidence}%"
    story.append(Paragraph(conf_text, ParagraphStyle('Conf', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 20))
    
    # Findings/Note Box
    story.append(Paragraph("Review Details:", heading_style))
    story.append(Paragraph(note, normal_style))
    story.append(Spacer(1, 20))

    # --- PROBABILITIES TABLE ---
    story.append(Paragraph("Differential Analysis (Probability Distribution)", heading_style))
    
    probs = diagnosis_data.get('probabilities', {})
    # Convert to list for table
    table_data = [['Condition', 'Likelihood']]
    
    # Sort by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    for disease, prob in sorted_probs:
        # Highlight the predicted one
        is_primary = (disease == label)
        prob_str = f"{prob*100:.2f}%"
        table_data.append([disease, prob_str])

    prob_table = Table(table_data, colWidths=[3*inch, 2*inch], hAlign='LEFT')
    
    # Style the probabilities table
    tbl_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.aliceblue), # Header bg
        ('TEXTCOLOR', (0,0), (-1,0), colors.navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
    ])
    
    # Add highlighting for the row that matches the diagnosis
    for i, (disease, _) in enumerate(sorted_probs):
        if disease == label:
             # Row index is i + 1 because of header
             row_idx = i + 1
             tbl_style.add('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold')
             tbl_style.add('TEXTCOLOR', (0, row_idx), (-1, row_idx), diag_color)
    
    prob_table.setStyle(tbl_style)
    story.append(prob_table)
    story.append(Spacer(1, 40))

    # --- FOOTER / DISCLAIMER ---
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 10))
    
    disclaimer_text = """
    <b>DISCLAIMER:</b> This report was generated by an Artificial Intelligence system (Microsoft Rad-DINO). 
    It is intended for screening and research purposes only. 
    <b>This result is NOT a confirmed medical diagnosis.</b> 
    Please consult a certified radiologist or medical professional for clinical interpretation.
    """
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()
