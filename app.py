import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import json
from typing import Dict, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from mistralai import Mistral

st.set_page_config(
    page_title="OCR Assurance - Analyse de Documents",
    page_icon="📄",
    layout="wide"
)

st.title("📄 OCR Assurance - Analyse de Documents")
st.markdown("---")

# Configuration Mistral AI
MISTRAL_API_KEY = "WOm8UvB9xLxxuyS3J7hSwLrBItAa979b"
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from PIL Image using OCR"""
    try:
        text = pytesseract.image_to_string(image, lang='fra')
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction OCR: {str(e)}")
        return ""

def process_pdf(pdf_bytes: bytes) -> List[str]:
    """Convert PDF to images and extract text from each page"""
    try:
        images = convert_from_bytes(pdf_bytes)
        texts = []
        for i, image in enumerate(images):
            text = extract_text_from_image(image)
            texts.append(text)
            st.write(f"Page {i+1} traitée")
        return texts
    except Exception as e:
        st.error(f"Erreur lors du traitement du PDF: {str(e)}")
        return []

def process_image(image_bytes: bytes) -> str:
    """Process uploaded image and extract text"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return extract_text_from_image(image)
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return ""

def analyze_insurance_document_with_llm(text: str) -> Dict[str, Optional[str]]:
    """Extract specific insurance information from text using Mistral AI"""

    try:
        prompt = f"""Tu es un expert en analyse de documents d'assurance. Analysez le texte suivant d'un document d'assurance et extrayez PRÉCISÉMENT les informations suivantes:

Éléments à identifier:
a) Numéro du contrat d'assurance et/ou avenant
b) Nom du propriétaire de la construction endommagée
c) Adresse de la construction endommagée
d) Date de réception du document
e) Date d'apparition des dommages
f) Description détaillée des dommages
g) Localisation précise des dommages
h) Présence d'une mise en demeure GPA (oui/non)

INSTRUCTIONS IMPORTANTES:
- Si une information n'est PAS trouvée, retournez exactement "Non trouvé"
- Retournez UNIQUEMENT les informations explicitement présentes dans le texte
- Ne pas inventer ou déduire d'informations
- Pour les dates, utilisez le format trouvé dans le document
- Soyez précis et factuel

TEXTE À ANALYSER:
{text}

RÉPONSE ATTENDUE (format JSON strict):
{{
    "numero_contrat": "valeur exacte trouvée ou Non trouvé",
    "nom_proprietaire": "valeur exacte trouvée ou Non trouvé",
    "adresse_construction": "valeur exacte trouvée ou Non trouvé",
    "date_reception": "valeur exacte trouvée ou Non trouvé",
    "date_dommages": "valeur exacte trouvée ou Non trouvé",
    "description_dommages": "valeur exacte trouvée ou Non trouvé",
    "localisation_dommages": "valeur exacte trouvée ou Non trouvé",
    "mise_en_demeure_gpa": "valeur exacte trouvée ou Non trouvé"
}}

Si tu trouve rien à la date de reception, regarde dans le début du PDF, y'a une mise en contexte (lieu et date)
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Appel à l'API Mistral
        response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.1,  # Très faible pour une analyse précise
            max_tokens=1000
        )

        response_text = response.choices[0].message.content.strip()

        # Extraire le JSON de la réponse
        try:
            # Chercher le JSON dans la réponse
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                # Nettoyer les résultats - remplacer "Non trouvé" par None
                for key, value in result.items():
                    if value == "Non trouvé" or value == "":
                        result[key] = None

                return result
            else:
                st.error("❌ Erreur: Format de réponse LLM invalide")
                return get_empty_result()

        except json.JSONDecodeError as e:
            st.error(f"❌ Erreur de parsing JSON: {e}")
            st.error(f"Réponse LLM: {response_text}")
            return get_empty_result()

    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel à Mistral AI: {str(e)}")
        return get_empty_result()

def get_empty_result() -> Dict[str, Optional[str]]:
    """Retourne un résultat vide en cas d'erreur"""
    return {
        "numero_contrat": None,
        "nom_proprietaire": None,
        "adresse_construction": None,
        "date_reception": None,
        "date_dommages": None,
        "description_dommages": None,
        "localisation_dommages": None,
        "mise_en_demeure_gpa": None
    }

def generate_pdf_report(analysis_result: Dict, filename: str, raw_text: str) -> bytes:
    """Generate a comprehensive PDF report of the analysis"""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Center
        textColor=colors.darkblue
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue
    )

    # Title
    story.append(Paragraph("RAPPORT D'ANALYSE OCR - DOCUMENT D'ASSURANCE", title_style))
    story.append(Spacer(1, 20))

    # Document info
    story.append(Paragraph("INFORMATIONS DU DOCUMENT", heading_style))
    doc_info = [
        ["Nom du fichier:", filename],
        ["Date d'analyse:", datetime.now().strftime("%d/%m/%Y à %H:%M")],
        ["Type d'analyse:", "OCR + Intelligence Artificielle"]
    ]

    doc_table = Table(doc_info, colWidths=[2.5*inch, 3.5*inch])
    doc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(doc_table)
    story.append(Spacer(1, 20))

    # Analysis results
    story.append(Paragraph("RÉSULTATS DE L'ANALYSE", heading_style))

    elements_to_check = [
        ("Numéro du contrat d'assurance", "numero_contrat"),
        ("Nom du propriétaire", "nom_proprietaire"),
        ("Adresse de la construction", "adresse_construction"),
        ("Date de réception", "date_reception"),
        ("Date d'apparition des dommages", "date_dommages"),
        ("Description des dommages", "description_dommages"),
        ("Localisation des dommages", "localisation_dommages"),
        ("Mise en demeure GPA", "mise_en_demeure_gpa")
    ]

    results_data = [["Élément", "Statut", "Valeur trouvée"]]
    missing_elements = []
    found_elements = []

    for element_name, key in elements_to_check:
        value = analysis_result.get(key)
        if value and value != "null" and value.strip():
            status = "✓ TROUVÉ"
            found_elements.append(element_name)
            # Truncate long values for table display
            display_value = value[:50] + "..." if len(str(value)) > 50 else str(value)
        else:
            status = "✗ MANQUANT"
            missing_elements.append(element_name)
            display_value = "Non trouvé"

        results_data.append([element_name, status, display_value])

    results_table = Table(results_data, colWidths=[2.5*inch, 1*inch, 2.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))

    # Color code the status column
    for i, (_, key) in enumerate(elements_to_check, 1):
        value = analysis_result.get(key)
        if value and value != "null" and value.strip():
            results_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, i), (1, i), colors.darkgreen)
            ]))
        else:
            results_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, i), (1, i), colors.darkred)
            ]))

    story.append(results_table)
    story.append(Spacer(1, 20))

    # Summary
    story.append(Paragraph("RÉSUMÉ DE L'ANALYSE", heading_style))

    total_elements = len(elements_to_check)
    found_count = len(found_elements)
    missing_count = len(missing_elements)
    completion_rate = (found_count / total_elements) * 100

    summary_text = f"""
    <b>Taux de complétion:</b> {completion_rate:.1f}% ({found_count}/{total_elements} éléments trouvés)<br/>
    <br/>
    <b>Éléments trouvés ({found_count}):</b><br/>
    {'<br/>'.join(['• ' + elem for elem in found_elements]) if found_elements else '• Aucun élément trouvé'}<br/>
    <br/>
    <b>Éléments manquants ({missing_count}):</b><br/>
    {'<br/>'.join(['• ' + elem for elem in missing_elements]) if missing_elements else '• Tous les éléments ont été trouvés'}
    """

    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Recommendations
    story.append(Paragraph("RECOMMANDATIONS", heading_style))

    if missing_count == 0:
        recommendation = "✅ <b>Excellent!</b> Tous les éléments requis ont été identifiés dans le document."
    elif missing_count <= 2:
        recommendation = "⚠️ <b>Attention:</b> Quelques éléments manquent. Vérifiez si ces informations sont présentes ailleurs dans le document ou dans des annexes."
    else:
        recommendation = "❌ <b>Action requise:</b> Plusieurs éléments importants sont manquants. Il pourrait être nécessaire de demander des documents complémentaires."

    story.append(Paragraph(recommendation, styles['Normal']))
    story.append(Spacer(1, 20))

    # Raw text preview (first 1000 characters)
    if raw_text:
        story.append(Paragraph("APERÇU DU TEXTE EXTRAIT", heading_style))
        preview_text = raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text
        story.append(Paragraph(f"<font size=8>{preview_text.replace('<', '&lt;').replace('>', '&gt;')}</font>", styles['Normal']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def is_damage_photo(image: Image.Image) -> bool:
    """Analyze if the image contains damage photos using basic image analysis"""
    try:
        # Convert to grayscale for analysis
        gray_image = image.convert('L')

        # Basic heuristics to detect damage photos:
        # 1. Check for high contrast areas (cracks, damage often create sharp edges)
        # 2. Look for irregular patterns
        # 3. Analyze color distribution for signs of damage (water stains, discoloration)

        # For now, we'll use a simple approach based on image characteristics
        # This could be enhanced with ML models for better accuracy

        import numpy as np
        img_array = np.array(gray_image)

        # Calculate edge density (damaged areas often have more edges)
        from PIL import ImageFilter
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges.convert('L'))
        edge_density = np.mean(edge_array > 50)  # Threshold for edge detection

        # Check for high variance (irregular patterns)
        variance = np.var(img_array)

        # Simple scoring system
        damage_score = 0

        # High edge density might indicate cracks or damage
        if edge_density > 0.15:
            damage_score += 1

        # High variance might indicate irregular surfaces
        if variance > 2000:
            damage_score += 1

        # Check image aspect ratio and size (damage photos are often close-ups)
        width, height = image.size
        aspect_ratio = width / height

        # Damage photos are often taken close-up with specific ratios
        if 0.5 < aspect_ratio < 2.0 and min(width, height) > 200:
            damage_score += 1

        # Return True if we have enough indicators of a damage photo
        return damage_score >= 2

    except Exception:
        # If analysis fails, assume it's not a damage photo
        return False

def analyze_damage_photo(image: Image.Image) -> Dict[str, str]:
    """Analyze damage photo for specific characteristics"""
    try:
        # Basic damage analysis
        width, height = image.size

        # Basic damage analysis without color space conversion

        analysis = {
            "image_size": f"{width}x{height}",
            "estimated_damage_type": "Analyse visuelle requise",
            "quality_assessment": "Bonne" if min(width, height) > 500 else "Faible résolution",
            "recommendations": []
        }

        # Basic quality checks
        if min(width, height) < 300:
            analysis["recommendations"].append("Image de faible résolution - considérer une photo de meilleure qualité")

        if width * height > 2000000:  # > 2MP
            analysis["recommendations"].append("Bonne qualité d'image pour l'analyse")

        # Check for blur (basic)
        gray = image.convert('L')
        import numpy as np
        img_array = np.array(gray)
        variance = np.var(img_array)

        if variance < 1000:
            analysis["recommendations"].append("Image possiblement floue - vérifier la netteté")

        return analysis

    except Exception as e:
        return {
            "image_size": "Analyse échouée",
            "estimated_damage_type": "Erreur d'analyse",
            "quality_assessment": "Erreur",
            "recommendations": [f"Erreur lors de l'analyse: {str(e)}"]
        }

def main():
    st.markdown("### 📤 Téléchargement de Document")

    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier PDF ou une image (PNG, JPG)",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Formats supportés: PDF, PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 🔄 Traitement en cours...")

        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type

        progress_bar = st.progress(0)

        # Process file based on type
        if file_type == "application/pdf":
            st.info("📄 Traitement du PDF...")
            progress_bar.progress(25)
            texts = process_pdf(file_bytes)
            combined_text = "\n\n".join(texts)
            progress_bar.progress(50)
        else:
            st.info("🖼️ Traitement de l'image...")
            progress_bar.progress(25)

            # Check if it's a damage photo
            image = Image.open(io.BytesIO(file_bytes))
            if is_damage_photo(image):
                st.warning("📸 Photo de dommages détectée")

                # Analyze the damage photo
                damage_analysis = analyze_damage_photo(image)

                with st.expander("🔍 Analyse de la photo de dommages"):
                    st.write(f"**Taille de l'image:** {damage_analysis['image_size']}")
                    st.write(f"**Type de dommage estimé:** {damage_analysis['estimated_damage_type']}")
                    st.write(f"**Qualité de l'image:** {damage_analysis['quality_assessment']}")

                    if damage_analysis['recommendations']:
                        st.write("**Recommandations:**")
                        for rec in damage_analysis['recommendations']:
                            st.write(f"• {rec}")

            combined_text = process_image(file_bytes)
            progress_bar.progress(50)

        if combined_text:
            st.success("✅ Texte extrait avec succès")

            # Show extracted text in expandable section
            with st.expander("📝 Texte extrait (cliquez pour voir)"):
                st.text_area("Texte OCR:", combined_text, height=200)

            progress_bar.progress(75)

            # Analyze with AI
            st.info("🤖 Analyse avec Mistral AI en cours...")
            analysis_result = analyze_insurance_document_with_llm(combined_text)
            progress_bar.progress(100)

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Résultats de l'Analyse")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 Informations Contractuelles")
                st.write(f"**Numéro de contrat:** {analysis_result.get('numero_contrat', 'Non trouvé')}")
                st.write(f"**Nom du propriétaire:** {analysis_result.get('nom_proprietaire', 'Non trouvé')}")
                st.write(f"**Adresse:** {analysis_result.get('adresse_construction', 'Non trouvée')}")
                st.write(f"**Date de réception:** {analysis_result.get('date_reception', 'Non trouvée')}")

            with col2:
                st.markdown("#### 🏗️ Informations sur les Dommages")
                st.write(f"**Date d'apparition:** {analysis_result.get('date_dommages', 'Non trouvée')}")
                st.write(f"**Description:** {analysis_result.get('description_dommages', 'Non trouvée')}")
                st.write(f"**Localisation:** {analysis_result.get('localisation_dommages', 'Non trouvée')}")
                st.write(f"**GPA - Mise en demeure:** {analysis_result.get('mise_en_demeure_gpa', 'Non spécifié')}")

            # Download results as JSON
            st.markdown("---")
            st.markdown("### 💾 Télécharger les Résultats")

            results_json = {
                "timestamp": datetime.now().isoformat(),
                "filename": uploaded_file.name,
                "analysis": analysis_result,
                "raw_text": combined_text
            }

            col_json, col_pdf = st.columns(2)

            with col_json:
                st.download_button(
                    label="📥 Télécharger l'analyse (JSON)",
                    data=json.dumps(results_json, ensure_ascii=False, indent=2),
                    file_name=f"analyse_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with col_pdf:
                try:
                    pdf_report = generate_pdf_report(analysis_result, uploaded_file.name, combined_text)
                    st.download_button(
                        label="📄 Télécharger le rapport (PDF)",
                        data=pdf_report,
                        file_name=f"rapport_analyse_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
                    st.info("Vous pouvez toujours télécharger les résultats en format JSON.")
        else:
            st.error("❌ Impossible d'extraire le texte du fichier")
            progress_bar.progress(100)

if __name__ == "__main__":
    main()