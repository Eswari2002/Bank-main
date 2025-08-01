import pdfplumber
import pandas as pd
import google.generativeai as genai
import re

# === Gemini API Setup ===
GEMINI_API_KEY = "AIzaSyDqAYD6aPGt9FNGI55rBFQDqPdcjwTNnCg"   #Ramya's API key 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_text(text, page_number):
    prompt = f"""You are a smart AI pdf analyzer assistant. Follow the instructions given below:
        -Summarize all the content present from page {page_number} of the given PDF in concise bullet points.
        -Focus on key facts, financial data, or important sections.
        -All the data should be present, do not use you own knowledge.
        -If the data is in a table, correlate all the values present with the attributes of the table.
        For example:                     Notes         2024 $'m            2023 $"m
                    Current assets
                    Inventories            17           174.3               120.5

                The above is the table, so the summarization should be as:
                >> The price of inventories for the year 2024 is $174.3m and 2023 is $120.5m in current assets, which can be referred from notes 17.
        -All the tables should be extracted as sentences in points, the table should not be obtained directly
Text:
{text}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def extract_table_headers(page):
    tables = page.extract_tables()
    headers = []
    for table in tables:
        if table and len(table) > 0:
            headers.append(table[0])  
    return headers

def extract_page_header(text, lines=2):
    if text:
        return '\n'.join(text.strip().split('\n')[:lines])
    return ""

def extract_notes(text):
    return ", ".join(re.findall(r"Note\s*\d+", text, re.IGNORECASE)) or "None"

def summarize_pdf_pages(pdf_path, page_numbers, output_csv):
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in page_numbers:
            if 1 <= page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                text = page.extract_text()

                if text:
                    summary = summarize_text(text, page_number)
                    page_header = extract_page_header(text)
                    note_refs = extract_notes(text)
                    table_headers = extract_table_headers(page)

                    results.append({
                        "Page Number": page_number,
                        "Page Header": page_header,
                        "Note References": note_refs,
                        "Table Headers": str(table_headers),
                        "Summary": summary
                    })
                    print(f"Page {page_number} summarized.")
                else:
                    results.append({
                        "Page Number": page_number,
                        "Page Header": "No text found",
                        "Note References": "None",
                        "Table Headers": "None",
                        "Summary": "No text found on page."
                    })
                    print(f"No text on page {page_number}")
            else:
                results.append({
                    "Page Number": page_number,
                    "Page Header": "Invalid page number",
                    "Note References": "None",
                    "Table Headers": "None",
                    "Summary": "Invalid page number."
                })
                print(f"Page {page_number} out of range.")

    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSummary with metadata saved to: {output_csv}")

# === NEW: Extract Financial Section Page Numbers from Excel ===
def get_financial_pages(excel_path):
    df = pd.read_excel(excel_path)
    financial_pages = df[df['Category'].str.strip().str.lower() == 'financial section']['Page']
    return sorted(financial_pages.dropna().astype(int).unique().tolist())

# === Run ===
pdf_path = r"C:\Users\EswariBabu\Downloads\volex.pdf"
excel_path = r"C:\Users\EswariBabu\WorkingFolder\bank-main\classification_output\classified_sections_output_all.xlsx"  
output_csv = r"C:\Users\EswariBabu\WorkingFolder\bank-main\summary_output_with_metadata.csv"


page_numbers = get_financial_pages(excel_path)


summarize_pdf_pages(pdf_path, page_numbers, output_csv)
