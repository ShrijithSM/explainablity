import fitz
import sys

papers = [
    r"c:\dev\explainablity\research_papers\pdfs\Non-stationary Transformers Exploring the Stationa.pdf",
    r"c:\dev\explainablity\research_papers\pdfs\Pattern Localization in Time Series through Signal.pdf",
    r"c:\dev\explainablity\research_papers\pdfs\Euler Characteristic Tools For Topological Data An.pdf",
]

for p in papers:
    print(f"\n--- {p} ---")
    try:
        doc = fitz.open(p)
        print(doc[0].get_text()[:1500])
    except Exception as e:
        print(f"Error: {e}")
