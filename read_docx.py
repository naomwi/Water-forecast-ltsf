import sys
import docx

def read_docx(path):
    try:
        doc = docx.Document(path)
        for para in doc.paragraphs:
            print(para.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_docx(sys.argv[1])
    else:
        print("Usage: python read_docx.py <path_to_docx>")
