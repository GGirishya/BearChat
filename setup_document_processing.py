"""
Quick setup script for document processing dependencies
"""

import subprocess
import sys
import platform

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n Checking Python packages...")
    
    required_packages = [
        ('PyPDF2', 'PyPDF2'),
        ('Pillow', 'PIL'),
        ('pdf2image', 'pdf2image'),
        ('pytesseract', 'pytesseract')
    ]
    
    missing = []
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f" {display_name}")
        except ImportError:
            print(f" {display_name}")
            missing.append(display_name)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n🔍 Checking Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f" Tesseract {version}")
        return True
    except Exception as e:
        print(f" Tesseract not found")
        print("\nInstall instructions:")
        
        os_type = platform.system()
        if os_type == "Darwin":  # macOS
            print("  brew install tesseract")
        elif os_type == "Linux":
            print("  sudo apt-get install tesseract-ocr")
        elif os_type == "Windows":
            print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        return False


def main():
    print("="*80)
    print("BEARCHAT DOCUMENT PROCESSING SETUP")
    print("="*80)
    
    # Check Python packages
    packages_ok = check_python_packages()
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    if packages_ok and tesseract_ok:
        print("\n All dependencies installed!")
        print("\nYou can now:")
        print("  1. Start the API server: python api_server.py")
        print("  2. Test document upload: python test_document_upload.py")
    else:
        print("\n Some dependencies are missing")
        print("\nFollow the installation instructions above and run this script again.")


if __name__ == "__main__":
    main()
