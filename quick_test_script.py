#!/usr/bin/env python3
"""
Quick test script to verify everything works before Streamlit deployment
Run this after fixing all files: python quick_test.py
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path.cwd()))

print("🧪 Minecraft Education Dashboard - Quick Test\n")
print("=" * 50)

# Test 1: Check directory structure
print("\n1️⃣ Checking directory structure...")
required_dirs = ['src', 'src/analysis', 'src/data_generation', 'src/visualization', 
                 'src/utils', 'pages', 'data', '.streamlit']
missing_dirs = [d for d in required_dirs if not os.path.exists(d)]

if missing_dirs:
    print(f"❌ Missing directories: {missing_dirs}")
else:
    print("✅ All directories present")

# Test 2: Check __init__.py files
print("\n2️⃣ Checking __init__.py files...")
init_files = ['src/__init__.py', 'src/analysis/__init__.py', 
              'src/data_generation/__init__.py', 'src/visualization/__init__.py',
              'src/utils/__init__.py']
missing_init = [f for f in init_files if not os.path.exists(f)]

if missing_init:
    print(f"❌ Missing __init__.py files: {missing_init}")
else:
    print("✅ All __init__.py files present")

# Test 3: Check page files
print("\n3️⃣ Checking page files...")
expected_pages = ['1_📊_Overview.py', '2_📈_Statistical_Analysis.py', 
                  '3_🤖_Predictive_Models.py', '4_🎮_3D_World_View.py',
                  '5_📚_Documentation.py']
pages_dir = Path('pages')
missing_pages = [p for p in expected_pages if not (pages_dir / p).exists()]

if missing_pages:
    print(f"❌ Missing pages: {missing_pages}")
else:
    print("✅ All page files present")

# Test 4: Test imports
print("\n4️⃣ Testing imports...")
import_results = []

# Test each module
modules_to_test = [
    ("Data Generation", "src.data_generation.simulator", "MinecraftEducationSimulator"),
    ("Statistical Analysis", "src.analysis.statistical", "EducationalStatisticsAnalyzer"),
    ("Time Series", "src.analysis.time_series", "TimeSeriesEducationAnalyzer"),
    ("Visualization", "src.visualization.plots", "EducationalVisualizer"),
    ("Helpers", "src.utils.helpers", "DataProcessor")
]

for name, module_path, class_name in modules_to_test:
    try:
        module = __import__(module_path, fromlist=[class_name])
        if hasattr(module, class_name):
            print(f"✅ {name}: Import successful")
            import_results.append(True)
        else:
            print(f"❌ {name}: Class {class_name} not found in module")
            import_results.append(False)
    except ImportError as e:
        print(f"❌ {name}: Import failed - {str(e)}")
        import_results.append(False)
    except Exception as e:
        print(f"❌ {name}: Unexpected error - {str(e)}")
        import_results.append(False)

# Test 5: Check required packages
print("\n5️⃣ Checking installed packages...")
required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'scipy', 
                    'sklearn', 'statsmodels', 'networkx', 'faker', 'yaml']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}: Installed")
    except ImportError:
        print(f"❌ {package}: Not installed")
        missing_packages.append(package)

# Final Report
print("\n" + "=" * 50)
print("📊 TEST SUMMARY")
print("=" * 50)

total_tests = 5
passed_tests = 0

if not missing_dirs:
    passed_tests += 1
if not missing_init:
    passed_tests += 1
if not missing_pages:
    passed_tests += 1
if all(import_results):
    passed_tests += 1
if not missing_packages:
    passed_tests += 1

print(f"\nTests Passed: {passed_tests}/{total_tests}")

if passed_tests == total_tests:
    print("\n🎉 ALL TESTS PASSED! Ready for deployment!")
    print("\nNext steps:")
    print("1. Run locally: streamlit run app.py")
    print("2. If successful, deploy to Streamlit Cloud")
else:
    print("\n⚠️  Some tests failed. Please fix the issues above.")
    print("\nCommon fixes:")
    print("- Run: python auto_fix_all.py")
    print("- Install missing packages: pip install -r requirements.txt")
    print("- Ensure all files are uploaded to GitHub")

print("\n" + "=" * 50)