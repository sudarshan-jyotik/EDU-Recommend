"""Download NLTK data required by StudyPlanGenerator."""
import nltk
packages = ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]
print("Downloading NLTK packages...")
for pkg in packages:
    result = nltk.download(pkg, quiet=False)
    if result:
        print(f"  OK  {pkg}")
    else:
        print(f"  --  {pkg} (may already exist or download failed)")
print("Done.")
