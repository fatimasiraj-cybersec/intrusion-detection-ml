# Intrusion Detection System — ML Comparison

Comparing supervised and unsupervised machine learning models for network intrusion detection on the NSL-KDD dataset.

# Models Compared
- Logistic Regression — linear baseline
- Random Forest — ensemble supervised classifier  
- Isolation Forest — unsupervised anomaly detector (trained on normal traffic only)

# Key Result
Evaluated not just on overall metrics, but on known vs. novel attack types - a critical distinction for real-world intrusion detection, where attacks evolve faster than labeled training data.

# Tech Stack
Python 3 · scikit-learn · pandas · NumPy · Matplotlib

## How to Run ? 

```bash
git clone https://github.com/YOUR_USERNAME/intrusion-detection-ml.git
cd intrusion-detection-ml

pip install -r requirements.txt

# Run the full experiment
python ids.py
```       

# Outputs
- `results.csv` — precision, recall, F1, FPR per model
- `confusion_matrices.png` — side-by-side confusion matrices
- `known_vs_novel.png` — detection rate comparison

## License
MIT