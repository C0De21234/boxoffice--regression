# 🎬 Predicting Box Office Revenue with Linear Regression  

This project demonstrates how to use **Linear Regression** (both simple and multiple) to predict movie **Box Office revenue** based on **Budget** and **Ratings**.  

## 📌 Project Overview  
- Generated **synthetic movie dataset** (Budget, BoxOffice, Rating).  
- Built two models using **scikit-learn**:  
  1. **Simple Linear Regression** → Budget → Box Office  
  2. **Multiple Linear Regression** → Budget + Rating → Box Office  
- Compared model performances with **visualizations and metrics**.  

## 📊 Results & Insights  
- **Simple Regression**: Budget explains part of the variance, but predictions are limited.  
- **Multiple Regression**: Adding Ratings improved accuracy — predictions aligned much more closely with actual values.  

### 📈 Model Performance  
| Model | R² | MAE | MSE |
|-------|----|-----|-----|
| Simple Linear Regression | ~0.8 | Higher | Higher |
| Multiple Linear Regression | ~0.95 | Lower | Lower |

*(Exact values are shown directly in the plots.)*  

![Regression Plots](plots.png)  
*(Left: Simple Regression | Right: Multiple Regression)*  

## 🛠️ Tech Stack  
- **Python**  
- **NumPy, Pandas** for data handling  
- **Matplotlib** for visualization  
- **scikit-learn** for regression modeling  

## 🚀 How to Run  
```bash
git clone https://github.com/yourusername/boxoffice-regression.git
cd boxoffice-regression
pip install -r requirements.txt
python regression_analysis.py
```

## 📂 Project Structure  
```
boxoffice-regression/
│── regression_analysis.py   # Main script
│── requirements.txt         # Dependencies
│── plots.png                # Visualization output
│── README.md                # Documentation
```

## ✅ Key Takeaways  
- More relevant predictors → better model accuracy.  
- Visualization makes models easier to understand.  
- Linear Regression, though simple, is a great starting point for real-world insights.  

---

🔗 Let’s connect: [LinkedIn](your-linkedin-url)  
