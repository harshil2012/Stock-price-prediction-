# Finance with AI: Stock Market Price Prediction

## Project Abstract
This project is an AI-based web application to predict the next 5 days' closing price of a given stock. It leverages historical data via the Yahoo Finance API to train predictive models (LSTM, Random Forest, Linear Regression). The project is designed with a modern user interface, showcasing robust data visualization, interactive controls, and advanced stock market analysis metrics.

## Problem Statement
Stock market price prediction is a highly challenging domain due to the volatility and non-linear nature of financial markets. Traditional statistical methods often fall short in capturing complex temporal patterns. This project aims to harness the power of deep learning and machine learning models to forecast short-term stock price trends, offering users an intuitive interface to access these predictions.

## Methodology
1. **Data Collection:** Historical prices are downloaded using `yfinance`.
2. **Preprocessing:** Data is cleaned, and features (Open, High, Low, Volume, Close, Adj Close) are normalized. Time-series windowing is applied for sequence-based models like LSTM.
3. **Model Training:** 
   - **LSTM**: Captures sequential dependencies.
   - **Random Forest / Linear Regression**: Establish reliable baseline and complementary predictions.
4. **Evaluation:** Models are rigorously evaluated using metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
5. **Deployment:** FastAPI serves the trained models and APIs, while a React/Vite frontend provides dynamic visualizations (Plotly/Chart.js).

## System Architecture
```
[ Frontend (React/Vite) ] <--> [ Backend API (FastAPI) ] <--> [ Model inference ]
                                                        <--> [ Training Scripts ]
                                                        <--> [ Yahoo Finance API ]
```

## Algorithms Used
* **Long Short-Term Memory (LSTM)**: A specialized Recurrent Neural Network architecture capable of learning order dependence in sequence prediction problems.
* **Random Forest**: An ensemble learning method leveraging multiple decision trees for regression.
* **Linear Regression**: A classic algorithm utilized for trend baseline comparisons.

## Setup Instructions

### Backend & Model Training
1. Navigate to the project root: `cd "finance with ai"`
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Run server: `cd backend` and `uvicorn main:app --reload`

### Frontend
1. Navigate: `cd frontend`
2. Install Node packages: `npm install`
3. Run: `npm run dev`

## Full File Structure
- `/backend`: FastAPI application and endpoints (`main.py`).
- `/frontend`: HTML/CSS/JS user interface.
- `/models`: Saved trained model files (`.h5`, `.joblib`).
- `/training`: Scripts for data fetching and model training (`data_processor.py`, `train_models.py`).
- `/data`: Directory for dynamically downloaded stock datasets.

## Deployment Guide

### Deploying the Backend (Render or Railway)
1. **Render**: 
   - Create a new Web Service.
   - Connect your GitHub repository.
   - Set the Build Command: `pip install -r requirements.txt`
   - Set the Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Ensure you use Python 3.10+ in the environment variables.

### Deploying the Frontend (Vercel)
1. Since the frontend is static (HTML/CSS/JS), you can deploy it easily on Vercel.
2. Sign in to Vercel, click "Add New" -> "Project" and import your GitHub repository.
3. In the "Build & Development Settings", select "Other" as the framework preset.
4. Set the Root Directory to `frontend`.
5. Click Deploy.
*Note: Make sure to update the `API_BASE_URL` in `app.js` to point to your live backend Render URL before deploying the frontend!*

## Future Improvements
- Incorporating sentiment analysis from stock news.
- Live streaming of stock prices via WebSockets.
- Portfolio tracking capabilities.
