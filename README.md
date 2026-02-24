 # Car Price Prediction

A Streamlit web app that predicts the selling price of a used car based on its specifications. The model is trained on real-world car listing data and exposes an interactive UI for exploration.

## Features

- Interactive sidebar to input car details (age, mileage, engine, fuel, transmission, etc.).
- Machine-learning model (Random Forest Regressor) pre-trained and saved as `model/model.pkl`.
- Radar chart visualization of your car’s specs using Plotly.
- Price estimate with an uncertainty range (lower/upper bound).
- Custom styling for a clean, dashboard-like UI.

## Project Structure

- `app/main.py` – Streamlit app entry point and UI + prediction logic.
- `assets/style.css` – Custom CSS styles applied to the Streamlit app.
- `data/cardekho_dataset.csv` – Car listings dataset used for training.
- `model/model.pkl` – Pre-trained regression model used by the app.
- `utils/data_cleaning.py` – Data cleaning and model training utilities.
- `requirements.txt` – Python dependencies for running the app.

## Model & Performance

The project utilizes a **Random Forest Regressor** to predict car prices. The model pipeline includes comprehensive data cleaning, feature engineering, and hyperparameter tuning.

### Data Processing
- **Encoding:** Categorical variables (`fuel_type`, `seller_type`, `transmission_type`) are one-hot encoded.
- **Feature Engineering:**
  - `power_per_cc`: Calculated as `max_power / engine` to capture engine efficiency.
  - `diesel_auto`: An interaction term capturing the premium often associated with automatic diesel vehicles.
- **Input Features:** The model is trained on 15 features including `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`, and various encoded categorical features.

### Performance Metrics
The model was evaluated on a held-out test set (20% split). The Random Forest Regressor (configured with `max_depth=20`, `min_samples_leaf=2`, `n_estimators=200`) achieved the following results:

- **Mean Absolute Error (MAE):** ₹103,792.41
- **Root Mean Squared Error (RMSE):** ₹233,617.44

*Note: Hyperparameter optimization was performed using GridSearchCV with 5-fold cross-validation to identify the best estimator configuration.*

## Prerequisites

- Python 3.8 or newer.
- pip (Python package manager).

It is recommended to use a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

From the project root (`Car_Price_Prediction`), run:

```bash
streamlit run app/main.py
```

Then open the URL that Streamlit prints in your terminal (typically http://localhost:8501).

Make sure you run the command from the project root so that the app can find:

- `data/cardekho_dataset.csv`
- `model/model.pkl`
- `assets/style.css`

## How It Works

1. The dataset is cleaned and preprocessed:
   - Drops non-essential columns such as name/brand/model identifiers.
   - Applies one-hot encoding to categorical features (fuel type, seller type, transmission).
   - Creates engineered features like `power_per_cc` and `diesel_auto`.
2. A Random Forest Regressor is trained to predict `selling_price`.
3. The trained model is serialized to `model/model.pkl`.
4. The Streamlit app:
   - Collects inputs from the sidebar.
   - Builds a feature vector matching the training schema.
   - Loads the model and returns an estimated price (and for ensembles, a mean ± standard deviation range).

The app requires only the existing `model.pkl`; you do not need to retrain the model to use it.

## Retraining the Model (Optional)

If you want to experiment with training:

- Inspect `utils/data_cleaning.py` to see the preprocessing and training pipeline.
- The current script uses absolute Windows paths; adjust them to your environment before running:
  - Path to `data/cardekho_dataset.csv`
  - Path to output `model/model.pkl`

After retraining, ensure that the new `model.pkl` is saved in the `model/` directory so the app can load it.

## Notes

- For best results, keep the working directory as the project root when running any scripts.
- If you change the dataset schema or add new features, make sure both the training code and `app/main.py` are updated consistently so that feature names and ordering match.

# Screenshots
<img width="2560" height="1440" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/da135f15-e236-469e-8f88-ea1a47ddedad" />
<img width="2560" height="1440" alt="Screenshot (2)" src="https://github.com/user-attachments/assets/30467fe2-0d6a-4635-ba34-98e8c72da35e" />


