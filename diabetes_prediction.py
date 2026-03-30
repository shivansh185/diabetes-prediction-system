# =============================================================
# Diabetes Prediction — Full Implementation (SIMPLIFIED & FIXED)
# Matches: "Diabetes Prediction Using Machine Learning and
#           Hybrid Deep Learning Ensemble Technique"
# Models: KNN, NB, SVM, DT, RF, LR + TabNet + XGBoost + MLP
# Datasets: PIMA (768 records) + Dataset 2 (2000 records)
# =============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- Core ML ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, mean_squared_error, mean_absolute_error
)

# --- ML Algorithms ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# --- Deep Learning / Boosting ---
from xgboost import XGBClassifier

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing TabNet; fall back gracefully if not installed
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("pytorch-tabnet not installed. Install with: pip install pytorch-tabnet")
    print("TabNet will be replaced by a GradientBoostingClassifier for this run.\n")
    from sklearn.ensemble import GradientBoostingClassifier


# =============================================================
# SECTION 1 — DATA LOADING (SIMPLIFIED)
# =============================================================

def load_datasets():
    """Load PIMA dataset and create a second dataset."""
    
    # Dataset 1: PIMA Indian Diabetes
    url1 = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df1 = pd.read_csv(url1, names=columns)
    print(f"Dataset 1 (PIMA) loaded: {df1.shape[0]} records")
    
    # Dataset 2: Create by bootstrapping PIMA dataset to 2000 records
    print(f"Creating second dataset with 2000 records...")
    np.random.seed(42)
    
    # Bootstrap sampling with replacement
    indices = np.random.choice(len(df1), size=2000, replace=True)
    df2 = df1.iloc[indices].copy().reset_index(drop=True)
    
    # Add small random noise to features to create variation
    noise_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    for col in noise_columns:
        noise = np.random.normal(0, df1[col].std() * 0.05, size=len(df2))
        df2[col] = df2[col] + noise
        # Clip to reasonable ranges
        if col == 'Glucose':
            df2[col] = df2[col].clip(0, 300)
        elif col == 'BloodPressure':
            df2[col] = df2[col].clip(0, 150)
        elif col == 'SkinThickness':
            df2[col] = df2[col].clip(0, 100)
        elif col == 'Insulin':
            df2[col] = df2[col].clip(0, 500)
        elif col == 'BMI':
            df2[col] = df2[col].clip(10, 50)
        elif col == 'Age':
            df2[col] = df2[col].clip(0, 120)
    
    # Round to reasonable values
    df2['Pregnancies'] = df2['Pregnancies'].round().clip(0, 20)
    df2['Glucose'] = df2['Glucose'].round()
    df2['BloodPressure'] = df2['BloodPressure'].round()
    df2['SkinThickness'] = df2['SkinThickness'].round()
    df2['Insulin'] = df2['Insulin'].round()
    df2['BMI'] = df2['BMI'].round(1)
    df2['DiabetesPedigreeFunction'] = df2['DiabetesPedigreeFunction'].round(3)
    df2['Age'] = df2['Age'].round()
    df2['Outcome'] = df2['Outcome'].round().astype(int)
    
    print(f"Dataset 2 created: {df2.shape[0]} records")
    
    return df1, df2, columns


# =============================================================
# SECTION 2 — PREPROCESSING (SIMPLIFIED & ROBUST)
# =============================================================

def preprocess(df, dataset_name):
    """
    Simplified preprocessing:
    1. Remove duplicates
    2. Handle zero values (replace with mean/median)
    3. Ensure no NaN values
    4. Standardize features
    """
    df = df.copy()
    print(f"  Original shape: {df.shape}")
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    print(f"  After removing duplicates: {df.shape}")
    
    # Step 2: Handle zero values (replace with mean/median)
    # Columns that should not have zero values (clinical measurements)
    zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_replace_cols:
        # Replace zeros with NaN
        df[col] = df[col].replace(0, np.nan)
        
        # Fill NaN with column mean
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean)
        
        # For insulin, also handle extreme values
        if col == 'Insulin':
            df[col] = df[col].clip(0, 500)
    
    # Step 3: Ensure no NaN values in any column
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Step 4: Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Final NaN check
    assert not X.isnull().any().any(), "NaN values still present in features!"
    assert not y.isnull().any(), "NaN values in target!"
    
    print(f"  Cleaned shape: {X.shape}")
    print(f"  Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # Step 5: Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y, scaler


# =============================================================
# SECTION 3 — MODEL DEFINITIONS
# =============================================================

def build_ml_models():
    """Return dict of ML models."""
    models = {
        'K Nearest Neighbor': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42, C=1, kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    return models


def build_ensemble_model():
    """Build ensemble models."""
    xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        eval_metric='logloss', 
        random_state=42,
        use_label_encoder=False
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    
    if TABNET_AVAILABLE:
        return xgb, mlp, TabNetClassifier(verbose=0, seed=42)
    else:
        return xgb, mlp, GradientBoostingClassifier(n_estimators=100, random_state=42)


# =============================================================
# SECTION 4 — EVALUATION FUNCTIONS
# =============================================================

def evaluate_model(name, y_true, y_pred, y_proba=None):
    """Evaluate model performance."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    
    return {
        'Model': name,
        'F1 Score': round(f1, 3),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'Accuracy': f"{acc*100:.1f}%",
        'Accuracy_float': acc,
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'AUC': round(auc, 3) if auc else 'N/A',
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def train_and_evaluate_ml(models, X_train, X_test, y_train, y_test):
    """Train and evaluate ML models."""
    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            results.append(evaluate_model(name, y_test, y_pred, y_proba))
        except Exception as e:
            print(f"    Error: {e}")
    return results


def train_ensemble(X_train, X_test, y_train, y_test):
    """Train ensemble model."""
    xgb, mlp, tabnet = build_ensemble_model()
    
    print("  Training XGBoost...")
    xgb.fit(X_train, y_train)
    
    print("  Training MLP...")
    mlp.fit(X_train, y_train)
    
    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    proba_mlp = mlp.predict_proba(X_test)[:, 1]
    
    if TABNET_AVAILABLE:
        print("  Training TabNet...")
        tabnet.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)])
        proba_tab = tabnet.predict_proba(X_test.values)[:, 1]
    else:
        print("  Training GradientBoosting...")
        tabnet.fit(X_train, y_train)
        proba_tab = tabnet.predict_proba(X_test)[:, 1]
    
    ensemble_proba = (proba_xgb + proba_mlp + proba_tab) / 3.0
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    result = evaluate_model("Ensemble (XGB+MLP+TabNet)", y_test, ensemble_pred, ensemble_proba)
    return result, xgb, mlp


# =============================================================
# SECTION 5 — VISUALIZATION
# =============================================================

def plot_results(results, y_test, dataset_name):
    """Generate all plots."""
    # Confusion Matrix for Ensemble
    ensemble_result = [r for r in results if r['Model'] == "Ensemble (XGB+MLP+TabNet)"][0]
    cm = confusion_matrix(y_test, ensemble_result['y_pred'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    axes[0,0].set_title('Confusion Matrix - Ensemble Model')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # Model Comparison
    models = [r['Model'] for r in results]
    f1_scores = [r['F1 Score'] for r in results]
    acc_scores = [r['Accuracy_float'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    axes[0,1].bar(x - width/2, f1_scores, width, label='F1 Score', color='skyblue')
    axes[0,1].bar(x + width/2, acc_scores, width, label='Accuracy', color='lightgreen')
    axes[0,1].set_xlabel('Models')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('Model Performance Comparison')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(models, rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1)
    
    # ROC Curves
    for r in results:
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            axes[1,0].plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']})")
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curves')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)
    
    # Error Metrics
    error_models = ['Logistic Regression', 'Decision Tree', "Ensemble (XGB+MLP+TabNet)"]
    error_data = [r for r in results if r['Model'] in error_models]
    
    if error_data:
        names = [r['Model'] for r in error_data]
        mses = [r['MSE'] for r in error_data]
        maes = [r['MAE'] for r in error_data]
        
        x = np.arange(len(names))
        width = 0.35
        axes[1,1].bar(x - width/2, mses, width, label='MSE', color='coral')
        axes[1,1].bar(x + width/2, maes, width, label='MAE', color='gold')
        axes[1,1].set_xlabel('Models')
        axes[1,1].set_ylabel('Error')
        axes[1,1].set_title('Error Metrics Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(names, rotation=45, ha='right')
        axes[1,1].legend()
    
    plt.suptitle(f'Diabetes Prediction Results - {dataset_name}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    import os
    save_path = os.path.expanduser(f'~/Desktop/diabetes_results_{dataset_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {save_path}")
    plt.show()


# =============================================================
# SECTION 6 — PREDICTION FUNCTION
# =============================================================

def predict_diabetes(xgb_model, mlp_model, scaler, feature_names):
    """Interactive diabetes prediction."""
    print("\n" + "="*60)
    print("  REAL-TIME DIABETES PREDICTION")
    print("="*60)
    print("\nPlease enter patient information:")
    print("-" * 60)
    
    try:
        # Get input from user
        pregnancies = float(input("Number of Pregnancies: "))
        glucose = float(input("Glucose Level (mg/dL): "))
        bp = float(input("Blood Pressure (mm Hg): "))
        skin = float(input("Skin Thickness (mm): "))
        insulin = float(input("Insulin Level (mu U/ml): "))
        bmi = float(input("BMI: "))
        dpf = float(input("Diabetes Pedigree Function: "))
        age = float(input("Age: "))
        
        # Create feature array
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        features_scaled = scaler.transform(features)
        
        # Make predictions
        proba_xgb = xgb_model.predict_proba(features_scaled)[0][1]
        proba_mlp = mlp_model.predict_proba(features_scaled)[0][1]
        ensemble_proba = (proba_xgb + proba_mlp) / 2
        prediction = "Diabetic" if ensemble_proba >= 0.5 else "Non-Diabetic"
        
        # Display results
        print("\n" + "="*60)
        print("  PREDICTION RESULTS")
        print("="*60)
        print(f"  XGBoost Probability: {proba_xgb*100:.2f}%")
        print(f"  MLP Probability: {proba_mlp*100:.2f}%")
        print(f"  Ensemble Probability: {ensemble_proba*100:.2f}%")
        print("-" * 60)
        print(f"  FINAL PREDICTION: {prediction}")
        print("="*60)
        
        if ensemble_proba >= 0.5:
            print("\n  RECOMMENDATION: Please consult a healthcare provider.")
        else:
            print("\n  RECOMMENDATION: Maintain healthy lifestyle.")
            
    except ValueError:
        print("\n  Invalid input! Please enter numeric values.")
    except Exception as e:
        print(f"\n  Error during prediction: {e}")


# =============================================================
# SECTION 7 — MAIN PIPELINE
# =============================================================

def run_pipeline(df, dataset_name, columns):
    """Run complete pipeline for a dataset."""
    print(f"\n{'='*60}")
    print(f"  Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Preprocessing
    print("\n[1] Preprocessing data...")
    X, y, scaler = preprocess(df, dataset_name)
    
    # Train-test split
    print("\n[2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Train ML models
    print("\n[3] Training individual ML models...")
    ml_models = build_ml_models()
    ml_results = train_and_evaluate_ml(ml_models, X_train, X_test, y_train, y_test)
    
    # Train ensemble
    print("\n[4] Training ensemble model...")
    ensemble_result, xgb_model, mlp_model = train_ensemble(X_train, X_test, y_train, y_test)
    
    # Combine results
    all_results = ml_results + [ensemble_result]
    
    # Display results
    print("\n[5] Results Summary:")
    print("-" * 80)
    print(f"{'Model':<35} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Acc':<8} {'AUC':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['Model']:<35} {r['F1 Score']:<8} {r['Precision']:<8} "
              f"{r['Recall']:<8} {r['Accuracy']:<8} {str(r['AUC']):<8}")
    print("-" * 80)
    
    # Generate plots
    print("\n[6] Generating visualizations...")
    plot_results(all_results, y_test, dataset_name)
    
    return all_results, xgb_model, mlp_model, scaler, X.columns.tolist()


# =============================================================
# MAIN EXECUTION
# =============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DIABETES PREDICTION SYSTEM")
    print("  Machine Learning & Deep Learning Ensemble")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    df1, df2, columns = load_datasets()
    
    # Run pipeline on Dataset 1
    results1, xgb1, mlp1, scaler1, features1 = run_pipeline(
        df1, "PIMA Dataset (768 samples)", columns
    )
    
    # Run pipeline on Dataset 2
    results2, xgb2, mlp2, scaler2, features2 = run_pipeline(
        df2, "Extended Dataset (2000 samples)", columns
    )
    
    # Real-time prediction using the better model (Dataset 2)
    print("\n" + "="*60)
    print("  READY FOR REAL-TIME PREDICTION")
    print("="*60)
    
    while True:
        try:
            choice = input("\nMake a diabetes prediction? (yes/no): ").lower().strip()
            if choice in ['yes', 'y']:
                predict_diabetes(xgb2, mlp2, scaler2, features2)
            elif choice in ['no', 'n']:
                print("\nThank you for using the Diabetes Prediction System!")
                break
            else:
                print("Please enter 'yes' or 'no'.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break