import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: imblearn and shap
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Employee Attrition — Advanced App", layout='wide')

# ----------------------------- Helpers -----------------------------
@st.cache_data
def load_sample_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'Age': rng.integers(20, 60, n),
        'Department': rng.choice(['Sales','Research & Development','Human Resources'], n, p=[0.45,0.45,0.10]),
        'MonthlyIncome': rng.integers(2000, 20000, n),
        'YearsAtCompany': rng.integers(0, 30, n),
        'YearsSinceLastPromotion': rng.integers(0,10,n),
        'JobSatisfaction': rng.integers(1,5,n),
        'WorkLifeBalance': rng.integers(1,4,n),
        'OverTime': rng.choice(['Yes','No'], n, p=[0.2,0.8]),
        'Education': rng.integers(1,5,n),
        'Attrition': rng.choice([0,1], n, p=[0.85,0.15])
    })
    mask = (df['YearsAtCompany']<3) & (df['OverTime']=='Yes')
    df.loc[mask, 'Attrition'] = (rng.random(mask.sum()) < 0.5).astype(int)
    return df


def read_uploaded_file(file):
    try:
        name = file.name.lower()
        if name.endswith('.csv'):
            return pd.read_csv(file)
        if name.endswith(('.xls','.xlsx')):
            return pd.read_excel(file)
        if name.endswith('.json'):
            return pd.read_json(file)
        st.error('Unsupported format. Use CSV, Excel (.xls/.xlsx) or JSON.')
        return None
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        return None


def display_download_link(df, filename='data.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ----------------------------- App UI -----------------------------
st.title('Employee Attrition — Advanced Interactive ML App')
st.markdown('Upload CSV/Excel/JSON or use a synthetic sample. Use the left panel to tweak everything — sliders update calculations in real time.')

# Sidebar: Data input
with st.sidebar.expander('Data — upload or sample', expanded=True):
    uploaded = st.file_uploader('Upload dataset (CSV / XLSX / JSON)', type=['csv','xlsx','xls','json'])
    use_sample = st.checkbox('Use synthetic sample dataset', value=(uploaded is None))
    if uploaded is not None:
        df = read_uploaded_file(uploaded)
    elif use_sample:
        df = load_sample_data()
    else:
        st.stop()
    st.write(f'Dataset: {df.shape[0]} rows × {df.shape[1]} cols')

if st.checkbox('Show raw data'):
    st.dataframe(df.head(200))

# Target detection and selection
default_target = None
for t in ['Attrition','attrition','target','Target']:
    if t in df.columns:
        default_target = t
        break

target_col = st.sidebar.selectbox('Target column (binary)', options=[None]+list(df.columns), index=list(df.columns).index(default_target)+1 if default_target else 0)
if target_col is None:
    st.sidebar.error('Choose a target column to continue')
    st.stop()

# Features selection
all_feats = [c for c in df.columns if c!=target_col]
selected_feats = st.sidebar.multiselect('Select features to use', options=all_feats, default=all_feats)

# Preprocessing controls
with st.sidebar.expander('Preprocessing & feature engineering', expanded=False):
    impute_num = st.selectbox('Numeric imputation', ['mean','median','constant'], index=0)
    fill_value = st.number_input('Fill constant value (if selected)', value=0.0)
    impute_cat = st.selectbox('Categorical imputation', ['most_frequent','constant'], index=0)
    encode_onehot = st.checkbox('One-hot encode categorical', value=True)
    scale_method = st.selectbox('Scaling', ['None','StandardScaler','MinMaxScaler'], index=1)
    add_interactions = st.checkbox('Add interaction (OverTime x YearsAtCompany)', value=True)
    add_derived = st.checkbox('Add derived features (IncomePerYearAtCompany)', value=True)

# Balancing & split
with st.sidebar.expander('Train/Test & balancing', expanded=False):
    balance = st.selectbox('Class imbalance handling', ['None','SMOTE (if available)','Class weight'], index=2)
    test_size = st.slider('Test size fraction', 0.1, 0.5, 0.2)
    random_state = int(st.number_input('Random seed', value=42, step=1))

# Model selection
with st.sidebar.expander('Model & hyperparameters', expanded=True):
    model_choice = st.selectbox('Model', ['Logistic Regression','Random Forest'])
    if model_choice=='Logistic Regression':
        C = st.number_input('Inverse regularization (C)', min_value=0.0001, max_value=1000.0, value=1.0, step=0.1)
        max_iter = int(st.number_input('Max iterations', value=200, step=10))
    else:
        n_estimators = int(st.slider('n_estimators', 10, 1000, 200))
        max_depth = int(st.slider('max_depth (0 => None)', 0, 50, 10))
        max_depth = None if max_depth==0 else max_depth
    cv_folds = int(st.slider('Cross-val folds', 2, 10, 5))

# ----------------------------- Feature engineering -----------------------------
df_work = df.copy()
if add_derived and 'MonthlyIncome' in df_work.columns and 'YearsAtCompany' in df_work.columns:
    df_work['IncomePerYearAtCompany'] = df_work['MonthlyIncome'] / (df_work['YearsAtCompany'] + 1)
if add_interactions and 'OverTime' in df_work.columns and 'YearsAtCompany' in df_work.columns:
    df_work['OverTime_Years'] = df_work['OverTime'].astype(str) + '_' + pd.cut(df_work['YearsAtCompany'], bins=[-1,1,3,6,30], labels=['<1','1-3','3-6','6+']).astype(str)

# Selected features split
numeric_cols = df_work[selected_feats].select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = [c for c in selected_feats if c not in numeric_cols]

st.subheader('Selected features and types')
col1, col2 = st.columns(2)
col1.write('Numeric')
col1.write(numeric_cols)
col2.write('Categorical')
col2.write(cat_cols)

# Build preprocessing pipeline
numeric_transformers = []
if numeric_cols:
    if impute_num=='constant':
        numeric_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value=fill_value)))
    else:
        numeric_transformers.append(('imputer', SimpleImputer(strategy=impute_num)))
    if scale_method=='StandardScaler':
        from sklearn.preprocessing import StandardScaler as _SS
        numeric_transformers.append(('scaler', _SS()))
    elif scale_method=='MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler as _MM
        numeric_transformers.append(('scaler', _MM()))

cat_transformers = []
if cat_cols:
    if impute_cat=='constant':
        cat_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
    else:
        cat_transformers.append(('imputer', SimpleImputer(strategy='most_frequent')))
    if encode_onehot:
        cat_transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

transformers = []
if numeric_transformers:
    transformers.append(('num', Pipeline([t for t in numeric_transformers]), numeric_cols))
if cat_transformers:
    transformers.append(('cat', Pipeline([t for t in cat_transformers]), cat_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

# Prepare X, y
def prepare_target(y_series):
    """Convert target column to numeric integers safely."""
    # Case 1: Already numeric
    if np.issubdtype(y_series.dtype, np.number):
        return y_series.astype(int)

    # Case 2: Binary strings (Yes/No, True/False)
    unique_vals = y_series.dropna().unique()
    if set(unique_vals) <= {'Yes','No'}:
        return y_series.map({'Yes':1,'No':0}).astype(int)
    if set(unique_vals) <= {'yes','no'}:
        return y_series.map({'yes':1,'no':0}).astype(int)
    if set(unique_vals) <= {True,False}:
        return y_series.astype(int)

    # Case 3: Multi-class categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(y_series)


X = df_work[selected_feats].copy()
y = prepare_target(df_work[target_col])

# Show target distribution
st.subheader('Target distribution')
st.write(y.value_counts().rename_axis('class').reset_index(name='count'))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# Preprocess and optional balancing
if balance=='SMOTE (if available)' and IMBLEARN_AVAILABLE:
    X_train_prep = preprocessor.fit_transform(X_train)
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)
    balancing_used = 'SMOTE'
else:
    X_train_res = preprocessor.fit_transform(X_train)
    y_train_res = y_train
    balancing_used = balance

X_test_prep = preprocessor.transform(X_test)

# Show preprocessing details
with st.expander('Preprocessing details (click to expand)'):
    st.write('Preprocessor pipeline:')
    st.write(preprocessor)
    st.write('Balancing used:', balancing_used)
    st.write('X_train shape (after prep):', getattr(X_train_res, 'shape', 'N/A'))
    st.write('X_test shape (after prep):', X_test_prep.shape)

# ----------------------------- Model training -----------------------------
st.subheader('Model training & evaluation')
if model_choice=='Logistic Regression':
    cls = LogisticRegression(C=float(C), max_iter=int(max_iter), class_weight='balanced' if balance=='Class weight' else None, solver='lbfgs')
else:
    cls = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth, class_weight='balanced' if balance=='Class weight' else None, random_state=random_state)

with st.spinner('Training model...'):
    cls.fit(X_train_res, y_train_res)

# Predictions
y_pred = cls.predict(X_test_prep)
try:
    y_proba = cls.predict_proba(X_test_prep)[:,1]
except Exception:
    y_proba = cls.predict(X_test_prep)

# Metrics (show calculations explicitly)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_proba)

st.metric('Accuracy', f'{acc:.4f}')
st.metric('Precision', f'{prec:.4f}')
st.metric('Recall', f'{rec:.4f}')
st.metric('F1-score', f'{f1:.4f}')
st.metric('ROC AUC', f'{roc:.4f}')

# Confusion matrix (detailed)
st.subheader('Confusion matrix (counts & normalized)')
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
col1, col2 = st.columns(2)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
ax1.set_title('Confusion matrix (counts)')
col1.pyplot(fig1)
fig2, ax2 = plt.subplots()
sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax2, cmap='Reds')
ax2.set_title('Confusion matrix (normalized by true class)')
col2.pyplot(fig2)

# ROC curve (plotly) and threshold slider
st.subheader('ROC and threshold analysis')
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':np.append(thresholds, np.nan)[:len(fpr)]})
fig_roc = px.line(roc_df, x='fpr', y='tpr', title=f'ROC Curve (AUC={roc:.3f})')
fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
st.plotly_chart(fig_roc, use_container_width=True)

threshold = st.slider('Choose classification threshold for positive class', 0.0, 1.0, 0.5, 0.01)
st.write('At threshold =', threshold)

y_pred_thresh = (y_proba >= threshold).astype(int)
acc_t = accuracy_score(y_test, y_pred_thresh)
prec_t = precision_score(y_test, y_pred_thresh, zero_division=0)
rec_t = recall_score(y_test, y_pred_thresh, zero_division=0)

st.write(f'Accuracy@thr={acc_t:.3f}, Precision@thr={prec_t:.3f}, Recall@thr={rec_t:.3f}')

# Feature importance or permutation importance
st.subheader('Feature importance / permutation importance')
try:
    if hasattr(cls, 'feature_importances_'):
        importances = cls.feature_importances_
        # attempt to recover feature names
        feat_names = []
        if hasattr(preprocessor, 'transformers_'):
            # numeric names
            feat_names.extend(numeric_cols)
            # for one-hot, try to get names
            try:
                ohe = preprocessor.named_transformers_.get('cat').named_steps.get('onehot')
                ohe_names = list(ohe.get_feature_names_out(cat_cols))
                feat_names.extend(ohe_names)
            except Exception:
                pass
        if len(feat_names) != len(importances):
            feat_names = [f'f{i}' for i in range(len(importances))]
        fi_df = pd.DataFrame({'feature':feat_names, 'importance':importances}).sort_values('importance', ascending=False)
        st.dataframe(fi_df.head(50))
        fig_fi = px.bar(fi_df.head(30), x='importance', y='feature', orientation='h')
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        perm = permutation_importance(cls, X_test_prep, y_test, n_repeats=10, random_state=42)
        fi_df = pd.DataFrame({'feature':[f'f{i}' for i in range(len(perm.importances_mean))], 'importance':perm.importances_mean}).sort_values('importance', ascending=False)
        st.dataframe(fi_df.head(50))
        st.plotly_chart(px.bar(fi_df.head(30), x='importance', y='feature', orientation='h'), use_container_width=True)
except Exception as e:
    st.write('Feature importance failed:', e)

# PCA 2D projection with interactive point hover
st.subheader('PCA 2D projection')
try:
    pca = PCA(n_components=2)
    X_all_prep = preprocessor.transform(X)
    X_pca = pca.fit_transform(X_all_prep)
    pca_df = pd.DataFrame(X_pca, columns=['PC1','PC2'])
    pca_df[target_col] = y.values
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=target_col, title='PCA projection', hover_data=pca_df.columns)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.write('Explained variance ratio:', pca.explained_variance_ratio_)
except Exception as e:
    st.write('PCA failed:', e)

# Optional SHAP explanation
if SHAP_AVAILABLE:
    with st.expander('SHAP explanation'):
        try:
            explainer = shap.Explainer(cls, X_train_res)
            shap_values = explainer(X_test_prep[:100])
            st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        except Exception as e:
            st.write('SHAP failed:', e)
else:
    st.info('SHAP not installed. Install `shap` if you want SHAP explanations.')

# Single-record interactive prediction with all calculation values visible
st.subheader('Single-record interactive prediction (enter values)')
with st.form('single_pred'):
    user_input = {}
    for c in selected_feats:
        if c in numeric_cols:
            user_input[c] = st.number_input(f'{c}', value=float(X[c].median()))
        else:
            opts = X[c].dropna().unique().tolist()
            if len(opts) < 50:
                user_input[c] = st.selectbox(f'{c}', options=opts, index=0)
            else:
                user_input[c] = st.text_input(f'{c}', '')
    submit = st.form_submit_button('Predict')
    if submit:
        single_df = pd.DataFrame([user_input])
        single_prep = preprocessor.transform(single_df)
        pred = cls.predict(single_prep)[0]
        try:
            proba = cls.predict_proba(single_prep)[0,1]
        except Exception:
            proba = None
        st.write('Predicted class (1=Attrition):', int(pred))
        if proba is not None:
            st.write('Predicted probability:', float(proba))

# Export predictions
st.subheader('Export predictions on test set')
export_df = X_test.copy().reset_index(drop=True)
export_df['y_true'] = y_test.reset_index(drop=True)
export_df['y_pred'] = y_pred
export_df['y_proba'] = y_proba
if st.button('Show prediction sample'):
    st.dataframe(export_df.head(50))
if st.button('Download predictions CSV'):
    display_download_link(export_df, 'attrition_predictions.csv')

# Final notes and requirements
with st.expander('Notes & recommended packages'):
    st.markdown("""
- For best results, install: pandas scikit-learn streamlit plotly matplotlib seaborn imbalanced-learn shap openpyxl
- Use GridSearchCV/RandomizedSearchCV for systematic hyperparameter tuning.
- Consider time-based split if your dataset contains time features.
- This app is a learning/experimental tool — for production serialize the pipeline and serve via API.
""")

st.success('Finished — adapt the app to your dataset and experiment with the controls!')
