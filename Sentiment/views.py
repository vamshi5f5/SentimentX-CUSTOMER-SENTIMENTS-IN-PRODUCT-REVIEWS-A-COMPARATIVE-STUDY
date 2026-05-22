from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
import pymysql
import pandas as pd
from joblib import load
import os
import pickle
import joblib
import numpy as np
from collections import Counter
from scipy.special import expit  # sigmoid
from imodels import BoostedRulesClassifier
from tqdm import tqdm
# ===============================
# NLP: NLTK & Gensim
# ===============================
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




# ===============================
# TensorFlow / Keras
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ===============================
# Transformers (Hugging Face)
# ===============================
from transformers import (
    AutoTokenizer, AutoModel,
    RobertaTokenizer, RobertaModel,
    BertTokenizer, BertForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification
)
import torch
from torch.optim import AdamW


MODEL_DIR = r"Sentiment\model"
os.makedirs(MODEL_DIR, exist_ok=True)
import sys
print(sys.version)

import nltk

# Tokenizer
nltk.download('punkt', quiet=True)

# Stopwords
nltk.download('stopwords', quiet=True)

# WordNet for lemmatizer
nltk.download('wordnet', quiet=True)

#  POS tagging for better lemmatization
nltk.download('averaged_perceptron_tagger', quiet=True)




def ensure_single_admin():    
    try:
        con = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='mydb'
        )
        with con:
            cur = con.cursor()

        
            cur.execute("SELECT id FROM emp_details3 WHERE role='admin'")
            admin = cur.fetchone()

            
            if not admin:
                cur.execute("""
                    INSERT INTO emp_details3
                    (username, email, password, role, approved, address, mobile)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    "admin",
                    "admin@gmail.com",
                    "admin",
                    "admin",
                    1,                     
                    "hyderabad",
                    "1234567890"
                ))
                con.commit()

    except Exception as e:
        print("Error ensuring admin:", e)



def index(request):
    return render(request, 'index.html')

def user_page(request):
    user = request.session.get('user')
    if not user:
        return redirect('login')  
    return render(request, 'user.html', {'user': user})


@csrf_exempt
def approve_user(request, username):
    user = request.session.get('user')
    if not user or user.get('role') != 'admin':
        return redirect('login')

    try:
        con = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='mydb'
        )
        with con:
            cur = con.cursor()
            cur.execute("UPDATE emp_details3 SET approved = 1 WHERE username = %s", (username,))
            con.commit()
    except Exception as e:
        print("Error approving user:", e)

    return redirect('admin_page')


def admin_page(request):
    user = request.session.get('user')
    if not user:
        return redirect('login')

    if user.get('role') != 'admin':
        return redirect('user_page')

    users_list = []
    try:
        con = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='mydb'
        )
        with con:
            cur = con.cursor(pymysql.cursors.DictCursor)
            cur.execute("SELECT * FROM emp_details3 WHERE role='user'")
            users_list = cur.fetchall()

    except Exception as e:
        return render(request, 'admin.html', {
            'user': user,
            'error': f'Database error: {str(e)}'
        })

    return render(request, 'admin.html', {'user': user, 'users_list': users_list})



def register_view(request):

    ensure_single_admin()

    message = None

    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        mobile = request.POST.get('mobile')          
        address = request.POST.get('address')        
        role = "user"                               

        if password != confirm_password:
            message = {'error': 'Passwords do not match'}

        else:
            try:
                con = pymysql.connect(
                    host='127.0.0.1',
                    port=3306,
                    user='root',
                    password='root',
                    database='mydb'
                )
                with con:
                    cur = con.cursor()

                    cur.execute("SELECT username FROM emp_details3 WHERE username=%s", (username,))
                    if cur.fetchone():
                        message = {'error': 'Username already exists'}


                    else:
                        cur.execute("SELECT email FROM emp_details3 WHERE email=%s", (email,))
                        if cur.fetchone():
                            message = {'error': 'Email already exists'}
                        cur.execute("SELECT mobile FROM emp_details3 WHERE mobile=%s", (mobile,))
                        if cur.fetchone():
                            message = {'error': 'mobile already exists'}
                        else:
                            cur.execute("""
                                INSERT INTO emp_details3
                                (username, email, password, role, approved, mobile, address)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (username, email, password, role, 0, mobile, address))

                            con.commit()
                            message = {'success': 'Account created successfully! Awaiting admin approval.'}

            except Exception as e:
                message = {'error': f'Database error: {str(e)}'}

    return render(request, 'register.html', message)



def login_view(request):
    context = {}
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            con = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='root',
                database='mydb'
            )
            with con:
                cur = con.cursor(pymysql.cursors.DictCursor)
                cur.execute( 
                    "SELECT * FROM emp_details3 WHERE username=%s AND password=%s",
                    (username, password)
                )
                user = cur.fetchone()

                if user:
                    if not user.get('approved'):
                        context['error'] = 'Your account is awaiting admin approval.'
                    else:
                        request.session['user'] = user
                        role = user['role']

                        if role == 'admin':
                            return redirect('admin_page')
                        elif role == 'user':
                            return redirect('user_page')
                        else:
                            context['error'] = 'Invalid role assigned to this user.'
                else:
                    context['error'] = 'Invalid login credentials'

        except Exception as e:
            context['error'] = f'Database error: {str(e)}'

    return render(request, 'login.html', context)


def preprocess_data(df, save_path=None, target_cols=None):

    global label_encoders
    label_encoders = {}  # dictionary to hold encoders for each target column

    if save_path and os.path.exists(save_path):
        print(f"Loading existing preprocessed file: {save_path}")
        df = pd.read_csv(save_path)
    else:
        print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
            return ' '.join(tokens)

        # Separate target columns
        target_df = None
        if target_cols:
            existing_targets = [col for col in target_cols if col in df.columns]
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

        # Process text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[f'processed_{col}'] = df[col].apply(clean_text)

        # Drop original text columns
        df.drop(columns=text_columns, inplace=True)

        # Reattach target columns
        if target_df is not None:
            for col in target_df.columns:
                df[col] = target_df[col]

        # Save only if path is specified
        if save_path:
            df.to_csv(save_path, index=False)

    # Select processed and numerical columns
    processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
    non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

    # Join processed text columns into one string
    X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

    # Combine with numerical columns if any
    X_numeric = df[non_text_cols].values if non_text_cols else None
    if X_numeric is not None and len(X_numeric) > 0:
        X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
    else:
        X = X_text.tolist()

    # Encode multiple target columns
    Y_dict = {}
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    return X, Y_dict



# Feature Extraction

def bert_feature_extraction(texts, model_name='bert-base-uncased', batch_size=32, pooling='mean'):
    """Extract BERT features from texts with tqdm progress bar."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        token_embeddings = model_output.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        if pooling == 'mean':
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
        elif pooling == 'cls':
            embeddings = token_embeddings[:, 0, :]  # CLS token
        else:
            raise ValueError("Pooling must be 'mean' or 'cls'")

        all_embeddings.append(embeddings.cpu().numpy())

    X = np.vstack(all_embeddings)
    return X, model

def feature_extraction(X_text, method='BERT_word_embeddings', model_dir=r'Sentiment\model', is_train=True):
    x_file = os.path.join(model_dir, f'X_{method}.pkl')

    print(f"[INFO] Feature extraction method: {method}, Train mode: {is_train}")
    model_name = 'bert-base-uncased'
    if os.path.exists(x_file):
        print(f"[INFO] Loading cached BERT features from {x_file}")
        X = joblib.load(x_file)
    else:
        print("[INFO] Computing BERT features...")
        X, model = bert_feature_extraction(X_text, model_name=model_name, pooling='mean')
        joblib.dump(X, x_file)
    # else:
    #     print("[INFO] Performing BERT feature extraction for testing...")
    #     X, model = bert_feature_extraction(X_text, model_name=model_name, pooling='mean')
    return X    


def test_DNN_feature_extraction(features_test, model_base_name="SBERT-WE DNN", model_dir=r"Sentiment\model"):
    feature_outputs = {}
    targets = ["Sentiment"]

    for target in targets:
        model_filename = f"{model_base_name}_{target}_dense_model.h5"
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            continue

        print(f" Loading model for {target}: {model_path}")
        model = load_model(model_path)

        try:
            feature_extractor = Model(inputs=model.input,
                                      outputs=model.get_layer("feature_layer").output)
        except:
            raise ValueError(f"Model does not contain a layer named 'feature_layer': {model_path}")

        print(f"Extracting features for {target}...")
        extracted_features = feature_extractor.predict(features_test)
        feature_outputs[target] = extracted_features

    return feature_outputs

Final_models = {}
model_path = os.path.join(MODEL_DIR, "SBERT-WE DNN_Sentiment_BoostedRules_model.pkl")
if os.path.exists(model_path):
    print(f"Loading Sentiment model from: {model_path}")
    Final_models["Sentiment"] = joblib.load(model_path)
else:
    print(" Sentiment model file not found. Please train and save it first.")


def prediction_page(request):
    prediction_table = None
    uploaded_filename = None
    

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        uploaded_filename = file.name
        df_test1 = pd.read_csv(file)
        df_result = df_test1.copy()

        #Preprocess uploaded file
        df_test, _ = preprocess_data(df_test1)

        #  Generate embeddings (set is_train=False)
        features_test = feature_extraction(df_test,method='SBERT with Word Embeddings',is_train=False)
        features_test = feature_extraction(df_test,method='SBERT-WE DNN',is_train=None)
        feature_outputs_dict = test_DNN_feature_extraction(features_test)
        classes = ["Cannot Say", "Negative", "Positive", "Neutral"]

# Loop over each target
        for target in ['Sentiment']:
             target_features = feature_outputs_dict[target]
             y_pred = Final_models[target].predict(target_features)
             mapped_labels = [classes[int(pred)] for pred in y_pred]
             df_result[f'Predicted_{target}'] = mapped_labels
             # 👉 SL.No starting from 0
             df_result.insert(0, "Sl.No", range(0, len(df_result)))

        #  Display prediction table
        prediction_table = df_result.to_html(
            classes='table table-bordered table-striped table-hover',
            index=False
        )

        messages.success(request, f"Predictions generated successfully for {uploaded_filename}")

    return render(
        request,
        'prediction.html',
        {
            'prediction_table': prediction_table,
            'uploaded_filename': uploaded_filename
        }
    )

