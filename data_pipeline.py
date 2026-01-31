import pandas as pd

def preprocess_dataframe(df, is_training=False):
    """
    Обрабатывает DataFrame для модели предсказания оттока клиентов.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Входной DataFrame с данными клиентов
    is_training : bool, default=False
        Если True, обрабатывает целевой признак Churn (для обучения)
        Если False, предполагает что Churn отсутствует (для предсказания)
    
    Returns:
    --------
    pandas.DataFrame
        Обработанный DataFrame готовый для модели
    """
    
    # Создаем копию, чтобы не изменять исходный DataFrame
    df_processed = df.copy()
    
    # 1. Удаляем колонку с уникальными значениями (если существует)
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
    
    # 2. Преобразуем колонки с Yes и No в Boolean
    # gender: Male -> False, Female -> True
    if 'gender' in df_processed.columns:
        df_processed['gender'] = df_processed['gender'].apply(
            lambda x: True if x == 'Female' else False if x == 'Male' else None
        ).astype(bool)
    
    # SeniorCitizen: 0 -> False, 1 -> True
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(bool)
    
    # Partner, Dependents, PaperlessBilling: Yes -> True, No -> False
    for col in ['Partner', 'Dependents', 'PaperlessBilling']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(
                lambda x: True if x == 'Yes' else False if x == 'No' else None
            ).astype(bool)
    
    # 3. Обработка PhoneService и MultipleLines
    if all(col in df_processed.columns for col in ['PhoneService', 'MultipleLines']):
        # Удаляем PhoneService
        df_processed = df_processed.drop('PhoneService', axis=1)
        
        # One-hot кодирование для MultipleLines
        df_processed = pd.get_dummies(df_processed, 
                                      columns=['MultipleLines'], 
                                      prefix=['MultipleLines'])
        
        # Удаляем колонку 'MultipleLines_No phone service', если существует
        drop_col = 'MultipleLines_No phone service'
        if drop_col in df_processed.columns:
            df_processed = df_processed.drop(drop_col, axis=1)
        
        # Переименовываем для ясности (опционально)
        rename_dict = {
            'MultipleLines_No': 'MultipleLines_No',
            'MultipleLines_Yes': 'MultipleLines_Yes'
        }
        df_processed = df_processed.rename(columns=rename_dict)
    
    # 4. Обработка InternetService, OnlineSecurity, OnlineBackup, 
    # DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    service_columns = [
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Проверяем, какие колонки существуют в DataFrame
    existing_service_cols = [col for col in service_columns if col in df_processed.columns]
    
    if existing_service_cols:
        # One-hot кодирование для существующих колонок
        df_processed = pd.get_dummies(df_processed, columns=existing_service_cols)
        
        # Удаляем ненужные колонки (если они существуют)
        columns_to_drop = [
            'InternetService_No',
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service'
        ]
        
        # Удаляем только те колонки, которые существуют
        columns_to_drop_existing = [col for col in columns_to_drop if col in df_processed.columns]
        if columns_to_drop_existing:
            df_processed = df_processed.drop(columns_to_drop_existing, axis=1)
    
    # 5. One-hot кодирование для PaymentMethod и Contract
    for col in ['PaymentMethod', 'Contract']:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, 
                                          columns=[col], 
                                          prefix=[col], 
                                          drop_first=True)
    
    # 6. Обработка TotalCharges и создание нового признака price_delta
    if all(col in df_processed.columns for col in ['tenure', 'MonthlyCharges', 'TotalCharges']):
        # Преобразуем TotalCharges в число
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        
        # Заполняем пропуски по формуле
        mask = df_processed['TotalCharges'].isna()
        df_processed.loc[mask, 'TotalCharges'] = (
            df_processed.loc[mask, 'tenure'] * df_processed.loc[mask, 'MonthlyCharges']
        )
        
        # Создаем новый признак
        df_processed['price_delta'] = (
            df_processed['tenure'] * df_processed['MonthlyCharges'] - df_processed['TotalCharges']
        )
        
        # Удаляем TotalCharges
        df_processed = df_processed.drop('TotalCharges', axis=1)
    
    # 7. Обработка целевой переменной Churn (только для обучения)
    if is_training and 'Churn' in df_processed.columns:
        df_processed['Churn'] = df_processed['Churn'].apply(
            lambda x: True if x == 'Yes' else False if x == 'No' else None
        ).astype(bool)
    
    # 8. Обеспечиваем одинаковый порядок колонок (опционально, но полезно)
    # Можно добавить сортировку колонок для консистентности
    
    return df_processed


