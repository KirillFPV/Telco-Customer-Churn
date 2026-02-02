import pandas as pd

def preprocess_dataframe(df):
    """
    Обрабатывает DataFrame с данными о клиентах телеком-компании.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Исходный DataFrame с сырыми данными
        
    Returns:
    --------
    pandas.DataFrame
        Обработанный DataFrame с преобразованными признаками
    """
    
    result_df = df.copy()
    
    result_df = result_df.drop('customerID', axis=1, errors='ignore')
    
    result_df['gender'] = result_df['gender'].apply(lambda x: False if x == 'Male' else True).astype(bool)
    result_df['SeniorCitizen'] = result_df['SeniorCitizen'].astype(bool)
    result_df['Partner'] = result_df['Partner'].apply(lambda x: True if x == 'Yes' else False).astype(bool)
    result_df['Dependents'] = result_df['Dependents'].apply(lambda x: True if x == 'Yes' else False).astype(bool)
    result_df['PaperlessBilling'] = result_df['PaperlessBilling'].apply(lambda x: True if x == 'Yes' else False).astype(bool)
    
    result_df = result_df.drop('PhoneService', axis=1, errors='ignore')
    result_df = pd.get_dummies(result_df, columns=['MultipleLines'], prefix='MultipleLines')
    result_df = result_df.drop('MultipleLines_No phone service', axis=1, errors='ignore')
    
    internet_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    result_df = pd.get_dummies(result_df, columns=internet_columns)
    
    columns_to_drop = [
        'InternetService_No', 
        'OnlineSecurity_No internet service', 
        'OnlineBackup_No internet service', 
        'DeviceProtection_No internet service',
        'TechSupport_No internet service', 
        'StreamingTV_No internet service', 
        'StreamingMovies_No internet service'
    ]
    
    for col in columns_to_drop:
        if col in result_df.columns:
            result_df = result_df.drop(col, axis=1)
    
    result_df = pd.get_dummies(result_df, columns=['PaymentMethod', 'Contract'], drop_first=True)
    
    result_df['TotalCharges'] = pd.to_numeric(result_df['TotalCharges'], errors='coerce')
    
    mask = result_df['TotalCharges'].isna()
    result_df.loc[mask, 'TotalCharges'] = result_df.loc[mask, 'tenure'] * result_df.loc[mask, 'MonthlyCharges']
    
    result_df['price_delta'] = result_df['tenure'] * result_df['MonthlyCharges'] - result_df['TotalCharges']
    result_df = result_df.drop('TotalCharges', axis=1)
    
    target_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PaperlessBilling', 'MonthlyCharges', 'MultipleLines_No',
        'MultipleLines_Yes', 'InternetService_DSL',
        'InternetService_Fiber optic', 'OnlineSecurity_No',
        'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'Contract_One year', 'Contract_Two year', 'price_delta'
    ]
    
    for col in target_columns:
        if col not in result_df.columns:
            result_df[col] = False if col.endswith(('_No', '_Yes', '_DSL', '_Fiber optic', 
                                                     'Credit card (automatic)', 'Electronic check', 
                                                     'Mailed check', 'One year', 'Two year')) else 0
    
    return result_df[target_columns]