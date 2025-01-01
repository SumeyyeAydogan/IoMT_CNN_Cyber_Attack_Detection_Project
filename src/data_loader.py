import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define attack categories
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(file_name, class_config): 
    """Get attack category from file name."""

    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes 
        categories = ATTACK_CATEGORIES_19  

    for key in categories:
        if key in file_name:
            return categories[key]
        
def balance_dataset(df, target_column, n_samples=None):
    """
    Her sınıftan eşit sayıda örnek alarak veri setini dengeler.
    n_samples belirtilmezse en az örneğe sahip sınıftaki örnek sayısı kullanılır.
    """
    # Her sınıftaki örnek sayısını say
    class_counts = df[target_column].value_counts()
    
    # Eğer n_samples belirtilmemişse, en az örneğe sahip sınıfın örnek sayısını kullan
    if n_samples is None:
        n_samples = class_counts.min()
    
    # Her sınıftan n_samples kadar rastgele örnek al
    balanced_dfs = []
    for class_label in class_counts.index:
        class_df = df[df[target_column] == class_label]
        sampled_df = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)
        balanced_dfs.append(sampled_df)
    
    # Tüm örnekleri birleştir
    balanced_df = pd.concat(balanced_dfs, axis=0)
    
    # Verileri karıştır
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_and_preprocess_data(data_dir, class_config=2):
    """
    Veri setini yükler, dengeler ve ön işleme yapar.
    """
    # Tüm CSV dosyalarını birleştir
    all_files = []
    for filename in os.listdir(os.path.join(data_dir, 'train')):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, 'train', filename)
            df = pd.read_csv(file_path)
            
            # Dosya adından saldırı türünü belirle
            attack_type = get_attack_category(filename, class_config)
            
            # Etiket sütunu ekle
            df['label'] = attack_type
            
            all_files.append(df)
    
    # Tüm verileri tek bir DataFrame'de birleştir
    combined_df = pd.concat(all_files, axis=0, ignore_index=True)
    
    # İlk DataFrame'in sütunlarını göster (debug için)
    print("Columns in the dataset:", combined_df.columns.tolist())
    print("\nUnique labels:", combined_df['label'].unique())
    
    # Sınıf yapılandırmasına göre etiketleri ayarla
    if class_config == 2:
        # İkili sınıflandırma: Normal vs Attack
        combined_df['label'] = combined_df['label'].apply(lambda x: 'Attack' if x != 'Benign' else 'Benign')
    elif class_config == 6:
        # 6 sınıflı sınıflandırma: Ana saldırı türlerine göre
        # Burada saldırı türlerini ana kategorilere ayırın
        pass
    # class_config == 19 durumunda zaten detaylı etiketler mevcut
    
    # Veri setini dengele
    balanced_df = balance_dataset(combined_df, 'label')
    
    # Özellikleri ve etiketleri ayır
    X = balanced_df.drop('label', axis=1)
    y = balanced_df['label']
    
    # Etiketleri sayısal değerlere dönüştür
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Verileri eğitim, doğrulama ve test setlerine böl
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Verileri normalize et
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Etiketleri kategorik formata dönüştür
    from tensorflow.keras.utils import to_categorical
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train, num_classes)
    y_val_categorical = to_categorical(y_val, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)
    
    # Verileri yeniden şekillendir
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder
