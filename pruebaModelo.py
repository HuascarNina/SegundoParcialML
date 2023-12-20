#importamos las librerias que utilizaremos
import datetime
import os
import ssl
import sys
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skfeature
import re
import pydotplus
import patoolib
from IPython.display import Image
from pathlib import Path
from pyunpack import Archive
from six.moves import urllib
from skfeature.function.statistical_based import CFS
from sklearn.utils import resample
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from time import time
from urllib.parse import urlparse
from urllib import parse

#Configuramos nuestras direcciones tanto de nuestro escritorio, como las direcciones de los datasets
BASE_DIR = 'C:/Users/huasc/OneDrive/Escritorio/ProyectoSegundoParcial' #ATENCION: PARA QUE FUNCIONE COLOCAR LA DIRECCION DE LA CARPETA DONDE TENDRA GUARDADO EL PROYECTO
DOWNLOAD_ROOT = "https://www.isi.csic.es/dataset/"
HTTPTRAFFIC_RAW_PATH = os.path.join("datasets/row", "http-traffic/")
HTTPTRAFFIC_PROCESSED_PATH = os.path.join("datasets/processed", "http-traffic")
TRAINING_NORMAL_HTTPTRAFFIC_URL = DOWNLOAD_ROOT + "normalTrafficTraining.rar"
TEST_NORMAL_HTTPTRAFFIC_URL = DOWNLOAD_ROOT + "normalTrafficTest.rar"
TEST_ANOMALOUS_HTTPTRAFFIC_URL = DOWNLOAD_ROOT + "anomalousTrafficTest.rar"

# hack para certificado SSL del CSIC
ssl._create_default_https_context = ssl._create_unverified_context

# recuperar datos de tráfico http
def fetch_httpTraffic_data(httpTraffic_path=HTTPTRAFFIC_RAW_PATH,
                           trainingNormalTraffic_url=TRAINING_NORMAL_HTTPTRAFFIC_URL,
                           testNormalTraffic_url=TEST_NORMAL_HTTPTRAFFIC_URL,
                           testAnomalousTraffic_url=TEST_ANOMALOUS_HTTPTRAFFIC_URL):
  if not os.path.isdir(httpTraffic_path):
      os.makedirs(httpTraffic_path)

  rar_trainingNormalTraffic_path = os.path.join(
  httpTraffic_path, "normalTrafficTraining.rar")
  rar_testNormalTraffic_path = os.path.join(
  httpTraffic_path, "normalTrafficTest.rar")
  rar_testAnomalousTraffic_path = os.path.join(
  httpTraffic_path, "anomalousTrafficTest.rar")

  urllib.request.urlretrieve(
  trainingNormalTraffic_url, rar_trainingNormalTraffic_path)
  patoolib.extract_archive(rar_trainingNormalTraffic_path, outdir=httpTraffic_path)
  os.remove(rar_trainingNormalTraffic_path)

  urllib.request.urlretrieve(
  testNormalTraffic_url, rar_testNormalTraffic_path)
  patoolib.extract_archive(rar_testNormalTraffic_path, outdir=httpTraffic_path)
  os.remove(rar_testNormalTraffic_path)

  urllib.request.urlretrieve(
  testAnomalousTraffic_url, rar_testAnomalousTraffic_path)
  patoolib.extract_archive(rar_testAnomalousTraffic_path, outdir=httpTraffic_path)
  os.remove(rar_testAnomalousTraffic_path)

def parse_datafile(dataset_filepath, csv_filepath, label):
    data = [] # crea una lista vacía para recopilar los datos
    method_is_get = False
    method_is_post_or_put = False
    if not os.path.isdir(os.path.dirname(os.path.abspath(csv_filepath))):
      os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)))

    # leer todo el archivo txt y guardar el conjunto de datos en la lista data[]
    with open(dataset_filepath, 'r') as file_object:
        lines = filter(None, (line.rstrip()
                              for line in file_object)) # elimina líneas en blanco
        try:
            line = next(lines)
        except StopIteration as e:
            print(e)
        while line:
            row = []
            if line.startswith('GET'):
                method_is_get = True
                method_is_post = False
                method_is_put = False
            elif line.startswith('POST'):
                method_is_get = False
                method_is_post = True
                method_is_put = False
            elif line.startswith('PUT'):
                method_is_get = False
                method_is_post = False
                method_is_put = True
            row.append(line.split(' ')[0]) # metodo
            url = line.split(' ')[1]
            row.append(url) # url
            row.append(line.split(' ')[2]) # protocolo
            row.append(next(lines).split('User-Agent: ', 1)[1])
            row.append(next(lines).split('Pragma: ', 1)[1])
            row.append(next(lines).split('Cache-control: ', 1)[1])
            if method_is_put:
                row.append(next(lines).split('Accept:', 1)[1])
            else:
                row.append(next(lines).split('Accept: ', 1)[1])
            row.append(next(lines).split('Accept-Encoding: ', 1)[1])
            row.append(next(lines).split('Accept-Charset: ', 1)[1])
            row.append(next(lines).split('Accept-Language: ', 1)[1])
            row.append(next(lines).split('Host: ', 1)[1])
            row.append(next(lines).split('Cookie: ', 1)[1])
            if method_is_get:
                connection = next(lines).split('Connection: ', 1)[1]
                # tipo de contenido vacio en el método GET
                row.append("notpresent")
                # longitud del contenido vacio en el método GET
                row.append("notpresent")
                url_get_parameters = "notpresent" if not urlparse(url).query else urlparse(url).query
                row.append(url_get_parameters) # obtener parámetros de URL
            elif method_is_post or method_is_put:
                content_type = next(lines).split('Content-Type: ', 1)[1]
                connection = next(lines).split('Connection: ', 1)[1]
                content_length = next(lines).split('Content-Length: ', 1)[1]
                url_parameters = next(lines)
                row.append(content_type)
                row.append(content_length)
                row.append(url_parameters)
            row.append(connection)
            row.append(label)
            data.append(row)
            try:
                line = next(lines)
            except StopIteration as e:
                print(e)
                break
    # guardar la lista de datos data[] en un archivo csv agregando los nombres de las columnas
    with open(csv_filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        # escribir encabezados
        writer.writerow(['feature-Method', 'feature-URL', 'feature-Protocol',
                        'feature-User-Agent', 'feature-Pragma', 'feature-Cache-Control',
                        'feature-Accept', 'feature-Accept-Encoding', 'feature-Accept-Charset',
                        'feature-Accept-Language', 'feature-Host', 'feature-Cookie',
                        'feature-Content-Type', 'feature-Content-Length', 'feature-Query',
                        'feature-Connection', 'Label'])
        # escribir datos en el csv
        writer.writerows(data)


def train_test_validation_split(full_dataset, train_size, test_size, validation_size, dir_name, verbose=True):
    proportion = 1 - validation_size
    # establece la semilla en un número fijo para muestrear siempre los mismos conjuntos de datos
    np.random.seed(0)
    total_rows_dataset = full_dataset.shape[0]
    if verbose:
        print("Dividiendo el conjunto de datos en conjuntos de entrenamiento, prueba y validación...")
    train_set, validation_set, test_set = np.split(full_dataset.sample(frac=1), [int(
        train_size * len(full_dataset)), int(proportion * len(full_dataset))])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    validation_set = validation_set.sample(frac=1).reset_index(drop=True)
    X_train = train_set.loc[:, train_set.columns != 'Label']
    y_train = train_set['Label']
    X_test = test_set.loc[:, test_set.columns != 'Label']
    y_test = test_set['Label']
    X_validation = validation_set.loc[:, validation_set.columns != 'Label']
    y_validation = validation_set['Label']
    if verbose:
        print(" - train-set: " +
              str(int(train_set.shape[0] * 100 / total_rows_dataset)) + "%" + " (" + str(train_set.shape[0]) + "ejemplos)")
        print(" - test-set: " + str(int(test_set.shape[0] * 100 / total_rows_dataset)
                                    ) + "%" + " (" + str(test_set.shape[0]) + " ejemplos)")
        print(" - validation-set: " +
              str(int(validation_set.shape[0] * 100 / total_rows_dataset)) + "%" + " (" + str(validation_set.shape[0]) + " ejemplos)")
    train_set.to_csv(
        BASE_DIR + '/datasets/' + dir_name + 'train_set.csv', sep='|', index=False)
    test_set.to_csv(
        BASE_DIR + '/datasets/' + dir_name + 'test_set.csv', sep='|', index=False)
    validation_set.to_csv(
        BASE_DIR + '/datasets/' + dir_name + 'validation_set.csv', sep='|', index=False)
    return X_train, y_train, X_test, y_test, X_validation, y_validation

def combine_datasets(full_dataset, dir_name):
    combined_set = full_dataset.sample(frac=1).reset_index(drop=True)
    combined_set.to_csv(
        BASE_DIR + '/datasets/' + dir_name + 'combined_set.csv', sep='|', index=False)
    X_test = combined_set.loc[:, combined_set.columns != 'Label']
    y_test = combined_set['Label']
    return X_test, y_test


def path_from_URL(url):
    return urlparse(url).path


def count_params(txt):
    """Devuelve el número de parámetros."""
    params = parse.parse_qs(txt)
    if not params: # la consulta no tiene ningún parámetro
        return 'notpresent'
    else:
        return len(params)


def count_dots(txt):
    """Devuelve la cantidad de "." en el texto."""
    return txt.count(".")

def count_percentage(txt):
    """Devuelve la cantidad de "%" en el texto."""
    return txt.count("%")


def count_specialChars(txt):
    """Devuelve la cantidad de caracteres especiales en el texto."""
    return len(re.sub('[\\w]+', '', txt))


def extract_label_values(lst):
    """Devuelve una lista con el último elemento de cada lista en lst"""
    return [item[-1] for item in lst]


def get_longest_param_value(query):
    dict = parse.parse_qs(query)
    all_values = dict.values()
    all_values_list = [item for elem in all_values for item in elem]
    if len(all_values_list):
        return len(max(all_values_list, key=len))
    else:
        return 0


def name_params(query):
    """Devuelve una cadena con la lista concatenada ordenada de nombres de parámetros"""
    params_list = parse.parse_qs(query)
    if not params_list: # la consulta no tiene ningún parámetro
        return 'notpresent'
    else:
        params_list_sorted = sorted(list(params_list))
        return '-'.join(params_list_sorted)


def get_resource_extension(URL_path):
    """Devuelve el archivo de extensión del recurso URL o 0 si es otro caso"""
    resource_extension = Path(URL_path).suffix
    if resource_extension:
        return resource_extension
    else:
        return 0


def get_resource_extension_length(URL_path):
    resource_extension = Path(URL_path).suffix
    if resource_extension:
        return len(resource_extension)
    else:
        return 0


def type_params(query):
    """Devuelve una cadena codificada que describe los tipos de parámetros."""
    type_params_list = ''
    params = parse.parse_qs(query)
    if not params: # la consulta no tiene ningún parámetro
        return 'notpresent'
    params_list = list(params.values())
    for p in params_list:
        if p[0].isdigit():
            type_params_list += str('[DIGIT]')
        else:
            type_params_list += str('[ALPHABET]')
    return type_params_list


def get_token_count(url):
    """Devuelve el número de tokens. Un token es un parámetro y el valor de un parámetro."""
    if url == '':
        return 0
    token_word = re.split('\\W+', url)
    no_ele = 0
    for ele in token_word:
        l = len(ele)
        if l > 0:
            no_ele += 1
    return no_ele


def get_token_largest(url):
    """Devuelve la longitud del token pargest"""
    if url == '':
        return 0
    token_word = re.split('\\W+', url)
    largest = 0
    for ele in token_word:
        l = len(ele)
        if largest < l:
            largest = l
    return largest


def get_token_avg(url):
    """Devuelve las longitudes promedio de todos los tokens en la URL"""
    if url == '':
        return 0
    token_word = re.split('\\W+', url)
    no_ele = sum_len = 0
    for ele in token_word:
        l = len(ele)
        sum_len += l
        if l > 0: # para exclusión de elementos vacíos en longitud promedio
            no_ele += 1
    try:
        return float(sum_len) / no_ele
    except:
        return 0

def auto_feature_selection(X_validation, y_validation):
    # para disminuir el tiempo transcurrido por CFS, autopreselección de características por Variance y SelectKBest
    sel = VarianceThreshold(threshold=(.6 * (1 - .6)))
    sel.fit(X_validation)
    X_validation = X_validation[X_validation.columns[sel.get_support(
        indices=True)]]
    sel2 = SelectKBest(chi2, k=10).fit(X_validation, y_validation)
    X_validation = X_validation[X_validation.columns[sel2.get_support(
        indices=True)]]
    # obtener todos los encabezados (características) en el marco de datos
    features = list(X_validation.columns.values)
    X_validation_np = X_validation.to_numpy()
    y_validation_np = y_validation.to_numpy()
    n_samples = X_validation_np.shape[0]
    print("\n\nCaracteristicas seleccionadas automáticamente mediante el método de selección de funciones de correlación (CFS):")
    # Aplicación del algoritmo CFS para seleccionar las mejores características.
    features_selected_idx = CFS.cfs(X_validation_np, y_validation_np)
    selected_features = []
    for idx in features_selected_idx:
        # cfs devuelve una lista de 6 elementos, si se seleccionan menos de 6 características, se usa -1 para completar la lista
        if idx >= 0:
            selected_features.append(features[idx])
            print(' - %s' % features[idx])
    return selected_features

def auto_hyperparameters_selection(X_validation, y_validation):
    ''' Ajuste de los hiperparámetros del DecisionTree '''
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = np.arange(2, 6)
    min_samples_split = [2, 6, 13, 20, 27, 34, 40] # best 2 to 40
    min_samples_leaf = [1, 5, 9, 13, 17, 20] # best: 1 to 20
    class_weight = ['balanced', None]
    param_grid = {'criterion': criterion,
                  'splitter': splitter,
                  'min_samples_split': min_samples_split,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'class_weight': class_weight}
    print("\n\nHiperparámetros seleccionados automáticamente (DecisionTreeClassifier) mediante búsqueda exhaustiva de cuadrícula: ")
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(X_validation, y_validation)
    for idx in clf.best_params_:
        print(f' - {idx}: {clf.best_params_[idx]}')
    return clf.best_params_


def main():
    """ Clasificador HTTP """
    if not os.path.isdir('datasets'):
        sys.exit("======\nERROR: El clasificador debe ejecutarse desde la carpeta raíz del proyecto.\n======")

    '''
    ====== RECOPILACIÓN Y ANÁLISIS DE CONJUNTOS DE DATOS ========================================
    '''

    # descarga de conjuntos de datos
    f_normal_test_exists = os.path.exists(BASE_DIR + "datasets/raw/normalTrafficTest.txt")
    f_normal_train_exists = os.path.exists(BASE_DIR + "datasets/raw/normalTrafficTraining.txt")
    f_anom_test_exists = os.path.exists(BASE_DIR + "datasets/raw/anomalousTrafficTest.txt")
    if (f_normal_test_exists and f_normal_train_exists and f_anom_test_exists) == False:
        fetch_httpTraffic_data()

    # analizar el archivo del conjunto de datos: normalTrafficTest

    parse_datafile(BASE_DIR + "/datasets/row/http-traffic/normalTrafficTest.txt",
                   BASE_DIR + "/datasets/processed/normalTrafficTest.csv",
                   "normal")
    # analizar el archivo del conjunto de datos: normalTrafficTraining
    parse_datafile(BASE_DIR + "/datasets/row/http-traffic/normalTrafficTraining.txt",
                   BASE_DIR + "/datasets/processed/normalTrafficTraining.csv",
                   "normal")
    # analizar el archivo del conjunto de datos: anómaloTrafficTest
    parse_datafile(BASE_DIR + "/datasets/row/http-traffic/anomalousTrafficTest.txt",
                   BASE_DIR + "/datasets/processed/anomalousTrafficTest.csv",
                   "anomalous")
    
    # combinar todos los archivos csv en uno solo
    csv_files = [BASE_DIR + "/datasets/processed/normalTrafficTest.csv",
                 BASE_DIR + "/datasets/processed/normalTrafficTraining.csv",
                 BASE_DIR + "/datasets/processed/anomalousTrafficTest.csv"]
    full_dataset_filename = BASE_DIR + "/datasets/processed/full_dataset.csv"
    df_norm_test = pd.read_csv(csv_files[0], sep='|')
    df_norm_train = pd.read_csv(csv_files[1], sep='|')
    df_anom_test = pd.read_csv(csv_files[2], sep='|')
    df_full_sataset = pd.concat(
        [df_norm_test, df_norm_train, df_anom_test], axis=0, join='inner', ignore_index=True)
    df_full_sataset.reset_index()
    df_full_sataset.to_csv(full_dataset_filename, index=False, sep='|')
    for csv in csv_files:
        os.remove(csv)

    '''
    ====== PREPROCESAMIENTO DE DATOS ==================================================
    '''
    df = pd.read_csv(full_dataset_filename, sep='|')
    df['feature-Query-Length'] = np.where(df['feature-Query']
                                          == 'notpresent', 0, df['feature-Query'].str.len())
    df['feature-Query-Longest-Value'] = df['feature-Query'].apply(
        get_longest_param_value)
    df['feature-URL-Path'] = df['feature-URL'].apply(path_from_URL)
    df['feature-URL-Path-Length'] = df['feature-URL-Path'].apply(len)
    df['feature-URL-Path-ResourceExtension'] = df['feature-URL-Path'].apply(
        get_resource_extension)
    df['feature-URL-Path-ResourceExtension-Length'] = df['feature-URL-Path'].apply(
        get_resource_extension_length)
    df['feature-URL-Parameters-Num'] = df['feature-Query'].apply(count_params)
    df['feature-URL-Parameters-Names'] = df['feature-Query'].apply(name_params)
    df['feature-URL-Parameters-Type'] = df['feature-Query'].apply(type_params)
    df['feature-URL-Param-Value-Count-Dot'] = df['feature-Query'].apply(
        count_dots)
    df['feature-URL-Param-Value-Count-Percentage'] = df['feature-Query'].apply(
        count_percentage)
    df['feature-URL-Param-Value-Count-SpecialChars'] = df['feature-Query'].apply(
        count_specialChars)
    df['feature-Query-token_avg'] = df['feature-Query'].apply(get_token_avg)
    df['feature-Query-token_count'] = df['feature-Query'].apply(
        get_token_count)
    df['feature-Query-token_largest'] = df['feature-Query'].apply(
        get_token_largest)
    # reordenar las columnas, estableciendo la columna 'label' como la última
    df = df[['feature-Method', 'feature-URL', 'feature-Host', 'feature-Cookie', 'feature-Content-Type',
             'feature-Content-Length', 'feature-Query', 'feature-Query-Length', 'feature-Query-Longest-Value',
             'feature-URL-Path', 'feature-URL-Path-Length', 'feature-URL-Path-ResourceExtension', 'feature-URL-Path-ResourceExtension-Length',
             'feature-URL-Parameters-Num', 'feature-URL-Parameters-Names', 'feature-URL-Parameters-Type', 'feature-URL-Param-Value-Count-Dot',
             'feature-URL-Param-Value-Count-Percentage', 'feature-URL-Param-Value-Count-SpecialChars',
             'feature-Query-token_avg', 'feature-Query-token_count', 'feature-Query-token_largest', 'Label']]
    # elimine algunas columnas porque su variación es lenta o está sesgada (otras características más valiosas ya se han extraído de ellas)
    df = df.drop(columns=['feature-Cookie', 'feature-URL'])
    df.to_csv(full_dataset_filename, sep='|', index=False)
    feature_columns = df.loc[:, df.columns != 'Label'].columns
    target_colum = df['Label']
    df_factorized = df # Copie el marco de datos para mantener una copia del df sin factorizar.
    stacked = df_factorized[feature_columns].stack()
    df_factorized = pd.Series(stacked.factorize()[0], index=stacked.index, dtype='str').unstack()
    df_factorized['Label'] = target_colum
    os.makedirs(BASE_DIR + '/datasets/factorized')
    df_factorized.to_csv(BASE_DIR + '/datasets/factorized/full_dataset.csv', sep='|', index=False)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(
        df_factorized, 0.8, 0.1, 0.1, '/factorized/')
    X_train_unfactorized, y_train_unfactorized, X_test_unfactorized, y_test_unfactorized, X_validation_unfactorized, y_validation_unfactorized = train_test_validation_split(
    df, 0.8, 0.1, 0.1, '/processed/', False)
    
    '''
    ====== SELECCIÓN DE CARACTERÍSTICAS ==================================================
    '''
    # Selección AUTOMÁTICA de funciones con el método de selección de funciones basado en correlación (CFS)
    auto_selected_features = auto_feature_selection(X_validation, y_validation)
    
    manual_selected_features = ['feature-Query-Length',
                                'feature-Query-Longest-Value',
                                'feature-URL-Parameters-Num',
                                'feature-URL-Path-Length',
                                'feature-URL-Param-Value-Count-SpecialChars']

    '''
    ====== SINTONIZACIÓN AUTOMÁTICA DE HIPERPARÁMETROS ====================================
    '''
    best_hyperparameters = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
    

    '''
    ====== ENTRENAMIENTO DEL MODELO ======================================
    '''
    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    # Entranemiento del clasificador del arbol de decisiones
    classifier = classifier.fit(X_train[manual_selected_features], y_train)
    
    
    
    '''
    ====== PREPROCESAMIENTO DE DATOS - prueba.csv ==================================================
    '''
    prueba_filename = BASE_DIR + "/prueba.csv"
    df = pd.read_csv(prueba_filename, sep='|')
    predictor = classifier.predict(df[manual_selected_features])
    
    print(f"Se clasifico que la solicitud http es: {predictor}")


    return 0


if __name__ == "__main__":
        main()