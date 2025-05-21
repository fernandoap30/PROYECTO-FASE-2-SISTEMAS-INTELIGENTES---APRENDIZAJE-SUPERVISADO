
# 1. Configuración e Importación de Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

# Para que los gráficos se muestren en Colab
%matplotlib inline

file_path = 'housing_price_dataset.csv'

try:
    df = pd.read_csv(file_path)
    print("¡Dataset cargado exitosamente!")
    print(df.head())
    print("\nInformación del Dataset:")
    df.info()
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no fue encontrado.")
    print("Asegúrate de que el archivo esté subido al entorno de Colab o que la ruta de Google Drive sea correcta.")
    exit() # Salir si no se encuentra el archivo

# 2. Preprocesamiento de Datos

min_price = df['Price'].min()
df['PriceCategory'] = pd.cut(df['Price'],
                                bins = [min_price - 1, 175000, 275000, df['Price'].max() + 1], # Ajusta estos rangos según la distribución de tus precios
                                labels=['Bajo', 'Medio', 'Alto'])

print("\nDistribución de las categorías de precio:")
print(df['PriceCategory'].value_counts())

# Definir características (X) y variables objetivo (y)
X = df.drop(['Price', 'PriceCategory'], axis=1) 
y_classification = df['PriceCategory']
y_regression = df['Price']

categorical_features = ['Neighborhood']
numerical_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']

# Crear un preprocesador usando ColumnTransformer
# Aplica OneHotEncoder a las características categóricas y StandardScaler a las numéricas.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Dividir los datos en conjuntos de entrenamiento y prueba 
X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(X, y_classification, y_regression, test_size=0.2, random_state=42, stratify=y_classification )

print(f"\nForma del conjunto de entrenamiento: {X_train.shape}, {y_clf_train.shape}, {y_reg_train.shape}")
print(f"Forma del conjunto de prueba: {X_test.shape}, {y_clf_test.shape}, {y_reg_test.shape}")

# 3. Implementación de modelos de aprendizaje (Algoritmos)

def run_logistic_regression():
    print("\n--- Ejecutando Regresión Logística ---")

    model_lr = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

    model_lr.fit(X_train, y_clf_train)
    y_pred_lr = model_lr.predict(X_test)

    print("\nReporte de Clasificación de Regresión Logística:")
    print(classification_report(y_clf_test, y_pred_lr))

    print("\nPrecisión (Accuracy) de Regresión Logística:", accuracy_score(y_clf_test, y_pred_lr))

    # Gráfico de Matriz de Confusión
    cm = confusion_matrix(y_clf_test, y_pred_lr, labels=['Bajo', 'Medio', 'Alto'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bajo', 'Medio', 'Alto'], yticklabels=['Bajo', 'Medio', 'Alto'])
    plt.title('Matriz de Confusión de Regresión Logística')
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.show()
    plt.close()

    # Gráfico de Coeficientes de Regresión Logística por Clase
    coefficients = model_lr.named_steps['classifier'].coef_
    classes = model_lr.named_steps['classifier'].classes_

    ohe_feature_names = model_lr.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names_out = numerical_features + list(ohe_feature_names)

    coef_df = pd.DataFrame(coefficients, columns=feature_names_out, index=classes)

    coef_df_T = coef_df.T
    coef_df_T.columns.name = 'Clase' 

    plt.figure(figsize=(12, 8))
   

    colors = {'Bajo': 'sandybrown', 'Medio': 'lightgreen', 'Alto': 'lightcoral'} 
    bar_height = 0.25 

    influential_features = coef_df_T.abs().max(axis=1).sort_values(ascending=False).index[:15] 

    plt.figure(figsize=(12, max(8, len(influential_features) * 0.5))) 
    y_pos = np.arange(len(influential_features))

    for i, cls in enumerate(classes):
        # Asegurarse de que el color se asigne correctamente si las clases no están en el orden esperado
        color = colors.get(cls, 'gray')
        plt.barh(y_pos + i * bar_height, coef_df_T.loc[influential_features, cls], height=bar_height, label=cls, color=color)

    plt.yticks(y_pos + bar_height, influential_features)
    plt.xlabel('Valor del Coeficiente (Estandarizado)')
    plt.ylabel('Característica')
    plt.title('Coeficientes de Regresión Logística por Clase')
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8) 
    plt.legend(title='Clase')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.show()
    plt.close('all')

def run_svm():
    print("\n--- Ejecutando Máquina de Soporte Vectorial (SVC) ---")

    model_svm = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', SVC(random_state=42))]) 

    model_svm.fit(X_train, y_clf_train)
    y_pred_svm = model_svm.predict(X_test)

    print("\nReporte de Clasificación de SVM:")
    print(classification_report(y_clf_test, y_pred_svm))

    print("\nPrecisión (Accuracy) de SVM:", accuracy_score(y_clf_test, y_pred_svm))

    cm = confusion_matrix(y_clf_test, y_pred_svm, labels=['Bajo', 'Medio', 'Alto'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Bajo', 'Medio', 'Alto'], yticklabels=['Bajo', 'Medio', 'Alto'])
    plt.title('Matriz de Confusión de SVM')
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.show()
    plt.close()

def run_decision_tree_regressor():
    print("\n--- Ejecutando Árbol de Decisión Regresor ---")

    model_dt_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', DecisionTreeRegressor(random_state=42))])

    model_dt_reg.fit(X_train, y_reg_train)
    y_pred_dt_reg = model_dt_reg.predict(X_test)

    print("\nEvaluación del Árbol de Decisión Regresor:")
    print(f"R-cuadrado (R-squared): {r2_score(y_reg_test, y_pred_dt_reg):.4f}")
    print(f"Error Absoluto Medio (MAE): {mean_absolute_error(y_reg_test, y_pred_dt_reg):.2f}")
    print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_reg_test, y_pred_dt_reg):.2f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {np.sqrt(mean_squared_error(y_reg_test, y_pred_dt_reg)):.2f}")

   
    ohe_feature_names = model_dt_reg.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names_out = numerical_features + list(ohe_feature_names)
    feature_importances = model_dt_reg.named_steps['regressor'].feature_importances_

    # Crear un DataFrame para las importancias de las características
    importance_df = pd.DataFrame({'Característica': feature_names_out, 'Importancia': feature_importances})
    importance_df = importance_df.sort_values(by='Importancia', ascending=False)

    print("\nImportancia de las Características del Árbol de Decisión:")
    print(importance_df.head(10)) # Mostrar las 10 características más importantes

    # Gráfico de Importancia de Características
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Importancia', y='Característica', data=importance_df.head(15), palette='viridis')
    plt.title('Importancia de las Características del Árbol de Decisión')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Gráfico de Precios Reales vs. Predichos
    plt.figure(figsize=(10, 7))
    plt.scatter(y_reg_test, y_pred_dt_reg, alpha=0.3)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.title('Árbol de Decisión Regresor: Precios Reales vs. Predichos')
    plt.xlabel('Precios Reales')
    plt.ylabel('Precios Predichos')
    plt.grid(True)
    plt.show()
    plt.close()

    # Gráfico de Residuos (para modelos de regresión)
    residuals = y_reg_test - y_pred_dt_reg
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True)
    plt.title('Árbol de Decisión Regresor: Distribución de Residuos')
    plt.xlabel('Residuos (Real - Predicho)')
    plt.ylabel('Frecuencia')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.scatter(y_pred_dt_reg, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Árbol de Decisión Regresor: Residuos vs. Valores Predichos')
    plt.xlabel('Precios Predichos')
    plt.ylabel('Residuos')
    plt.grid(True)
    plt.show()
    plt.close()

    # Visualizar el Árbol de Decisión 
    plt.figure(figsize=(20, 10))
    plot_tree(model_dt_reg.named_steps['regressor'],
              feature_names=feature_names_out,
              filled=True, rounded=True,
              fontsize=8,
              max_depth=3) # Limitar la profundidad para mejor legibilidad
    plt.title('Estructura del Árbol de Decisión Regresor (Profundidad Máxima 3)')
    plt.show()
    
def run_pca_visualization():
    print("\n--- Ejecutando Visualización PCA (Análisis de Componentes Principales) ---")

    # Aplicar el preprocesador a todo el conjunto de datos (X) para PCA
    X_transformed = preprocessor.fit_transform(X)

    # Crear el modelo PCA con 2 componentes
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_transformed)

    # Crear un DataFrame con las componentes principales y la categoría de precio
    pca_df = pd.DataFrame(data = components, columns = ['Componente Principal 1', 'Componente Principal 2'])
    pca_df['PriceCategory'] = y_classification.reset_index(drop=True) # Asegurar que los índices coincidan

    print("\Varianza explicada por cada componente principal:")
    print(f"Componente Principal 1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"Componente Principal 2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")


    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Componente Principal 1', y='Componente Principal 2', hue='PriceCategory', data=pca_df,
                    palette='viridis', s=50, alpha=0.6)
    plt.title('PCA 2D: Componentes Principales por Categoría de Precio')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}% de varianza explicada)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}% de varianza explicada)')
    plt.grid(True)
    plt.legend(title='Categoría de Precio')
    plt.show()
    plt.close('all')


# 4. Interfaz de Usuario (Menú)
def main_menu():
    while True:
        print("\n--- Algoritmos de Aprendizaje Supervisado para Precios de Viviendas ---")
        print("1. Ejecutar Regresión Logística (Clasificación de Precios)")
        print("2. Ejecutar Máquina de Soporte Vectorial (SVM - Clasificación de Precios)")
        print("3. Ejecutar Árbol de Decisión Regresor (Predicción de Precio Exacto)")
        print("4. Visualización PCA (Componentes Principales)") # Nueva opción en el menú
        print("5. Salir")

        choice = input("Ingresa tu elección (1-5): ")

        if choice == '1':
            run_logistic_regression()
        elif choice == '2':
            run_svm()
        elif choice == '3':
            run_decision_tree_regressor()
        elif choice == '4': # Opción para PCA
            run_pca_visualization()
        elif choice == '5':
            print("Saliendo del programa. ¡Hasta luego!")
            break
        else:
            print("Opción inválida. Por favor, ingresa un número entre 1 y 4.")

if __name__ == "__main__":
    main_menu()