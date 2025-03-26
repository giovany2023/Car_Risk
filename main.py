import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn # type: ignore

#Configurar la página
st.set_page_config(page_title='Predicción riesgo de seguros', page_icon ='🚗', layout='centered', initial_sidebar_state='auto')
st.image('logo.jpeg')
#st.title('Predicción riesgo de seguros')

#punto de entrada
def main():
    #Cargar el modelo
    filename = 'modelo-clas-tree-RL-RF.pkl'
    modelTree, model_RL, model_knn, model_SVM,model_RF, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

    # Titulo principal
   # st.title('Predicción de riesgo de seguros')
    st.sidebar.title('Ingresar datos del cliente')

    # Entradas del usuario en el sidebar
    def user_input_features():
        #edad como valor entero entre 0 y 50 años
        edad = st.sidebar.slider('Edad', min_value=18, max_value=50, value=25, step=1)#step = 1 para que se mueva de 1 en 1
        #Entrada variables Cartype
        options = ['combi','minivan','sport','family']
        cartype = st.sidebar.selectbox('Tipo de carro', options)
        #Crear un diccionario con los valores de entrada
        data = {'age': edad,
                'cartype': cartype}
        #Crear un DataFrame a partir del diccionario
        features = pd.DataFrame(data, index=[0])
       # st.subheader('Datos del cliente')
       # st.write(features)

        #Preparar los datos de entrada
        data_preparada = features.copy()
       #st.write(data_preparada)

       #Crear las variables dummies de la variable cartype
        data_preparada = pd.get_dummies(data_preparada,  columns=['cartype'], drop_first=False)
        #st.subheader('Datos del cliente con dummies')
        #st.write(data_preparada)
        # Realizar reindexación para añadir columnas faltantes
        data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
       # st.subheader('Datos del cliente con reindex')
        #st.write(data_preparada)
        return data_preparada
    # Llamada a la función user_input_features
    df= user_input_features()
    
    #Selector de modelos
    options = ['DT','Rl','KNN','SVM','RF']
    model = st.sidebar.selectbox('Seleccional Modelo', options)
    #st.caption('Modelo seleccionado: ' + model)
    #st.write(model)

    #Boton de predicción
    if st.button('Realizar predicción'):
        if model == 'DT':
            y_fut = modelTree.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo de seguro es {}'.format(resultado[0]))
        elif model == 'RF':
            y_fut = model_RF.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo de seguro es {}'.format(resultado[0]))
        elif model == 'Rl':
            y_fut = model_RL.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo de seguro es {}'.format(resultado[0]))
        elif model == 'KNN':
            y_fut = model_knn.predict(df)
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo de seguro es {}'.format(resultado[0]))
        elif model == 'SVM':
           # df['age'] = min_max_scaler.transform(df[['age']])
            #st.write(df)
            y_fut = model_SVM.predict(df) 
            resultado = labelencoder.inverse_transform(y_fut)
            st.success('El riesgo de seguro es {}'.format(resultado[0]))
    


#Crear formulario

if __name__ == '__main__':
    main()


