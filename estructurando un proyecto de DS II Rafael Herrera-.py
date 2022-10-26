#!/usr/bin/env python
# coding: utf-8

# 
# 
# # **Encuesta de satisfacción de pasajeros aéreos**
# ## *Abstract*
# 
#   Una empresa que trabaja con clientes necesita indefectiblemente, para crecer, generar que los clientes tengan una mayor preferencia hacia su aerolinea. Por ello, es importante conocer al pasajero que elige volar con una cierta aerolinea. Cuando intentamos conocer el pensamiento del pasajero lo que vamos a realizar es una encuesta que intente reflejar los puntos más importantes (datos e información) de la relación que existe entre el cliente y la empresa. Para saber cual es el pensamiento del cliente también es necesario conocer ciertos aspectos clasicos de su persona (como ser género, edad), asi como también el proposito de su viaje o que tipo de asiento prefiere para viajar.
# 
#   En los últimos años se ha puesto foco en la perspectiva de género, lo que permite intentar mejorar la vida de las personas en general. A su vez, "ayuda a comprender más profundamente tanto la vida de las mujeres como la de los hombres y las relaciones que se dan entre ambos. Este enfoque cuestiona los estereotipos con que somos educados y abre la posibilidad de elaborar nuevos contenidos de socialización y relación entre los seres humanos." [(cit.)](https://www.gob.mx/conavim/articulos/que-es-la-perspectiva-de-genero-y-por-que-es-necesario-implementarla) La idea de darle una mirada de género a la información es intentar que los servicios sean un poco mejor tanto para los hombres como para las mujeres y lograr el objetivo de llegar a la igualdad entre el hombre y la mujer.
# 
#   "Conocer a los clientes nos permite desarrollar productos y servicios adecuadamente. Los ciclos de consumo actuales son cada vez más cortos. El cliente ya no tiene tiempo para buscar. Les gusta que les ofrezcan los productos y servicios que les convienen y que se sientan únicos y privilegiados."
# 
#   Este trabajo esta centrado en una encuesta realizada a los pasajeros de la linea aérea y muestra su nivel de conformidad con respecto al servicio prestado por ella. 
#   
#   El dataset fue extraido desde la página [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv) contiene mas de 100000 encuestas realizadas a pasajeros para medir el nivel de satisfacción de los mismos, a los servicios prestados por las lineas aereas tanto abordo como fuera del avión.
# 
# 
#  ## *Objetivo*
#  El objetivo del presente trabajo es lograr entender los problemas que pueden tener los pasajeros de nuestra linea aérea con el fin de poder mejorar nuestro servicio.
# 
# ##*Contexto comercial*
# 
#   Es importante recalcar que el mercado aeronautico en los Estados Unidos mueve miles de millones de dólares al año, siendo uno de los mercados más competitivos del mundo. Por lo que se hace necesario, para poder triunfar, estar a la vanguardia en el sector de entrega de servicios de calidad. Por ello, es necesario contar con un paquete de servicios innovadores y que resuelvan los problemas de los usuarios.
# 
#   Se ha hecho una recopilación de las encuestas realizadas a los pasajeros con el fin de poder conocer la opinión de los mismos con respecto al servicio que se brinda. 
#  
# ## *Problema comercial*
# 
#  Teniendo en cuenta una perspectiva de género se desea conocer la situación general de los pasajeros encuestados. Para ello, se consulta las siguientes cuestiones:
# 
#  1) ¿Cual es el proposito de viaje de nuestros pasajeros? 
#  
#  2) ¿Cuál es la proporción de recuento de pasajeros por edad? ¿Cual es el sexo que posee mayor número de pasajeros?
# 
#  3) ¿Existe algún tipo de preferencia en cuanto a la selección de los asientos, según el proposito del viaje?
# 
#  4)¿cual es el nivel de satisfacción de nuestros clientes teniendo en cuenta su opinión con respecto a nuestros servicios?
# 
# 
# 

# #Imports y lecturas iniciales

# In[1]:


#DATOS
import numpy as np
import pandas as pd
#Visual
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Archivo de datos
aerolinea=pd.read_csv("aerolinea.csv")


# In[2]:


aerolinea.head()


# In[3]:


aerolinea.shape


# #**Exploración de los datos**

# In[4]:


aerolinea.describe()


# In[5]:


aerolinea.info()


# In[6]:


#La primer y segunda columna no tiene sentido tenerla por lo que la borro
aerolinea=aerolinea.drop(columns=['Unnamed: 0','id'])
aerolinea.columns


# In[7]:


#Al estar las columnas en ingles las voy a renombrar
aerolinea.columns=['genero','tipo_cliente','edad','tipo_viaje','clase','distancia_vuelo','wifi_abordo','horario','facilita_compra','puerta_embarque','comida_bebida','checkin_online','comodidad_asiento','entretenimiento_vuelo','servicio_abordo','espacio_piernas','equipaje','servicio_checkin','inflight_service','limpieza','retraso_salida','retraso_llegada','satisfaccion']


# In[8]:


#Reviso en busqueda de nulos
aerolinea.isnull().sum()


# In[9]:


#Debido a que la columna 'retraso_llegada' tiene datos nulos debo revisarla para ver si es necesario realizarle algun cambio
aerolinea['retraso_llegada'].unique()


# In[10]:


#Con el fin de saber si hay algún valor que haya que limpiar reviso los valores de la columna 'Age'
aerolinea['edad'].unique()


# In[11]:


aerolinea['tipo_cliente'].value_counts()


# In[12]:


aerolinea['tipo_viaje'].value_counts()


# In[13]:


aerolinea['clase'].value_counts()


# In[14]:


aerolinea['genero'].value_counts()


# In[15]:


aerolinea['satisfaccion'].unique()


# #**Limpieza de los datos**

# In[16]:


#cambio el tipo de viaje a castellano
dict_tipo={'Business travel':'Viaje de negocios','Personal Travel':'vacaciones'}
aerolinea['tipo_viaje'] = aerolinea.tipo_viaje.replace(dict_tipo)
aerolinea['tipo_viaje'].value_counts()


# In[17]:


#cambio el nivel de satisfaccion a castellano
dict_satisfaccion={'neutral or dissatisfied':'neutral o no satisfecho','satisfied':'satisfecho'}
aerolinea['satisfaccion'] = aerolinea.satisfaccion.replace(dict_satisfaccion)
aerolinea['satisfaccion'].value_counts()


# In[18]:


#cambio el genero a castellano
dict_gen={'Male':'Masculino','Female':'Femenino'}
aerolinea['genero'] = aerolinea.genero.replace(dict_gen)


# In[19]:


#Los elementos nan deben ser limpiados por lo que realizo la limpieza y reviso si despues de ella no quedan mas por realizar, los que son nan los pondre como 0
def limpia_delai(x):
    if np.isnan(x):
      return 0
    return float(x)

aerolinea.retraso_llegada= aerolinea.retraso_llegada.apply(limpia_delai)


# # **Insights**

# ##Analisis de la distribución según el género del pasajero
# 
# Se busca saber la distribución general según el genero pero también se hará una evaluación de la preferencia de los tipos de asiento.

# In[20]:


#Debemos establecer el criterio de que se contaran los pasajeros por género
distribucion_genero_general = (aerolinea.genero.value_counts()
                                    .to_frame('CANTIDAD').reset_index().rename(columns={'index':'genero'}))
#Criterio por clase de asiento
distribucion_genero_asiento= (aerolinea.groupby(['clase','genero']).genero.count()
                               .to_frame('CANTIDAD').reset_index().rename(columns={'index':'genero'}))


# ### Gráfico correspondiente a la distribución a nivel general

# In[21]:


figura_genero_general=px.pie(distribucion_genero_general, values='CANTIDAD', names='genero', title='Distribución de pasajeros por genero',height=400,width=400)
figura_genero_general.show()


# ### Gráfico correspondiente a la distribución a nivel de clase de asientos

# In[22]:


fig_distribucion_genero_asientos = px.bar(distribucion_genero_asiento, y='clase', x='CANTIDAD', 
                                          color='genero', barmode='group',
                                            height=850,width=1500, text_auto=True,
                                          title="Distribución de los pasajeros con respecto a su genero en las clases de asientos",
                labels={'CANTIDAD':'Cantidad de pasajeros','clase':'clase de asiento','genero':'Género'})

fig_distribucion_genero_asientos.show()


# ### Análisis de los gráficos de distribución general en razon del género
# 
# De los gráficos surge que hubieron más pasajeros mujeres que hombres, tendencia que también se ve reflejada en la eleccion de los tipos de clase de asientos.

# ##Análisis de los tipos de cliente según el género

# In[23]:


import warnings
warnings.filterwarnings("ignore")


# In[24]:


fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

for i, gen_pas, genero in zip([0,1,2], [['Masculino','Femenino'], ['Masculino'], ['Femenino']], ['General', 'Hombres', 'Mujeres']):
    
    # preparacion de la data
    df_gen_pas = aerolinea[ (aerolinea.genero.isin(gen_pas))]
    
    # grafico
    sns.countplot(df_gen_pas.tipo_cliente, ax=axs[i])
    axs[i].title.set_text('Distribución de los pasajeros '+genero)


# ### Análisis de los gráficos
# 
# En los gráficos se muestra una marcada presencia de los pasajeros leales, que serían aquellos que se encuentran dentro de los programas de fidelidad de las aerolineas.

# ##Retraso de vuelos
# 
# Se necesita saber la distribución de los retrasos y el tiempo de retraso que se tuvo teniendo en cuenta la clase de asiento elegida

# In[25]:


#Criterio por clase de asiento
distribucion_viaje_asiento= (aerolinea.groupby(['clase','tipo_viaje']).tipo_viaje.count()
                               .to_frame('CANTIDAD').reset_index().rename(columns={'index':'tipo_viaje'}))
distribucion_viaje_asiento


# In[26]:


asientovuelos =distribucion_viaje_asiento.pivot("tipo_viaje", "clase", 'CANTIDAD')
f, ax = plt.subplots(figsize=(10, 10))
sns.set_theme(style="ticks", font_scale=1)
mapacalorcodigo=sns.heatmap(asientovuelos, annot=True, fmt=".0f", linewidths=.2, ax=ax)
mapacalorcodigo.set(xlabel="Clase de asiento",ylabel="Tipo de viaje",title="Clase de asiento seleccionado según tipo de viaje (MAPA DE CALOR)")
plt.show()


# ### Análisis del gráfico
# 
# Marcada tendencia de la selección de asientos de clase Business en los viajes de negocios mientras que en los viajes de vacaciones se ve una marcada selección de los asientos de tipo Eco (clase económica).

# ##Se desea conocer la distribución de los pasajeros según su edad y genero

# In[27]:


#Analisis de edad generalizado
f, ax = plt.subplots(figsize=(10, 10))
histogramaedad=sns.histplot(data=aerolinea, x="edad")
histogramaedad.set(xlabel="Edades pasajeros",ylabel="Cantidad de pasajeros",title="Cantidad de pasajeros según la edad")
plt.show()


# In[28]:


fig_edad_pas_gral, axes = plt.subplots(1, 2,figsize=(20,15))
sns.set_theme(style="ticks", font_scale=1.5)
sns.boxplot(x="genero", y="edad",data=aerolinea,ax=axes[0])
sns.violinplot(x="genero", y="edad",data=aerolinea, 
               split=True, inner="quart", linewidth=1,ax=axes[1])
fig_edad_pas_gral.suptitle("Análisis de la edad de los pasajeros")
axes[0].set_ylabel("Edad de los pasajeros")
axes[1].set_ylabel("Edad de los pasajeros")
axes[0].set_xlabel("Género de los pasajeros")
axes[1].set_xlabel("Género de los pasajeros")
plt.show()


# ###Análisis de los gráficos
# 
# De los gráficos se desprende que este dataset es uno muy balanceado. Para realizar el analisis por edades de los pasajeros utilizamos dos graficos para representar los resultados, al lado izquierdo el de caja y al lado derecho el de violin. El grafico de caja nos indica que para el sexo femenino la edad en cuartil inferior estan igualados por debajo de los 10 años por lo tanto la mediana también se encuentra igualada en los 40 años y en caso de analizar los cuartiles superiores se puede ver el mismo tipo de situacion donde los valores estan igualados en una edad mayor a 50 años.
# 
# El grafico de violin nos sirve para complementar mucho mas la informacion y entender en donde se encuentra el mayor grupo de pasajeros respecto a su edad, para los dos generos vemos una homogeneidad respecto a la edad, ya que por los 25 años y tambien entre los 40 y 60 años años se encuentra la mayor parte de la muestra. 
# 
# En el histograma se puede ver la tendencia pero ya generalizada sin tener en cuenta el género del pasajero, aqui vemos que las tendencias son como se venia charlando y el gráfico nos muestra que se dan picos de cantidad de pasajeros en los 38 y 25 años aproximadamente. Teniendo una cantidad minima de pasajeros por encima de los 70 años.
# 

# In[29]:


fig_edad_pas_tipo_viaje, axes = plt.subplots(1, 2,figsize=(20,15))
sns.set_theme(style="ticks", font_scale=1.5)
sns.boxplot(x="tipo_viaje", y="edad",data=aerolinea,ax=axes[0])
sns.violinplot(x="tipo_viaje", y="edad",data=aerolinea, 
               split=True, inner="quart", linewidth=1,ax=axes[1])
fig_edad_pas_tipo_viaje.suptitle("Análisis de la edad de los pasajeros según clase de viaje")
axes[0].set_ylabel("Edad de los pasajeros")
axes[1].set_ylabel("Edad de los pasajeros")
axes[0].set_xlabel("Tipo de viaje")
axes[1].set_xlabel("Tipo de viaje")
plt.show()


# ##Nivel de satisfacción

# In[30]:


fig, axs = plt.subplots(3, 3, figsize=(50, 50), sharey=True)

for variable,titulo,ax in zip(['wifi_abordo','facilita_compra','comida_bebida','comodidad_asiento','entretenimiento_vuelo','servicio_abordo','espacio_piernas','inflight_service','limpieza'], 
                              ['servicio de wifi abordo','si se facilita la compra de pasajes','la comida a bordo','la comodidad del asiento','entretenimiento a bordo','servicio abordo','el espacio para piernas','inflight service','la limpieza'],axs.flat):
  # grafico
  sns.countplot(x=aerolinea[variable],hue=aerolinea['satisfaccion'], ax=ax)
  ax.title.set_text('Satisfacción del cliente según su opinión sobre '+titulo)
  ax.set(xlabel='Puntaje asignado por los pasajeros',ylabel='Cantidad de pasajeros')


# ## Ánalisis de los gráficos
# 
# De los gráficos se desprende que hay una tendencia hacia la disconformidad con los servicios prestados, siendo una mayor cantidad de pasajeros los que optaron por la opción neutral o no satisfecho. Igualmente se puede ver una mayor concentración de las personas que han elegido la opción satisfecho, dentro de los puntajes mas altos de los diferentes servicios prestados.
# 

# ## Conclusiones generales de lo analizado
# 
# El dataset se encuentra muy equilibrado, dentro de el me encuentro que la tendencia de pasajeros hombres como la de pasajeros mujeres es casi la misma destacandose que hay mas mujeres que hombres (pero la diferencia es de tan solo 1,4%). De los encuestados hay una proporcion muy alta de los pasajeros que utilizan la clase bussiness para viajar (siguiendo la tendencia general, las mujeres son mas que los hombres dentro de las 3 clases de asientos), seguido por la clase economica, mientras que el economy plus es la de menor preferencia. Las preferencias de asientos business principalmente estan dentro de los que realizan viajes de negocios, mientras que la clase economica en general es elegida por los que realizan viajes de vacaciones aunque si se miran los numeros no es muy alejado de la cantidad de elecciones para el viaje de negocios. En cuanto a la edad de los pasajeros la mediana se encuentra en los 40 años tanto para hombres como para mujeres.
# 
# Por último, del análisis realizado se desprende que hay una mayor proporción de pasajeros disconformes con los servicios prestado por parte de la aerolinea siendo los peores servicios rankeados el wifi, la forma de compra de los pasajes y el espacio para las piernas.

# # Recomendaciones en base a lo observado
# 
# En base a la información extraida del set de datos se recomienda un profundo upgrade en los servicios prestados por la compañia, comenzando por las areas que más han sido criticadas como ser el wifi, el sistema de compras de pasajes y tambien el espacio de piernas en los asientos. También al destacarse la cantidad de pasajeros que utilizan el servicio para viajes de negocios, se recomienda realizar campañas y mejoras de servicio dirigidas principalmente a este publico ya que son los que utilizan los servicios mas caros de la empresa (asientos business) y sumado a ello, el principal caudal de pasajeros, esta dentro de la orbita de los que realizan viajes de negocios.
