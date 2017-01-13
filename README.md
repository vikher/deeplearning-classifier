#Construcción de un clasificador utilizando Caffe y una red neuronal convolucional

Implementaremos un clasificador de perros y gatos usando una red neuronal convolucional. Utilizaremos un conjunto de datos y para implementar la red neuronal convolucional, usaremos un framework de aprendizaje profundo llamado Caffe y código Python

##Información acerca de Caffe

Caffe es un framework de aprendizaje profundo desarrollado por el Centro de Visión y Aprendizaje de Berkeley (BVLC). Está escrito en C ++ y tiene enlaces de Python y Matlab.

Hay 4 pasos en el entrenamiento de una CNN usando Caffe:

Paso 1 - Preparación de datos : En este paso, limpiaremos las imágenes y las almacenaremos en un formato que puede ser utilizado por Caffe. Escribiremos un script de Python que manejará tanto el preprocesamiento como el almacenamiento de imágenes.

Paso 2 - Definición del modelo: En este paso, elegimos una arquitectura CNN y definimos sus parámetros en un archivo de configuración con extensión .prototxt.

Paso 3 - Definición del Solver: El solver es responsable de la optimización del modelo. Definimos los parámetros del solver en un archivo de configuración con la extensión .prototxt.

Paso 4 - Entrenamiento del modelo: Entrenamos el modelo ejecutando un comando Caffe desde la terminal. Después de entrenar el modelo, obtendremos el modelo entrenado en un archivo con extensión .caffemodel.

Después de la fase de entrenamiento, usaremos el modelo .caffemodel entrenado para hacer predicciones de nuevos datos jamas vistos porla red neuronal. Escribiremos un script de Python para esto.

###Obtener el conjunto de datos

Primero, necesitamos descargar 2 conjuntos de datos de perros y gatos de la pagina de kaggle, debes registrarte para poder descargar el conjunto de datos [aqui] (https://www.kaggle.com/c/dogs-vs-cats/data): train.zip y test.zip. El archivo train.zip contiene imágenes etiquetadas que usaremos para entrenar la red. El archivo test.zip contiene imágenes sin etiqueta que clasificaremos usando el modelo entrenado. 

###Configuración de la máquina

Para entrenar la red neuronal convolucional, necesitamos una máquina con una poderosa GPU.Si aun no has instalado el framework Caffe puedes ir a la pagina oficial para la guia de instalacion [aqui] (http://caffe.berkeleyvision.org/installation.html)

Yo utilicé una instancia de AWS EC2 de tipo p2.xlarge Esta instancia tiene 1 GPU NVIDIA K80 de alto rendimiento con 61GB de memoria RAM y 4 vCPUs. La máquina cuesta $ 0.900 / hora

Después de configurar una instancia AWS, nos conectamos a ella y clonamos el repositorio github que contiene el código Python y los archivos de configuración Caffe necesarios para el tutorial. Desde tu terminal, ejecuta el siguiente comando.

    
    git clone https://github.com/vikher/deeplearning-classifier.git
    
A continuación, creamos una carpeta de nombre *input* para almacenar las imágenes de entrenamiento y de prueba.

    
    cd deeplearning-classifier
    mkdir input
    
###Preparación de datos

Comenzamos copiando los archivos train.zip y test.zip (que descargamos en nuestra máquina local) a la carpeta de entrada en la instancia de AWS. Después de copiar los datos, descomprimimos los archivos ejecutando los siguientes comandos:

    unzip ~/tutorial/input/train.zip
    unzip ~/tutorial/input/test.zip
    rm ~/tutorial/input/*.zip

A continuación, ejecutamos create_lmdb.py
    
    cd ~/deeplearning-classifier/code
    python create_lmdb.py

El script create_lmdb.py hace lo siguiente:

 - Ejecuta la ecualización del histograma en todas las imágenes de entrenamiento. La ecualización del histograma es una técnica para ajustar el contraste de las imágenes.

 - Cambia el tamaño de todas las imágenes de entrenamiento a un formato 227x227.

 - Divide los datos de entrenamiento en 2 grupos: uno para entrenamiento (5/6 de imágenes) y otro para validación (1/6 de imágenes). El conjunto de entrenamiento se utiliza para entrenar el modelo, y el conjunto de validación se utiliza para calcular la precisión del modelo.

 - Almacena el entrenamiento y validación en 2 bases de datos LMDB. Train_lmdb para el entrenamiento del modelo y validation_lmbd para la evaluación del modelo.

###Generar media de la imagen de los datos de entrenamiento

Ejecutamos el siguiente comando para generar media de la imagen de los datos de entrenamiento. Substraeremos la imagen media de cada imagen de entrada para asegurar que cada píxel tenga media cero. Este es un paso común de preprocesamiento en el aprendizaje supervisado de máquinas.

    /home/ubuntu/caffe-master/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/deeplearning-classifier/input/inputtrain_lmdb 
    /home/ubuntu/deeplearning-classifier/input/mean.binaryproto

###Definición del modelo

Después de decidir sobre la arquitectura CNN, necesitamos definir sus parámetros en un archivo .prototxt train_val. Caffe viene con algunos [modelos](https://github.com/BVLC/caffe/tree/master/models) de CNN populares como Alexnet y GoogleNet.Usare el modelo [bvlc_reference_caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) que es una réplica de AlexNet con algunas modificaciones. Se debe tener el mismo archivo en deeplearning-classifier/caffe_models/caffe_model_1 .

Necesitamos hacer las modificaciones en el archivo original de protxtxt bvlc_reference_caffenet:
 - Cambie la ruta de datos de entrada y la imagen media: Líneas 13, 24, 40 y 51.
 - Cambie el número de salidas de 1000 a 2: Línea 373. El original bvlc_reference_caffenet fue diseñado para un problema de clasificación con 1000 clases.

Opcional: Podemos imprimir la arquitectura del modelo ejecutando el siguiente comando. La imagen de la arquitectura del modelo se almacenará en deeplearning-classifier/caffe_models/caffe_model_1 /caffe_model_1.png

###Definición del Solver 

El solver es responsable de la optimización del modelo. Definimos los parámetros del solver en un archivo .prototxt. Puede encontrar el solver en deeplearning-classifier/caffe_models/caffe_model_1/solver_1.prototxt.

Este solver calcula la exactitud del modelo usando el conjunto de validación cada 1000 iteraciones. El proceso de optimización se ejecutará para un máximo de 40000 iteraciones y tomará una foto del modelo entrenado cada 5000 iteraciones.

###Entrenamiento del Modelo

Después de definir el modelo y el solver, podemos comenzar a entrenar el modelo ejecutando el siguiente comando:

/home/ubuntu/caffe-master/build/tools/caffe train --solver home/ubuntu/deeplearning-classifier/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee /home/ubuntu/deeplearning-classifier/caffe_models/caffe_model_1/model_1_train.log

Durante el proceso de entrenamiento, necesitamos monitorear la pérdida y la precisión del modelo. Podemos detener el proceso en cualquier momento presionando Ctrl + c. Caffe salvara el modelo entrenado cada 5000 iteraciones y lo almacenará en la carpeta caffe_model_1.

El modelo salvado tiene extensión .caffemodel. Por ejemplo, El modelo de 10000 iteraciones se llamará: caffe_model_1_iter_10000.caffemodel.

###Predicción sobre nuevos datos

Ahora que tenemos un modelo entrenado, podemos usarlo para hacer predicciones sobre nuevos datos (imágenes de test1). El código de Python para hacer las predicciones es make_predictions_1.py y se almacena en deeplearning-classifier/code. El código necesita 4 archivos para ejecutarse:

 - Imágenes de prueba: Utilizaremos imágenes de test1.
 - Imagen media: La imagen media que hemos calculado
 - Archivo de modelo de arquitectura: Llamaremos a este archivo caffenet_deploy_1.prototxt. Se almacena en deeplearning-classifier / caffe_models/caffe_model_1. Está estructurado de forma similar a caffenet_train_val_1.prototxt, pero con algunas modificaciones. Necesitamos eliminar las capas de datos, añadir una capa de entrada y cambiar el último tipo de capa de SoftmaxWithLoss a Softmax.
 - Pesos del modelo entrenado: Este es el archivo que calculamos en la fase de entrenamiento. Usaremos caffe_model_1_iter_10000.caffemodel.

Para ejecutar el código Python, necesitamos ejecutar el siguiente comando.

    cd /home/ubuntu/deeplearning-classifier/code
    python make_predictions_1.py

Las predicciones se almacenarán en /deeplearning-classifier/caffe_models/caffe_model_1/submission_model_1.csv.

##Construyendo un clasificador usando el aprendizaje de transferencia

Las redes neuronales convolucionales requieren grandes conjuntos de datos y un montón de tiempo computacional para entrenar. Algunas redes pueden tardar hasta 2-3 semanas en varias GPU para entrenar. La transferencia de aprendizaje es una técnica muy útil que trata de abordar ambos problemas. En lugar de entrenar la red desde cero, el aprendizaje de transferencia utiliza un modelo entrenado en un conjunto de datos diferente y lo adapta al problema que estamos tratando de resolver.

Hay 2 estrategias para el aprendizaje de transferencia:

 - Utilizar el modelo entrenado como un extractor de características: En esta estrategia, eliminamos la última capa conectada del modelo entrenado, congelamos los pesos de las capas restantes y entrenamos un clasificador en la salida de las capas restantes.

 - Afinar el modelo entrenado: En esta estrategia, afinamos el modelo entrenado en el nuevo conjunto de datos continuando la retropropagación. Podemos ajustar la red completa o congelar algunas de sus capas

###Entrenamiento del Clasificador usando Aprendizaje de Transferencia

Caffe tiene un repositorio que es utilizado por los investigadores y profesionales para compartir sus modelos entrenados. Esta biblioteca se llama [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).

Utilizaremos el [bvlc_reference_caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) como punto de partida para construir nuestro clasificador utilizando el aprendizaje de transferencia. Este modelo fue entrenado en el conjunto de datos [ImageNet](http://www.image-net.org/) que contiene millones de imágenes en 1000 categorías.

Utilizaremos la estrategia de afinar para entrenar nuestro modelo.

Podemos descargar el modelo entrenado ejecutando el siguiente comando

    cd /home/ubuntu/caffe-master/models/bvlc_reference_caffenet
    wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

###Definición del modelo

Los archivos de configuración del modelo y del solver se almacenan en deeplearning-classifier/caffe_models/caffe_model_2. Necesitamos realizar el siguiente cambio en el archivo caffenet_train_val_2.prototxt.

 - Cambie la ruta de datos de entrada y la imagen media: Líneas 13, 24, 40 y 51.
 - Cambie el nombre de la última capa completamente conectada de fc8 a fc8_c. Líneas 360, 363, 387 y 397.
 - Cambie el número de salidas de 1000 a 2: Línea 373. El modelo original bvlc_reference_caffenet fue diseñado para un problema de clasificación con 1000 clases

###Definición del Solver 

Usaremos un solver similar al utilizado antes. Se encuentra en /deeplearning-classifier/caffe_models/caffe_model_2/solver_2.prototxt

###Entrenamiento del Modelo con Aprendizaje de Transferencia

Después de definir el modelo y el solver, podemos comenzar a entrenar el modelo ejecutando el siguiente comando. Tenga en cuenta que podemos pasar los pesos del modelo entrenado usando el argumento --weights.

/home/ubuntu/caffe-master/build/tools/caffe train --solver=/home/ubuntu/deeplearning-classifier/caffe_models/caffe_model_2/solver_2.prototxt --weights /home/ubuntu/deeplearning-classifier/caffe_models/caffe_model_2/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/deeplearning-classifier/caffe_models/caffe_model_2/model_2_train.log

###Predicción sobre nuevos datos

El código para hacer las predicciones está en deeplearning-classifier/code/make_predictions_2.py

