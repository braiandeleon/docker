
# Zahoree Challenge

#### Candidato: **Braian de Leon**, Fecha: **8/5/2023**

**Puntos de salida**

> http://127.0.0.1:5000/turnover

> http://127.0.0.1:5000/turnover?employeenumber=601

**Importante**: Se incluye un archivo de pruebas en la carpeta tests. Basta con correrlo para determinar la funcionalidad del servidor una vez montado en la maquina local con el puerto 5000 expuesto sin redireccionamiento. El siguiente comando ejecuta las pruebas:

> docker exec -it *nombre_delcontenedor* python tests/test_app.py 

La primera prueba es de conexion, la segunda sobre manejar valores no guardados en el modelo y el tercero de prueba de un nuevo trabajador a traves de un metodo POST. 

El programa carga en cada inicializacion el archivo que contiene los datos de los empleados. 

Luego, calcula en base a estos la probabilidad de permanencia y los guarda en un archivo .csv externo diferente del original con dos columnas (EmployeeNumber y TurnoverRate).

Se inicializa un endpoint en /turnover que acepta Requests GET y POST: 

Para el metodo GET el programa recoge el parametro employeenumber y devuelve la probabilidad de permanencia en porcentaje.

Para el metodo POST este acepta un arreglo en formato JSON con las llaves necesarias para que el modelo pueda calcular en base a estos datos la probabilidad de permanencia de un empleado que aun no esta contenido en el archivo de datos. 

Para que los socios de recursos humanos puedan comprender mejor esta informacion se presenta en termino de porcentaje de permanencia actual relativo a el tiempo maximo de permanencia (tomado como infinito por referencia). 


 