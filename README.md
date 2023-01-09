# MNIST_Classification_AMSD
Une Webapp Django qui permet d'entrainer, de tester et de faire des prédictions sur le Dataset de chiffres manuscrits MNIST à l'aide de deux modèles. Le premier modèle étant un CNN 2D et le deuxième un auto-encodeur.

## Lancer l'application sans docker
* installer les packages nécessaires

```
pip install -r requirements.txt
```

* Lancer l'application 
```
python .\manage.py runserver
```
Ouvrir `localhost:8000` sur votre navigateur web pour ouvrir l'application.

## Lancer l'application avec Docker

* Builder une image
```
docker build -t nom_conteneur .
```
* Ou bien utiliser une image depuis Dockerhub
```
docker pull farahboun/mnist_class
```

* Lancer l'application
```
docker run -it -p 8000:8000 -v /path/to/MNIST_Classification_AMSD:/code nom_conteneur
```
