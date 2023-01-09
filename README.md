# MNIST_Classification_AMSD
Une web app Django qui permet d'entrainer, de tester et de faire des prédictions sur le dataset de chiffres manuscrits MNIST à l'aide de deux modèles. Le premier modèle étant un CNN 2D et le deuxieme un autoencodeur.

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
docker build -t nom_contenaire .
```
* Lancer l'application
```
docker run -it -p 8000:8000 -v /path/to/MNIST_Classification_AMSD:/code nom_contenaire
```
