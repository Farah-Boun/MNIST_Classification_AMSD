FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . code
WORKDIR /code
EXPOSE 8000
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]


#      docker run -it -p 8000:8000 mnist_class_wv





FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /code
EXPOSE 8000
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]
#      docker run -it -p 8000:8000 -v C:/Users/Mehdi/Documents/GitHub/MNIST_Classification_AMSD:/code mnist_class3







#FROM python:3.8
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
#COPY ./mnist code/mnist
#COPY ./models code/models
#WORKDIR /code
#EXPOSE 8000
#ENTRYPOINT ["python", "manage.py"]
#CMD ["runserver", "0.0.0.0:8000"]
#      docker run -it -p 8000:8000 -v C:/Users/Mehdi/Documents/GitHub/MNIST_Classification_AMSD:/code mnist_class2


#FROM python:3.8
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
#COPY . code
#WORKDIR /code
#EXPOSE 8000
#ENTRYPOINT ["python", "manage.py"]
#CMD ["runserver", "0.0.0.0:8000"]
#      docker run -it -p 8000:8000 -v C:/Users/Mehdi/Documents/GitHub/saved_models:/code/saved_models mnist_class
