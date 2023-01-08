from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

def index(request):
    template = loader.get_template('templates/home.html')
    context = {
        'latest_question_list': 0,
    }
    return HttpResponse(template.render(context, request))


def traincnn(request):
    template = loader.get_template('templates/traincnn.html')
    context={}

    return HttpResponse(template.render(context, request))


def trainingcnn(request):
    template = loader.get_template('templates/traincnn.html')
    context={}
    request.GET['epochs']
    epochs=int(request.GET['epochs'])
    lr=request.GET['lr']
    steps=request.GET['step']
    opt=request.GET['opt']

    print(epochs)

    return HttpResponse(template.render(context, request))
def trainingauc(request):
    template = loader.get_template('templates/traincnn.html')
    context={}
    request.GET['epochs']
    epochs=int(request.GET['epochs'])
    lr=request.GET['lr']
    steps=request.GET['step']
    opt=request.GET['opt']

    print(epochs)

    return HttpResponse(template.render(context, request))

def trainauc(request):
    template = loader.get_template('templates/home.html')
    context = {
        'latest_question_list': 0,
    }
    return HttpResponse(template.render(context, request))

def predcnn(request):
    template = loader.get_template('templates/home.html')
    context = {
        'latest_question_list': 0,
    }
    return HttpResponse(template.render(context, request))

def predauc(request):
    template = loader.get_template('templates/home.html')
    context = {
        'latest_question_list': 0,
    }
    return HttpResponse(template.render(context, request))