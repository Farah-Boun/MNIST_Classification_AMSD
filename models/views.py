from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from main import train_cnn,train_autoencoder,train_classifier
from predict import test, predict
import mock
from keras.datasets import mnist


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


def index(request):
    template = loader.get_template('templates/home.html')
    context = {}
    return HttpResponse(template.render(context, request))


def traincnn(request):
    template = loader.get_template('templates/traincnn.html')
    context={}

    return HttpResponse(template.render(context, request))


def trainingcnn(request):
    
    context={}
    request.GET['epochs']
    epochs=int(request.GET['epochs'])
    lr=float(request.GET['lr'])
    steps=int(request.GET['step'])
    opt=int(request.GET['opt'])
    model=train_cnn(opt,epochs,lr,steps)
    model.save("CNNtrained.h5")


    args = mock.Mock()
    args.type=2
    args.model="CNNtrained.h5"
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'nc':len(cm),
        'type':0,
        'model':"classifier_CNN.h5",
        'i':0
    }
    template = loader.get_template('templates/test_predict.html')
    return HttpResponse(template.render(context, request))

def trainauc(request):
    template = loader.get_template('templates/trainauc.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def trainingauc(request):
    context={}
    epochs=int(request.GET['epochs'])
    lr=float(request.GET['lr'])
    steps=int(request.GET['step'])
    opt=int(request.GET['opt'])

    aucepochs=int(request.GET['aucepochs'])
    auclr=float(request.GET['auclr'])
    aucsteps=int(request.GET['aucstep'])
    aucopt=int(request.GET['aucopt'])
    auto=train_autoencoder(opt,epochs,lr,steps)
    model=train_classifier(auto,opt,epochs,lr,steps)
    model.save("AUCtrained.h5")
    
    
    args = mock.Mock()
    args.type=2
    args.model="AUCtrained.h5"
    
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'nc':len(cm),
        'type':0,
        'model':"classifier_au.h5",
        'image':False,
        'id':0,
        'pred':-1
    }
    template = loader.get_template('templates/test_predict.html')
    

    return HttpResponse(template.render(context, request))



def predcnn(request):    
    args = mock.Mock()
    args.type=0
    
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'nc':len(cm),
        'type':0,
        'model':"classifier_CNN.h5",
        'image':False,
        'id':0,
        'pred':-1
    }
    template = loader.get_template('templates/test_predict.html')

    return HttpResponse(template.render(context, request))


def predauc(request):    
    args = mock.Mock()
    args.type=1
    
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'nc':len(cm),
        'type':0,
        'model':"classifier_au.h5",
        'image':False,
        'id':0,
        'pred':-1
    }
    template = loader.get_template('templates/test_predict.html')

    return HttpResponse(template.render(context, request))



def predauctrain(request):    
    args = mock.Mock()
    args.type=2
    args.model="AUCtrained.h5"
    
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'nc':len(cm),
        'type':0,
        'model':"classifier_au.h5",
        'image':False,
        'id':0,
        'pred':-1
    }
    template = loader.get_template('templates/test_predict.html')

    return HttpResponse(template.render(context, request))

def predcnntrain(request):    
    args = mock.Mock()
    args.type=2
    args.model="CNNtrained.h5"
    
    cm,acc=test(args)
    context = {
        'cm':cm,
        'acc':acc,
        'type':0,
        'model':"classifier_CNN.h5",
        'image':False,
        'id':0,
        'pred':-1
    }
    template = loader.get_template('templates/test_predict.html')

    return HttpResponse(template.render(context, request))

def predictimage(request):
    
    id=int(request.GET['imid'])
    model=(request.GET['model'])
    args = mock.Mock()
    args.type=int(request.GET['type'])+2
    args.model=model
    cm,acc=test(args)
    
    args.input=X_test[id]
    print("please")
    print(args.input)
    print("this")
    args.array=False
    prediction=predict(args)
    
    context = {
        'cm':cm,
        'acc':acc,
        'type':args.type,
        'model':args.model,
        'image':True,
        'id':id,
        'pred':prediction
    }
    template = loader.get_template('templates/test_predict.html')
    return HttpResponse(template.render(context, request))
