from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
from rest_framework import status

from drf_yasg.utils import swagger_auto_schema
import joblib

from .serializers import *
from .models import *


#---------------------------------------------------------
# Logistic Regression
#---------------------------------------------------------
class LogisticRegressionView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою Логістичної регресії

        Predicts your survival results on Logistic regression model
        '''
        model = joblib.load('ml_models/LogisticRegressionCV.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



#---------------------------------------------------------
# Adaptive boosting
#---------------------------------------------------------
class AdaBoostClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі Адаптивного бустингу

        Predicts your survival results on Adaptive boosting model
        '''
        model = joblib.load('ml_models/AdaBoostClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Decision tree
#---------------------------------------------------------
class DecisionTreeClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою Дерева рішень

        Predicts your survival results on Decision tree model
        '''
        model = joblib.load('ml_models/DecisionTreeClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Extra trees
#---------------------------------------------------------
class ExtraTreesClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі Екстра дерев

        Predicts your survival results on Extra trees model
        '''
        model = joblib.load('ml_models/ExtraTreesClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
#---------------------------------------------------------
# Gaussian Naive Bayes
#---------------------------------------------------------
class GaussianNBView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою Гаусового наївного байєсівського класифікатора

        Predicts your survival results on Gaussian Naive Bayes model
        '''
        model = joblib.load('ml_models/GaussianNB.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# K Nearest neighbours
#---------------------------------------------------------
class KNeighboursClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі К Найближчих сусідів

        Predicts your survival results on K Nearest neighbours model
        '''
        model = joblib.load('ml_models/KNeighboursClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Random forest
#---------------------------------------------------------
class RandomForestClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі Випадкового лісу

        Predicts your survival results on Random forest model
        '''
        model = joblib.load('ml_models/RandomForestClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Support vector machines
#---------------------------------------------------------
class SVCView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою методі Опорних векторів

        Predicts your survival results on Support vector machines model
        '''
        model = joblib.load('ml_models/SVC.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Voting
#---------------------------------------------------------
class VotingClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою Голосуючого ансамблю моделей

        Predicts your survival results on Voting model
        '''
        model = joblib.load('ml_models/VotingClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Extreme boosting
#---------------------------------------------------------
class XGBClassifierView(ViewSet):
    serializer_class = BloodTestSerializer

    @swagger_auto_schema(method='post', request_body=BloodTestSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі екстримального бустингу

        Predicts your survival results on Extreme boosting model
        '''
        model = joblib.load('ml_models/XGBClassifier.pcl')
        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            result = PredictionResultSerializer(data=serializer.instance.predict(models=[model]))
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#---------------------------------------------------------
# Multiple models
#---------------------------------------------------------
class MultipleModelsView(ViewSet):
    serializer_class = ManyTestsSerializer

    @swagger_auto_schema(method='post', request_body=ManyTestsSerializer, responses={ 200: PredictionResultSerializer} )
    @action(detail=True, methods=['post'])
    def predict(self, request, **kwargs):
        '''
        Прогнозування за допомогою моделі екстримального бустингу

        Predicts your survival results on Extreme boosting model
        '''
        models = []
        model_names = request.POST.getlist('models')
        print(model_names)
        for model_name in model_names:
            models.append(joblib.load('ml_models/' + model_name + '.pcl'))

        serializer = BloodTestSerializer(data=request.data)
        if(serializer.is_valid()):
            serializer.save()
            preds = serializer.instance.predict(models=models, return_mean=False)
            result = PredictionResultSerializer(data=preds, many=True)
            if(result.is_valid()):
                return Response(result.data, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)