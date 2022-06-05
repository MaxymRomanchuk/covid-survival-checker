from cProfile import label
from rest_framework import serializers
from .models import BloodTest

class BloodTestSerializer(serializers.ModelSerializer):

    class Meta:
        model = BloodTest
        fields = '__all__'
        read_only_fields = ['id']

class ManyTestsSerializer(serializers.ModelSerializer):
    models = serializers.MultipleChoiceField(choices=[
        ('AdaBoostClassifier', 'Адаптивний бустинг'),
        ('DecisionTreeClassifier', 'Дерево рішень'),
        ('ExtraTreesClassifier', 'Ектра дерева'),
        ('GaussianNB', 'Гаусовий наївний байєсів класифікатор'),
        ('KNeighborsClassifier', 'К найближчих сусідів'),
        ('LogisticRegressionCV', 'Логістична регресія'),
        ('RandomForestClassifier', 'Випадковий ліс'),
        ('SVC', 'Метод опорних векторів'),
        ('VotingClassifier', 'Голосуючий класифікатор'),
        ('XGBClassifier', 'Ектримальний бустинг'),
    ])

    class Meta:
        model = BloodTest
        fields = '__all__'
        read_only_fields = ['id']


class PredictionResultSerializer(serializers.Serializer):
    model_name = serializers.CharField(label='Модель прогнозування')
    death_probability = serializers.FloatField(label='Ймовірність загинути')
    cure_probability = serializers.FloatField(label='Ймовірність одужати')