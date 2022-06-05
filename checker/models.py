from django.db import models
import numpy as np
import pandas as pd

class BloodTest(models.Model):
    age = models.PositiveSmallIntegerField(verbose_name='Вік', null=True, default=0)
    hematocrit = models.FloatField(verbose_name='Гематокрит', null=True, default=0)
    red_blood_cells = models.FloatField(verbose_name='Еритроцити', null=True, default=0)
    lymphocytes = models.FloatField(verbose_name='Лімфоцити', null=True, default=0)
    leukocytes = models.FloatField(verbose_name='Лейкоцити', null=True, default=0)
    basophils = models.FloatField(verbose_name='Базофіли', null=True, default=0)
    eosinophils = models.FloatField(verbose_name='Еозинофіли', null=True, default=0)
    monocytes = models.FloatField(verbose_name='Моноцити', null=True, default=0)
    glucose = models.FloatField(verbose_name='Глюкоза', null=True, default=0)
    neutrophils = models.FloatField(verbose_name='Нейтрофіли', null=True, default=0)
    c_reactive_protein = models.FloatField(verbose_name='С-реактивний білок', null=True, default=0)
    creatinine = models.FloatField(verbose_name='Креатинін', null=True, default=0)
    potassium = models.FloatField(verbose_name='Калій', null=True, default=0)
    sodium = models.FloatField(verbose_name='Натрій', null=True, default=0)
    alat = models.FloatField(verbose_name='АЛАТ (Аланінамінотрансфераза)', null=True, default=0)
    asat = models.FloatField(verbose_name='АСАТ (Аспартатамінотрансфераза)', null=True, default=0)
    bilirubine = models.FloatField(verbose_name='Білірубін', null=True, default=0)

    def predict(self, models, return_mean=True):
        result = []
        data = pd.DataFrame([[
            self.age // 4,          self.hematocrit,        self.red_blood_cells,
            self.lymphocytes,       self.leukocytes,        self.basophils,
            self.eosinophils,       self.monocytes,         self.glucose,
            self.c_reactive_protein,self.creatinine,        self.potassium,
            self.sodium,            self.alat,              self.asat,          
            self.bilirubine
        ]], columns=[
            'patient_age_quantile', 'hematocrit', 'red_blood_cells', 'lymphocytes', 
            'leukocytes', 'basophils', 'eosinophils', 'monocytes', 'serum_glucose', 
            'c_reactive_protein', 'creatinine', 'potassium', 'sodium', 'alanine_transaminase',
            'aspartate_transaminase', 'total_bilirubin']
        )
        
        for model in models:
            result.append(model.predict_proba(data)[0].tolist())

        if(return_mean):
            result = np.mean(result, axis=0)
        
        if(len(models) == 1):
            return {
                'model_name': type(models[0].steps[1][1]).__name__,
                'death_probability': result[1],
                'cure_probability': result[0]
            }
        elif(return_mean):
            return {
                'model_name': 'multiple',
                'death_probability': result[1],
                'cure_probability': result[0]
            }
        else:
            return [{
                'model_name': type(models[idx].steps[1][1]).__name__,
                'death_probability': result[idx][1],
                'cure_probability': result[idx][0]
            } for idx in range(len(models))]


    class Meta:
        verbose_name_plural = 'Аналізи крові'
        verbose_name = 'Аналіз крові'
        ordering = ['age']
