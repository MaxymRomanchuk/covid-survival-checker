from django.urls import path
from .views import *
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title='COVID-19 survival checker',
        default_version='v0.2',
        description=''' 
        <b>Сервіс з прогнозування ймовірності одужати/загинути від COVID-19</b>
        Прогнозування відбувається за результатами аналізів крові
        <br>
        COVID-19 blood test based mortality prediction service
        '''
    ),
    public=True
)

urlpatterns = [
    path('', schema_view.with_ui('swagger', cache_timeout=0)),
    path('model/svc', SVCView.as_view({'post': 'predict'})),
    path('model/xgb', XGBClassifierView.as_view({'post': 'predict'})),
    path('model/gaussian-nb', GaussianNBView.as_view({'post': 'predict'})),
    path('model/voting', VotingClassifierView.as_view({'post': 'predict'})),
    path('model/knn', KNeighboursClassifierView.as_view({'post': 'predict'})),
    path('model/ada-boost', AdaBoostClassifierView.as_view({'post': 'predict'})),
    path('model/extra-trees', ExtraTreesClassifierView.as_view({'post': 'predict'})),
    path('model/decision-tree', DecisionTreeClassifierView.as_view({'post': 'predict'})),
    path('model/random-forest', RandomForestClassifierView.as_view({'post': 'predict'})),
    path('model/logistic-regression', LogisticRegressionView.as_view({'post': 'predict'})),
    path('models/multiple', MultipleModelsView.as_view({'post': 'predict'}))
]