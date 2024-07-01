from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

def models():
    models = {
              "k-NN": KNeighborsClassifier(),
              "Decision Tree": DecisionTreeClassifier(random_state=42),
              "SVM": svm.SVC(),
              "BernoulliNB": BernoulliNB(),
              "LogisticRegression": LogisticRegression(random_state=42),
              "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
              "XGBClassifier": XGBClassifier(),
              "GradientBoostClassifier": GradientBoostingClassifier()

          }
    return models