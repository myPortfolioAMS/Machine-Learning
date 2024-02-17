# check kaggle for charts

#Data Treatment 
import numpy as np
import pandas as pd
import warnings

# Vizualization
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#Hypothesis Tests
from scipy.stats import ranksums
from scipy.stats import normaltest
from statsmodels.stats.weightstats import ztest

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Stratification strategy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

# Assessment metrics
from sklearn.metrics import accuracy_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Customizing views
matplotlib.rcParams['figure.figsize'] = [14, 8]
sns.set_theme(style='whitegrid')
warnings.simplefilter(action='ignore', category=FutureWarning)

