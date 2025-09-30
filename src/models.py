from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_models_and_grids():
    models = {
        "logreg": LogisticRegression(max_iter=500, class_weight="balanced"),
        "rf":     RandomForestClassifier(class_weight="balanced", random_state=42),
        "gb":     GradientBoostingClassifier(random_state=42),
        "svc":    SVC(probability=True, class_weight="balanced", random_state=42),
        "knn":    KNeighborsClassifier(),
    }
    grids = {
        "logreg": {"clf__C":[0.1,1.0,3.0], "clf__penalty":["l2"], "clf__solver":["lbfgs"]},
        "rf":     {"clf__n_estimators":[200,400], "clf__max_depth":[None,8,16], "clf__min_samples_split":[2,10]},
        "gb":     {"clf__n_estimators":[150,300], "clf__learning_rate":[0.05,0.1], "clf__max_depth":[2,3]},
        "svc":    {"clf__C":[0.5,1.0,3.0], "clf__kernel":["rbf"], "clf__gamma":["scale",0.01]},
        "knn":    {"clf__n_neighbors":[5,11,21], "clf__weights":["uniform","distance"]},
    }
    return models, grids
