from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.tree import export_graphviz
import os
from sklearn.model_selection import train_test_split
from scipy import sparse
import json
from graphviz import Digraph
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import random
from PIL import Image

class RandomForest:

    #Initializing the class
    def __init__(self,df, predict_col, estimators, max_depth, max_samples, test_size):
        self.df = df
        self.predict_col = predict_col
        self.estimators = estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.test_size = test_size
        self.get_model()
        self.main_test_value = self.x_test[random.randrange(len(self.x_test))]
        self.generate_rf_pngs(self.main_test_value)

    # Splits data and creates the random forest regression model
    def get_model(self):
        self.train, self.test = train_test_split(self.df, test_size=self.test_size)
        self.y_train = np.array(self.train[self.predict_col]).tolist()
        self.x_train = np.array(self.train.drop([self.predict_col], axis=1)).tolist()
        self.x_names = self.train.drop([self.predict_col], axis=1).columns.tolist()
        self.x_test = np.array(self.test.drop([self.predict_col], axis=1))
        self.y_test = np.array(self.test[self.predict_col]).tolist()
        self.model = RandomForestRegressor(n_estimators=self.estimators, bootstrap=True, max_depth=self.max_depth, max_samples=self.max_samples)
        self.model = self.model.fit(self.x_train, self.y_train)
        return self.model

    # Gets amount of estimators passed to the class
    def get_estimators(self):
        return self.estimators

    # Gets max depth that was passed to the class
    def get_max_depth(self):
        return self.max_depth

    # Gets max_samples that was passed to the class
    def get_max_samples(self):
        return self.max_samples

    # Gets the accuracy of the model based on the test data
    def get_accuracy(self):
        return self.model.score(self.x_test,self.y_test)

    # Gets feature importance of the model
    def get_feature_importance(self):
        importances = self.model.feature_importances_
        x = self.test.drop([self.predict_col], axis=1).columns.values
        y = importances
        return x,y

    # Calls model prediction on 'pred'
    def get_prediction(self,pred):
        return self.model.predict(pred.reshape(1,-1))

    # We establish a main test value for the dashboard, Gets that value
    def get_main_test_value(self):
        return self.main_test_value

    # Gets feature name excluding the column that it is trying to predict used in dashboard
    def get_feature_names(self):
        return self.x_names

    # Gets a random value from the test dataset
    def new_random_test_value(self):
        return self.x_test[random.randrange(len(self.x_test))]

    # Gets path to a specific lime model created by passing a path_name and the value that is to be tested
    def get_lime(self, path_name, test_value):
        x_test_local = self.train.drop([self.predict_col], axis=1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(x_test_local),
                                                           mode = 'regression',
                                                           feature_names=self.x_names,
                                                           categorical_features=[0]
                                                           )
        exp = explainer.explain_instance(data_row= np.array(test_value),
                                         predict_fn=self.model.predict)
        exp.as_pyplot_figure(label=1)
        plt.tight_layout()
        plt.savefig(f'assets/rflime_{path_name}')
        return f'assets/rflime_{path_name}.png'

    # Generate Graphviz PNG's of trees, recolors the specific instance path for test value and resizes the pictures
    def generate_rf_pngs(self,test_value):

        # For loop that loops through the last 3 estinators
        length_estimators = range(len(self.model.estimators_))
        for i in length_estimators[-3:]:
            # Gets estimator
            estimator = self.model.estimators_[i]
            # Exports estimator to Graphviz
            export_graphviz(estimator,
                            out_file=fr'dotfiles\tree{i}.dot',
                            feature_names=self.x_names,
                            filled=True,
                            rounded=True)

            # Saves the Graphviz file as a .dot
            os.system(fr'dot -Txdot_json -ojsonfiles\tree{i}.json dotfiles\tree{i}.dot')

            # Gets the matrix of the specific prediction and converts it to an array
            a = self.model.estimators_[i].decision_path(test_value.reshape(1, -1))
            b = sparse.csr_matrix(a).toarray()
            # Loops through the matrix
            for j in range(len(b[0])):
                # Loops through the json file
                with open(fr'jsonfiles\tree{i}.json', 'r+') as f:
                    data = json.load(f)
                    # If the value in the matrix is 1 we recolor the corresponding value in the json
                    if b[0][j] == 1:
                        data['objects'][j]['fillcolor'] = 'green'
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()

            # Creates a new graphviz digraph
            digraph = Digraph()
            # These two loops recreates the entire tree based on the json fgile that we edited above
            for a in range(len(data['objects'])):
                digraph.node(str(data['objects'][a]['_gvid']), str(data['objects'][a]['label']),
                             style='filled',
                             fillcolor=str(data['objects'][a]['fillcolor']))

            for b in range(len(data['edges'])):
                digraph.edge(str(data['edges'][b]['tail']), str(data['edges'][b]['head']))

            # Saves these new recolored trees as .dot files
            digraph.render(filename=f'digraphfiles\digraph{i}.dot')

            # Saves the .dot files as PNG's
            os.system(fr'dot -Tpng digraphfiles\digraph{i}.dot -o assets\tree{i}.png')

            # Resizes PNG's to fit in the dashboard and replace the old files so we only have the resized version
            image = Image.open(fr'assets\tree{i}.png')
            resized_image = image.resize((1600,300))
            resized_image.save(fr'assets\tree{i}.png')