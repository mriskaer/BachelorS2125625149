import plotly.express as px
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class MultipleLR:

    # Initializing the class
    def __init__(self,data_set,predict_column):
        self.data_set = data_set
        self.predict_column = predict_column
        self.X = self.data_set.drop(columns=[self.predict_column])
        self.y = self.data_set[self.predict_column]

    # Creates a linear regression model with data we split in the __init__
    def get_model(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model

    # Gets a bar chart of feature importance of the selected features.
    def get_model_bar(self,model):
        colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]
        fig = px.bar(
            x=self.X.columns, y=self.get_model().coef_, color=colors,
            color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Feature', y='Linear coefficient'),
            title='Weight of each feature for predicting movie rating'
        )
        return fig

    # Creates a pearson heatmap and returns the path for it
    def get_pearson_heatmap(self):
        plt.figure(figsize=(20, 10))
        cor = self.data_set.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Greens)
        plt.savefig('assets/pearson_heatmap.png')
        return 'assets/pearson_heatmap.png'