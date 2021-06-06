import dash as dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from MultipleLR import *
from RandomForest import *

class DashboardController:
    # Initializing the class
    def __init__(self, df):
        self.df = df
        self.rf = RandomForest(df,'avg_vote',400,5,1000,0.2)
        self.randvalue1 = self.rf.new_random_test_value()
        self.randvalue2 = self.rf.new_random_test_value()
        self.randvalue3 = self.rf.new_random_test_value()
        self.mlr = MultipleLR(df,'avg_vote')
        self.mlr.get_pearson_heatmap()

    # all code that is needed for the dashboard html and to launch
    def dash_frontend(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        #
        app.layout = dbc.Container([
            self.tabs()
        ])

        tab_one = self.tab_one()
        tab_two = self.tab_two()

        # Using callback for interacting
        @app.callback(Output('tab_page', 'children'),
                      Input('tabs', 'value'))
        def render_content(tab):
            if tab == 'tab_1':
                return tab_one
            elif tab == 'tab_2':
                return tab_two

        # Starting server for Dashboard
        if __name__ == 'DashboardController':
            app.run_server(debug=True)

    # dashboard core component tab menu:
    def tabs(self):
        return html.Div([
            dcc.Tabs(id='tabs', value='tab_1', children=[
                dcc.Tab(label='Multiple Linear Regression', value='tab_1'),
                dcc.Tab(label='Random Forest', value='tab_2'),
            ]),
            html.Div(id='tab_page')
        ])
    # Function for showing Multi linear regression model at the dashboard in tab 1 at the top
    def tab_one(self):
        return html.Div([html.Div(html.H3('Multiple Linear Regression'), style={'text-align': 'center'}),
        html.Div([
            html.Img(src='assets/parson_heatmap.png',className='align-self-center')
        ],style={'text-align': 'center'}),
        html.Div([html.H5('Feature importance for the chosen model: ')],style={'text-align': 'center'}),
            html.Div(
                dcc.Graph(figure=self.mlr.get_model_bar(self.mlr.get_model()))
            )
        ])

    #Function for show contents in tab 2
    def tab_two(self):
        return html.Div([html.Div(html.H3('Random Forest'), style={'text-align': 'center'}),
        html.Div([
            html.Span('In this particular instance a random forest has been train with '),
            html.Span(f'{self.rf.get_estimators()}',style={'color':'#228B22','font-weight':'bold'}),
            html.Span(' trees. The maximum amount of depth for each tree is: '),
            html.Span(f'{self.rf.get_max_depth()}',style={'color':'#228B22','font-weight':'bold'}),
            html.Span('. The model has an accuracy of: '),
            html.Span(f'{round(self.rf.get_accuracy(),4) * 100}',style={'color':'#228B22','font-weight':'bold'}),
            html.Span('%. Lets explorer the model! :)'),
            html.P(''),
            html.P('First and foremost we can have a look at the feature importance of the model itself,'
                   'the feature importance of the global model are:'),
        ]),
        html.Div([
            dcc.Graph(figure=px.bar(x=self.rf.get_feature_importance()[0],
                                    y=self.rf.get_feature_importance()[1],
                                    template= 'simple_white',
                                    text=self.rf.get_feature_importance()[1]))
        ]),
        html.Div([
            html.Span('As you can see above what features that are important to the model can vary alot,'
                   'but lets take a closer look. Below we will take a look at a specific case and how it travels'
                   'through some of the '),
            html.Span(f'{self.rf.get_estimators()}', style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' trees that make up the random forest.'),
            html.P(''),
            html.Span('For this specific test the following test data has been selected at random from our test dataset'),
            html.Span(f'{list(zip(self.rf.get_feature_names(),self.rf.get_main_test_value()))}',
                      style={'color':'#228B22','font-weight':'bold'}),
            html.Span('This random test value has been predicted to to be the following value: '),
            html.Span(f'{self.rf.get_prediction(self.rf.get_main_test_value())}',
                      style={'color':'#228B22','font-weight':'bold'}),
            html.Span('But how did the model get to this specific value?, lets take a look at a tree or three:')
        ]),
        html.Div([
            html.Img(src=f'assets/tree{self.rf.get_estimators() - 1}.png',className='align-self-center')
        ],style={'text-align': 'center'}),
        html.Div([
            html.Img(src=f'assets/tree{self.rf.get_estimators() - 2}.png',className='align-self-center')
        ],style={'text-align': 'center'}),
        html.Div([
            html.Img(src=f'assets/tree{self.rf.get_estimators() - 3}.png',className='align-self-center')
        ],style={'text-align': 'center'}),
        html.Div([
            html.P(''),
            html.Span('These are just 3 trees out of the '),
            html.Span(f'{self.rf.get_estimators()}', style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' that were used to construct the Random Forest, unfortunately going through every tree is'
                      'not a feasible way of understanding the Random Forest.'),
            html.P(''),
            html.Span('One way we can clarify what factors that whent into this specific prediction is by using LIME'
                      'LIME summarizes what features that specifically had impact on this prediction and how much'
                      'impact they had!:')

        ]),
        html.Div([
            html.Img(src=self.rf.get_lime('1', self.rf.get_main_test_value()))
        ],style={'text-align': 'center'}),
        html.Div([
            html.Span('The above LIME model is still for the: '),
            html.Span(f'{self.rf.get_main_test_value()}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' value, with the prediction: '),
            html.Span(f'{self.rf.get_prediction(self.rf.get_main_test_value())}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.P(''),
            html.Span('Lets take a look at some other examples to see how LIME changes and if the feature '
                      'importance stays consistent: '),
            html.P('')
        ]),
        html.Div([
            html.P(''),
            html.Span('The value of:'),
            html.P(f'{list(zip(self.rf.get_feature_names(), self.randvalue1))}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' has been predicted to have the following value: '),
            html.Span(f'{self.rf.get_prediction(self.randvalue1)}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.P(''),
            html.Img(src=self.rf.get_lime('randvalue1', self.randvalue1)),
            html.P('')
        ], style={'text-align': 'center'}),
        html.Div([
            html.P(''),
            html.Span('The value of:'),
            html.P(f'{list(zip(self.rf.get_feature_names(), self.randvalue2))}',
                   style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' has been predicted to have the following value: '),
            html.Span(f'{self.rf.get_prediction(self.randvalue2)}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.P(''),
            html.Img(src=self.rf.get_lime('randvalue2', self.randvalue2)),
            html.P('')
        ], style={'text-align': 'center'}),
        html.Div([
            html.P(''),
            html.Span('The value of:'),
            html.P(f'{list(zip(self.rf.get_feature_names(), self.randvalue3))}',
                   style={'color': '#228B22', 'font-weight': 'bold'}),
            html.Span(' has been predicted to have the following value: '),
            html.Span(f'{self.rf.get_prediction(self.randvalue3)}',
                      style={'color': '#228B22', 'font-weight': 'bold'}),
            html.P(''),
            html.Img(src=self.rf.get_lime('randvalue3', self.randvalue3)),
            html.P(''),
        ], style={'text-align': 'center'})
    ])
