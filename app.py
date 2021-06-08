import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from loaders import load_model
from models import CifarModel
import numpy as np
import os
from PIL import Image
import datetime
import matplotlib.pyplot as plt
image_directory = 'assets/images/'
app = dash.Dash(__name__, assets_url_path='assets/')
app.layout = html.Div(
    id='app',
    children=[
        html.Div(
            id='app-container',
            children=[
                html.Div(
                    id='images_upload_container',
                    children=[
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop image to make prediction',
                            ]),
                            multiple=True
                        ),
                    ],
                ),

            ]
        ),
        html.Div(
                 id='flex-container',
                 children=[
                    html.Div(className='flex-item componentWrapper ',
                             children=[
                                 html.Div(className="header",
                                          children=[html.Div("Input data")]
                                          ),
                                 html.Div(id='output-image-upload')
                             ]),
                    html.Div(className='flex-item componentWrapper ',
                             children=[
                                 html.Div(className="header",
                                          children=[html.Div("Output")]
                                          ),
                                 html.Div(id='output-prediction', className='center_vertically'),
                             ]),

                ],
        ),
    ],

)


def parse_contents(contents, filename):
    return html.Div([
        html.H4(filename),
        html.Img(src=contents, className='image'),
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


def load_and_preprocess(image):
    img = plt.imread(os.path.join(image_directory, image))
    return img


model = load_model(CifarModel, 'cifar_model')


@app.callback(Output('output-prediction', 'children'),
              Input('upload-image', 'filename'))
def prediction(img):
    if img is None:
        raise dash.exceptions.PreventUpdate
    else:
        img = load_and_preprocess(img[0])
        img = np.expand_dims(img, axis=0)
        classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck'])
        y_preds = model.predict(img[:, :, :, :3])
        predicted_classes = classes[np.argmax(y_preds)]

    return html.H3(predicted_classes)


if __name__ == '__main__':
    app.run_server(debug=True)