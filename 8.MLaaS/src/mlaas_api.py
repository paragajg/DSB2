# Load Libraries
import flask
from flask import request, url_for
import json
import pandas as pd
import numpy as np
import cherrypy
from paste.translogger import TransLogger
import joblib
import time
import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# Initialize Flask App
app = flask.Flask(__name__)

logging.info("Workign dir:%s"%os.getcwd())
model = joblib.load("email_classfierV1.0.model")

@app.route("/", methods = ['GET' , 'POST'])
def open_html():
    return flask.render_template("index.html")

@app.route("/predict_spam", methods = ['GET' , 'POST'])
def predict_spam():
    data = {'success': False}
    #params = flask.request.get_json(force = True)
    #print("Input data for prediction")
    #print(params)
    '''
    if (params == None):
        params = flask.request.args
    '''
    
    if request.method == "POST":
        emailText = request.form["text"]
        y_predict = model.predict([emailText])
        dict_predict = {'predictions':y_predict[0]}
        data.update(dict_predict)
        data["success"] = True
        logging.info("Sent Prediciton as: %s"%(y_predict[0]))

    '''    
    if (params != None):
        y_predict = model.predict([params["text"]])
        dict_predict = {'predictions':y_predict[0]}
        data.update(dict_predict)
        data["success"] = True
        print(data)'''
    
    #return flask.jsonify(data)
    return flask.render_template("pass_values.html",inpText = emailText, result = y_predict[0])

class FotsTransLogger(TransLogger):
    def write_log(self, environ, method, req_uri, start, status, bytes):
        if bytes is None:
            bytes = '-'
        remote_addr = '-'
        if environ.get('HTTP_X_FORWARDED_FOR'):
            remote_addr = environ['HTTP_X_FORWARDED_FOR']
        elif environ.get('REMOTE_ADDR'):
            remote_addr = environ['REMOTE_ADDR']
        d = {
            'REMOTE_ADDR': remote_addr,
            'REMOTE_USER': environ.get('REMOTE_USER') or '-',
            'REQUEST_METHOD': method,
            'REQUEST_URI': req_uri,
            'HTTP_VERSION': environ.get('SERVER_PROTOCOL'),
            'time': time.strftime('%d/%b/%Y:%H:%M:%S', start),
            'status': status.split(None, 1)[0],
            'bytes': bytes,
            'HTTP_REFERER': environ.get('HTTP_REFERER', '-'),
            'HTTP_USER_AGENT': environ.get('HTTP_USER_AGENT', '-'),
        }
        message = self.format % d
        self.logger.log(self.logging_level, message)


def run_server():
    # Enable custom Paste access logging
    log_format = (
        '[%(time)s] REQUES %(REQUEST_METHOD)s %(status)s %(REQUEST_URI)s '
        '(%(REMOTE_ADDR)s) %(bytes)s'
    )
    app_logged = FotsTransLogger(app, format=log_format)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    run_server()
