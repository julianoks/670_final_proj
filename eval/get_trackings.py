from flask import Flask, request, send_from_directory
import webbrowser
import os, shutil
import json
import urllib.parse
from multiprocessing import Process
import time
import numpy as np
import datetime

def setup_flask_app(res_dir):
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    @app.route('/post_results', methods=['POST'])
    def post_results():
        result = request.get_json(force = True)
        name = result['experimentName']
        trackings = result['trackingsOverTime']
        with open(f'{res_dir}/{name}.json', 'w') as f:
            json.dump(trackings, f)
        return 'OK', 200

    @app.route('/eval/algorithms/<path:path>')
    def eval_algorithms_file(path):
        return send_from_directory(os.getcwd(), os.path.join('algorithms', path))

    @app.route('/<path:path>')
    def static_file(path):
        return send_from_directory(os.getcwd(), path)
    
    return app

def get_trackings(tracker_weight_params = {}, train=True, n_datasets=1):
    train_or_test = 'train' if train else 'test'
    res_dir = f'eval/results/{datetime.datetime.now()}'
    os.mkdir(res_dir)
    app = setup_flask_app(res_dir)
    # run training
    ds_names = os.listdir('eval/2DMOT2015/' + train_or_test)
    n_images = [
        len([x for x in os.listdir('eval/2DMOT2015/' + train_or_test + '/' + ds + '/img1') if x[-4:].lower() == '.jpg'])
        for ds in ds_names
    ]

    if n_datasets != -1:
        selected_idxs = np.argsort(n_images)[:n_datasets]
        ds_names = np.array(ds_names)[selected_idxs].tolist()
        n_images = np.array(n_images)[selected_idxs].tolist()

    # run flask server in a new process
    app.use_reloader = False
    app.debug = False
    proc = Process(target=app.run)
    proc.start()
    time.sleep(1) # wait for server process to startup

    for n_ims, ds_name in zip(n_images, ds_names):
        payload = {
            'nImages': n_ims,
            'datasetName': ds_name,
            'datasetTrainOrTest': train_or_test,
            'trackerWeightParams': tracker_weight_params
        }
        url = 'http://127.0.0.1:5000/eval/get_trackings.html?' + urllib.parse.urlencode({'payload': json.dumps(payload)})
        webbrowser.open_new(url)
    
    # wait for all datasets to be processed
    while len([x for x in os.listdir(res_dir) if x[-5:].lower() == '.json']) != len(ds_names):
        time.sleep(1)
    
    time.sleep(2) # let the writers finish
    proc.terminate()
    proc.join()

    return res_dir
