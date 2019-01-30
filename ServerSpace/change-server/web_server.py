# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import flask
import redis
import time
import io
import random
import pickle
from urllib.request import urlopen

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def read_image(image_url, is_oss=True, resize_h=settings.IMAGE_HEIGHT):
    start_time = time.time()
    if is_oss:
        image_url = image_url + "?x-oss-process=image/resize,h_" + str(resize_h)
    image_read = urlopen(image_url).read()
    image_time = time.time() - start_time
    return image_read, image_time


def prepare_image(image, target=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)):
    image = Image.open(io.BytesIO(image))
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # prepeocess input, used for xception and inception model
    image = imagenet_utils.preprocess_input(image, mode='tf')
    image = pickle.dumps(image)
    return image


@app.route("/")
def homepage():
    return "Welcome to change_detect Server!"


@app.route("/change_detect", methods=["POST"])
def change_detect():
    # time
    start_time = time.time()
    # initialize the result_data dictionary that will be returned from the view
    result_data = {'request_success': False,
                   'error_code': 0,
                   'error_description': 'no error',
                   "predict_result": {},
                   "save_result": {},
                   'web_time': {}}
    # ensure request was properly uploaded to our endpoint
    # noinspection PyBroadException
    try:
        request_data = flask.request.json
        request_id_diy = str(request_data["request_id"]) + '_' + str(time.time())
        request_data["request_id_diy"] = request_id_diy
        request_type = request_data["request_type"]
        image_a_url = request_data["image_a"]
        image_b_url = request_data["image_b"]
    except Exception as e:
        result_data['error_code'] = 1
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # download and read the image in PIL format and prepare it
    try:
        image_a, image_a_time = read_image(image_a_url)
        image_a_key = request_id_diy + '_img_a'
        request_data['image_a_key'] = image_a_key
        db.set(image_a_key, prepare_image(image_a))
        image_b, image_b_time = read_image(image_b_url)
        image_b_key = request_id_diy + '_img_b'
        request_data['image_b_key'] = image_b_key
        db.set(image_b_key, prepare_image(image_b))
    except Exception as e:
        result_data['error_code'] = 2
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # put into the queue
    if request_type == 'change_detect':
        task_queue = random.choice(settings.QUEUE_GPUS_CD)
    else:
        result_data['error_code'] = 3
        result_data['error_description'] = 'not support request type'
        # return the result_data dictionary as a JSON response
        return flask.jsonify(result_data)
    # begin
    db.rpush(task_queue, pickle.dumps(request_data))
    prepare_time = time.time() - start_time
    # time
    get_start_time = time.time()
    get_count = 0
    # keep looping until our model server returns the output
    # predictions
    while True:
        # sleep for a small amount to give the model time
        time.sleep(settings.CLIENT_SLEEP)
        # attempt to grab the output predictions
        output = db.get(request_id_diy)
        get_count += 1
        # check to see if our model has done the prediction
        if output is not None:
            # add the output predictions to our result_data
            # dictionary so we can return it to the client
            pred = pickle.loads(output)["pred_change"]
            # {'change': 0, 'same': 1}
            if pred > settings.Threshold_Change_Detect:
                result_data["predict_result"]["change"] = 0
                result_data["predict_result"]["prob"] = "%.3f" % pred
            else:
                result_data["predict_result"]["change"] = 1
                result_data["predict_result"]["prob"] = "%.3f" % (1 - pred)
            # save real model result
            if pred > 0.5:
                result_data["save_result"]["change"] = 0
                result_data["save_result"]["prob"] = "%.3f" % pred
            else:
                result_data["save_result"]["change"] = 1
                result_data["save_result"]["prob"] = "%.3f" % (1 - pred)
            # delete the result from the result_database and
            db.delete(request_id_diy)
            db.delete(image_a_key)
            db.delete(image_b_key)
            # time
            get_time = time.time() - get_start_time
            run_time = time.time() - start_time
            result_data['web_time']['server_time'] = "%.3f" % run_time
            result_data['web_time']['prepare_time'] = "%.3f" % prepare_time
            result_data['web_time']['image_a_time'] = "%.3f" % image_a_time
            result_data['web_time']['image_b_time'] = "%.3f" % image_b_time
            result_data['web_time']['get_time'] = "%.3f" % get_time
            result_data['web_time']['get_count'] = get_count
            # indicate that the request was a success
            result_data["request_success"] = True
            # break from the polling loop
            break
        elif get_count >= settings.GET_MAX:
            # delete the result from the result_database and
            db.delete(image_a_key)
            db.delete(image_b_key)
            # reach max get count, force to stop
            result_data['error_code'] = 4
            result_data['error_description'] = "cost time too long"
            # break from the polling loop
            break
    # return the result_data dictionary as a JSON response
    return flask.jsonify(result_data)


# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production)
if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
