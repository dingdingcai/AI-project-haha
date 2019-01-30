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
import json
from urllib.request import urlopen


# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def read_image(image_url, is_oss=True, resize_h=settings.IMAGE_HEIGHT):
    start_time = time.time()
    if is_oss:
        image_url = image_url + "?x-oss-process=image/resize,h_" + str(resize_h)
    image = urlopen(image_url).read()
    image_time = time.time() - start_time
    return image, image_time


def save_image(image, image_path):
    with open(image_path, 'wb') as f:
        f.write(image)


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


def generate_heatmap_with_sku_points(sku_points):
    heatmap = np.zeros((settings.HEATMAP_HEIGHT, settings.HEATMAP_WIDTH), dtype=np.uint8)
    for sku, point_list in sku_points.items():
        if len(point_list) > 0:
            for point in point_list:
                h, w = float(point[1]), float(point[0])
                if 0 <= h < 1 and 0 <= w < 1:
                    h, w = int(h * settings.HEATMAP_HEIGHT), int(w * settings.HEATMAP_WIDTH)
                    heatmap[h][w] = 1
    return heatmap[:, :, np.newaxis]


def res_sku_points(sku_points, heatmap):
    heatmap = np.squeeze(heatmap, axis=-1)
    predict_res = {}
    b_count = 0
    save_res = {}
    min_change_prob = 1.0
    min_unchange_prob = 1.0
    for sku, point_list in sku_points.items():
        if len(point_list) > 0:
            predict_point_list = []
            save_point_list = []
            for point in point_list:
                h, w = float(point[1]), float(point[0])
                if 0 <= h < 1 and 0 <= w < 1:
                    h, w = int(h * settings.HEATMAP_HEIGHT), int(w * settings.HEATMAP_WIDTH)
                    x = heatmap[h][w]
                    if x < settings.Threshold_point_change_one:
                        # unchange
                        predict_point_list.append(point)
                        unchange_prob = 1 - x
                        if unchange_prob < min_unchange_prob:
                            min_unchange_prob = unchange_prob
                    else:
                        change_prob = x
                        if change_prob < min_change_prob:
                            min_change_prob = change_prob
                    save_point_list.append((*point, "%.3f" % x))
            if len(predict_point_list) > 0:
                predict_res[sku] = predict_point_list
                b_count += len(predict_point_list)
                save_res[sku] = save_point_list
    return predict_res, b_count, min_change_prob, min_unchange_prob, save_res


@app.route("/")
def homepage():
    return "Welcome to m3 Server!"


@app.route("/predict", methods=["POST"])
def predict():
    # time
    start_time = time.time()
    # initialize the result_data dictionary that will be returned from the view
    result_data = {"request_success": False,
                   "error_code": 0,
                   "error_description": "no error",
                   "predict_result": {},
                   "save_result": {},
                   "web_time": {}}
    # ensure request was properly uploaded to our endpoint
    # noinspection PyBroadException
    try:
        request_data = flask.request.json
        request_id_diy = str(request_data["request_id"]) + '_' + str(time.time())
        request_data["request_id_diy"] = request_id_diy
        request_type = request_data["request_type"]
        if request_type == "take_goods_simple":
            image_b_url = request_data["image_b"]
            image_c_url = request_data["image_c"]
        else:
            result_data['error_code'] = 3
            result_data['error_description'] = 'not support request type'
            # return the result_data dictionary as a JSON response
            return flask.jsonify(result_data)
        device_id = request_data["device_id"]
        shelve_level = request_data["shelve_level"]
        shelve_id = str(device_id) + '_' + str(shelve_level)
    except Exception as e:
        result_data['error_code'] = 1
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # download and read the image in PIL format and prepare it
    try:
        # image_b is also used as b2,(count model) so should download original size
        image_b, image_b_time = read_image(image_b_url, is_oss=False)
        image_b_key = request_id_diy + '_img_b'
        request_data['image_b_key'] = image_b_key
        db.set(image_b_key, prepare_image(image_b))
        # temp for all count
        image_b2_key = request_id_diy + '_img_b2'
        request_data['image_b2_key'] = image_b2_key
        db.set(image_b2_key, prepare_image(image_b, target=(640, 480)))
        # deal with image c
        image_c_key = shelve_id + '_img_c'
        request_data['image_c_key'] = image_c_key
        image_c_url_old = db.get(shelve_id)
        image_c_path = "/var/www/html/m3_server/image_c/"+shelve_id+'.jpg'
    except Exception as e:
        result_data['error_code'] = 5
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    try:
        # check if image_c_url exist in the database
        if image_c_url_old is None or (image_c_url_old is not None and image_c_url_old != image_c_url):
            image_c, image_c_time = read_image(image_c_url)
            save_image(image_c, image_c_path)
            db.set(shelve_id, image_c_url)
        else:
            image_c, image_c_time = read_image("file://"+image_c_path, is_oss=False)
        db.set(image_c_key, prepare_image(image_c))
        # deal with pointmap_c
        c_sku_points = json.loads(request_data["pointmap_c"])
        pointmap_c = generate_heatmap_with_sku_points(c_sku_points)
        pointmap_c_key = request_id_diy + '_pointmap_c'
        request_data['pointmap_c_key'] = pointmap_c_key
        db.set(pointmap_c_key, pickle.dumps(pointmap_c))
    except Exception as e:
        result_data['error_code'] = 2
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # put into the queue
    task_queue = random.choice(settings.QUEUE_GPUS_R1)
    # begin
    db.rpush(task_queue, pickle.dumps(request_data))
    prepare_time = time.time() - start_time
    # time
    get_start_time = time.time()
    get_count = 0
    # keep looping until our model server returns the output predictions
    while True:
        # sleep for a small amount to give the model time
        time.sleep(settings.CLIENT_SLEEP)
        # attempt to grab the output predictions
        out = db.get(request_id_diy)
        get_count += 1
        # check to see if our model has done the prediction
        if out is not None:
            # add the output predictions to our result_data
            # dictionary so we can return it to the client
            out = pickle.loads(out)
            pred_heatmap = out["point_change"]
            b_sku_points, b_count, min_change_prob, min_unchange_prob, save_res_pointmap = res_sku_points(c_sku_points, pred_heatmap)
            skucount_b = {}
            for k, v in b_sku_points.items():
                skucount_b[k] = len(v)
            result_data["predict_result"]["skucount_b"] = json.dumps(skucount_b)
            result_data["save_result"]["pointmap_b"] = json.dumps(save_res_pointmap)
            result_data["save_result"]["pointmap_b_count"] = b_count
            result_data["save_result"]["min_change_prob"] = "%.3f" % min_change_prob
            result_data["save_result"]["min_unchange_prob"] = "%.3f" % min_unchange_prob
            pred_all_count = out["all_count"]
            count = np.round(pred_all_count)
            count_prob = 1.0 - np.absolute(count - pred_all_count)
            count = count.astype(int)
            result_data["save_result"]["all_count"] = "%d" % count
            result_data["save_result"]["all_count_prob"] = "%.3f" % count_prob
            if min(min_change_prob, min_unchange_prob) < settings.Threshold_point_change_min:
                # point_change prob low
                result_data["predict_result"]["if_use"] = 0
                result_data["predict_result"]["why_not"] = "point_change prob low"
            elif count_prob < settings.Threshold_all_count:
                # all_count prob low
                result_data["predict_result"]["if_use"] = 0
                result_data["predict_result"]["why_not"] = "all_count prob low"
            elif b_count != count:
                # count not equal
                result_data["predict_result"]["if_use"] = 0
                result_data["predict_result"]["why_not"] = "count not equal"
            else:
                result_data["predict_result"]["if_use"] = 1
                result_data["predict_result"]["why_not"] = "use"
            # delete the result from the result_database
            db.delete(request_id_diy)
            # delete image a and b
            db.delete(image_b_key)
            db.delete(image_b2_key)
            db.delete(image_c_key)
            get_time = time.time() - get_start_time
            run_time = time.time() - start_time
            result_data["request_success"] = True
            result_data['web_time']['server_time'] = "%.3f" % run_time
            result_data['web_time']['prepare_time'] = "%.3f" % prepare_time
            result_data['web_time']['image_b_time'] = "%.3f" % image_b_time
            result_data['web_time']['image_c_time'] = "%.3f" % image_c_time
            result_data['web_time']['get_time'] = "%.3f" % get_time
            result_data['web_time']['get_count'] = get_count
            # indicate that the request was a success
            result_data["request_success"] = True
            # break from the polling loop
            break
        elif get_count >= settings.GET_MAX:
            # delete the result from the result_database
            db.delete(image_b_key)
            db.delete(image_b2_key)
            db.delete(image_c_key)
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
