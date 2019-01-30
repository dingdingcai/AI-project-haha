# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import flask
import redis
import time
import json
import io
import random
import pickle
from collections import defaultdict
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


def postprocess_v3(y_pred, lc):
    """use mean of lc
    model output (pred): out_count
    # lc:
    # 0 -> left 0, right 0
    # 1 -> left 1, right 0
    # 2 -> left 1, right 1
    # 3 -> left 0, right 1
    # left, right, left ,right ... for column_num
    """
    # do with lc
    lc = lc[::2]
    i = 0
    todo_dict = defaultdict(list)
    start = 0
    flag_in = False
    while i < len(lc):
        if lc[i]:
            if not flag_in:
                start = i
                flag_in = True
            todo_dict[start].append(i)
        else:
            if flag_in:
                todo_dict[start].append(i)
                flag_in = False
        i += 1
    # do pred with lc
    if todo_dict:
        for x in todo_dict.values():
            if len(x) == 2:
                # 2 column, use max num
                save_number = np.max([y_pred[j] for j in x])
            else:
                # 3 or more column, use middle column to average
                x_temp = x[1:-1]
                save_number = np.mean([y_pred[j] for j in x_temp])
            for i in range(len(x)):
                y_pred[x[i]] = save_number
    count = np.round(y_pred).astype(int)
    prob = np.ones_like(y_pred) - np.absolute(np.round(y_pred) - y_pred)
    return count, prob, y_pred


def compare_empty(count_x, count_y, sub_prob):
    check_flag = False
    for i in range(len(count_y)):
        if sub_prob[i] >= settings.Threshold_Single_Count:
            if count_y[i] != 0 and count_x[i] == 0:
                check_flag = True
            elif count_y[i] == 0 and count_x[i] != 0 :
                count_x[i] = 0
    return count_x, check_flag


def sub_cunt_postprocess(pred_a, pred_b):
    sub = pred_a - pred_b
    sub_count = np.round(sub).astype(int)
    sub_prob = np.ones_like(sub) - np.absolute(sub - sub_count)
    # improve prob for zero count, plus 0.05
    for i in range(len(sub_count)):
        if sub_count[i] == 0 and sub_prob[i] < settings.Threshold_Single_Count:
            sub_prob[i] += 0.05
    return sub_count, sub_prob, sub_prob.min()


@app.route("/")
def homepage():
    return "Welcome to Ai Server!"


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
        if request_type == "take_goods":
            # only take_goods need image_a and image_c
            image_a_url = request_data["image_a"]
            # Temporally delete double_abnormal
            # image_c_url = request_data["image_c"]
        else:
            image_a_url = None
            # Temporally delete double_abnormal
            # image_c_url = None
        image_b_url = request_data["image_b"]
        # Temporally delete double_abnormal, use for temp check
        device_id = request_data["device_id"]
        shelve_level = request_data["shelve_level"]
        shelve_id = str(device_id) + '_' + str(shelve_level)
        # deal with empty_check
        check_empty_key = shelve_id + '_check_empty'
        check_empty = db.get(check_empty_key)
        # deal with column_attribute
        if request_type != "renew_goods_abnormal":
            column_attribute = request_data["column_attribute"]
            # change form of column_attribute
            column_lc = [0] * (2 * settings.COLUMN_NUM)
            lc = json.loads(column_attribute)
            for z in range(settings.COLUMN_NUM):
                # multi colomun attribute should be transformed to 2 attribute
                # 0 -> left 0, right 0
                # 1 -> left 1, right 0
                # 2 -> left 1, right 1
                # 3 -> left 0, right 1
                mc = int(lc[z])
                if mc == 1 or mc == 2:
                    column_lc[z * 2] = 1  # left, right, left ,right ... for column_num
                if mc == 2 or mc == 3:
                    column_lc[z * 2 + 1] = 1
            request_data["column_lc"] = json.dumps(column_lc)
        else:
            column_lc = None
    except Exception as e:
        result_data['error_code'] = 1
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # download and read the image in PIL format and prepare it
    try:
        image_b, image_b_time = read_image(image_b_url)
        image_b_key = request_id_diy + '_img_b'
        request_data['image_b_key'] = image_b_key
        db.set(image_b_key, prepare_image(image_b))
        # deal with image a and c
        if request_type == "take_goods":
            image_a, image_a_time = read_image(image_a_url)
            image_a_key = request_id_diy + '_img_a'
            request_data['image_a_key'] = image_a_key
            db.set(image_a_key, prepare_image(image_a))
            # Temporally delete double_abnormal
            # # deal with image_b for double_abnormal
            # image_b2_key = request_id_diy + '_img_b2'
            # request_data['image_b2_key'] = image_b2_key
            # db.set(image_b2_key, prepare_image(image_b, target=(480, 360)))
            # # deal with image_c
            # image_c_key = shelve_id + '_img_c'
            # request_data['image_c_key'] = image_c_key
            # image_c_url_old = db.get(shelve_id)
            # image_c_path = "/var/www/html/ai_server/image_c/"+shelve_id+'.jpg'
            # # check if image_c_url exist in the database
            # if image_c_url_old is None or (image_c_url_old is not None and image_c_url_old != image_c_url):
            #     image_c = read_image(image_c_url)
            #     save_image(image_c, image_c_path)
            #     db.set(shelve_id, image_c_url)
            #     image_c = prepare_image(image_c, target=(480, 360))
            # else:
            #     image_c = prepare_image(read_image("file://"+image_c_path, is_oss=False), target=(480, 360))
            # db.set(image_c_key, image_c)
        else:
            image_a_key = None
            image_a_time = 0
            # Temporally delete double_abnormal
            # image_c_key = None
            # image_b2_key = None

    except Exception as e:
        result_data['error_code'] = 2
        result_data['error_description'] = str(e)
        return flask.jsonify(result_data)
    # put into the queue
    if request_type == 'take_goods':
        task_queue = random.choice(settings.QUEUE_GPUS_R1)
    elif request_type == 'renew_goods_abnormal':
        task_queue = random.choice(settings.QUEUE_GPUS_R2)
    elif request_type == 'renew_goods_count':
        task_queue = random.choice(settings.QUEUE_GPUS_R3)
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
        out = db.get(request_id_diy)
        get_count += 1
        # check to see if our model has done the prediction
        if out is not None:
            # add the output predictions to our result_data
            # dictionary so we can return it to the client
            out = pickle.loads(out)
            if request_type == "take_goods":
                # single abnormal
                pred_single_abnormal = out["single_abnormal"]
                if pred_single_abnormal < settings.Threshold_Single_Abnormal:
                    # normal
                    result_data["predict_result"]["single_abnormal"] = 0
                    result_data["predict_result"]["single_abnormal_prob"] = "%.3f" % (1 - pred_single_abnormal)
                else:
                    # abnormal
                    result_data["predict_result"]["single_abnormal"] = 1
                    result_data["predict_result"]["single_abnormal_prob"] = "%.3f" % pred_single_abnormal
                # Temporally delete double_abnormal
                # double abnormal
                # pred_double_abnormal = out["double_abnormal"]
                # if pred_double_abnormal < settings.Threshold_Double_Abnormal:
                #     # normal
                #     result_data["predict_result"]["double_abnormal"] = 0
                #     result_data["predict_result"]["double_abnormal_prob"] = "%.3f" % (1 - pred_double_abnormal)
                # else:
                #     # abnormal
                #     result_data["predict_result"]["double_abnormal"] = 1
                #     result_data["predict_result"]["double_abnormal_prob"] = "%.3f" % pred_double_abnormal
                # count
                pred_count_a = out["single_count_a"]
                pred_count_b = out["single_count_b"]
                count_a, prob_a, pred_a = postprocess_v3(pred_count_a, column_lc)
                count_b, prob_b, pred_b = postprocess_v3(pred_count_b, column_lc)
                result_data["predict_result"]["a_count_pred"] = json.dumps(pred_a.tolist())
                result_data["predict_result"]["b_count_pred"] = json.dumps(pred_b.tolist())
                sub_count, sub_prob, sub_min_prob = sub_cunt_postprocess(pred_a, pred_b)
                result_data["predict_result"]["sub_count"] = json.dumps(sub_count.tolist())
                result_data["predict_result"]["sub_prob"] = json.dumps(sub_prob.tolist())
                result_data["predict_result"]["sub_count_prob"] = "%.3f" % sub_min_prob
                # store b count
                result_data["predict_result"]["count"] = json.dumps(count_b.tolist())
                result_data["predict_result"]["count_prob"] = "%.3f" % prob_b.min()
                # init double_abnormal
                result_data["predict_result"]["double_abnormal"] = 0
                result_data["predict_result"]["double_abnormal_prob"] = "%.1f" % 0.5
                # check_empty
                if check_empty is None:
                    check_empty, check_flag = compare_empty(count_a, count_b, sub_prob)
                else:
                    check_empty, check_flag = compare_empty(pickle.loads(check_empty), count_b, sub_prob)
                db.set(check_empty_key, pickle.dumps(check_empty))
                result_data["predict_result"]["check_empty"] = json.dumps(check_empty.tolist())
                if check_flag:
                    result_data["predict_result"]["double_abnormal"] = 1
                    result_data["predict_result"]["double_abnormal_prob"] = "%.1f" % 1.0
                # save result
                if pred_single_abnormal < 0.5:
                    # normal
                    result_data["save_result"]["single_abnormal"] = 0
                    result_data["save_result"]["single_abnormal_prob"] = "%.3f" % (1 - pred_single_abnormal)
                    # save result
                    # Temporally delete double_abnormal
                    # result_data["save_result"]["double_abnormal"] = 0
                    # result_data["save_result"]["double_abnormal_prob"] = "%.1f" % 0.5
                    # save result
                    result_data["save_result"]["count_a"] = json.dumps(count_a.tolist())
                    result_data["save_result"]["count_a_prob"] = "%.3f" % prob_a.min()
                    result_data["save_result"]["count_b"] = json.dumps(count_b.tolist())
                    result_data["save_result"]["count_b_prob"] = "%.3f" % prob_b.min()
                else:
                    # abnormal
                    result_data["save_result"]["single_abnormal"] = 1
                    result_data["save_result"]["single_abnormal_prob"] = "%.3f" % pred_single_abnormal
            elif request_type == "renew_goods_abnormal":
                pred_single_abnormal = out["single_abnormal"]
                if pred_single_abnormal < settings.Threshold_Single_Abnormal:
                    # normal
                    result_data["predict_result"]["single_abnormal"] = 0
                    result_data["predict_result"]["single_abnormal_prob"] = "%.3f" % (1 - pred_single_abnormal)
                else:
                    # abnormal
                    result_data["predict_result"]["single_abnormal"] = 1
                    result_data["predict_result"]["single_abnormal_prob"] = "%.3f" % pred_single_abnormal
                # save result
                if pred_single_abnormal < 0.5:
                    # normal
                    result_data["save_result"]["single_abnormal"] = 0
                    result_data["save_result"]["single_abnormal_prob"] = "%.3f" % (1 - pred_single_abnormal)
                else:
                    # abnormal
                    result_data["save_result"]["single_abnormal"] = 1
                    result_data["save_result"]["single_abnormal_prob"] = "%.3f" % pred_single_abnormal
            elif request_type == "renew_goods_count":
                pred_count = out["single_count"]
                count, prob, pred = postprocess_v3(pred_count, column_lc)
                result_data["predict_result"]["count_pred"] = json.dumps(pred.tolist())
                result_data["predict_result"]["count"] = json.dumps(count.tolist())
                result_data["predict_result"]["count_prob"] = "%.3f" % prob.min()
                result_data["save_result"]["count_b"] = json.dumps(count.tolist())
                result_data["save_result"]["count_b_prob"] = "%.3f" % prob.min()
                # init empty_check
                db.delete(check_empty_key)
            # delete the result from the result_database
            db.delete(request_id_diy)
            # delete image a and b
            db.delete(image_b_key)
            if image_a_key:
                db.delete(image_a_key)
            # Temporally delete double_abnormal
            # if image_c_key:
            #     db.delete(image_c_key)
            # if image_b2_key:
            #     db.delete(image_b2_key)
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
            # delete the result from the result_database
            db.delete(image_b_key)
            if image_a_key:
                db.delete(image_a_key)
            # Temporally delete double_abnormal
            # if image_c_key:
            #     db.delete(image_c_key)
            # if image_b2_key:
            #     db.delete(image_b2_key)
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
