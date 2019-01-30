# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time

# initialize the server URL along with the input data
AI_SERVER_URL = "http://localhost/change_server/change_detect"
IMAGE_A = "http://localhost/change_server/static/a.jpg"
IMAGE_B = "http://localhost/change_server/static/b.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 10
SLEEP_COUNT = 0.0125
# NUM_REQUESTS = 16
# SLEEP_COUNT = 0


def call_predict_endpoint(n):
    # request_id should be identity, below is just a sample
    request_id = str(n)
    # load the input image and construct the payload for the request
    request_data = {"request_id": request_id,  # request_id should be identity
                    "request_type": "change_detect",  # should be change_detect
                    "image_a": IMAGE_A,  # image url before openning door of this action
                    "image_b": IMAGE_B  # image url after closing door of this action
                    }
    # time
    start_time = time.time()
    # submit the request
    try:
        response = requests.post(AI_SERVER_URL, json=request_data, timeout=25)
    except requests.exceptions.ReadTimeout:
        print('[ERROE] thread {} timeout error'.format(n))
        return None
    except requests.exceptions.ConnectionError:
        print('[ERROE] thread {} connection error'.format(n))
        return None
    if not response.status_code == requests.codes.ok:
        print("[ERROE] thread {} reponse fail code: {}".format(n, response.status_code))
        return None
    # time
    run_time = time.time() - start_time
    # read result json
    r = response.json()
    if r is None:
        print('[ERROE] thread {} server model error'.format(n))
        return None
    # ensure the request was sucessful
    if r["request_success"]:
        print("[INFO] thread {} OK, use time {}, time : {}".format(n, str(run_time), r['web_time']))
        print(r["predict_result"])
    # otherwise, the request failed
    else:
        print(
            "[INFO] thread {} FAILED, error code {}, error des: {}".format(n, r['error_code'], r['error_description']))


# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)
