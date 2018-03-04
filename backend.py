#!/usr/bin/python
# -*- coding: utf-8 -*-

import fcntl
import os
import os.path
import threading
import base64
from PIL import Image
import io
import numpy as np
import re
import time

import tornado.web
import tornado.ioloop
import tornado.options
import tornado.httpserver
import json
import collections
from tornado.options import define, options

from skimage.transform import resize
import skimage
import skimage.io

from eye_predictor import net, mean_face, mean_right, mean_left
from face_detector import face_detector, face_predictor


DEBUG = False
MONGO_DBNAME = 'test'

# deque to smooth the movement of the mouse
K = 1 # 1 means smooth disabled
dq = collections.deque()
for i in range(K):
    dq.append((0.0, 0.0)) # the output of model, rather than point on the screen


class APIHandler(tornado.web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(APIHandler, self).__init__(app, request, **kwargs)
        self.data = None


class GetHandler(APIHandler):
    @tornado.web.asynchronous
    def get(self):
        self.write('fuck')


class PostHandler(APIHandler):
    def get(self):
        self.write('Joint Diagnosis dev') # for dev  version
        # self.render("index.html") # for official version

    def post(self):
        resp = self.predict()
        # return a json data with x, y as str(float)
        print(json.dumps(resp))
        self.write(json.dumps(resp))

    def predict(self):
        st = time.time()
        # to get image and other ancillary data from web, web need to use ajax
        data = self.request.arguments
        # print(type(self.request.arguments))
        
        # TEST CODE
        print('Data received:')
        for key in list(data.keys()):
            print('key: %s , value: %s' % (key, data[key][0][:20]))
        
        user_id = data['user_id'][0]
        # make game display fit the resolution and F11 to fillscreen
        resolution = data['resolution'][0]  # screen resolution
        device = data['device'][0]  # device name

        px = data['point_x'][0]  # x center of target
        py = data['point_y'][0]  # y center of target
        delay = data['delay'][0]  # how much time has elapsed(in ms)
        base64_str = data['base64_image'][0]  # image in base64 format

        screen_width, screen_height = map(int, resolution.split('x'))
        delay = int(delay)

        # maybe too slow using re
        # print(type(base64_str))
        img_str = re.search('base64,(.+)', base64_str).group(1)
        # print(img_str)
        
        # default initial coordinate
        resp = {}
        resp['x'] = str(screen_width / 2)
        resp['y'] = str(screen_height / 2)

        # convert base64 data to img
        byte_content = base64.b64decode(img_str)
        # may be slow
        img = Image.open(io.BytesIO(byte_content))
        # pay attention to whether it needs flop
        # flop, if needed
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # img.show() # show image
        img = np.array(img)
        # print(img, img.shape)
        
        # pay attention to the values in the images
        # 0-255
        
        # face detection and crop using landmarks
        dets = face_detector(img, 1)
        # TEST CODE
        print('%d Faces are detected.' % len(dets))
        if len(dets) == 1:
            det = dets[0]
            height, width = img.shape[0], img.shape[1]
            # crop face
            face_img = img[det.top():det.bottom(), det.left():det.right()]
            # get landmarks
            shape = face_predictor(img, det)
            # crop right eye
            rightx_min = width
            righty_min = height
            rightx_max = 0
            righty_max = 0
            # 37-42(index starts with 1), while index of array starts with 0
            for i in range(36, 42):
                rightx_min = min(rightx_min, shape.part(i).x)
                righty_min = min(righty_min, shape.part(i).y)
                rightx_max = max(rightx_max, shape.part(i).x)
                righty_max = max(righty_max, shape.part(i).y)
            
            # correction for eyes
            # !!!! may need to be changed
            d_x = 5
            d_y = 9
            
            base_dir = './tmp' # save for test
            # make sure index will not be out of range
            righteye_img = img[righty_min - d_y:righty_max + d_y, rightx_min - d_x: rightx_max + d_x]
            # TEST CODE
            skimage.io.imsave(base_dir + '/test_righteye.jpg', righteye_img) # save for test

            # crop left eye
            leftx_min = width
            lefty_min = height
            leftx_max = 0
            lefty_max = 0
            # 43-48
            for i in range(42, 48):
                leftx_min = min(leftx_min, shape.part(i).x)
                lefty_min = min(lefty_min, shape.part(i).y)
                leftx_max = max(leftx_max, shape.part(i).x)
                lefty_max = max(lefty_max, shape.part(i).y)

            # make sure index will not be out of range
            lefteye_img = img[lefty_min - d_y:lefty_max + d_y, leftx_min - d_x: leftx_max + d_x]
            # TEST CODE
            skimage.io.imsave(base_dir + '/test_lefteye.jpg', lefteye_img) # save for test

            # resize face, lefteye, righteye
            r_face = resize(face_img, (224, 224), preserve_range=True)
            r_righteye = resize(righteye_img, (224, 224), preserve_range=True)
            r_lefteye = resize(lefteye_img, (224, 224), preserve_range=True)
            # TEST CODE, now range(float): 0-255
            skimage.io.imsave(base_dir + '/r_face.jpg', r_face.astype(np.uint8)) # save for test
            # need convert??

            # swap axes & reshape face, lefteye, righteye
            r_face = np.transpose(r_face, (2, 0, 1))
            face = r_face.reshape(1, 3, 224, 224)

            r_lefteye = np.transpose(r_lefteye, (2, 0, 1))
            lefteye = r_lefteye.reshape(1, 3, 224, 224)
            
            r_righteye = np.transpose(r_righteye, (2, 0, 1))
            righteye = r_righteye.reshape(1, 3, 224, 224)

            # facegrid
            min_x = int(1.0 * det.left() / width * 25)
            max_x = int(1.0 * det.right() / width * 25)
            min_y = int(1.0 * det.top() / height * 25)
            max_y = int(1.0 * det.bottom() / height * 25)
            # print(min_x, max_x, min_y, max_y)
            facegrid = [0 for k in range(625)]
            for k in range(625):
                x, y = k % 25, k // 25
                if min_x <= x and x < max_x and min_y <= y and y < max_y:
                    facegrid[k] = 1
            facegrid = np.array(facegrid).reshape(1, 625, 1, 1)


            # eye prediction
            # image_XXX shape: (1, 3, 224, 224), range: 0-255(uint8/float)
            # facegrid shape: (1, 625, 1, 1)
            net.blobs['image_face'].data[...] = face - mean_face
            net.blobs['image_left'].data[...] = lefteye - mean_left
            net.blobs['image_right'].data[...] = righteye - mean_right
            net.blobs['facegrid'].data[...] = facegrid
            out = net.forward()
            x, y = out['fc3'][0][0], out['fc3'][0][1]
            rx, ry = x, y

            # smooth movement
            dq.popleft()
            dq.append((x, y))
            for i in range(K - 1):
                x += dq[i][0] + dq[i][0]
                y += dq[i][1] + dq[i][1]
            x /= K
            y /= K
            
            # print(x, y)
            # print(rx, ry)

            # calculate coordinate according to resolution
            # !!!!!! calibrate using data of mine, MACHENIKE LAPTOP M511
            fx = (x - (-5)) / 11 * screen_width
            fy = -(y - 0) / 7 * screen_height
            resp['x'] = str(min(max(10, fx), screen_width - 10))
            resp['y'] = str(min(max(10, fy), screen_height - 10))

            # TEST CODE
            print('x: %s, y: %s' % (x, y))
            print('fx: %s, fy: %s' % (fx, fy))

            # save data async, if delay >= 500 ms
            if delay >= 500:
                # convert x, y from pixel to cm
                # 15.6 inches = 345mm * 195mm ~= 34cm * 19cm
                # rx =
                #
                #
                #
                t = threading.Thread(target=self.save,
                                     args=(px, py, base64_str, user_id, device))
                t.start()

        ed = time.time()
        # TEST CODE
        print('Time: ', ed - st)

        # return a dict
        return resp

    def save(self, x, y, base64_image_with_header, user_id, device):
        if not os.path.isdir('./pic/%s' % user_id):
            os.mkdir('./pic/%s' % user_id)
        
        json_data = {'user_id': user_id,
                     'device': device,
                     'x': x,
                     'y': y,
                     'base64_image_with_header': base64_image_with_header
                     }
        # save data
        with open('./pic/%s/%s.json' % (user_id, time.clock()), 'w') as op_file:
            fcntl.flock(op_file.fileno(), fcntl.LOCK_EX) # ref: https://www.jianshu.com/p/05491c692e96
            json.dump(json_data, op_file)
        # LOCK released out of the with block

        # !!!!! need a better way to save data, seperate image and other data
        # skimage.io.imsave('./pic/%f-%f.jpg' % (x, y), img)
        print('Image saved successfully. Thead: %s' % threading.currentThread().getName() )



class Application(tornado.web.Application):
    def __init__(self):
        tornado_settings = dict(
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            debug = DEBUG
        )
        handlers = [
            (r'/', GetHandler),
            (r'/get', GetHandler),
            (r'/post', PostHandler)
        ]
        #? structure func, why not super()
        # tornado.web.Application.__init__(self, handlers, **tornado_settings)
        super(Application, self).__init__(handlers, **tornado_settings)


app = Application()


define("port", default=18888, help="run on the given port", type=int)

def main():
    # mkdir
    if not os.path.isdir('./pic'):
        os.mkdir('./pic')
    if not os.path.isdir('./tmp'):
        os.mkdir('./tmp')

    
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(app, ssl_options={
		"certfile": "./ssl_file/server.csr",
		"keyfile": "./ssl_file/server.key",
	})
    http_server.listen(options.port)

    print("Development server is running at https://127.0.0.1:%s/post" % options.port)
    print("Quit the server with Control-C")

    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()


