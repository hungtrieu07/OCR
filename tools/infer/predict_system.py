import os
import re
import sys
import subprocess
import json 

from dateutil.parser import parse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import time
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = args.drop_score

        self.args = args
        self.crop_image_res_index = 0

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def ocr_left(text_sys, img):
        _, rec_res, _ = text_sys(img)
            
        text_data = [item[0] for item in rec_res]
        
        print(text_data)

        start_keyword = "SOLDTO"
        end_keyword = "PAYMENT"
        extracted_text = []

        for i, text in enumerate(text_data):
            if start_keyword in text:
                extracted_text.append(text)
            elif end_keyword in text:
                extracted_text.append(text)
                if i+1 < len(text_data):
                    extracted_text.append(text_data[i+1])
                break
            elif extracted_text:
                extracted_text.append(text)
        
        new_list = []
        
        for item in extracted_text:
            if ":" in item:
                new_list.extend(item.split(":"))
            elif "." in item:
                new_list.extend(item.split("."))
            else:
                new_list.append(item)
                
        new_list = [item for item in new_list if item != ""]
        
        required_keywords = ["SOLDTO", "SHIPTO", "SHIPMENTTERMS", "CUSTOMERORDERNO", "SALESORDERNO", "SHIPMENT", "PAYMENT"]
        result_dict = {}

        for i, keyword in enumerate(required_keywords):
            if i == len(required_keywords) - 1:
                result_dict[keyword] = " ".join(new_list[new_list.index(keyword)+1:])
            else:
                next_keyword = required_keywords[i+1]
                result_dict[keyword] = " ".join(new_list[new_list.index(keyword)+1:new_list.index(next_keyword)])
        return result_dict
    
    def ocr_right(text_sys, img):
        _, rec_res, _ = text_sys(img)
            
        text_data = [item[0] for item in rec_res]

        start_keyword = "DATE"
        end_keyword = "DISCHARGEPORT"
        extracted_text = []

        for i, text in enumerate(text_data):
            if start_keyword in text:
                extracted_text.append(text)
            elif end_keyword in text:
                extracted_text.append(text)
                if i+1 < len(text_data):
                    extracted_text.append(text_data[i+1])
                break
            elif extracted_text:
                extracted_text.append(text)
        
        new_list = []
        
        for item in extracted_text:
            if ".:" in item:
                new_list.extend(item.split(".:"))
            # elif "." in item:
                # new_list.extend(item.split("."))
            elif ":" in item:
                new_list.extend(item.split(":"))
            else:
                new_list.append(item)
                
        new_list = [item for item in new_list if item != ""]
        
        required_keywords = ["DATE", "INVOICENO", "VESSELNAME", "DEPARTINGON/ABOUT", "LOADINGPORT", "DISCHARGEPORT"]
        result_dict = {}

        for i, keyword in enumerate(required_keywords):
            if i == len(required_keywords) - 1:
                result_dict[keyword] = " ".join(new_list[new_list.index(keyword)+1:])
            else:
                next_keyword = required_keywords[i+1]
                result_dict[keyword] = " ".join(new_list[new_list.index(keyword)+1:new_list.index(next_keyword)])
        return result_dict

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def check_date(str):
    if not is_date(str, fuzzy=False):
        # Use regular expressions to extract the text and date
        match = re.match(r'([A-Z]+)(\d{4}-\d{2}-\d{2}|\d{1,2}/[A-Za-z]+/\d{4})', str)

        if match:
            day = match.group(2)
            
            return day
            
    else:
        return str


# def main2(args):
#     image_file_list = get_image_file_list(".")
#     image_file_list = image_file_list[args.process_id::args.total_process_num]
#     text_sys = TextSystem(args)
#     final_result_dict = {}
#     dict1 = {}
#     dict2 = {}
#     for image in image_file_list:
#         img = cv2.imread(image)
        
#         img_order = ((os.path.splitext(os.path.basename(image))[0]).split("_")[1])
        
#         if img_order == "1":
#             dict1 = TextSystem.ocr_left(text_sys, img)
#         else:
#             dict2 = TextSystem.ocr_right(text_sys, img)
            
#         final_result_dict.update(dict1)
#         final_result_dict.update(dict2)
        
#     # del final_result_dict['SHIPTO']
#     # del final_result_dict['CUSTOMERORDERNO']
#     # del final_result_dict['SALESORDERNO']
#     # del final_result_dict['SHIPMENT']
#     # del final_result_dict['VESSELNAME']
#     # del final_result_dict['DEPARTINGON/ABOUT']
#     # del final_result_dict['DISCHARGEPORT']
    
#     print(final_result_dict)
    
def main(args):
    image_file_list = get_image_file_list("cropped.jpg")
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    for image in image_file_list:
        img = cv2.imread(image)
        bboxes, rec_res, _ = text_sys(img)
    
        text_data = [item[0] for item in rec_res]
        
        print(text_data)
        
        # for text in text_data:
        #     if check_date(text):
        #         print(text)
                
        #         # Regular expression pattern to match strings with - or /
        #         regex_pattern = r'[-/]'

        #         # Using the re.findall() function to find all matches
        #         matches = re.findall(regex_pattern, text)
                
        #         # print(matches)

        bounding_boxes = []

        for arr in bboxes:
            x_min = arr[:, 0].min()
            y_min = arr[:, 1].min()
            x_max = arr[:, 0].max()
            y_max = arr[:, 1].max()
            
            bounding_boxes.append((x_min, y_min, x_max, y_max))
            
        print(bounding_boxes)
        
        # box_text = []
        
        # date_pattern = r'\d{4}-\d{2}-\d{2}'

        # for text, box in zip(text_data, bounding_boxes):
        #     if re.search(date_pattern, text):
        #         box_text.append(f"{text}, {box}")
        
        # # Create a list to store bounding boxes on the left and right sides
        # left_side_boxes = []
        # right_side_boxes = []

        # # Define the center of the image
        # image_width = img.shape[1]
        # center_x = image_width // 2

        # # Iterate through the bounding boxes
        # for box in box_text:
        #     # Use regular expressions to extract the text and bounding box
        #     if check_date(box):
        #         text = box

        #     x_min = float(re.search(r'\((\d+\.\d+),', box).group(1))

        #     if x_min < center_x:
        #         left_side_boxes.append(text)
        #     else:
        #         right_side_boxes.append(text)

        # total_prepaid = {"Total Prepaid": None}
        # date_of_issue = {"Place and date of issue": None}
        
        # # Print or process the bounding boxes on the left and right sides
        # for box in left_side_boxes:
        #     total_prepaid["Total Prepaid"] = check_date(box)

        # for box in right_side_boxes:
        #     date_of_issue["Place and date of issue"] = check_date(box)
        
        # result = {}
        # result.update(total_prepaid)
        # result.update(date_of_issue)
        # print(result)
    
    # sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    
    
    # Draw bounding boxes on the image
    for box, text in zip(bounding_boxes, text_data):
        x_min, y_min, x_max, y_max = box
        print(box, text)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 1)  # You can adjust color and thickness

    # Save or display the image with bounding boxes
    cv2.imwrite('image_with_bboxes.jpg', img)
        
# def main(args):
#     image_file_list = get_image_file_list(args.image_dir)
#     image_file_list = image_file_list[args.process_id::args.total_process_num]
#     text_sys = TextSystem(args)
#     is_visualize = True
#     font_path = args.vis_font_path
#     drop_score = args.drop_score
#     draw_img_save_dir = args.draw_img_save_dir
#     os.makedirs(draw_img_save_dir, exist_ok=True)
#     save_results = []

#     logger.info(
#         "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
#         "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
#     )

#     # warm up 10 times
#     if args.warmup:
#         img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
#         for i in range(10):
#             res = text_sys(img)

#     total_time = 0
#     cpu_mem, gpu_mem, gpu_util = 0, 0, 0
#     _st = time.time()
#     count = 0
#     for idx, image_file in enumerate(image_file_list):

#         img, flag_gif, flag_pdf = check_and_read(image_file)
#         if not flag_gif and not flag_pdf:
#             img = cv2.imread(image_file)
#         if not flag_pdf:
#             if img is None:
#                 logger.debug("error in loading image:{}".format(image_file))
#                 continue
#             imgs = [img]
#         else:
#             page_num = args.page_num
#             if page_num > len(img) or page_num == 0:
#                 page_num = len(img)
#             imgs = img[:page_num]
#         for index, img in enumerate(imgs):
#             starttime = time.time()
#             dt_boxes, rec_res, time_dict = text_sys(img)
#             elapse = time.time() - starttime
#             total_time += elapse
#             if len(imgs) > 1:
#                 logger.debug(
#                     str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
#                     % (image_file, elapse))
#             else:
#                 logger.debug(
#                     str(idx) + "  Predict time of %s: %.3fs" % (image_file,
#                                                                 elapse))
#             for text, score in rec_res:
#                 logger.debug("{}, {:.3f}".format(text, score))

#             res = [{
#                 "transcription": rec_res[i][0],
#                 "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
#             } for i in range(len(dt_boxes))]
#             if len(imgs) > 1:
#                 save_pred = os.path.basename(image_file) + '_' + str(
#                     index) + "\t" + json.dumps(
#                         res, ensure_ascii=False) + "\n"
#             else:
#                 save_pred = os.path.basename(image_file) + "\t" + json.dumps(
#                     res, ensure_ascii=False) + "\n"
#             save_results.append(save_pred)

#             if is_visualize:
#                 image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 boxes = dt_boxes
#                 txts = [rec_res[i][0] for i in range(len(rec_res))]
#                 scores = [rec_res[i][1] for i in range(len(rec_res))]

#                 draw_img = draw_ocr_box_txt(
#                     image,
#                     boxes,
#                     txts,
#                     scores,
#                     drop_score=drop_score,
#                     font_path=font_path)
#                 if flag_gif:
#                     save_file = image_file[:-3] + "png"
#                 elif flag_pdf:
#                     save_file = image_file.replace('.pdf',
#                                                    '_' + str(index) + '.png')
#                 else:
#                     save_file = image_file
#                 cv2.imwrite(
#                     os.path.join(draw_img_save_dir,
#                                  os.path.basename(save_file)),
#                     draw_img[:, :, ::-1])
#                 logger.debug("The visualized image saved in {}".format(
#                     os.path.join(draw_img_save_dir, os.path.basename(
#                         save_file))))

#     logger.info("The predict total time is {}".format(time.time() - _st))
#     if args.benchmark:
#         text_sys.text_detector.autolog.report()
#         text_sys.text_recognizer.autolog.report()

#     with open(
#             os.path.join(draw_img_save_dir, "system_results.txt"),
#             'w',
#             encoding='utf-8') as f:
#         f.writelines(save_results)


if __name__ == "__main__":
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
