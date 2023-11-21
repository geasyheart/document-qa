from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
app = Flask(__name__)


@app.route('/', methods=['POST'])
def infer():
    img = request.files.get('file')
    ocr_res = ocr.ocr(img.read(), cls=True)[0]
    segments = []
    for rst in ocr_res:
        left = min(rst[0][0][0], rst[0][3][0])
        top = min(rst[0][0][-1], rst[0][1][-1])
        width = max(rst[0][1][0], rst[0][2][0]) - min(rst[0][0][0], rst[0][3][0])
        height = max(rst[0][2][-1], rst[0][3][-1]) - min(rst[0][0][-1], rst[0][1][-1])
        segments.append({"bbox": [left, top, left + width, top + height], "text": rst[-1][0]})
    return jsonify(segments)


if __name__ == '__main__':
    app.run()
