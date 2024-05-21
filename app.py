from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
app = Flask(__name__)
video_camera = None
global_frame = None

@app.route("/")
#定義方法 **用jinjia2引擎來渲染頁面**，并回傳一個index.html頁面
def root():
    return render_template("tt.html")

#app的路由地址"/submit"即為ajax中定義的url地址，采用POST、GET方法均可提交
@app.route("/submit",methods=["GET", "POST"])
#從這里定義具體的函式 回傳值均為json格式
def submit():
    #由于POST、GET獲取資料的方式不同，需要使用if陳述句進行判斷
    if request.method == "POST":
        # 從前端拿數據
        name = request.form.get("name")
        age = request.form.get("age")
    if request.method == "GET":
        name = request.args.get("name")
        age = request.args.get("age")
    #如果獲取的資料為空
    if len(name) == 0 or len(age) == 0:
        # 回傳的形式為 json
        return {'message':"error!"}
    else:
        return {'message':"success!",'name':name,'age':age}
#往html找















@app.route('/register', methods=['GET','POST'])
def register():
    return
    print (request.headers)
    print (request.form)
    print (request.form['name'])
    print (request.form.get('name'))
    print (request.form.getlist('name'))
    print (request.form.get('nickname', default='little apple'))
    return 'welcome'

@app.route('/get/<name>', methods=['GET'])
def queryDataMessageByName(name):
    print("type(name) : ", type(name))
    return 'String => {}'.format(name)
@app.route("/home")
def index():
    return ('87')

@app.route('/v')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
app.run(host = '0.0.0.0',debug=True) #才可以讓其他PC連線。
# app.run(host="localhost",port=8000,debug=True)
