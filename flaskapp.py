from flask import Flask, render_template, Response, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from YOLO_Video import video_detection

app = Flask(__name__)

app.config["SECRET_KEY"] = "mkrykhovetskyi"
app.config["UPLOAD_FOLDER"] = "static/files"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=""):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        _, buffer = cv2.imencode(".jpg", detection_)

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        _, buffer = cv2.imencode(".jpg", detection_)

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def home():
    session.clear()
    return render_template("home.html")


@app.route("/detectByWebcam", methods=["GET", "POST"])
def detectByWebcam():
    session.clear()
    return render_template("webcam.html")


@app.route("/detectByVideo", methods=["GET", "POST"])
def detectByVideo():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["UPLOAD_FOLDER"],
                secure_filename(file.filename),
            )
        )

        session["video_path"] = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config["UPLOAD_FOLDER"],
            secure_filename(file.filename),
        )
    return render_template("video.html", form=form)


@app.route("/video")
def video():
    return Response(
        generate_frames(path_x=session.get("video_path", None)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/webcam")
def webcam():
    return Response(
        generate_frames_web(path_x=0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
