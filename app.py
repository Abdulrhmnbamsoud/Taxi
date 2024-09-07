import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO, solutions

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


model = YOLO("C:\\Users\\abdau\\OneDrive\\Desktop\\c\\custom_yolov8_model.pt")
line_points = [(100, 400), (1200, 400)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
  
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"Uploaded file saved at: {filepath}")  
        
        processed_path = process_video(filepath, file.filename)
        
        return redirect(url_for('download_file', filename=os.path.basename(processed_path)))

def process_video(filepath, filename):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{filename}")
    video_writer = cv2.VideoWriter(processed_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Processing video: {filepath}...")  
    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=line_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )
    speed_obj = solutions.SpeedEstimator(
        reg_pts=line_points,
        names=model.names,
        view_img=False,
    )
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        tracks = model.track(im0, persist=True, show=False)

     
        im0 = counter.start_counting(im0, tracks)
        im0 = speed_obj.estimate_speed(im0, tracks)
        
    
        for track in tracks:
            if 'speed' in track:
                x1, y1, x2, y2 = track['bbox']  
                speed = track['speed']
                cv2.putText(im0, f"Speed: {speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        video_writer.write(im0)


    cap.release()
    video_writer.release()

    print(f"Processed video saved at: {processed_filename}")  
    return processed_filename
@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
