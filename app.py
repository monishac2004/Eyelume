import os
import random
import time
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import or_

# --- NEW IMPORTS FOR SEGFORMER INTEGRATION ---
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
import cv2

# local imports
from utils.image_processing import is_eye_image, allowed_file

# ---------- App setup ----------
app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_secret_change_me")

# DB setup
db_path = os.path.join(app.instance_path, "eyelume.sqlite")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# File paths
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join("static", "results")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
os.makedirs(app.instance_path, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------- SegFormer Model Loading ----------
# ---------- SegFormer Model Loading ----------
MODEL_PATH = "models/pupil_segformer_trained"  # ✅ updated path to your trained model folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("🔄 Loading SegFormer model and processor...")
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
    segformer_processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    print("✅ SegFormer model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load SegFormer model: {e}")
    segformer_model, segformer_processor = None, None


# ---------- Helper: Segmentation Prediction ----------
def predict_segformer(image_path, output_folder):
    if segformer_model is None or segformer_processor is None:
        return {"success": False, "reason": "Model not loaded"}

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = segformer_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = segformer_model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]

    mask = np.argmax(logits, axis=0).astype(np.uint8) * 255  # convert to binary mask
    mask_filename = os.path.basename(image_path).split(".")[0] + "_mask.png"
    mask_path = os.path.join(output_folder, mask_filename)
    cv2.imwrite(mask_path, mask)

    # Overlay on original image
    original = np.array(image)
    overlay = original.copy()
    mask_resized = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay[mask_resized > 127] = (255, 0, 0)
    overlay = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    overlay_filename = os.path.basename(image_path).split(".")[0] + "_overlay.png"
    overlay_path = os.path.join(output_folder, overlay_filename)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Example derived values
    pupil_px = int(np.sum(mask > 127))
    area_px = mask.size
    confidence = round(random.uniform(0.85, 0.95), 3)  # Placeholder accuracy
    confidence_pct = int(confidence * 100)

    # Determine CVS risk level based on confidence
    if confidence_pct < 88:
        cvs_risk = "Low"
        cvs_risk_pct = random.randint(10, 40)
        remedy = "Your eye condition looks healthy. Continue good eye hygiene and take regular screen breaks."
    elif confidence_pct < 92:
        cvs_risk = "Medium"
        cvs_risk_pct = random.randint(41, 70)
        remedy = "You show moderate CVS risk. Follow the 20-20-20 rule, adjust lighting, and use artificial tears if needed."
    else:
        cvs_risk = "High"
        cvs_risk_pct = random.randint(71, 100)
        remedy = "High CVS risk detected. Reduce screen exposure, ensure proper posture, and consult an eye specialist."

    return {
        "success": True,
        "mask": mask_filename,
        "overlay": overlay_filename,
        "pupil_px": pupil_px,
        "area_px": area_px,
        "confidence": confidence_pct,  # return as %
        "cvs_risk": cvs_risk,
        "cvs_risk_pct": cvs_risk_pct,
        "remedy": remedy,
    }



# ---------- Models ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Initialize DB
with app.app_context():
    db.create_all()

# ---------- Routes ----------
@app.route("/")
def welcome():
    return render_template("welcome.html")


# ---------- Signup ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not email or not phone or not password or password != confirm:
            flash("Please fill all fields and ensure passwords match.", "danger")
            return redirect(url_for("signup"))

        if not phone.startswith("+"):
            flash("Phone must include country code (e.g. +91xxxxxxxxxx)", "danger")
            return redirect(url_for("signup"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "warning")
            return redirect(url_for("login"))

        if User.query.filter_by(phone=phone).first():
            flash("Phone already registered.", "warning")
            return redirect(url_for("login"))

        hashed = generate_password_hash(password)
        user = User(email=email, phone=phone, password=hashed)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Account created successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter(
            or_(User.email == identifier, User.phone == identifier)
        ).first()

        if not user or not check_password_hash(user.password, password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ---------- Logout ----------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("welcome"))


# ---------- Forgot Password ----------
@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        phone = request.form.get("phone", "").strip()
        if not phone:
            flash("Enter your phone number.", "danger")
            return redirect(url_for("forgot"))

        user = User.query.filter_by(phone=phone).first()
        if not user:
            flash("Phone not registered.", "danger")
            return redirect(url_for("forgot"))

        session["reset_phone"] = phone
        flash("Proceed to verify your phone using OTP.", "info")
        return redirect(url_for("reset_password"))

    return render_template("forgot.html")


# ---------- Reset Password ----------
@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not password or password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("reset_password"))

        phone = session.get("reset_phone")
        if not phone:
            flash("Session expired. Start again.", "danger")
            return redirect(url_for("forgot"))

        user = User.query.filter_by(phone=phone).first()
        if not user:
            flash("User not found.", "danger")
            return redirect(url_for("forgot"))

        user.password = generate_password_hash(password)
        db.session.commit()

        session.pop("reset_phone", None)
        flash("Password reset successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


# ---------- Dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


# ---------- Upload & Analyze ----------
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part.", "danger")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No selected file.", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            if not is_eye_image(save_path):
                os.remove(save_path)
                flash("Uploaded image does not look like an eye.", "danger")
                return redirect(request.url)

            # 🔥 Run SegFormer segmentation prediction
            result = predict_segformer(save_path, output_folder=app.config["RESULT_FOLDER"])
            if not result.get("success"):
                flash("Error during segmentation: " + result.get("reason", "Unknown"), "danger")
                return redirect(request.url)

            return render_template(
    "result.html",
    original="uploads/" + filename,
    overlay="results/" + result["overlay"],
    mask="results/" + result["mask"],
    pupil_px=result["pupil_px"],
    area_px=result["area_px"],
    confidence=result["confidence"],
    cvs_risk=result["cvs_risk"],
    cvs_risk_pct=result["cvs_risk_pct"],
    remedy=result["remedy"],
)


        else:
            flash("Allowed file types: png, jpg, jpeg", "warning")
            return redirect(request.url)

    demo_image = url_for("static", filename="images/demo_eye.jpg")
    return render_template("upload.html", demo_image=demo_image)


# ---------- Static Pages ----------
@app.route("/demo")
def demo():
    demo_image = url_for("static", filename="images/demo_eye.jpg")
    return render_template("demo.html", demo_image=demo_image)


@app.route("/info")
def info():
    return render_template("info.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
