// ================== FILE UPLOAD VALIDATION ==================
const fileInput = document.querySelector('input[type=file]');
if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('Please choose an image file.');
      e.target.value = '';
    }
  });
}

// ================== FIREBASE OTP AUTH ==================

// Load Firebase modules dynamically
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { getAuth, RecaptchaVerifier, signInWithPhoneNumber } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

// ---------- YOUR FIREBASE CONFIG ----------
const firebaseConfig = {
  apiKey: "YOUR_API_KEY_HERE",
  authDomain: "eyelume-aede7.firebaseapp.com",
  projectId: "eyelume-aede7",
  storageBucket: "eyelume-aede7.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID_HERE"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// reCAPTCHA setup
window.recaptchaVerifier = new RecaptchaVerifier(auth, 'recaptcha-container', {
  'size': 'invisible',
  'callback': () => { console.log("reCAPTCHA solved"); },
  'expired-callback': () => { alert("reCAPTCHA expired, try again"); }
});

let confirmationResult;

// Send OTP
const sendOtpBtn = document.getElementById('sendOtpBtn');
if (sendOtpBtn) {
  sendOtpBtn.addEventListener('click', async () => {
    const phone = document.getElementById('phone').value.trim();
    if (!phone.startsWith("+")) {
      alert("Enter phone number with country code (e.g., +91XXXXXXXXXX)");
      return;
    }

    try {
      confirmationResult = await signInWithPhoneNumber(auth, phone, window.recaptchaVerifier);
      document.getElementById("otpSection").style.display = "block";
      document.getElementById("message").innerText = "OTP sent successfully!";
    } catch (error) {
      console.error(error);
      alert("Error sending OTP: " + error.message);
    }
  });
}

// Verify OTP
const verifyOtpBtn = document.getElementById('verifyOtpBtn');
if (verifyOtpBtn) {
  verifyOtpBtn.addEventListener('click', async () => {
    const otp = document.getElementById('otp').value.trim();
    if (!otp) {
      alert("Enter OTP");
      return;
    }

    try {
      await confirmationResult.confirm(otp);
      alert("Phone verified successfully!");
      window.location.href = "/reset_password";  // Flask endpoint
    } catch (error) {
      console.error(error);
      alert("Invalid OTP: " + error.message);
    }
  });
}
