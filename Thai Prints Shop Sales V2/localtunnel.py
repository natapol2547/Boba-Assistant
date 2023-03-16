import subprocess
import os

# Set working directory to ~/myproject


# Start Flask application from ~/myproject/app.py
# subprocess.Popen(["python", "app.py"])

python_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(python_dir)
# app_dir = python_dir + "/RUN V2.py/"

# print(app_dir)
# Start Flask application
# subprocess.Popen(["python", "RUN V2.py"])

# Create local tunnel for Flask application
subprocess.Popen(["lt", "--port", "5000"])
