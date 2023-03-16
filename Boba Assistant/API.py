from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/myapi', methods=['POST'])
def myapi():
    data = request.json
    # Process the data here and generate a response
    # response = "Hello, " + data['name']
    return "พี่ Mos สุดตึง"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)