from flask import Flask, request, Response,json
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
from werkzeug.utils import secure_filename
from requests_toolbelt import MultipartEncoder
import os
app = Flask(__name__, static_url_path='/static')
api = Api(app)

class Image(Resource):
    def post(self):
        image = request.files.to_dict() #convert multidict to dict
        print(image.keys())
        file_name = image["Input"].filename
        if not os.path.isdir("images"):
                
            os.mkdir("images")
        image["Input"].save("images/"+file_name)
                
        fields=({'Name': '3',
                   'Data':'static/homer.jpg'})
        response = app.response_class(
            response=json.dumps(fields),
            status=200,
            mimetype='application/json'
            )
        return response





        
    
api.add_resource(Image, '/')


if __name__ == '__main__':
     app.run(port='5002')
