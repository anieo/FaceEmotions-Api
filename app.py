from flask import Flask, request, Response,json
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
from werkzeug.utils import secure_filename
from requests_toolbelt import MultipartEncoder
import os
import subprocess
app = Flask(__name__, static_url_path='/static')
api = Api(app)

class Image(Resource):
    def post(self):
        file= open('static/result.txt', 'w')
        file.write('9')
        file.close()
        #contain the input image from the http request
        image = request.files.to_dict() 
        #Image Input Name
        file_name = image["Input"].filename
        #the directory in which the model start its operation on
        inputFolder = "OSC-Project-EmoRecog/Input/"
        
        #check IF the Input folder exits
        if not os.path.isdir(inputFolder):
            os.mkdir(inputFolder)
        #saves the Input Image in the Input Folder
        image["Input"].save(inputFolder+file_name)
        #change directory because i hate python 2
        mydir = os.getcwd() # would be the MAIN folder
        mydir_tmp = mydir + "/OSC-Project-EmoRecog" # add the testA folder name
        os.chdir(mydir_tmp) # change the current working directory
        
        #call For The Model To start processing the Images
        subprocess.run("python2 Core.py",shell=True)

        #get back to python 2
        os.chdir(mydir) # change the current working directory
        print("Finished and now in "+ os.getcwd())
        #delete The Input Images Form the Imput  Folder
        os.remove(inputFolder+file_name)
        #get the Model Result
        with open('static/result.txt') as text:
            res = text.readline().split()
        if res[0]=='9':
            response = app.response_class(
                status=422,
                mimetype='application/json'
                )
            return response
        #Data to be sent to the App
        fields=({'emotion': res[0],
                   'Data':'static/'+file_name,
                   'status':200})
        response = app.response_class(
            response=json.dumps(fields),
            status=200,
            mimetype='application/json'
            )
        return response





        
    
api.add_resource(Image, '/')


if __name__ == '__main__':
     app.run(port='5002')
