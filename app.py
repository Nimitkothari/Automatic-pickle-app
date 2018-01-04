from flask import Flask,Response,render_template,send_from_directory
from flask import request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import json
import pickle
path = os.getcwd()
print(path)
port = int(os.getenv("PORT", 3000))
upload_folder = path
ALLOWED_EXTENSIONS = set(['pkl','txt','csv'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/upload')
def redirect():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
    try:
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print('file uploaded successfully')
        return 'file uploaded successfully'
    except Exception as e:
        print(e)

@app.route('/pickleOne',methods = ['GET','POST'])
def download():
    try:
        return send_from_directory(path,'predict1.pkl', as_attachment=True)
    except Exception as e:
        print(e)

@app.route('/pickleTwo',methods = ['GET','POST'])
def download_2():
    try:
        return send_from_directory(path,'predict2.pkl', as_attachment=True)
    except Exception as e:
        print(e)

@app.route('/pickleThree',methods = ['GET','POST'])
def download_3():
    try:
        return send_from_directory(path,'predict3.pkl', as_attachment=True)
    except Exception as e:
        print(e)

@app.route('/pickleFour',methods = ['GET','POST'])
def download_4():
    try:
        return send_from_directory(path,'predict4.pkl', as_attachment=True)
    except Exception as e:
        print(e)

@app.route('/pickleFive',methods = ['GET','POST'])
def download_5():
    try:
        return send_from_directory(path,'predict5.pkl', as_attachment=True)
    except Exception as e:
        print(e)


@app.route('/train',methods=['GET'])
def train_data():
    try:
        column_data = pd.read_csv('./columns.csv')
        column_1 = (column_data.columns[0])
        print("column 1", column_1)  #lat
        column_2 = (column_data.columns[1])
        print("column 2", column_2)  #lon
        column_3 = (column_data.columns[2])
        print("column 3", column_3)  # Jan
        column_4 = (column_data.columns[3])
        print("column 4", column_4)  # Dec
        column_5 = (column_data.columns[4])
        print("column 5", column_5)  # Ann

        print("Reading data")
        data = pd.read_csv('./data.csv')
        print("data.shape", data.shape)
        # Splitting Data :
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        # Making copies below:
        train_set_data_1=train_set_data_2=train_set_data_3=train_set_data_4=train_set_data_5=train_set
        test_set_data_1 = test_set_data_2 = test_set_data_3 = test_set_data_4 = test_set_data_5 = test_set
        print("After making training copies")
        lin_reg = LinearRegression()
        print("linear model imported")


        # For 1st variable
        test_set_1 = test_set_data_1.drop([column_1], axis=1)
        print("test_set_after_drop_1", test_set_1)
        train_labels_1 = train_set_data_1[column_1]
        print("train_lables_1", train_labels_1)
        train_set_1 = train_set_data_1.drop([column_1], axis=1)
        print("train_set_after_drop_1", train_set_1)
        lin_reg.fit(train_set_1, train_labels_1)
        print("linear fit done 1")
        pickle.dump(lin_reg, open('predict1.pkl', 'wb'))

        # For 2nd variable
        test_set_2 = test_set_data_2.drop([column_2], axis=1)
        print("test_set_after_drop_2", test_set_2)
        train_labels_2 = train_set_data_2[column_2]
        print("train_lables_2", train_labels_2)
        train_set_2 = train_set_data_2.drop([column_2], axis=1)
        print("train_set_after_drop_2", train_set_2)
        lin_reg.fit(train_set_2, train_labels_2)
        print("linear fit done 2")
        pickle.dump(lin_reg, open('predict2.pkl', 'wb'))

        # For 3rd variable
        test_set_3 = test_set_data_3.drop([column_3], axis=1)
        print("test_set_after_drop_3", test_set_3)
        train_labels_3 = train_set_data_3[column_3]
        print("train_lables_3", train_labels_3)
        train_set_3 = train_set_data_3.drop([column_3], axis=1)
        print("train_set_after_drop_3", train_set_3)
        lin_reg.fit(train_set_3, train_labels_3)
        print("linear fit done 3")
        pickle.dump(lin_reg, open('predict3.pkl', 'wb'))

        # For 4th variable
        test_set_4 = test_set_data_4.drop([column_4], axis=1)
        print("test_set_after_drop_4", test_set_4)
        train_labels_4 = train_set_data_4[column_4]
        print("train_lables_4", train_labels_4)
        train_set_4 = train_set_data_4.drop([column_4], axis=1)
        print("train_set_after_drop_4", train_set_4)
        lin_reg.fit(train_set_4, train_labels_4)
        print("linear fit done 4")
        pickle.dump(lin_reg, open('predict4.pkl', 'wb'))

        # For 5th variable
        test_set_5 = test_set_data_5.drop([column_5], axis=1)
        print("test_set_after_drop_5", test_set_5)
        train_labels_5 = train_set_data_5[column_5]
        print("train_lables_5", train_labels_5)
        train_set_5 = train_set_data_5.drop([column_5], axis=1)
        print("train_set_after_drop_5", train_set_5)
        lin_reg.fit(train_set_5, train_labels_5)
        print("linear fit done 5")
        pickle.dump(lin_reg, open('predict5.pkl', 'wb'))
        result = 'done'
        msg = {
            "Pickle file is": "%s" % (result)
        }
        resp = Response(response=json.dumps(msg),
                        status=200,
                        mimetype="application/json")
        return resp
    except Exception as e:
        print(e)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port)
