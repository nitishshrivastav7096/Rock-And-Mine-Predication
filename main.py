import os
import numpy as np
import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import *

def welcome():
    print("Welcome to Rock vs Mine predication System")
    print("Press ENTER key to proceed")
    input()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file in the directory'
    else:
        return csv_files

def display_and_select(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'....',file_name)
        i+=1
    
    return csv_files[int(input("Select file to create a ML model"))]


def main():
    welcome()
    try:
      csv_files=checkcsv()
      if csv_files=='No csv file in the directory':
          raise FileNotFoundError('No csv file in the directory')
      csv_file=display_and_select(csv_files)
      print(csv_file,' is selected')
      print("Reading csv file")
      print("Creating dataset")
      dataset=pd.read_csv(csv_file,header=None)
      #print(dataset.head())
      #print(dataset.shape)
      #print(dataset.describe())  -->used for statical data
      #print(dataset[60].value_counts)
      print(dataset.groupby(60).mean())
      x=dataset.drop(columns=60,axis=1)
      y=dataset[60]
      s=float(input("Enter the test size "))
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s,stratify=y,random_state=1)
      print("Model i creating")
      linearObject=LogisticRegression()
      linearObject.fit(x_train,y_train)
      print("Modei is created")
      print("Press Enter to Test Model")
      input()

      y_pred=linearObject.predict(x_test)
      training_data_accuracy=accuracy_score(y_pred,y_test)

      
      print("Our Model is %2.2f%% accurat "%(training_data_accuracy*100))

      print("now you can use our Model ")
      print("Enter the details of shonar,separed by comma")
      input_data=list(map(float,input().split()))
      input_data_as_numpy_array=np.asarray(input_data)  #changing input data into numpy
      input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
      
      pred= linearObject.predict(input_data_reshaped)

      root=Tk()
      root.geometry('400x400')
      if(pred=='R'):
          #print("The Object Is Rock")
          un=Label(root,text="The Object Is Rock",font=('algerian',25))
          un.grid(pady=25)
      else:
          un=Label(root,text="The Object Is Mine",font=('algerian',25))
          un.grid(pady=25)
      
      
      
      root.mainloop()
    except  FileNotFoundError:
        print('No csv file in the directory')
        print("Press ENTER key to exit")
        input()
        exit()       

if __name__=="__main__":
      main()
      input()
