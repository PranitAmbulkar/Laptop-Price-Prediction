import streamlit as st 
import pickle
import numpy as np

df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

st.title("Laptop Price Predictor")
company = st.selectbox("Brand",df['Company'].unique(), index=4)
type = st.selectbox("Laptop Type",df['TypeName'].unique() , index=1)
ram = st.selectbox("Ram in GB",[2,4,6,8,12,16,24,32,64,128], index=3)
gpu = st.selectbox("GPU", df['Gpu'].unique(), index=0)
os = st.selectbox("Operating System" ,df['OpSys'].unique() , index=2)
weight = st.slider("Weight of the Laptop in Kg.",min_value=0.65,max_value=5.0,value=2.0,step=0.5)
touchscreen = st.selectbox("Touchscreen", ['Yes','No'] , index=1)
ips = st.selectbox("IPS Display",['Yes','No'] , index=0)
cpu = st.selectbox("Processor", df['CPU'].unique(), index=0)
hdd = st.selectbox("Hard Drive in GB(Select the 0 for only SSD in system)", [0,128,256,512,1024,2048], index=0)
ssd = st.selectbox("SSD size in GB", [0,8,16,32,64,128,180,240,256,512,768,1024,2048] , index=9)
screen_size = st.slider("Screen size (in inches.)",min_value=10.5,max_value=18.5,value=15.6,step=0.1)
resolution = st.selectbox("Screen Resolution", 
["2560x1600","1440x900","1920x1080","2880x1800","1366x768","2304x140","3200x1800","1920x1200","2256x1504",
"3840x2160","2160x1440","2560x1440","1600x900","2736x1824","2400x1600"], index=2)

if st.button("PREDICT PRICE"):
    ppi=None
    if(touchscreen=="Yes"):
        touchscreen=1
    else:
        touchscreen=0
    if(ips=="Yes"):
        ips=1
    else:
        ips=0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi =((X_res**2)+(Y_res**2)**0.5)/screen_size
    query=np.array([[company,type,ram,gpu,os,weight,touchscreen,ips,cpu,hdd,ssd,ppi]])
    op=np.exp(pipe.predict(query))
    st.subheader("The Predicted price of the laptop with the above configuration is ₹"+str(round(op[0])))
