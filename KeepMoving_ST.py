#Importing the libraries

import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import warnings
import pydeck as pdk
import datetime
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts

#Defining some usefull function 
def get_dom(dt): #This function is used to extract the day of the run
    return dt.day

def get_dow(dt): #This function is used to extract the weekday of the run
    return dt.weekday() + 1

def get_hour(dt): #This function is used to extract the hour of the run
    return dt.hour

def get_min(dt): #This function is used to extract the minute of the extracted data
    return dt.minute

def count_rows(rows): #This function is used to count the number of rows of my dataframe
    return len(rows)


def put_in_form_df(df): #This function is used to clear the dataframe and use only the datas that I wanted
    df['loggingTime'] = pd.to_datetime(df['locationTimestamp_since1970'], unit='s') #I convert the logging time into a date
    df['loggingTime'] += pd.DateOffset(hours=2)
    df1 = df[['loggingTime', 'loggingSample','locationLatitude', 'locationLongitude', 'locationAltitude', 'locationSpeed',
        'locationHeadingX', 'locationHeadingY', 'locationHeadingZ', 'accelerometerAccelerationX', 'accelerometerAccelerationY',
        'accelerometerAccelerationZ', 'motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ',
        'pedometerNumberofSteps', 'pedometerAverageActivePace', 'pedometerCurrentPace', 'pedometerCurrentCadence',
        'pedometerDistance', 'pedometerFloorAscended','pedometerFloorDescended']]

    df1['hour']= df1['loggingTime'].map(get_hour)
    df1['jour'] = df1['loggingTime'].map(get_dow)
    df1['minute'] = df1['loggingTime'].map(get_min)
    df1['velocity km/h'] = df1['locationSpeed'] * 3.6
    df1['kilometer'] = (df1['pedometerDistance'].astype(int) / 1000 + 1).astype(int)

    df1['runningMinute'] = (df1['loggingTime'] - df1['loggingTime'].iloc[0])
    df1['runningMinute'] = (df1['runningMinute'].dt.total_seconds()/60).astype(int) + 1
    
    return df1

def define_state(row, newCol):
    if (row['locationSpeed'] < 2.222222222):
        newCol.append(1)
    elif (row['locationSpeed'] < 2.777777778):
        newCol.append(2)
    elif (row['locationSpeed'] < 3.333333333):
        newCol.append(3)
    elif (row['locationSpeed'] < 3.888888889):
        newCol.append(4)
    else:
        newCol.append(5)
        
def choose_color(row):
    if (row['running_state']==1):
        return tuple((94, 175, 1))
    if (row['running_state']==2):
        return tuple((212, 218, 3))
    if (row['running_state']==3):
        return tuple((218, 110, 3))
    if (row['running_state']==4):
        return tuple((240, 82, 3))
    if (row['running_state']==5):
        return tuple((255, 0, 0))

@st.cache
def get_important_info(filename1, filename2, filename3, filename4):
    filenames = []
    filenames.append(filename1)
    filenames.append(filename2)
    filenames.append(filename3)
    filenames.append(filename4)
    
    dfData = pd.DataFrame(columns={'average_speed', 'total_distance', 'total_time', 'average_cadence', 'location', 'hour'})
    for f in filenames:
    
        df = pd.read_csv(f)
        df = put_in_form_df(df)
        if (f==filename1):
            df=df[::4]
        average_speed = df['locationSpeed'].mean() * 3.6
        first = df.iloc[0]
        last = df.iloc[-1,:]
        total_distance = last['pedometerDistance']
        total_time = pd.Timedelta(last['loggingTime'] - first['loggingTime']).seconds / 60.0
        average_cadence = df['pedometerCurrentCadence'].mean()
        location = (df['locationLongitude'].mean(), df['locationLatitude'].mean())
        hour = (df['hour'].mean()).astype(int)
        
        dfData = dfData.append({'average_speed':average_speed, 'total_distance': total_distance, 'total_time' : total_time, 'average_cadence':average_cadence, 'location':location, 'hour':hour}, ignore_index=True)
    return dfData
 
def analyse_run(filename, dfData, weight):
    df = pd.read_csv(filename)
    df = put_in_form_df(df)
    if (filename==filename1):
        df = df[::4]

    average_speed = df['locationSpeed'].mean() * 3.6
    first = df.iloc[0]
    last = df.iloc[-1,:]
    total_distance = last['pedometerDistance']
    total_time = pd.Timedelta(last['loggingTime'] - first['loggingTime']).seconds / 60.0
    average_cadence=df['pedometerCurrentCadence'].mean()

    rankDist = []
    rankSpeed = []
    rankCadence = []
    rankTime = []
    for index,row in dfData.iterrows():
        rankDist.append(dfData['total_distance'][index])
        rankSpeed.append(dfData['average_speed'][index])
        rankCadence.append(dfData['average_cadence'][index])
        rankTime.append(dfData['total_time'][index])

    rankDist = sorted(rankDist)
    rankDist= rankDist[::-1]
    rankSpeed=sorted(rankSpeed)
    rankSpeed=rankSpeed[::-1]
    rankCadence=sorted(rankCadence)
    rankCadence=rankCadence[::-1]
    rankTime=sorted(rankTime)
    rankTime=rankTime[::-1]

    for i in range(len(rankDist)):
        if (total_distance==rankDist[i]):
            posDist=i+1
        if (average_speed==rankSpeed[i]):
            posSpeed=i+1
        if (average_cadence==rankCadence[i]):
            posCadence=i+1
        if (total_time==rankTime[i]):
            posTime=i+1

    dfSummary = pd.DataFrame(columns={'average speed (km/h)', 'total distance (m)', 'total time (min)', 'average cadence (step/sec)'})
    dfSummary = dfSummary.append({'average speed (km/h)': average_speed, 'total distance (m)':total_distance, 'total time (min)':total_time,'average cadence (step/sec)':average_cadence},ignore_index=True)

    dfRanking =pd.DataFrame(columns={'rank in terms of speed', 'rank in terms of distance', 'rank in terms of time', 'rank in terms of cadence'})
    dfRanking = dfRanking.append({'rank in terms of speed': posSpeed, 'rank in terms of distance':posDist, 'rank in terms of time':posTime,'rank in terms of cadence':posCadence},ignore_index=True)   

    st.header("First of all, CONGRATULATIONS !")

    st.markdown("## Main informations of your run")
    st.table(dfSummary)

    st.markdown("## Rank of your run")
    st.table(dfRanking)

    if st.checkbox('Do you want to see more details ?'):
        newCol = []
        dff = df[::6]
        for index,row in dff.iterrows():
            define_state(row,newCol)
            
        dff['running_state'] = newCol

        view_state = pdk.ViewState(latitude=df['locationLatitude'].mean(), longitude=df['locationLongitude'].mean(), zoom=14)

        newPath = []
        state=1
        df3 = pd.DataFrame(columns = {'path','color'})
        for index,row in dff.iterrows():    
            if (state == row['running_state']):
                prev = (row['locationLongitude'], row['locationLatitude'])
                newPath.append((row['locationLongitude'], row['locationLatitude']))  
            else :
                df3 = df3.append({'path':newPath,'color':choose_color(row)},ignore_index=True)
                state = row['running_state']        
                newPath = []
                newPath.append(prev)
                newPath.append((row['locationLongitude'], row['locationLatitude']))
    
        df3 = df3.append({'path':newPath,'color':choose_color(row)},ignore_index=True)

        layer = pdk.Layer(
            type="PathLayer",
            data=df3,
            pickable=True,
            get_color="color",
            width_scale=5,
            width_min_pixels=5,
            get_path="path",
            get_width=3)

        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
        st.markdown("# The plot of your run")
        st.caption('With : green when speed < 8Km/h, yellow when speed between 8Km/h and 10Km/h')
        st.caption('orange when speed between 10Km/h and 12km/h, dark orange when speed between 12Km/h and 14Km/h, red when speed > 14Km/h')
        st.pydeck_chart(r)



        fig = plt.figure(figsize=(7,4))
        ax = fig.gca()
        ax.plot(df['loggingTime'], df['velocity km/h']);
        ax.set_title("Evolution of your pace");
        ax.set_xlabel("Time");
        ax.set_ylabel("Speed in km/h");

        st.pyplot(fig)

        df2 = df.groupby('runningMinute').mean()
        df3 = df.groupby('kilometer').mean()

        st.markdown("## What about your speed ?")
        fig1, axs = plt.subplots(1,2)
        fig1.set_size_inches(18, 8)

        axs[0].bar(df2.index,df2['velocity km/h']);
        axs[0].set_title("Evolution of your velocity every minute");
        axs[0].set_xlabel("minute");
        axs[0].set_ylabel("velocity in km/h");

        axs[1].bar(df3.index, df3['velocity km/h']);
        axs[1].set_title("Evolution of your velocity every kilometer");
        axs[1].set_xlabel("kilometer");
        axs[1].set_ylabel("velocity in km/h");

        st.pyplot(fig1)

        st.header("Let's get interested to the difference in altitude")
        fig2, axs = plt.subplots(2,1, sharex = True);    
        axs[0].plot(df2.index, df2['pedometerFloorAscended'] * 3);
        axs[0].set_title("Metters acsended all along your run");
        axs[0].set_ylabel("metters");

        axs[1].plot(df2.index, df2['pedometerFloorDescended'] * (-3));
        axs[1].set_title("Metters descended all along your run");
        axs[1].set_xlabel("minute");
        axs[1].set_ylabel("metters");

        st.pyplot(fig2)

        fig3 = plt.figure(figsize=(8,5))
        ax = fig3.gca()
        ax.bar(df2.index,df2['pedometerCurrentCadence']);
        ax.set_title("Evolution of your cadence every minute");
        ax.set_xlabel("minute");
        ax.set_ylabel("steps / s");

        st.pyplot(fig3)
        
        df5 = df.drop(columns=['loggingSample', 'jour','hour','motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ','locationSpeed','locationLongitude', 'locationLatitude', 'locationHeadingX', 'locationHeadingY', 'locationHeadingZ', 'accelerometerAccelerationX', 'accelerometerAccelerationY', 'accelerometerAccelerationZ'])
        fig4, ax = plt.subplots()
        sns.heatmap(df5.corr())
        ax.set_title("Let's try to see a correlation between those variables");
        st.write(fig4)

        grade = (average_speed*20/13 + total_distance/1000*20/8 + total_time*20/40 + df['pedometerCurrentCadence'].mean()*20/3 + last['pedometerFloorAscended']*20/8)/5
        grade=((grade*100)//1)/100
        cal = (0.5*weight*(total_distance/1000) + 0.3*weight*average_speed).astype(int)
        st.header("We gave your run the grade of {}/20".format(grade))
        dfCriteria = pd.DataFrame(columns={'Speed', 'Distance', 'Time', 'Cadence','DifferenceAltitude'})
        dfCriteria= dfCriteria.append({'Speed': "20/20 = 13km/h", 'Distance':"20/20 = 8km", 'Time':"20/20 = 37 min",'Cadence':"20/20 = 3 step/sec",'DifferenceAltitude':"20/20 = 25m"},ignore_index=True)
        st.markdown("### The rating criteria are :")
        st.table(dfCriteria)
        st.header("Congratulations, you lost approximately {} calories during this run".format(cal))


        if st.checkbox('See DataFrame'):
            st.write("You can see the dataframe :")
            st.write(df)



st.sidebar.markdown("# Welcome to Keep Moving")

st.sidebar.markdown("## In order to calculate your performance, we need to know your weight")
weight = st.sidebar.slider('Select your weigth', min_value=30, max_value=140)

st.sidebar.markdown("## Which run do you want to see in details ?")
run = st.sidebar.slider('Select the run', min_value=1, max_value=4)

filename1 = 'second-training.csv'
filename2 = 'third-training.csv'
filename3 = 'fourth-training.csv'
filename4 = 'fifth-training.csv'

dfData = get_important_info(filename1, filename2, filename3, filename4)

st.title("Welcome to Keep Moving !");
st.markdown("# Benoit CLAUDIC - September 2021")
st.markdown("https://github.com/BenoitClaudic")
st.image('linkedin-profile.png')
st.image('KeepMoving.png')

st.header("Here is your history of running :")


fig, axs = plt.subplots(2, 2)
fig.set_size_inches(12,12)

axs[0, 0].bar(dfData.index, dfData['average_speed'], width=0.6);
axs[0,0].set_title("Evolution of your velocity");
axs[0,0].set_xlabel('Your different runs');
axs[0,0].set_ylabel('velocity km/h');

axs[0, 1].bar(dfData.index, dfData['total_distance'], width=0.6);
axs[0,1].set_title("Evolution of your total distance");
axs[0,1].set_xlabel('Your different runs');
axs[0,1].set_ylabel('Distance in meters');

axs[1, 0].bar(dfData.index, dfData['total_time'], width=0.6);
axs[1, 0].set_title("Evolution of your total time running");
axs[1,0].set_xlabel('Your different runs');
axs[1,0].set_ylabel('Time in minutes');

axs[1, 1].bar(dfData.index, dfData['average_cadence'], width=0.6);
axs[1, 1].set_title("Evolution of your cadence");
axs[1,1].set_xlabel('Your different runs');
axs[1,1].set_ylabel('Number of steps/sec');

st.pyplot(fig)

if st.checkbox('Do you wanna see other informations ?'):
    st.markdown("## But where do you run ?")
    layer = pdk.Layer(
        "GridLayer", dfData, pickable=True, extruded=True, cell_size=500, elevation_scale=10, get_position="location",
        )
    view_state = pdk.ViewState(latitude=48.95386804630998, longitude=1.2322168507535871, zoom=7, bearing=0, pitch=45)

    r1 = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"},)
    st.pydeck_chart(r1)
    st.markdown("## But when do you run ?")
    fig5 = plt.figure(figsize=(10, 4))
    plt.hist(dfData['hour'])
    st.pyplot(fig5)


#Put checkbox pour voir des autres plot sur tous le résumé des courses (Les heures où je cours, map avec les lieux où je cours le plus)

st.markdown("## Analyse of a specific run")

if (run == 1):
    analyse_run(filename1, dfData, weight)
elif (run ==2):
    analyse_run(filename2, dfData, weight)
elif (run ==3):
    analyse_run(filename3, dfData, weight)
elif (run ==4):
    analyse_run(filename4, dfData, weight)

